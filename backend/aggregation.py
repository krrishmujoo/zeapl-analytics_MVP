from datetime import datetime
import pandas as pd

_MONTH_ORDER = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Calendar quarter each month belongs to (Jan-Mar = Q1, Apr-Jun = Q2, ...)
_MONTH_TO_QUARTER = {month: (num - 1) // 3 + 1 for month, num in _MONTH_ORDER.items()}


def _month_sort_key(record):

    return _MONTH_ORDER[record["month"].strip().lower()]


# ---------------------------------------------------------------------------
# Scan-activity rollup
#
# Everything below _get_metric_value() in this file assumes one record per
# month with fields like "month"/"revenue"/"units_sold" — the data.json
# shape. fact_scan_activity.csv is raw, one row per scan. This turns the
# raw rows into that same monthly shape (reusing the field names revenue
# and units_sold that METRIC_TO_FIELD already points "revenue" and "sales"
# at), so growth_by_month / total_metric / highest_metric / etc. keep
# working unchanged no matter which data source is live. It's a no-op on
# data that's already monthly-shaped (e.g. data.json itself).
#
# Known limitations, not solved here:
#   - Not year-aware for month-level lookups (same limitation the rest of
#     this file already has for growth_by_month/highest_metric/etc.) — if
#     the dataset ever spans the same month name in two different years,
#     those rows get merged into one bucket. quarterly_growth avoids this
#     via its own year handling; growth_by_month does not.
#   - Only rolls up by month. Breakdowns by region/channel/etc. need a
#     separate rollup (group by month + that column) — this one alone
#     won't feed a "revenue by region" chart correctly.
#   - "profit", "operating_expenses", "active_customers", "churned_customers"
#     have no equivalent in the scan_activity schema, so those metrics will
#     just come back as None (total_metric/average_metric already handle a
#     missing field by returning None rather than raising).
# ---------------------------------------------------------------------------

def rollup_scan_activity_to_monthly(data):
    if not data:
        return data

    first = data[0]
    if "scan_date" not in first or "month" in first:
        return data  # already monthly-shaped (or nothing usable to roll up)

    

    df = pd.DataFrame(data)
    df["scan_date"] = pd.to_datetime(df["scan_date"])

    df["month"] = df["scan_date"].dt.strftime("%B").str.lower()
    df["year"] = df["scan_date"].dt.year
    df["quarter_num"] = (df["scan_date"].dt.month - 1) // 3 + 1

    grouped = df.groupby("month", sort=False).agg(
        revenue=("invoice_value", "sum"),
        units_sold=("quantity", "sum"),
        points_earned=("points_earned", "sum"),
        scan_count=("scan_id", "count"),
        year=("year", "first"),
        quarter_num=("quarter_num", "first"),
    ).reset_index()

    grouped["quarter"] = "Q" + grouped["quarter_num"].astype(str)
    grouped = grouped.drop(columns=["quarter_num"])
    grouped = grouped.sort_values("month", key=lambda col: col.map(_MONTH_ORDER))

    return grouped.to_dict(orient="records")


def _parse_date(record):
    """
    Parses a record's 'date' field (expected format "YYYY-MM-DD") into a
    date object. Returns None if the field is missing or not a valid date,
    so daily/weekly functions can just skip that record instead of crashing.
    """
    date_str = record.get("date")
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Generic, metric-agnostic helpers
#
# Every aggregation used to be hard-coded to "revenue". These versions take
# a `metric` argument (e.g. "revenue" or "profit") so the same function can
# aggregate any numeric field on a record. The original revenue-only
# functions (cal_avg, highest_rev, lowest_rev, total_rev,
# month_over_month_growth) are kept below as thin wrappers around these, so
# nothing that already imports them from this file breaks.
# ---------------------------------------------------------------------------

def _get_metric_value(record, metric):
    """
    Safely reads a numeric field off a record. Returns None if the field is
    missing or isn't a number, rather than raising, so a record missing
    "profit" (for example) is just skipped instead of crashing the whole
    aggregation.
    """
    value = record.get(metric)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def total_metric(data, metric="revenue"):
    """Sum of `metric` across every record that has it. None if none do."""
    if not data:
        return None
    values = [v for v in (_get_metric_value(r, metric) for r in data) if v is not None]
    if not values:
        return None
    return sum(values)


def average_metric(data, metric="revenue"):
    """Average of `metric` across every record that has it. None if none do."""
    if not data:
        return None
    values = [v for v in (_get_metric_value(r, metric) for r in data) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def highest_metric(data, metric="revenue"):
    """Full record with the highest `metric` value, or None."""
    if not data:
        return None
    best = None
    best_val = None
    for record in data:
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        if best_val is None or val > best_val:
            best, best_val = record, val
    return best


def lowest_metric(data, metric="revenue"):
    """Full record with the lowest `metric` value, or None."""
    if not data:
        return None
    worst = None
    worst_val = None
    for record in data:
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        if worst_val is None or val < worst_val:
            worst, worst_val = record, val
    return worst


def growth_by_month(data, metric="revenue"):
    if not data:
        return []

    totals = {}
    for record in data:
        month = record["month"].strip().lower()
        month_num = _MONTH_ORDER[month]
        year = record.get("year", 0)
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        key = (year, month_num)
        totals[key] = totals.get(key, 0) + val

    if not totals:
        return []

    ordered_keys = sorted(totals.keys())
    growth = []
    for prev_key, curr_key in zip(ordered_keys, ordered_keys[1:]):
        prev_val = totals[prev_key]
        curr_val = totals[curr_key]
        pct = None if prev_val == 0 else round((curr_val - prev_val) / prev_val * 100, 2)
        year, month_num = curr_key
        month_name = next(m for m, n in _MONTH_ORDER.items() if n == month_num).capitalize()
        label = f"{month_name} {year}" if year else month_name
        growth.append({"month": label, "growth_pct": pct})
    return growth


def quarterly_growth(data, metric="revenue"):
    """
    Groups monthly records into calendar quarters (Q1-Q4), sums `metric`
    within each quarter, then returns quarter-over-quarter growth.
    Requires a "month" field on every record. An optional "year" field lets
    this order quarters correctly once the dataset spans more than one
    year — without it, quarters are just ordered Q1 -> Q4 assuming a single
    year, which matches the current dataset.
    """
    if not data:
        return []

    # Sum the metric per (year, quarter) bucket.
    totals = {}
    for record in data:
        month = record["month"].strip().lower()
        quarter = _MONTH_TO_QUARTER[month]
        year = record.get("year", 0)
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        key = (year, quarter)
        totals[key] = totals.get(key, 0) + val

    if not totals:
        return []

    ordered_keys = sorted(totals.keys())
    growth = []
    for prev_key, curr_key in zip(ordered_keys, ordered_keys[1:]):
        prev_val = totals[prev_key]
        curr_val = totals[curr_key]
        pct = None if prev_val == 0 else round((curr_val - prev_val) / prev_val * 100, 2)
        year, quarter = curr_key
        label = f"Q{quarter} {year}" if year else f"Q{quarter}"
        growth.append({
            "quarter": label,
            f"{metric}_total": curr_val,
            "growth_pct": pct,
        })
    return growth


def yearly_growth(data, metric="revenue"):
    """
    Groups records by their "year" field and returns year-over-year growth
    for `metric`. Every record needs a "year" field for this to produce
    anything — the current dataset only has "month", so this returns an
    empty list until "year" is added to each record.
    """
    if not data or not all("year" in r for r in data):
        return []

    totals = {}
    for record in data:
        year = record["year"]
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        totals[year] = totals.get(year, 0) + val

    if not totals:
        return []

    years = sorted(totals.keys())
    growth = []
    for prev_year, curr_year in zip(years, years[1:]):
        prev_val = totals[prev_year]
        curr_val = totals[curr_year]
        pct = None if prev_val == 0 else round((curr_val - prev_val) / prev_val * 100, 2)
        growth.append({
            "year": curr_year,
            f"{metric}_total": curr_val,
            "growth_pct": pct,
        })
    return growth


def weekly_growth(data, metric="revenue"):
    """
    Groups records by ISO week (year + week number), sums `metric` per
    week, then returns week-over-week growth. Requires a "date" field
    (format "YYYY-MM-DD") on every record — the current dataset only has
    "month", so this returns an empty list until daily-level records with
    dates are added.
    """
    totals = {}
    for record in data:
        day = _parse_date(record)
        if day is None:
            continue
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        iso_year, iso_week, _ = day.isocalendar()
        key = (iso_year, iso_week)
        totals[key] = totals.get(key, 0) + val

    if not totals:
        return []

    weeks = sorted(totals.keys())
    growth = []
    for prev_key, curr_key in zip(weeks, weeks[1:]):
        prev_val = totals[prev_key]
        curr_val = totals[curr_key]
        pct = None if prev_val == 0 else round((curr_val - prev_val) / prev_val * 100, 2)
        year, week = curr_key
        growth.append({
            "week": f"{year}-W{week:02d}",
            f"{metric}_total": curr_val,
            "growth_pct": pct,
        })
    return growth


def daily_growth(data, metric="revenue"):
    """
    Day-over-day growth for `metric`. Requires a "date" field
    (format "YYYY-MM-DD") on every record — the current dataset only has
    "month", so this returns an empty list until daily-level records with
    dates are added.
    """
    dated = []
    for record in data:
        day = _parse_date(record)
        if day is None:
            continue
        val = _get_metric_value(record, metric)
        if val is None:
            continue
        dated.append((day, val))

    if not dated:
        return []

    dated.sort(key=lambda pair: pair[0])
    growth = []
    for (prev_day, prev_val), (curr_day, curr_val) in zip(dated, dated[1:]):
        pct = None if prev_val == 0 else round((curr_val - prev_val) / prev_val * 100, 2)
        growth.append({
            "date": curr_day.isoformat(),
            f"{metric}_total": curr_val,
            "growth_pct": pct,
        })
    return growth


# ---------------------------------------------------------------------------
# Profit — same shape as the existing revenue functions, just pointed at
# the "profit" field instead. Returns None / empty results until records
# in the dataset actually include a "profit" field.
# ---------------------------------------------------------------------------

def total_profit(data):
    return total_metric(data, "profit")


def average_profit(data):
    return average_metric(data, "profit")


def highest_profit(data):
    record = highest_metric(data, "profit")
    return record["profit"] if record else None


def lowest_profit(data):
    return lowest_metric(data, "profit")


def monthly_profit_growth(data):
    return growth_by_month(data, metric="profit")

def churn_rate(data):
    total_churned = total_metric(data, "churned_customers")
    total_active = total_metric(data, "active_customers")
    if total_churned is None or total_active is None or total_active == 0:
        return None
    return round(total_churned / total_active * 100, 2)

def group_totals(data, metric, dimension):
    groups = {}
    for record in data:
        key = record.get(dimension)
        if key is None:
            continue
        value = _get_metric_value(record, metric)
        if value is None:
            continue
        groups[key] = groups.get(key, 0) + value
    return groups

def correlation_coefficient(data, metric_a, metric_b):
    pairs = []
    for record in data:
        a = _get_metric_value(record, metric_a)
        b = _get_metric_value(record, metric_b)
        if a is not None and b is not None:
            pairs.append((a, b))

    n = len(pairs)
    if n < 2:
        return None

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = (variance_x * variance_y) ** 0.5

    if denominator == 0:
        return None

    return round(covariance / denominator, 4)


def profit_margin(data):
    """
    Overall profit margin as a percentage: total profit / total revenue * 100.
    None if either total can't be computed (e.g. no "profit" field yet).
    """
    total_p = total_metric(data, "profit")
    total_r = total_metric(data, "revenue")
    if total_p is None or total_r is None or total_r == 0:
        return None
    return round(total_p / total_r * 100, 2)


def rank_users(
    data,
    metric,
    n=5,
    ascending=False,
    area_field=None,
    area_value=None,
):
    df = pd.DataFrame(data)

    if area_field and area_value:
        df = df[
            df[area_field]
            .astype(str)
            .str.lower()
            == area_value.lower()
        ]

    grouped = (
        df.groupby("user_id")[metric]
        .sum()
        .sort_values(ascending=ascending)
        .head(n)
        .reset_index()
    )

    return grouped.to_dict(orient="records")

def top_users(
    data,
    metric,
    top_n=5,
    area_field=None,
    area_value=None,
):
    return rank_users(
        data,
        metric,
        n=top_n,
        ascending=False,
        area_field=area_field,
        area_value=area_value,
    )


def bottom_users(
    data,
    metric,
    top_n=5,
    area_field=None,
    area_value=None,
):
    return rank_users(
        data,
        metric,
        n=top_n,
        ascending=True,
        area_field=area_field,
        area_value=area_value,
    )

def unique_users(
    data,
    area_field=None,
    area_value=None,
):
    df = pd.DataFrame(data)

    if area_field and area_value:
        df = df[
            df[area_field]
            .astype(str)
            .str.lower()
            == area_value.lower()
        ]

    return int(df["user_id"].nunique())

def distinct_count(data, field):
    """
    Generic distinct count for any real dataset field.
    Does not assume a specific column such as user_id or product_id.
    """
    if not data:
        return 0

    if not field:
        raise ValueError("distinct_count requires a field")

    df = pd.DataFrame(data)

    if field not in df.columns:
        raise ValueError(
            f"Field '{field}' does not exist in the dataset"
        )

    return int(df[field].dropna().nunique())
# ---------------------------------------------------------------------------
# Multi-metric aggregation
#
# Lets a caller ask for several metrics at once (e.g. revenue AND profit)
# and get totals/average/highest/lowest for each back in one call, instead
# of only ever being able to aggregate a single metric at a time.
# ---------------------------------------------------------------------------

def summarize_metrics(data, metrics=("revenue",)):
    """
    Computes total, average, highest, and lowest for each metric in
    `metrics`. Example: summarize_metrics(data, ["revenue", "profit"])
    returns a dict keyed by metric name, each holding its own summary —
    used to answer questions that reference more than one metric at once
    (e.g. "show revenue and profit for Q1") so both can be aggregated (and
    later charted) together.
    """
    summary = {}
    for metric in metrics:
        summary[metric] = {
            "total": total_metric(data, metric),
            "average": average_metric(data, metric),
            "highest": highest_metric(data, metric),
            "lowest": lowest_metric(data, metric),
        }
    return summary




# ---------------------------------------------------------------------------
# Original revenue-only functions, kept as-is for backward compatibility —
# anything already importing these from app.py / LLM.py keeps working
# unchanged. Internally they now just call the generic metric versions.
# ---------------------------------------------------------------------------

def month_over_month_growth(data):
    return growth_by_month(data, metric="revenue")


def total_rev(data):
    return total_metric(data, "revenue")


def lowest_rev(data):
    return lowest_metric(data, "revenue")


def cal_avg(data):
    return average_metric(data, "revenue")


def highest_rev(data):
    record = highest_metric(data, "revenue")
    return record["revenue"] if record else None


def get_quarterly_growth(data):
    quarterly = {}

    for item in data:
        quarter = item["quarter"]

        if quarter not in quarterly:
            quarterly[quarter] = {
                "quarter": quarter,
                "revenue": 0,
                "cost_of_product": 0,
                "profit": 0
            }

        quarterly[quarter]["revenue"] += item.get("revenue", 0)
        quarterly[quarter]["cost_of_product"] += item.get("cost_of_product", 0)
        quarterly[quarter]["profit"] += item.get("profit", 0)

    quarter_order = ["Q1", "Q2", "Q3", "Q4"]
    result = []

    previous_revenue = None

    for quarter in quarter_order:
        if quarter in quarterly:
            current = quarterly[quarter]

            if previous_revenue is None:
                growth_pct = None
            else:
                growth_pct = round(
                    ((current["revenue"] - previous_revenue) / previous_revenue) * 100,
                    2
                )

            current["quarterly_growth_pct"] = growth_pct
            result.append(current)

            previous_revenue = current["revenue"]

    return result