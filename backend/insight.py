

from aggregation import (
    _MONTH_ORDER,
    total_metric,
    average_metric,
    highest_metric,
    lowest_metric,
    growth_by_month,
    quarterly_growth,
    yearly_growth,
    weekly_growth,
    daily_growth,
    churn_rate,
    correlation_coefficient,
    profit_margin,
    group_totals,
    summarize_metrics,
)
from request_analyser import (
    extract_metrics,
    extract_operations,
    extract_dimension,
    choose_primary_operation,
    determine_needs_llm,
)
from registries import METRIC_TO_FIELD

# ---------------------------------------------------------------------------
# request_analyser.extract_metrics() returns broad categories (from
# registries.Metrics) meant for natural-language matching -- "revenue",
# "profit", "expenses", "customers", "sales", "churn" -- not necessarily
# literal field names on a record. This maps each category down to the
# one dataset field that best represents it, so the aggregation.py
# functions (which operate on literal field names) can be called directly.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Month scoping -- not handled anywhere else in the project yet, so it
# lives here. Reuses aggregation._MONTH_ORDER as the canonical month list
# rather than redefining month names.
# ---------------------------------------------------------------------------

def _extract_requested_months(question):
    """
    Scans a question for month names, with simple range support
    ("January to March" expands to every month in between). Returns a
    list of lowercase month names in chronological order, or [] if none
    were mentioned.
    """
    lower = question.lower()
    found = [m for m in _MONTH_ORDER if m in lower]
    if not found:
        return []

    if len(found) >= 2 and any(word in lower for word in (" to ", " through ", " until ", "-")):
        indices = sorted(_MONTH_ORDER[m] for m in found)
        start, end = indices[0], indices[-1]
        return [m for m, n in sorted(_MONTH_ORDER.items(), key=lambda kv: kv[1]) if start <= n <= end]

    return sorted(found, key=lambda m: _MONTH_ORDER[m])


def _filter_by_months(data, months):
    """Records whose "month" is in `months` (case-insensitive). Returns `data` unchanged if `months` is empty."""
    if not months or not data:
        return data
    wanted = {m.strip().lower() for m in months}
    matched = [r for r in data if str(r.get("month", "")).strip().lower() in wanted]
    return matched if matched else data


# ---------------------------------------------------------------------------
# Metric-agnostic single-fact insights
# ---------------------------------------------------------------------------

def total_insight(data, metric="revenue"):
    """Hardcoded insight for "what was total <metric>" style questions."""
    total = total_metric(data, metric)
    if total is None:
        return None
    return {
        "metric": metric,
        "value": total,
        "summary": f"Total {metric.replace('_', ' ')} across the dataset was {total:,.2f}.",
    }


def average_insight(data, metric="revenue"):
    """Hardcoded insight for "what's the average <metric>" style questions."""
    avg = average_metric(data, metric)
    if avg is None:
        return None
    return {
        "metric": metric,
        "value": avg,
        "summary": f"Average {metric.replace('_', ' ')} across the dataset was {avg:,.2f}.",
    }


def highest_insight(data, metric="revenue"):
    """Hardcoded insight for "highest/peak <metric>" style questions."""
    record = highest_metric(data, metric)
    if record is None:
        return None
    value = record[metric]
    month = record.get("month", "an unknown period")
    return {
        "metric": metric,
        "month": month,
        "value": value,
        "summary": f"The highest {metric.replace('_', ' ')} was in {month}, at {value:,.2f}.",
    }


def lowest_insight(data, metric="revenue"):
    """Hardcoded insight for "lowest/weakest <metric>" style questions."""
    record = lowest_metric(data, metric)
    if record is None:
        return None
    value = record[metric]
    month = record.get("month", "an unknown period")
    return {
        "metric": metric,
        "month": month,
        "value": value,
        "summary": f"The lowest {metric.replace('_', ' ')} was in {month}, at {value:,.2f}.",
    }

def churn_rate_insight(data):
    """Hardcoded insight for "churn rate" style questions."""
    rate = churn_rate(data)
    if rate is None:
        return None
    return {
        "value": rate,
        "summary": f"Overall churn rate was {rate:.2f}% of active customers.",
    }

def correlation_insight(data, metric_a, metric_b):
    r = correlation_coefficient(data, metric_a, metric_b)
    if r is None:
        return None

    if r >= 0.7:
        strength = "a strong positive"
    elif r >= 0.3:
        strength = "a moderate positive"
    elif r > -0.3:
        strength = "a weak or negligible"
    elif r > -0.7:
        strength = "a moderate negative"
    else:
        strength = "a strong negative"

    label_a = metric_a.replace("_", " ").capitalize()
    label_b = metric_b.replace("_", " ").capitalize()

    return {
        "value": r,
        "metrics": [metric_a, metric_b],
        "summary": f"{label_a} and {label_b} show {strength} correlation (r = {r:+.2f}).",
    }

def breakdown_insight(data, metric, dimension):
    groups = group_totals(data, metric, dimension)
    if not groups:
        return None

    total = sum(groups.values())
    ordered = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
    leader, leader_val = ordered[0]

    metric_label = metric.replace("_", " ")
    dim_label = dimension.replace("_", " ")
    leader_share = round(leader_val / total * 100, 1) if total else None

    summary = (
        f"{leader} leads in {metric_label} by {dim_label} with {leader_val:,.2f}"
        + (f" ({leader_share}% of total)" if leader_share is not None else "")
        + "."
    )

    if len(ordered) > 1:
        trailer, trailer_val = ordered[-1]
        summary += f" {trailer} trails with {trailer_val:,.2f}."

    return {
        "metric": metric,
        "dimension": dimension,
        "breakdown": groups,
        "summary": summary,
    }

def profit_margin_insight(data):
    """Hardcoded insight for "what's our profit margin" style questions."""
    margin = profit_margin(data)
    if margin is None:
        return None
    return {
        "value": margin,
        "summary": f"Overall profit margin was {margin:.2f}% of total revenue.",
    }


def multi_metric_insight(data, metrics=("revenue", "profit")):
    """
    Hardcoded insight for questions naming more than one metric at once
    (e.g. "revenue and profit totals"). Summarizes total + average for
    each metric side by side.
    """
    summary = summarize_metrics(data, metrics)
    sentences = []
    for metric, stats in summary.items():
        if stats["total"] is None:
            continue
        sentences.append(
            f"{metric.replace('_', ' ').capitalize()}: total {stats['total']:,.2f}, "
            f"average {stats['average']:,.2f}."
        )
    if not sentences:
        return None
    return {
        "summary_by_metric": summary,
        "summary": " ".join(sentences),
    }


# ---------------------------------------------------------------------------
# Growth insights (month/quarter/year/week/day)
#
# growth_by_month / quarterly_growth / yearly_growth / weekly_growth /
# daily_growth each return a list of {<period>: ..., "growth_pct": ...}
# records with no summary sentence attached. _describe_growth_series turns
# any of those lists into one plain-English summary: overall direction,
# plus the single best and worst period.
# ---------------------------------------------------------------------------

def _describe_growth_series(records, label_key, period_word, metric="revenue"):
    valid = [r for r in records if r.get("growth_pct") is not None]
    if not valid:
        return None

    avg_growth = sum(r["growth_pct"] for r in valid) / len(valid)
    best = max(valid, key=lambda r: r["growth_pct"])
    worst = min(valid, key=lambda r: r["growth_pct"])
    direction = "up" if avg_growth >= 0 else "down"

    summary = (
        f"{metric.replace('_', ' ').capitalize()} growth averaged {abs(round(avg_growth, 2))}% "
        f"{direction} per {period_word}. The strongest {period_word} was "
        f"{best[label_key]} ({best['growth_pct']:+.2f}%), and the weakest "
        f"was {worst[label_key]} ({worst['growth_pct']:+.2f}%)."
    )
    return {
        "average_growth_pct": round(avg_growth, 2),
        "best": best,
        "worst": worst,
        "records": records,
        "summary": summary,
    }


def monthly_growth_insight(data, metric="revenue"):
    """Hardcoded insight for "month over month growth" style questions."""
    return _describe_growth_series(growth_by_month(data, metric), "month", "month", metric)


def quarterly_growth_insight(data, metric="revenue"):
    """Hardcoded insight for "quarterly growth" style questions."""
    return _describe_growth_series(quarterly_growth(data, metric), "quarter", "quarter", metric)


def yearly_growth_insight(data, metric="revenue"):
    """Hardcoded insight for "year over year growth" style questions."""
    return _describe_growth_series(yearly_growth(data, metric), "year", "year", metric)


def weekly_growth_insight(data, metric="revenue"):
    """Hardcoded insight for "week over week growth" style questions."""
    return _describe_growth_series(weekly_growth(data, metric), "week", "week", metric)


def daily_growth_insight(data, metric="revenue"):
    """Hardcoded insight for "day over day growth" style questions."""
    return _describe_growth_series(daily_growth(data, metric), "date", "day", metric)


# ---------------------------------------------------------------------------
# Router
#
# Given a raw question string, figures out which (if any) of the hardcoded
# insights above answers it, with zero LLM calls, by leaning on
# request_analyser.py's existing question parsing rather than
# re-implementing it. Returns None when the question needs real reasoning
# (determine_needs_llm) or doesn't match any known lookup pattern --
# callers should fall back to LLM.analyze_data in that case.
# ---------------------------------------------------------------------------

_GROWTH_PERIOD_KEYWORDS = {
    "quarterly": ("quarter", "quarterly"),
    "yearly": ("year over year", "yearly", "annual"),
    "weekly": ("week over week", "weekly"),
    "daily": ("day over day", "daily"),
}


def get_basic_insight(question, data):
    if not question or determine_needs_llm(question):
        return None

    lower = question.lower()

    requested_months = _extract_requested_months(question)
    scoped_data = _filter_by_months(data, requested_months)

    if "churn rate" in lower or ("churn" in lower and "rate" in lower):
        return churn_rate_insight(scoped_data)

    categories = extract_metrics(question)
    metrics = [METRIC_TO_FIELD[c] for c in categories if c in METRIC_TO_FIELD]
    if not metrics:
        metrics = ["revenue"]

    operations = extract_operations(question)

    if "correlation" in operations and len(metrics) >= 2:
        return correlation_insight(scoped_data, metrics[0], metrics[1])

    if len(metrics) > 1:
        return multi_metric_insight(scoped_data, metrics)
    metric = metrics[0]

    if "margin" in lower:
        return profit_margin_insight(scoped_data)

    primary_operation = choose_primary_operation(operations)

    if primary_operation in ("growth", "trend"):
        if any(word in lower for word in _GROWTH_PERIOD_KEYWORDS["quarterly"]):
            return quarterly_growth_insight(scoped_data, metric)
        if any(word in lower for word in _GROWTH_PERIOD_KEYWORDS["yearly"]):
            return yearly_growth_insight(scoped_data, metric)
        if any(word in lower for word in _GROWTH_PERIOD_KEYWORDS["weekly"]):
            return weekly_growth_insight(scoped_data, metric)
        if any(word in lower for word in _GROWTH_PERIOD_KEYWORDS["daily"]):
            return daily_growth_insight(scoped_data, metric)
        return monthly_growth_insight(scoped_data, metric)

    if primary_operation == "maximum":
        return highest_insight(scoped_data, metric)
    if primary_operation == "minimum":
        return lowest_insight(scoped_data, metric)
    if primary_operation == "average":
        return average_insight(scoped_data, metric)
    if primary_operation == "total":
        return total_insight(scoped_data, metric)

    dimension = extract_dimension(question)
    if dimension is not None:
        return breakdown_insight(scoped_data, metric, dimension)

    return None