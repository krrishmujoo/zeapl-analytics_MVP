"""
forecastability.py -- Phase 6

Not every discovered metric should be forecast, and not at every
granularity. This assesses, per metric x candidate granularity, whether
there's enough clean chronological signal to forecast responsibly, and
recommends a granularity + a horizon ceiling based on actual data density
rather than a fixed assumption.
"""
import numpy as np
import pandas as pd

GRANULARITY_FREQ = {"day": "D", "week": "W", "month": "MS"}
GRANULARITY_ORDER = ["day", "week", "month"]

# Minimum number of resampled periods needed before a granularity is even
# considered -- below this, cross-validation/features can't be built at all.
MIN_PERIODS = {"day": 14, "week": 8, "month": 6}

# Absolute ceiling on how far out to forecast at each granularity, regardless
# of how much history exists.
ABSOLUTE_MAX_HORIZON = {"day": 90, "week": 26, "month": 24}

MAX_MISSING_PERIOD_RATIO = 0.4
MAX_ZERO_RATIO = 0.7
MIN_UNIQUE_VALUES = 3


def _resample(df: pd.DataFrame, date_col: str, value_col: str, agg: str, granularity: str) -> pd.Series:
    freq = GRANULARITY_FREQ[granularity]
    indexed = df.set_index(date_col)[value_col]
    if agg == "count":
        return indexed.resample(freq).count()
    if agg == "nunique":
        return indexed.resample(freq).nunique()
    if agg == "mean":
        return indexed.resample(freq).mean()
    if agg in ("min", "max"):
        return getattr(indexed.resample(freq), agg)()
    return indexed.resample(freq).sum()  # default: sum


def assess_forecastability(data, metric_spec: dict, date_column: str) -> dict:
    """
    `data` is the raw (or already-filtered) list[dict]/DataFrame containing
    at least `date_column` and `metric_spec["field"]`.
    """
    reasons, warnings = [], []
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    if not metric_spec.get("forecastable", True):
        return {
            "forecastable": False, "score": 0.0, "supported_granularities": [],
            "recommended_granularity": None, "max_horizon": {},
            "reasons": metric_spec.get("forecast_rejection_reasons", ["metric flagged non-forecastable"]),
            "warnings": [],
        }

    if date_column not in df.columns or metric_spec["field"] not in df.columns:
        return {
            "forecastable": False, "score": 0.0, "supported_granularities": [],
            "recommended_granularity": None, "max_horizon": {},
            "reasons": [f"missing required column(s): {date_column!r} and/or {metric_spec['field']!r}"],
            "warnings": [],
        }

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    if df.empty:
        return {
            "forecastable": False, "score": 0.0, "supported_granularities": [],
            "recommended_granularity": None, "max_horizon": {},
            "reasons": ["no rows with a parsable date"], "warnings": [],
        }

    span_days = (df[date_column].max() - df[date_column].min()).total_seconds() / 86400
    if span_days < 1:
        return {
            "forecastable": False, "score": 0.0, "supported_granularities": [],
            "recommended_granularity": None, "max_horizon": {},
            "reasons": ["chronological span is under 1 day -- not enough history to forecast"],
            "warnings": [],
        }

    agg = metric_spec.get("default_aggregation", "sum")
    supported = []
    max_horizon = {}
    granularity_scores = {}

    for granularity in GRANULARITY_ORDER:
        try:
            series = _resample(df, date_column, metric_spec["field"], agg, granularity)
        except Exception as error:
            warnings.append(f"{granularity}: resampling failed ({error})")
            continue

        n_periods = len(series)
        if n_periods < MIN_PERIODS[granularity]:
            reasons.append(f"{granularity}: only {n_periods} periods, need at least {MIN_PERIODS[granularity]}")
            continue

        missing_ratio = float(series.isna().mean())
        filled = series.fillna(0)
        nunique = int(filled.nunique())
        zero_ratio = float((filled == 0).mean())

        if missing_ratio > MAX_MISSING_PERIOD_RATIO:
            reasons.append(f"{granularity}: {missing_ratio:.0%} of periods have no data")
            continue
        if nunique < MIN_UNIQUE_VALUES:
            reasons.append(f"{granularity}: fewer than {MIN_UNIQUE_VALUES} distinct values across periods (flat series)")
            continue
        if zero_ratio > MAX_ZERO_RATIO:
            reasons.append(f"{granularity}: {zero_ratio:.0%} of periods are zero -- too sparse to forecast reliably at this granularity")
            continue

        # crude density/quality score for this granularity, used to rank
        # candidates once several qualify.
        density_score = min(n_periods / (MIN_PERIODS[granularity] * 3), 1.0)
        completeness_score = 1.0 - missing_ratio
        variability_score = min(nunique / max(n_periods * 0.3, 1), 1.0)
        score = round(0.4 * density_score + 0.35 * completeness_score + 0.25 * variability_score, 3)

        supported.append(granularity)
        granularity_scores[granularity] = score
        # horizon ceiling: never forecast further than half the observed
        # history, and never past the absolute per-granularity cap.
        max_horizon[granularity] = max(1, min(n_periods // 2, ABSOLUTE_MAX_HORIZON[granularity]))

        if zero_ratio > 0.5:
            warnings.append(f"{granularity}: over half of periods are zero -- forecast may be dominated by sparsity")

    if not supported:
        return {
            "forecastable": False, "score": 0.0, "supported_granularities": [],
            "recommended_granularity": None, "max_horizon": {},
            "reasons": reasons or ["no granularity had sufficient history"], "warnings": warnings,
        }

    recommended_granularity = max(supported, key=lambda g: granularity_scores[g])
    overall_score = round(max(granularity_scores.values()), 3)

    return {
        "forecastable": True,
        "score": overall_score,
        "supported_granularities": supported,
        "recommended_granularity": recommended_granularity,
        "max_horizon": max_horizon,
        "reasons": [f"supports {', '.join(supported)} granularity" if supported else ""],
        "warnings": warnings,
    }