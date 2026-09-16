"""
dataset_profiler.py -- Phase 2

Inspects an arbitrary tabular dataset (list[dict] or pandas DataFrame) and
produces a DatasetProfile: per-column semantic typing, date-column
detection/scoring, identifier detection, and candidate dimension/metric
lists. This is the single place that looks at raw column shapes; every
downstream module (metric_discovery, dataset_runtime_config, forecastability,
...) consumes its output instead of re-deriving semantics from column names.

Nothing in here is dataset-specific. It makes zero assumptions about any
particular column name -- every decision is made from dtype, parsability,
distribution, and generic naming *hints* (never a naming requirement).
"""
import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROFILER_VERSION = 2

# ---------------------------------------------------------------------------
# Naming hints -- used as WEAK signals only, combined with distribution-based
# signals. Never the sole basis for a semantic-type decision.
# ---------------------------------------------------------------------------
DATE_NAME_HINTS = ("date", "time", "timestamp", "created", "updated", "day", "month", "year", "period")
IDENTIFIER_NAME_HINTS = ("id", "uuid", "guid", "code", "number", "no", "key")
MONETARY_NAME_HINTS = ("revenue", "profit", "price", "cost", "expense", "spend", "amount",
                        "value", "invoice", "fee", "salary", "income", "budget", "sales")
PERCENTAGE_NAME_HINTS = ("rate", "ratio", "pct", "percent", "percentage", "utilization", "margin")
GEOGRAPHIC_NAME_HINTS = ("city", "state", "country", "region", "zone", "zip", "postal", "geo", "location")
BOOLEAN_NAME_HINTS = ("is_", "has_", "flag", "returned", "valid", "active", "churned", "cancelled")

MAX_CATEGORICAL_CARDINALITY = 50
HIGH_CARDINALITY_RATIO = 0.9
IDENTIFIER_UNIQUE_RATIO = 0.95
DATE_MIN_CONFIDENCE = 0.5
DATE_PARSE_SUCCESS_FLOOR = 0.8


def _to_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _is_string_like_dtype(series: pd.Series) -> bool:
    """pandas >=2.x can infer a dedicated 'str'/StringDtype for text columns
    instead of the classic numpy 'object' dtype -- checking `== object` alone
    misses those entirely. This catches both."""
    return series.dtype == object or pd.api.types.is_string_dtype(series)


# A numeric column only gets flagged as an identifier from uniqueness alone
# (no name hint) once there's enough data for "every value is basically
# unique" to be a meaningful signal rather than a coincidence of a tiny
# sample (e.g. 3 distinct customer counts in a toy dataset aren't an ID).
MIN_ROWS_FOR_UNNAMED_IDENTIFIER_HEURISTIC = 20


def _name_hint(name: str, hints: tuple) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in hints)


def _safe_float(value):
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-column profiling
# ---------------------------------------------------------------------------

def _looks_like_sequential_key(values: pd.Series) -> bool:
    """True if sorted unique values step by ~1 most of the time -- the
    actual signature of an autoincrement/row-number ID, as opposed to a
    business metric that merely happens to have no duplicate values (e.g. a
    steadily trending revenue column over enough rows)."""
    sorted_vals = np.sort(values.unique())
    if len(sorted_vals) < MIN_ROWS_FOR_UNNAMED_IDENTIFIER_HEURISTIC:
        return False
    diffs = np.diff(sorted_vals)
    if len(diffs) == 0:
        return False
    return float((diffs == 1).mean()) > 0.9


def _profile_numeric_column(col: str, series: pd.Series) -> dict:
    non_null = series.dropna()
    count = len(series)
    n_non_null = len(non_null)
    unique_count = int(non_null.nunique())
    unique_ratio = (unique_count / n_non_null) if n_non_null else 0.0

    negative_ratio = float((non_null < 0).mean()) if n_non_null else 0.0
    zero_ratio = float((non_null == 0).mean()) if n_non_null else 0.0
    is_integer_valued = n_non_null > 0 and bool(((non_null.astype(float) % 1) == 0).all())

    is_constant = unique_count <= 1
    has_business_metric_hint = _name_hint(col, MONETARY_NAME_HINTS) or _name_hint(col, PERCENTAGE_NAME_HINTS)
    is_identifier = unique_ratio >= IDENTIFIER_UNIQUE_RATIO and (
        _name_hint(col, IDENTIFIER_NAME_HINTS)
        or (
            not has_business_metric_hint
            and is_integer_valued
            and unique_ratio >= 0.99
            and _looks_like_sequential_key(non_null)
        )
    )
    is_high_cardinality = unique_ratio >= HIGH_CARDINALITY_RATIO and unique_count > MAX_CATEGORICAL_CARDINALITY

    # semantic type ------------------------------------------------------
    all_bool_like = n_non_null > 0 and non_null.dropna().isin([0, 1]).all()
    in_unit_interval = n_non_null > 0 and non_null.between(0, 1).mean() > 0.9

    if is_identifier:
        semantic_type = "identifier"
    elif all_bool_like and (unique_count <= 2):
        semantic_type = "boolean"
    elif in_unit_interval and _name_hint(col, PERCENTAGE_NAME_HINTS):
        semantic_type = "percentage"
    elif _name_hint(col, MONETARY_NAME_HINTS):
        semantic_type = "monetary"
    elif is_integer_valued:
        semantic_type = "discrete_numeric"
    else:
        semantic_type = "continuous_numeric"

    forecast_rejection_reasons = []
    if is_identifier:
        forecast_rejection_reasons.append("looks like an identifier column")
    if is_constant:
        forecast_rejection_reasons.append("column has zero variance (constant)")
    if unique_count < 3:
        forecast_rejection_reasons.append("fewer than 3 distinct values")
    is_forecast_candidate = not forecast_rejection_reasons

    return {
        "original_dtype": str(series.dtype),
        "semantic_type": semantic_type,
        "nullable": bool(series.isna().any()),
        "missing_count": int(series.isna().sum()),
        "missing_ratio": float(series.isna().mean()) if count else 0.0,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "sample_values": [x for x in non_null.head(3).tolist()],
        "min": _safe_float(non_null.min()) if n_non_null else None,
        "max": _safe_float(non_null.max()) if n_non_null else None,
        "mean": _safe_float(non_null.mean()) if n_non_null else None,
        "median": _safe_float(non_null.median()) if n_non_null else None,
        "std": _safe_float(non_null.std()) if n_non_null else None,
        "negative_ratio": round(negative_ratio, 4),
        "zero_ratio": round(zero_ratio, 4),
        "is_identifier": is_identifier,
        "is_constant": is_constant,
        "is_high_cardinality": is_high_cardinality,
        "is_forecast_candidate": is_forecast_candidate,
        "forecast_rejection_reasons": forecast_rejection_reasons,
    }


def _profile_object_column(col: str, series: pd.Series, date_parse_rate: float) -> dict:
    non_null = series.dropna().astype(str)
    count = len(series)
    n_non_null = len(non_null)
    unique_count = int(non_null.nunique())
    unique_ratio = (unique_count / n_non_null) if n_non_null else 0.0

    is_bool_like = n_non_null > 0 and non_null.str.lower().isin(
        ["true", "false", "yes", "no", "y", "n", "0", "1"]
    ).mean() > 0.95
    avg_len = non_null.str.len().mean() if n_non_null else 0

    normalized_col = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")

    strong_identifier_name = bool(
        normalized_col in {"id", "uuid", "guid", "key"}
        or normalized_col.endswith("_id")
        or normalized_col.endswith("_uuid")
        or normalized_col.endswith("_guid")
        or normalized_col.endswith("_key")
    )

    is_identifier = (
        strong_identifier_name
        and unique_count > 1
        and n_non_null > 0
    )
    is_constant = unique_count <= 1
    is_high_cardinality = unique_ratio >= HIGH_CARDINALITY_RATIO and unique_count > MAX_CATEGORICAL_CARDINALITY

    if date_parse_rate >= DATE_PARSE_SUCCESS_FLOOR:
        semantic_type = "datetime"
        is_identifier = False  # a date column isn't an "identifier" in the ID sense
    elif is_identifier:
        semantic_type = "identifier"
    elif is_bool_like:
        semantic_type = "boolean"
    elif _name_hint(col, GEOGRAPHIC_NAME_HINTS):
        semantic_type = "geographic"
    elif unique_count <= MAX_CATEGORICAL_CARDINALITY and unique_ratio <= 0.5:
        semantic_type = "categorical"
    else:
        semantic_type = "free_text"

    return {
        "original_dtype": str(series.dtype),
        "semantic_type": semantic_type,
        "nullable": bool(series.isna().any()),
        "missing_count": int(series.isna().sum()),
        "missing_ratio": float(series.isna().mean()) if count else 0.0,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "sample_values": non_null.head(3).tolist(),
        "min": None, "max": None, "mean": None, "median": None, "std": None,
        "negative_ratio": 0.0, "zero_ratio": 0.0,
        "is_identifier": is_identifier,
        "is_constant": is_constant,
        "is_high_cardinality": is_high_cardinality,
        "is_forecast_candidate": False,
        "forecast_rejection_reasons": ["non-numeric column"],
    }


def _profile_datetime_column(col: str, series: pd.Series) -> dict:
    non_null = series.dropna()
    return {
        "original_dtype": str(series.dtype),
        "semantic_type": "datetime",
        "nullable": bool(series.isna().any()),
        "missing_count": int(series.isna().sum()),
        "missing_ratio": float(series.isna().mean()) if len(series) else 0.0,
        "unique_count": int(non_null.nunique()),
        "unique_ratio": round((non_null.nunique() / len(non_null)) if len(non_null) else 0.0, 4),
        "sample_values": [str(x) for x in non_null.head(3).tolist()],
        "min": str(non_null.min()) if len(non_null) else None,
        "max": str(non_null.max()) if len(non_null) else None,
        "mean": None, "median": None, "std": None,
        "negative_ratio": 0.0, "zero_ratio": 0.0,
        "is_identifier": False, "is_constant": non_null.nunique() <= 1,
        "is_high_cardinality": False, "is_forecast_candidate": False,
        "forecast_rejection_reasons": ["datetime column, not a metric"],
    }


# ---------------------------------------------------------------------------
# Date-column detection and scoring
# ---------------------------------------------------------------------------

def _score_date_candidate(col: str, parse_success_rate: float, unique_ratio: float, span_days: float) -> float:
    name_bonus = 1.0 if _name_hint(col, DATE_NAME_HINTS) else 0.0
    uniqueness_score = min(unique_ratio * 2, 1.0)  # a date column need not be 100% unique (daily rollups repeat)
    range_score = min(span_days / 365.0, 1.0) if span_days else 0.0
    return round(0.45 * parse_success_rate + 0.2 * uniqueness_score + 0.2 * range_score + 0.15 * name_bonus, 4)


def _detect_date_columns(df: pd.DataFrame) -> tuple[list[dict], dict[str, float]]:
    """Returns (ranked candidates, {column: parse_success_rate}) -- the
    parse-rate map lets the main loop reuse this work instead of re-parsing
    every string column twice."""
    candidates = []
    parse_rates = {}

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_datetime64_any_dtype(series):
            parse_rates[col] = 1.0
            non_null = series.dropna()
            if non_null.empty:
                continue
            unique_ratio = non_null.nunique() / len(non_null)
            span_days = (non_null.max() - non_null.min()).total_seconds() / 86400
            score = _score_date_candidate(col, 1.0, unique_ratio, span_days)
            candidates.append({"column": col, "score": score, "parse_success_rate": 1.0})
            continue

        if series.dtype == object or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                parse_rates[col] = 0.0
                continue
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            valid = parsed.dropna()
            # Guard against dateutil's over-eager fuzzy parsing turning a
            # short alphanumeric code (e.g. "M01") into an "implausible but
            # technically valid" date like year 1. A real date column's
            # successfully-parsed values should mostly land in a sane range.
            plausible = valid[(valid.dt.year >= 1900) & (valid.dt.year <= 2100)]
            success_rate = (len(plausible) / len(non_null)) if len(non_null) else 0.0
            parse_rates[col] = success_rate

            if success_rate >= DATE_PARSE_SUCCESS_FLOOR:
                unique_ratio = plausible.nunique() / len(plausible) if len(plausible) else 0.0
                span_days = (plausible.max() - plausible.min()).total_seconds() / 86400 if len(plausible) else 0.0
                score = _score_date_candidate(col, success_rate, unique_ratio, span_days)
                candidates.append({"column": col, "score": score, "parse_success_rate": round(success_rate, 4)})
        else:
            parse_rates[col] = 0.0

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates, parse_rates


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile_dataset(data, dataset_id: str | None = None) -> dict:
    """
    Profiles a dataset (list[dict] or DataFrame). Never raises on "weird"
    data -- an empty/unparseable dataset simply produces a profile with
    warnings and no candidates, so callers can present a clear message
    rather than crash.
    """
    df = _to_dataframe(data)
    warnings = []

    if df.empty:
        return {
            "dataset_id": dataset_id,
            "profiler_version": PROFILER_VERSION,
            "row_count": 0,
            "column_count": 0,
            "columns": {},
            "detected_date_columns": [],
            "primary_date_column": None,
            "candidate_dimensions": [],
            "candidate_metrics": [],
            "forecastable_metrics": [],
            "warnings": ["dataset is empty"],
        }

    date_candidates, parse_rates = _detect_date_columns(df)

    columns_profile = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            columns_profile[col] = _profile_datetime_column(col, series)
        elif pd.api.types.is_bool_dtype(series):
            non_null = series.dropna()
            columns_profile[col] = {
                "original_dtype": str(series.dtype), "semantic_type": "boolean",
                "nullable": bool(series.isna().any()), "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()), "unique_count": int(non_null.nunique()),
                "unique_ratio": round((non_null.nunique() / len(non_null)) if len(non_null) else 0.0, 4),
                "sample_values": non_null.head(3).tolist(),
                "min": None, "max": None, "mean": _safe_float(non_null.mean()) if len(non_null) else None,
                "median": None, "std": None, "negative_ratio": 0.0, "zero_ratio": 0.0,
                "is_identifier": False, "is_constant": non_null.nunique() <= 1,
                "is_high_cardinality": False, "is_forecast_candidate": False,
                "forecast_rejection_reasons": ["boolean column"],
            }
        elif pd.api.types.is_numeric_dtype(series):
            # A numeric column that also scored as a strong date candidate
            # (e.g. a unix timestamp or a YYYYMMDD int) is treated as a date
            # signal, not a metric -- checked via the same candidate list.
            columns_profile[col] = _profile_numeric_column(col, series)
        elif _is_string_like_dtype(series):
            rate = parse_rates.get(col, 0.0)
            columns_profile[col] = _profile_object_column(col, series, rate)
        else:
            non_null = series.dropna()
            columns_profile[col] = {
                "original_dtype": str(series.dtype), "semantic_type": "unknown",
                "nullable": bool(series.isna().any()), "missing_count": int(series.isna().sum()),
                "missing_ratio": float(series.isna().mean()), "unique_count": int(non_null.nunique()),
                "unique_ratio": 0.0, "sample_values": [], "min": None, "max": None,
                "mean": None, "median": None, "std": None, "negative_ratio": 0.0, "zero_ratio": 0.0,
                "is_identifier": False, "is_constant": False, "is_high_cardinality": False,
                "is_forecast_candidate": False, "forecast_rejection_reasons": ["unrecognized dtype"],
            }

    primary_date_column = None
    if date_candidates and date_candidates[0]["score"] >= DATE_MIN_CONFIDENCE:
        primary_date_column = date_candidates[0]["column"]
    elif date_candidates:
        warnings.append(
            f"date-like column(s) found ({', '.join(c['column'] for c in date_candidates)}) "
            f"but none scored above the confidence threshold ({DATE_MIN_CONFIDENCE}) -- "
            "no primary date column was selected"
        )
    else:
        warnings.append("no reliable date/time column found -- forecasting will not be available")

    candidate_dimensions = [
        col for col, prof in columns_profile.items()
        if prof["semantic_type"] in ("categorical", "boolean", "geographic")
        and not prof["is_identifier"] and not prof["is_constant"]
    ]
    candidate_metrics = [
        col for col, prof in columns_profile.items()
        if prof["semantic_type"] in ("continuous_numeric", "discrete_numeric", "monetary", "percentage")
        and not prof["is_identifier"]
    ]
    forecastable_metrics = [col for col in candidate_metrics if columns_profile[col]["is_forecast_candidate"]]

    if not candidate_metrics:
        warnings.append("no numeric, non-identifier columns found -- nothing to aggregate or forecast")

    return {
        "dataset_id": dataset_id,
        "profiler_version": PROFILER_VERSION,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_profile,
        "detected_date_columns": date_candidates,
        "primary_date_column": primary_date_column,
        "candidate_dimensions": candidate_dimensions,
        "candidate_metrics": candidate_metrics,
        "forecastable_metrics": forecastable_metrics,
        "warnings": warnings,
    }