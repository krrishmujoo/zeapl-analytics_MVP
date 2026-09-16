"""
dynamic_router.py -- Phase 12

Orchestrates the dataset-agnostic path end to end: active dataset -> runtime
config -> metric resolution -> generic dimension filtering -> forecast ->
response. This is what app.py's new /api/v2/forecast endpoint calls; the
original router.py/app.py /api/analyze path is untouched and keeps working
exactly as before for the current dataset.
"""
import logging
import re

import pandas as pd

from dataset_runtime_config import build_runtime_config
from metric_resolver import resolve_metric
from dynamic_prediction import run_forecast

logger = logging.getLogger(__name__)

def _coerce_filter_value(value, series: pd.Series):
    """
    Convert an LLM-produced filter value to the datatype used by the
    actual dataset column.
    """
    if isinstance(value, list):
        return [
            _coerce_filter_value(item, series)
            for item in value
        ]

    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, str):
            normalized = value.strip().casefold()

            if normalized in {"true", "1", "yes", "valid"}:
                return True

            if normalized in {"false", "0", "no", "invalid"}:
                return False

        return bool(value)

    if pd.api.types.is_integer_dtype(series):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value

    if pd.api.types.is_float_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    if pd.api.types.is_datetime64_any_dtype(series):
        converted = pd.to_datetime(
            value,
            errors="coerce",
        )

        if not pd.isna(converted):
            return converted

    return value

def apply_generic_filters(
    data,
    filters,
    config: dict,
    filter_logic: str = "and",
):
    """
    Apply validated filters dynamically to the current dataset.

    Supports both the older dictionary format:

        {"state": "Karnataka"}

    and the new condition-list format:

        [
            {
                "field": "state",
                "operator": "equals",
                "value": "Karnataka",
            }
        ]
    """
    if not filters or data is None:
        return data

    df = (
        data.copy()
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data)
    )

    if df.empty:
        return data

    date_column = config.get("date_column")

    conditions = []

    # Preserve compatibility with existing dictionary filters.
    if isinstance(filters, dict):
        for field, value in filters.items():
            if field == "date_from":
                conditions.append(
                    {
                        "field": date_column,
                        "operator": "date_from",
                        "value": value,
                    }
                )
            elif field == "date_to":
                conditions.append(
                    {
                        "field": date_column,
                        "operator": "date_to",
                        "value": value,
                    }
                )
            elif isinstance(value, (list, tuple, set)):
                conditions.append(
                    {
                        "field": field,
                        "operator": "in",
                        "value": list(value),
                    }
                )
            else:
                conditions.append(
                    {
                        "field": field,
                        "operator": "equals",
                        "value": value,
                    }
                )

    elif isinstance(filters, list):
        conditions = [
            condition
            for condition in filters
            if isinstance(condition, dict)
        ]

    else:
        raise ValueError(
            "Unsupported filters format: "
            f"{type(filters).__name__}"
        )
    normalized_filter_logic = str(
        filter_logic or "and"
    ).lower()

    if normalized_filter_logic not in {"and", "or"}:
        raise ValueError(
            f"Unsupported filter logic "
            f"{normalized_filter_logic!r}."
        )

    if normalized_filter_logic == "or":
        combined_mask = pd.Series(
            False,
            index=df.index,
        )

        for condition in conditions:
            matching_rows = apply_generic_filters(
                df,
                [condition],
                config,
                filter_logic="and",
            )

            combined_mask.loc[
                combined_mask.index.isin(
                    matching_rows.index
                )
            ] = True

        filtered_df = df.loc[combined_mask].copy()

        if isinstance(data, pd.DataFrame):
            return filtered_df

        return filtered_df.to_dict("records")

    for condition in conditions:
        field = condition.get("field")
        operator = condition.get("operator", "equals")
        value = condition.get("value")

        if not field:
            raise ValueError(
                "A filter is missing its field."
            )

        if field not in df.columns:
            raise ValueError(
                f"Filter field {field!r} does not exist in the dataset."
            )

        series = df[field]

        if operator in ("date_from", "date_to"):
            dates = pd.to_datetime(series, errors="coerce")
            filter_date = pd.to_datetime(value, errors="coerce")

            if pd.isna(filter_date):
                raise ValueError(
                    f"Invalid date filter value {value!r}."
                )

            if operator == "date_from":
                df = df[dates >= filter_date]
            else:
                value_text = str(value).strip()

                # A date-only upper bound means the complete calendar day.
                if ":" not in value_text:
                    next_day = filter_date + pd.Timedelta(days=1)
                    df = df[dates < next_day]
                else:
                    df = df[dates <= filter_date]

        elif operator == "equals":
            coerced_value = _coerce_filter_value(
                value,
                series,
            )

            if (
                series.dtype == object
                or pd.api.types.is_string_dtype(series)
            ):
                df = df[
                    series.astype(str).str.casefold()
                    == str(coerced_value).casefold()
                ]
            else:
                df = df[series == coerced_value]

        elif operator == "not_equals":
            coerced_value = _coerce_filter_value(
                value,
                series,
            )

            if (
                series.dtype == object
                or pd.api.types.is_string_dtype(series)
            ):
                df = df[
                    series.astype(str).str.casefold()
                    != str(coerced_value).casefold()
                ]
            else:
                df = df[series != coerced_value]

        elif operator in ("in", "not_in"):
            accepted_values = (
                value
                if isinstance(value, list)
                else [value]
            )

            accepted_values = _coerce_filter_value(
                accepted_values,
                series,
            )

            if (
                series.dtype == object
                or pd.api.types.is_string_dtype(series)
            ):
                normalized_values = {
                    str(item).casefold()
                    for item in accepted_values
                }

                mask = (
                    series.astype(str)
                    .str.casefold()
                    .isin(normalized_values)
                )
            else:
                mask = series.isin(accepted_values)

            if operator == "in":
                df = df[mask]
            else:
                df = df[~mask]

        elif operator == "contains":
            df = df[
                series.astype(str).str.contains(
                    str(value),
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        elif operator in {
            "greater_than",
            "greater_than_or_equal",
            "less_than",
            "less_than_or_equal",
        }:
            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Filter value {value!r} for field "
                    f"{field!r} must be numeric."
                ) from error

            if operator == "greater_than":
                df = df[numeric_series > numeric_value]
            elif operator == "greater_than_or_equal":
                df = df[numeric_series >= numeric_value]
            elif operator == "less_than":
                df = df[numeric_series < numeric_value]
            else:
                df = df[numeric_series <= numeric_value]

        elif operator == "between":
            if (
                not isinstance(value, list)
                or len(value) != 2
            ):
                raise ValueError(
                    "The between operator requires exactly two values."
                )                

            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            )

            try:
                lower_value = float(value[0])
                upper_value = float(value[1])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Between-filter values for {field!r} "
                    "must be numeric."
                ) from error

            df = df[
                numeric_series.between(
                    lower_value,
                    upper_value,
                    inclusive="both",
                )
            ]

        else:
            raise ValueError(
                f"Unsupported filter operator {operator!r}."
            )

    if isinstance(data, pd.DataFrame):
        return df

    return df.to_dict("records")


def resolve_granularity(query: str) -> str | None:
    q = query.lower()
    if any(w in q for w in ("daily", "day over day", "per day", "next 1 day", "days")):
        return "day"
    if any(w in q for w in ("weekly", "week over week", "per week", "weeks")):
        return "week"
    if any(w in q for w in ("monthly", "month over month", "per month", "months", "quarter")):
        return "month"
    return None  # let forecastability recommend one


def resolve_horizon(query: str) -> int | None:
    match = re.search(r"next\s+(\d+)\s*(day|week|month)", query.lower())
    if match:
        return int(match.group(1))
    return None


def run_dynamic_forecast(
    data,
    query: str,
    dataset_id: str | None = None,
    source_path: str | None = None,
    requested_metrics: list[str] | None = None,
    filters: dict | None = None,
    metric_key_override: str | None = None,
    granularity_override: str | None = None,
    horizon_override: int | None = None,
    enable_llm_fallback: bool = False,
) -> dict:
    config = build_runtime_config(data, dataset_id=dataset_id, source_path=source_path)

    if metric_key_override:
        if metric_key_override not in config["metrics"]:
            return {"status": "error", "message": f"unknown metric_key {metric_key_override!r} for this dataset",
                     "dataset": {"dataset_id": dataset_id, "fingerprint": config["dataset_fingerprint"]}}
        resolution = {"resolved": True, "metric_key": metric_key_override,
                      "resolution_method": "explicit_override", "resolution_confidence": 1.0}
    else:
        resolution = resolve_metric(query, config, requested_metrics)

    llm_used, llm_purpose = False, None

    # LLM fallback is scoped to exactly one thing: picking among candidates
    # that deterministic matching already narrowed down but couldn't decide
    # between -- never invoked for a clean no_match with zero candidates,
    # never sees raw data, never invents a metric outside the candidate list.
    if not resolution.get("resolved") and enable_llm_fallback and resolution.get("candidates"):
        from llm_metric_resolver import resolve_ambiguous_metric_with_llm
        llm_result = resolve_ambiguous_metric_with_llm(query, resolution["candidates"])
        if llm_result is not None:
            chosen_spec = config["metrics"][llm_result["metric_key"]]
            resolution = {
                "resolved": True, "metric_key": chosen_spec["metric_key"],
                "resolution_method": "llm_disambiguation", "resolution_confidence": 0.65,
            }
            llm_used, llm_purpose = True, "metric_disambiguation"

    if not resolution.get("resolved"):
        return {
            "status": "ambiguous" if resolution.get("candidates") else "error",
            "message": resolution.get("message", "could not resolve a metric for this query"),
            "resolution_method": resolution.get("resolution_method"),
            "candidates": resolution.get("candidates", []),
            "dataset": {"dataset_id": dataset_id, "fingerprint": config["dataset_fingerprint"]},
            "llm_used": llm_used,
            "llm_purpose": llm_purpose,
            "errors": [],
        }

    filtered = apply_generic_filters(data, filters or {}, config)

    granularity = granularity_override or resolve_granularity(query)
    horizon = horizon_override or resolve_horizon(query)

    result = run_forecast(
        filtered, config, metric_key=resolution["metric_key"],
        granularity=granularity, horizon=horizon,
        dataset_fingerprint=config["dataset_fingerprint"],
    )
    if result.get("status") == "success":
        result["metric"]["resolution_method"] = resolution["resolution_method"]
        result["metric"]["resolution_confidence"] = resolution["resolution_confidence"]
        if llm_used:
            result["llm_used"] = True
            result["llm_purpose"] = llm_purpose
    return result