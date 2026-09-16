"""
dynamic_prediction.py -- Phases 7 & 8

The dataset-agnostic forecasting entry point. Everything that used to be a
hardcoded constant in prediction.py (date column, non-negative fields,
which metric, which granularity) is now read from a MetricSpec /
DatasetRuntimeConfig produced by dataset_profiler / metric_discovery /
dataset_runtime_config / forecastability.

Reuses the leakage-safe feature-engineering shape from prediction.py
(lag/rolling/trend/calendar features, all computed on the *lagged* series)
and TimeSeriesSplit validation, but drives it from model_selection.py's
candidate-model framework instead of a single fixed LinearRegression.
"""
import logging

import numpy as np
import pandas as pd

from model_manager import (
    load_model, save_model, get_model_metrics, update_metadata,
    should_retrain, compute_dataset_fingerprint, FEATURE_ENGINEERING_VERSION,
)
from model_selection import select_model, NaiveLastValueModel, SeasonalNaiveModel
from forecastability import assess_forecastability, GRANULARITY_FREQ
from forecast_insights import build_forecast_insights
from recommendation_engine import build_recommendations

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "t", "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7", "rolling_std_3",
    "pct_growth", "prev_growth", "cumulative_trend",
    "month", "quarter", "weekday", "week_of_year", "year",
]

MAX_HORIZON_ABS = {"day": 90, "week": 26, "month": 24}
Z_80 = 1.2816  # z-score for an ~80% prediction interval under a normal-residual assumption


# ---------------------------------------------------------------------------
# Cleaning -- driven by MetricSpec instead of a hardcoded field whitelist
# ---------------------------------------------------------------------------

def _clean_dataframe(
    data,
    date_column: str,
    metric_field: str,
    non_negative: bool,
    aggregation: str = "sum",
) -> pd.DataFrame:
    if not data:
        raise ValueError("no records supplied")

    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    if date_column not in df.columns:
        raise KeyError(f"missing expected date column: {date_column!r}")
    if metric_field not in df.columns:
        raise KeyError(f"missing expected metric column: {metric_field!r}")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no rows with a parsable {date_column!r} value")

    df = df.drop_duplicates().reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    if aggregation in {"count", "nunique"}:
        # Identifier-backed count metrics may contain strings such as scan IDs.
        # Preserve them until resampling performs count/nunique aggregation.
        df = df.dropna(subset=[metric_field]).reset_index(drop=True)
    else:
        df[metric_field] = pd.to_numeric(
            df[metric_field],
            errors="coerce",
        )

        if non_negative:
            df.loc[df[metric_field] < 0, metric_field] = np.nan

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=[metric_field]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"every row had an unusable value in {metric_field!r}")

    return df.rename(columns={date_column: "_date"})


def _resample(df: pd.DataFrame, field: str, agg: str, granularity: str) -> pd.DataFrame:
    freq = GRANULARITY_FREQ[granularity]
    indexed = df.set_index("_date")[field]
    if agg == "count":
        series = indexed.resample(freq).count()
    elif agg == "nunique":
        series = indexed.resample(freq).nunique()
    elif agg == "mean":
        series = indexed.resample(freq).mean()
    elif agg in ("min", "max"):
        series = getattr(indexed.resample(freq), agg)()
    else:
        series = indexed.resample(freq).sum()
    out = series.reset_index()
    out.columns = ["period", "value"]
    return out


def _create_features(series: pd.DataFrame) -> pd.DataFrame:
    """Same leakage-safe shape as prediction.py's create_features: every
    rolling/trend feature is built on the *lagged* series so a feature at
    row i only reflects periods before i."""
    df = series.copy()
    df["t"] = np.arange(len(df))
    lagged = df["value"].shift(1)

    df["lag_1"] = lagged
    df["lag_2"] = df["value"].shift(2)
    df["lag_3"] = df["value"].shift(3)

    df["rolling_mean_3"] = lagged.rolling(3, min_periods=1).mean()
    df["rolling_mean_7"] = lagged.rolling(7, min_periods=1).mean()
    df["rolling_std_3"] = lagged.rolling(3, min_periods=1).std().fillna(0.0)

    df["pct_growth"] = ((df["lag_1"] - df["lag_2"]) / df["lag_2"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100
    df["prev_growth"] = ((df["lag_2"] - df["lag_3"]) / df["lag_3"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 100
    df["cumulative_trend"] = lagged.expanding().mean()

    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter
    df["weekday"] = df["period"].dt.weekday
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    df["year"] = df["period"].dt.year

    return df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)


def _forecast_forward(model, model_name: str, feature_df: pd.DataFrame, granularity: str,
                       horizon: int, non_negative: bool, rmse: float | None) -> list[dict]:
    history = feature_df[["period", "value"]].copy()
    freq = GRANULARITY_FREQ[granularity]

    forecasts = []
    for step in range(1, horizon + 1):
        last_period = history["period"].iloc[-1]
        next_period = pd.Timestamp(last_period) + pd.tseries.frequencies.to_offset(freq)

        tail = history["value"]
        lag_1 = tail.iloc[-1]
        lag_2 = tail.iloc[-2] if len(tail) >= 2 else lag_1
        lag_3 = tail.iloc[-3] if len(tail) >= 3 else lag_2
        pct_growth = ((lag_1 - lag_2) / lag_2 * 100) if lag_2 else 0.0
        prev_growth = ((lag_2 - lag_3) / lag_3 * 100) if lag_3 else 0.0

        next_features = {
            "t": len(history), "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
            "rolling_mean_3": tail.tail(3).mean(), "rolling_mean_7": tail.tail(7).mean(),
            "rolling_std_3": (tail.tail(3).std() if len(tail) >= 2 else 0.0) or 0.0,
            "pct_growth": pct_growth, "prev_growth": prev_growth, "cumulative_trend": tail.mean(),
            "month": next_period.month, "quarter": next_period.quarter, "weekday": next_period.weekday(),
            "week_of_year": int(next_period.isocalendar()[1]), "year": next_period.year,
        }
        X_future = pd.DataFrame([next_features])[FEATURE_COLUMNS]
        predicted = float(model.predict(X_future)[0])
        if non_negative:
            predicted = max(0.0, predicted)

        lower = upper = None
        if rmse is not None:
            width = Z_80 * rmse * (step ** 0.5)  # widen with sqrt(step) -- random-walk-style growth
            lower = predicted - width
            upper = predicted + width
            if non_negative:
                lower = max(0.0, lower)

        forecasts.append({
            "period": next_period.strftime("%Y-%m-%d"),
            "predicted_value": round(predicted, 4),
            "lower_bound": round(lower, 4) if lower is not None else None,
            "upper_bound": round(upper, 4) if upper is not None else None,
        })

        new_row = {"period": next_period, "value": predicted}
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return forecasts


# ---------------------------------------------------------------------------
# Quality assessment (Phase 8)
# ---------------------------------------------------------------------------

def assess_quality(
    validation: dict, history_periods: int, horizon: int,
    volatility_cv: float | None, missing_ratio: float = 0.0, recursive: bool = True,
) -> dict:
    reasons = []
    score = 0.5  # neutral baseline

    improvement = validation.get("improvement_over_baseline")
    if improvement is not None:
        if improvement > 10:
            score += 0.2; reasons.append(f"model beats naive baseline by {improvement}%")
        elif improvement > 0:
            score += 0.05; reasons.append(f"model marginally beats naive baseline ({improvement}%)")
        else:
            score -= 0.15; reasons.append("model does not outperform a naive last-value baseline")
    else:
        score -= 0.1; reasons.append("no baseline comparison available")

    if history_periods >= 60:
        score += 0.15; reasons.append(f"{history_periods} historical periods available")
    elif history_periods >= 20:
        score += 0.05; reasons.append(f"{history_periods} historical periods (moderate history)")
    else:
        score -= 0.15; reasons.append(f"only {history_periods} historical periods (thin history)")

    if horizon > history_periods * 0.5:
        score -= 0.15; reasons.append("forecast horizon is long relative to available history")

    if volatility_cv is not None:
        if volatility_cv >= 0.5:
            score -= 0.1; reasons.append("high historical volatility")
        elif volatility_cv < 0.2:
            score += 0.05

    if missing_ratio > 0.2:
        score -= 0.1; reasons.append(f"{missing_ratio:.0%} of periods had missing data")

    if recursive and horizon > 1:
        reasons.append("forecast beyond the first period is recursively generated -- errors can compound")

    n_folds = validation.get("n_folds", 0)
    if n_folds and n_folds < 3:
        score -= 0.05; reasons.append(f"only {n_folds} cross-validation folds")

    score = max(0.0, min(1.0, round(score, 3)))
    if score >= 0.7:
        level = "high"
    elif score >= 0.45:
        level = "moderate"
    elif score >= 0.25:
        level = "low"
    else:
        level = "unreliable"

    return {
        "quality_score": score,
        "quality_level": level,
        "reasons": reasons,
        "safe_to_act_on": level in ("high", "moderate"),
        "baseline_mae": validation.get("baseline_mae"),
        "model_mae": validation.get("mae"),
        "history_periods": history_periods,
        "score": score,  # alias used by forecast_insights
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_forecast(
    data,
    config: dict,
    metric_key: str,
    granularity: str | None = None,
    horizon: int | None = None,
    dataset_fingerprint: str | None = None,
) -> dict:
    metric_spec = config["metrics"].get(metric_key)
    if metric_spec is None:
        return {"status": "error", "message": f"unknown metric_key {metric_key!r} for this dataset"}

    date_column = config.get("date_column")
    if not date_column:
        return {"status": "error", "message": "this dataset has no reliable date column -- forecasting unavailable"}

    if not metric_spec.get("forecastable"):
        return {"status": "error", "message": f"{metric_spec['display_name']} is not forecastable: "
                                                + "; ".join(metric_spec.get("forecast_rejection_reasons", []))}

    fingerprint = dataset_fingerprint or config.get("dataset_fingerprint") or compute_dataset_fingerprint(data)

    try:
        df = _clean_dataframe(
            data,
            date_column,
            metric_spec["field"],
            metric_spec.get("non_negative", False),
            metric_spec.get("default_aggregation", "sum"),
        )
    except (KeyError, ValueError) as error:
        return {"status": "error", "message": f"data cleaning failed: {error}"}

    fc = assess_forecastability(df.rename(columns={"_date": date_column}), metric_spec, date_column)
    if not fc["forecastable"]:
        return {"status": "error", "message": "not forecastable at any granularity: " + "; ".join(fc["reasons"])}

    granularity = granularity or fc["recommended_granularity"]
    if granularity not in fc["supported_granularities"]:
        return {
            "status": "error",
            "message": f"granularity {granularity!r} isn't supported for this metric given available history; "
                       f"supported: {fc['supported_granularities']}",
        }

    max_horizon = fc["max_horizon"].get(granularity, MAX_HORIZON_ABS.get(granularity, 12))
    if horizon is None:
        horizon = min(max_horizon, {"day": 7, "week": 4, "month": 3}.get(granularity, 4))
    warnings = []
    if horizon > max_horizon:
        warnings.append(f"requested horizon {horizon} exceeds the safe maximum ({max_horizon}) for this "
                         f"amount of history; capped to {max_horizon}")
        horizon = max_horizon
    if horizon <= 0:
        return {"status": "error", "message": "forecast horizon must be a positive integer"}

    series = _resample(df, metric_spec["field"], metric_spec["default_aggregation"], granularity)
    feature_df = _create_features(series)
    if len(feature_df) < 5:
        return {"status": "insufficient_data", "message": "not enough usable history after feature engineering"}

    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["value"]

    # retraining decision + model I/O, reusing model_manager's existing
    # fingerprinted, atomic-write, corrupted-file-safe machinery.
    retrain, reason = should_retrain(metric_key, granularity, len(df), fingerprint,
                                      config.get("newest_timestamp"))
    existing_metrics = get_model_metrics(fingerprint, metric_key, granularity)

    if retrain or existing_metrics is None:
        selection = select_model(X, y)
        model = selection["model"]
        update_metadata(
            metric_key, granularity, len(df), selection["validation"], fingerprint,
            newest_timestamp=config.get("newest_timestamp"),
            algorithm=selection["algorithm"], feature_count=len(FEATURE_COLUMNS),
        )
        save_model(model, metric_key, granularity, fingerprint)
        validation = selection["validation"]
        algorithm = selection["algorithm"]
        model_version = 1 if existing_metrics is None else existing_metrics.get("version", 0) + 1
        selection_reason = selection["selection_reason"]
    else:
        model = load_model(metric_key, granularity, fingerprint)
        if model is None:  # metadata says trained, but .pkl missing/corrupted -- retrain now
            selection = select_model(X, y)
            model = selection["model"]
            update_metadata(metric_key, granularity, len(df), selection["validation"], fingerprint,
                             newest_timestamp=config.get("newest_timestamp"),
                             algorithm=selection["algorithm"], feature_count=len(FEATURE_COLUMNS))
            save_model(model, metric_key, granularity, fingerprint)
            validation = selection["validation"]
            algorithm = selection["algorithm"]
            model_version = 1
            selection_reason = selection["selection_reason"] + " (model file was missing/corrupted)"
        else:
            validation = existing_metrics
            algorithm = existing_metrics.get("algorithm", "unknown")
            model_version = existing_metrics.get("version", 1)
            selection_reason = "reused existing trained model (no retrain triggered)"

    rmse = validation.get("rmse")
    forecast_periods = _forecast_forward(model, algorithm, feature_df, granularity, horizon,
                                          metric_spec.get("non_negative", False), rmse)

    hist_vals = feature_df["value"].tolist()
    volatility_cv = None
    if hist_vals and np.mean(hist_vals) != 0:
        volatility_cv = float(np.std(hist_vals) / abs(np.mean(hist_vals)))

    quality = assess_quality(
        validation, history_periods=len(feature_df), horizon=horizon,
        volatility_cv=volatility_cv, missing_ratio=0.0, recursive=True,
    )

    insights = build_forecast_insights(
        history_periods=feature_df["period"].astype(str).tolist(),
        history_values=hist_vals,
        forecast_periods=forecast_periods,
        quality=quality,
        metric_spec=metric_spec,
    )
    recommendations = build_recommendations(metric_spec, insights, quality)

    total = round(sum(p["predicted_value"] for p in forecast_periods), 2)
    average = round(total / len(forecast_periods), 2) if forecast_periods else None

    # Return only the most recent history so the chart stays readable.
    # Daily: 30 points, weekly: 20 points, monthly: 12 points.
    history_window = {
        "day": 30,
        "week": 20,
        "month": 12,
    }.get(granularity, 20)

    historical_periods = [
        {
            "period": str(row["period"]),
            "actual_value": round(float(row["value"]), 2),
        }
        for _, row in series.tail(history_window).iterrows()
    ]



    return {
        "status": "success",
        "dataset": {
            "dataset_id": config.get("dataset_id"),
            "fingerprint": fingerprint,
            "date_column": date_column,
            "time_range": config.get("time_range"),
        },
        "metric": {
            "key": metric_key, "field": metric_spec["field"], "display_name": metric_spec["display_name"],
            "semantic_type": metric_spec["semantic_type"], "aggregation": metric_spec["default_aggregation"],
        },
        "forecast_config": {
            "granularity": granularity, "horizon": horizon,
            "recommended_granularity": fc["recommended_granularity"],
        },
        "model": {
            "algorithm": algorithm, "version": model_version,
            "feature_version": FEATURE_ENGINEERING_VERSION,
            "training_periods": len(feature_df),
            "selection_reason": selection_reason,
            "validation": {
                "mae": validation.get("mae"), "rmse": validation.get("rmse"), "r2": validation.get("r2"),
                "smape": validation.get("smape"), "baseline_mae": validation.get("baseline_mae"),
                "improvement_over_baseline": validation.get("improvement_over_baseline"),
                "n_folds": validation.get("n_folds"),
            },
        },
        "quality": {
            "score": quality["quality_score"], "level": quality["quality_level"],
            "safe_to_act_on": quality["safe_to_act_on"], "reasons": quality["reasons"],
        },
        "history": {
        "periods": historical_periods,
        },
        "forecast": {
            "total": total,
            "average": average,
            "periods": forecast_periods,
        },
        "insights": insights,
        "recommendations": recommendations,
        "warnings": warnings + fc.get("warnings", []),
        "llm_used": False,
        "errors": [],
        # --- legacy-shaped aliases kept temporarily for any older frontend
        # code expecting prediction.py's original response shape; marked
        # for deprecation, remove once the frontend migrates to the fields
        # above. ---
        "confidence": validation.get("r2"),
        "low_confidence": quality["quality_level"] in ("low", "unreliable"),
        "forecast_total": total,
        "historical_periods": historical_periods,
        "forecast_periods": forecast_periods,
    }