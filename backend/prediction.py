import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from model_manager import (
    load_model,
    get_model_confidence,
    get_model_metrics,
    compute_dataset_fingerprint,
    LOW_CONFIDENCE_THRESHOLD,
)
from registries import DatasetConfig

# ---------------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------------

GRANULARITY_FREQ = {"day": "D", "week": "W", "month": "MS"}
MIN_PERIODS_NEEDED = {"day": 14, "week": 8, "month": 6}
MAX_HORIZON = {"day": 90, "week": 26, "month": 24}

# Fields that can never legitimately be negative in this dataset. Rows with
# negative values in these columns are treated as data errors (Task 1).
NON_NEGATIVE_FIELDS = {"invoice_value", "quantity", "points_earned", "revenue", "profit"}

FEATURE_COLUMNS = [
    "t",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_mean_7", "rolling_std_3",
    "pct_growth", "prev_growth", "cumulative_trend",
    "month", "quarter", "weekday", "week_of_year", "year",
]


# ---------------------------------------------------------------------------
# Task 1: Data cleaning / preprocessing pipeline
#
# Runs on the raw scan-level rows before any time-series work happens.
# Every step is defensive: missing/garbage input produces a clear,
# descriptive error rather than a crash deeper in the pipeline.
# ---------------------------------------------------------------------------

def prepare_dataframe(
    data: list[dict],
    date_column: str | None = None,
    valid_column: str | None = None,
    value_column: str | None = None,
) -> pd.DataFrame:
    """
    Cleans raw records into a sorted, de-duplicated, gap-aware dataframe
    indexed by a canonical "scan_date" column.

    Steps: parse dates -> sort chronologically -> drop exact duplicate rows
    -> apply the validity flag (if any) -> forward-fill missing numeric
    values along the time axis -> strip impossible values (negative counts
    /amounts, NaN, +/-inf) out of the target metric column.
    """
    date_column = date_column or DatasetConfig["date_column"]
    valid_column = (
        valid_column if valid_column is not None else DatasetConfig.get("valid_column")
    )

    if not data:
        raise ValueError("no records supplied")

    df = pd.DataFrame(data)

    if date_column not in df.columns:
        raise KeyError(f"missing expected date column: {date_column!r}")

    if value_column is not None and value_column not in df.columns:
        raise KeyError(f"missing expected metric column: {value_column!r}")

    # --- parse + sort -----------------------------------------------------
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    unparsable = int(df[date_column].isna().sum())
    if unparsable:
        df = df.dropna(subset=[date_column])

    if df.empty:
        raise ValueError(
            f"all rows had an unparsable {date_column!r} value ({unparsable} rows dropped)"
        )

    df = df.sort_values(date_column).reset_index(drop=True)

    # --- de-duplicate -------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)

    # --- validity flag ------------------------------------------------------
    if valid_column and valid_column in df.columns:
        df = df[df[valid_column] == 1].reset_index(drop=True)

    if df.empty:
        raise ValueError("no valid rows remained after applying the validity flag")

    # --- missing-value handling ---------------------------------------------
    # Forward-fill numeric gaps along the time axis (the data is already
    # sorted chronologically), which is appropriate for a value that
    # persists between observations. Any values still missing at the very
    # start of the series (nothing to forward-fill from) are back-filled;
    # rows where a value is missing with nothing to fill from on either side
    # simply carry NaN forward and are excluded by NON_NEGATIVE_FIELDS
    # cleanup / resampling below rather than crashing here.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # --- impossible values ---------------------------------------------------
    for col in NON_NEGATIVE_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)

    if value_column is not None:
        before_value_drop = len(df)
        df = df.dropna(subset=[value_column]).reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"every remaining row had an unusable value in {value_column!r} "
                f"({before_value_drop} rows considered)"
            )

    df = df.rename(columns={date_column: "scan_date"})
    return df


# ---------------------------------------------------------------------------
# Task 2 (part 1): roll raw rows up into a clean, gap-free time series
# ---------------------------------------------------------------------------

def build_time_series(df: pd.DataFrame, field: str, agg: str, granularity: str) -> pd.DataFrame:
    freq = GRANULARITY_FREQ[granularity]
    indexed = df.set_index("scan_date")[field]

    # resample() creates a bucket for every period in the date range and
    # sums/counts to 0 for empty ones, so gap days/weeks/months are
    # included as zero-activity periods rather than silently disappearing.
    series = indexed.resample(freq).count() if agg == "count" else indexed.resample(freq).sum()

    out = series.reset_index()
    out.columns = ["period", "value"]
    return out


# ---------------------------------------------------------------------------
# Task 2 (part 2): feature engineering
# ---------------------------------------------------------------------------

def create_features(series: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Builds one feature row per period, predicting df["value"] at row i from
    information available strictly BEFORE period i (i.e. periods < i only).

    IMPORTANT (data-leakage guard): pandas' .rolling()/.expanding()/
    .pct_change() all include the *current* row by default. Naively calling
    them on df["value"] would let e.g. rolling_mean_3 or pct_growth encode
    today's value directly (rolling_mean_3 combined with the two prior lags
    lets you algebraically solve for value[i] almost exactly) -- which is
    exactly why an earlier version of this pipeline was seeing suspicious
    R²=1.0 / MAE=0.0 scores at day granularity. Every rolling/trend feature
    below is therefore computed on the *lagged* series (df["value"].shift(1))
    so a feature at row i only ever reflects periods 0..i-1, matching what
    forecast_forward() can actually know at prediction time.
    """
    df = series.copy()

    df["t"] = np.arange(len(df))

    lagged = df["value"].shift(1)  # everything below is built on this

    # Lags
    df["lag_1"] = lagged
    df["lag_2"] = df["value"].shift(2)
    df["lag_3"] = df["value"].shift(3)

    # Rolling statistics over the lagged series -- min_periods=1 so early
    # rows get a (partial) estimate instead of NaN; only the lag columns
    # are allowed to force row-drops below, per spec.
    df["rolling_mean_3"] = lagged.rolling(3, min_periods=1).mean()
    df["rolling_mean_7"] = lagged.rolling(7, min_periods=1).mean()
    df["rolling_std_3"] = lagged.rolling(3, min_periods=1).std().fillna(0.0)

    # Trend features -- growth momentum from history only: pct_growth is
    # the most recent known period-over-period change (lag_1 vs lag_2),
    # prev_growth is the change before that (lag_2 vs lag_3). This mirrors
    # forecast_forward()'s recursive step exactly.
    df["pct_growth"] = ((df["lag_1"] - df["lag_2"]) / df["lag_2"]).replace([np.inf, -np.inf], np.nan) * 100
    df["pct_growth"] = df["pct_growth"].fillna(0.0)
    df["prev_growth"] = ((df["lag_2"] - df["lag_3"]) / df["lag_3"]).replace([np.inf, -np.inf], np.nan) * 100
    df["prev_growth"] = df["prev_growth"].fillna(0.0)
    df["cumulative_trend"] = lagged.expanding().mean()

    # Calendar features (these describe the target period itself, which is
    # legitimately known in advance -- not leakage).
    df["month"] = df["period"].dt.month
    df["quarter"] = df["period"].dt.quarter
    df["weekday"] = df["period"].dt.weekday
    df["week_of_year"] = df["period"].dt.isocalendar().week.astype(int)
    df["year"] = df["period"].dt.year

    # Only the lag features (which need real history to exist) force a
    # row-drop -- this trims exactly the first 3 rows of the series.
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Task 3 + 4: chronological validation and honest evaluation
#
# Replaces the old single 80/20 split (which, on the small monthly series
# in this dataset, often had a 1-2 row test fold -- easy to "ace" by luck,
# which is why revenue_week/quantity_week/etc. were showing confidence
# 1.0). TimeSeriesSplit with multiple folds, each respecting chronological
# order (train is always strictly earlier than test), gives an averaged,
# more honest score. It also always produces a number when there are at
# least 5 usable rows, instead of silently falling back to None whenever
# len(feature_df) < 10 -- this is the fix for the null-confidence models
# at month granularity.
# ---------------------------------------------------------------------------

def _flat_series_metrics() -> dict:
    return {"r2": None, "mae": 0.0, "rmse": 0.0, "n_folds": 0, "note": "flat series (no variance to evaluate)"}


def train_model(feature_df: pd.DataFrame):
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["value"]

    if y.nunique() <= 1:
        # A constant target has nothing for R²/MAE/RMSE to meaningfully
        # measure -- fit on everything and say so explicitly instead of
        # reporting a fake-precise number.
        model = LinearRegression().fit(X, y)
        return model, _flat_series_metrics()

    n = len(feature_df)
    n_splits = min(5, max(2, n // 5))
    n_splits = min(n_splits, n - 1) if n > 1 else 1

    fold_r2, fold_mae, fold_rmse = [], [], []

    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(X):
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if y_train.nunique() <= 1:
                continue

            fold_model = LinearRegression().fit(X_train, y_train)
            preds = fold_model.predict(X_test)

            if y_test.nunique() > 1:
                fold_r2.append(r2_score(y_test, preds))
            fold_mae.append(mean_absolute_error(y_test, preds))
            fold_rmse.append(mean_squared_error(y_test, preds) ** 0.5)

    if fold_mae:
        metrics = {
            "r2": round(float(np.mean(fold_r2)), 4) if fold_r2 else None,
            "mae": round(float(np.mean(fold_mae)), 4),
            "rmse": round(float(np.mean(fold_rmse)), 4),
            "n_folds": len(fold_mae),
            "note": None,
        }
    else:
        # Too little history for even a 2-fold chronological split --
        # rather than silently returning None (the old bug), say why.
        metrics = {
            "r2": None, "mae": None, "rmse": None, "n_folds": 0,
            "note": "insufficient history for cross-validated evaluation",
        }

    # Refit on everything for the actual forecast -- validation above is
    # only used to get an honest read on accuracy, never for prediction.
    model = LinearRegression().fit(X, y)
    return model, metrics


# ---------------------------------------------------------------------------
# Recursive multi-step forecast -- rebuilds every engineered feature at
# each step exactly the way create_features() does, so the model always
# sees feature vectors from the same distribution it was trained on.
# ---------------------------------------------------------------------------

def forecast_forward(model, feature_df: pd.DataFrame, granularity: str, horizon: int) -> list[dict]:
    history = feature_df[["period", "value"]].copy()
    freq = GRANULARITY_FREQ[granularity]

    forecasts = []
    for _ in range(horizon):
        last_period = history["period"].iloc[-1]
        next_period = pd.Timestamp(last_period) + pd.tseries.frequencies.to_offset(freq)

        tail = history["value"]
        lag_1 = tail.iloc[-1]
        lag_2 = tail.iloc[-2] if len(tail) >= 2 else lag_1
        lag_3 = tail.iloc[-3] if len(tail) >= 3 else lag_2

        pct_growth = ((lag_1 - lag_2) / lag_2 * 100) if lag_2 else 0.0
        prev_growth = ((lag_2 - lag_3) / lag_3 * 100) if lag_3 else 0.0

        next_features = {
            "t": len(history),
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_3": tail.tail(3).mean(),
            "rolling_mean_7": tail.tail(7).mean(),
            "rolling_std_3": tail.tail(3).std() if len(tail) >= 2 else 0.0,
            "pct_growth": pct_growth,
            "prev_growth": prev_growth,
            "cumulative_trend": tail.mean(),
            "month": next_period.month,
            "quarter": next_period.quarter,
            "weekday": next_period.weekday(),
            "week_of_year": int(next_period.isocalendar()[1]),
            "year": next_period.year,
        }
        if pd.isna(next_features["rolling_std_3"]):
            next_features["rolling_std_3"] = 0.0

        X_future = pd.DataFrame([next_features])[FEATURE_COLUMNS]
        predicted = max(0.0, float(model.predict(X_future)[0]))  # can't have negative scans/revenue

        forecasts.append({
            "period": next_period.strftime("%Y-%m-%d"),
            "predicted_value": round(predicted, 2),
        })

        history = pd.concat(
            [history, pd.DataFrame([{"period": next_period, "value": predicted}])],
            ignore_index=True,
        )

    return forecasts


# ---------------------------------------------------------------------------
# Task 7: deterministic (non-LLM) business explanation + recommendation
# ---------------------------------------------------------------------------

def _confidence_label(metrics: dict) -> str:
    r2 = metrics.get("r2") if metrics else None
    if r2 is None:
        return "unknown"
    if r2 >= 0.7:
        return "high"
    if r2 >= LOW_CONFIDENCE_THRESHOLD:
        return "moderate"
    return "low"


def generate_explanation(metric_key: str, feature_df: pd.DataFrame, metrics: dict) -> str:
    label = metric_key.replace("_", " ")
    recent = feature_df["pct_growth"].tail(5)
    recent = recent[recent != 0] if len(recent) > 1 else recent
    avg_growth = float(recent.mean()) if len(recent) else 0.0

    if metrics.get("note") == "flat series (no variance to evaluate)":
        return f"{label.capitalize()} has been flat with no meaningful movement in the recent history used for this forecast."

    if avg_growth > 3:
        trend_sentence = f"{label.capitalize()} has increased consistently over recent periods."
    elif avg_growth < -3:
        trend_sentence = f"{label.capitalize()} has declined over recent periods."
    else:
        trend_sentence = f"{label.capitalize()} has stayed roughly stable over recent periods."

    confidence = _confidence_label(metrics)
    if confidence == "low":
        confidence_sentence = "Prediction confidence is low because historical variance is high relative to the signal, so treat this forecast as directional rather than exact."
    elif confidence == "moderate":
        confidence_sentence = "Prediction confidence is moderate; the forecast is influenced by the recent trend but short-term swings could shift the actual outcome."
    elif confidence == "high":
        confidence_sentence = "Prediction confidence is high, based on a consistent historical pattern."
    else:
        confidence_sentence = "There wasn't enough history to cross-validate this forecast, so treat it as a rough estimate."

    return f"{trend_sentence} {confidence_sentence}"


def generate_recommendation(metric_key: str, forecasts: list[dict], metrics: dict) -> str:
    label = metric_key.replace("_", " ")
    if len(forecasts) < 2:
        return f"Monitor {label} closely as more data becomes available before acting on this forecast."

    first_val = forecasts[0]["predicted_value"]
    last_val = forecasts[-1]["predicted_value"]
    change = ((last_val - first_val) / first_val * 100) if first_val else 0.0
    confidence = _confidence_label(metrics)

    if change > 5:
        action = f"Consider preparing capacity/inventory for growing {label} over the forecast horizon."
    elif change < -5:
        action = f"Consider investigating the drivers behind declining {label} and planning mitigations."
    else:
        action = f"{label.capitalize()} looks steady; focus resources on initiatives with clearer upside."

    if confidence == "low":
        action += " Given the low confidence, validate with additional data before committing resources."

    return action


# ---------------------------------------------------------------------------
# Public entry point -- called by router.py as:
#   run_prediction(data, metric_key=..., field=..., agg=..., granularity=..., horizon=...)
# ---------------------------------------------------------------------------

def run_prediction(
    data: list[dict],
    metric_key: str,
    field: str,
    agg: str,
    granularity: str,
    horizon: int,
    date_column: str | None = None,
    valid_column: str | None = None,
) -> dict:
    if not data:
        return {"status": "insufficient_data", "message": "No data available to forecast from."}

    if granularity not in GRANULARITY_FREQ:
        return {"status": "error", "message": f"Unsupported granularity: {granularity!r}."}

    if not isinstance(horizon, int) or horizon <= 0:
        return {"status": "error", "message": "Forecast horizon must be a positive integer."}

    max_horizon = MAX_HORIZON[granularity]
    if horizon > max_horizon:
        return {
            "status": "error",
            "message": f"Forecast horizon of {horizon} {granularity}s exceeds the maximum supported ({max_horizon}).",
        }

    try:
        df = prepare_dataframe(data, date_column, valid_column, value_column=field)
    except KeyError as error:
        return {
            "status": "error",
            "message": (
                f"Prediction needs the raw activity records ({error}). "
                "Check that registries.DatasetConfig matches this dataset's column names."
            ),
        }
    except ValueError as error:
        return {"status": "error", "message": f"Data cleaning failed: {error}"}

    if df.empty:
        return {"status": "insufficient_data", "message": "No valid records to forecast from."}

    series = build_time_series(df, field, agg, granularity)

    min_needed = MIN_PERIODS_NEEDED[granularity]
    if len(series) < min_needed:
        return {
            "status": "insufficient_data",
            "message": (
                f"At least {min_needed} {granularity}s of history are needed to forecast "
                f"{metric_key}; only {len(series)} available."
            ),
        }

    feature_df = create_features(series, granularity)
    if len(feature_df) < 5:
        return {"status": "insufficient_data", "message": "Not enough usable history after feature preparation."}

    dataset_fingerprint = compute_dataset_fingerprint(data)
    model = load_model(metric_key, granularity, dataset_fingerprint)

    if model is None:
        return {
            "status": "error",
            "message": (
                f"No trained model exists for {metric_key} ({granularity}) on this dataset. "
                "Please run train_models.py first."
            ),
        }

    saved_metrics = get_model_metrics(dataset_fingerprint, metric_key, granularity) or {}
    confidence = saved_metrics.get("r2")
    low_confidence = confidence is None or confidence < LOW_CONFIDENCE_THRESHOLD

    try:
        forecasts = forecast_forward(model, feature_df, granularity, horizon)
    except Exception as error:
        return {"status": "error", "message": f"Forecast generation failed: {error}"}

    total = round(sum(f["predicted_value"] for f in forecasts), 2)

    explanation = generate_explanation(metric_key, feature_df, saved_metrics)
    recommendation = generate_recommendation(metric_key, forecasts, saved_metrics)

    return {
        "status": "success",
        "metric": metric_key,
        "granularity": granularity,
        "horizon": horizon,
        "model": saved_metrics.get("algorithm", "linear_regression"),
        "model_version": saved_metrics.get("version"),
        "confidence": confidence,
        "confidence_level": _confidence_label(saved_metrics),
        "low_confidence": low_confidence,
        "r2": saved_metrics.get("r2"),
        "mae": saved_metrics.get("mae"),
        "rmse": saved_metrics.get("rmse"),
        "forecast_total": total,
        "forecast_periods": forecasts,
        "training_periods": len(feature_df),
        "explanation": explanation,
        "recommendation": recommendation,
    }