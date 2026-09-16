"""
forecast_insights.py -- Phase 9

Deterministic, non-LLM forecast insight engine. Computes a structured
insight object first (direction, magnitude, peaks, volatility, seasonality,
anomalies, quality-aware caveats), then renders it into short sentences.
Never states a cause -- only what the numbers show.
"""
import numpy as np
import pandas as pd


def _direction(change_pct: float) -> str:
    if change_pct is None:
        return "mixed"
    if change_pct > 3:
        return "increasing"
    if change_pct < -3:
        return "decreasing"
    return "stable"


def _volatility_level(cv: float | None) -> str:
    if cv is None:
        return "low"
    if cv >= 0.5:
        return "high"
    if cv >= 0.2:
        return "moderate"
    return "low"


def _detect_seasonality(values: np.ndarray, max_period: int = 12) -> dict:
    """Cheap deterministic seasonality check via autocorrelation -- not a
    full STL decomposition, but enough to flag a repeating pattern without
    an LLM or heavy dependency."""
    n = len(values)
    if n < 8:
        return {"detected": False, "period": None, "strength": None}

    centered = values - values.mean()
    denom = float((centered ** 2).sum())
    if denom == 0:
        return {"detected": False, "period": None, "strength": None}

    best_period, best_corr = None, 0.0
    for period in range(2, min(max_period, n // 2) + 1):
        num = float((centered[:-period] * centered[period:]).sum())
        corr = num / denom
        if corr > best_corr:
            best_period, best_corr = period, corr

    detected = best_corr >= 0.4
    return {
        "detected": detected,
        "period": best_period if detected else None,
        "strength": round(best_corr, 3) if detected else round(best_corr, 3),
    }


def _detect_anomalies(values: np.ndarray, periods: list[str], z_threshold: float = 2.5) -> list[dict]:
    if len(values) < 4:
        return []
    mean, std = values.mean(), values.std()
    if std == 0:
        return []
    anomalies = []
    for period, value in zip(periods, values):
        z = (value - mean) / std
        if abs(z) >= z_threshold:
            anomalies.append({"period": str(period), "value": round(float(value), 2), "z_score": round(float(z), 2)})
    return anomalies


def build_forecast_insights(
    history_periods: list,
    history_values: list[float],
    forecast_periods: list[dict],
    quality: dict,
    metric_spec: dict,
) -> dict:
    """
    history_periods/history_values: the cleaned, aggregated actuals used for
    training (chronological order).
    forecast_periods: [{"period", "predicted_value", "lower_bound", "upper_bound"}, ...]
    """
    label = metric_spec.get("display_name", metric_spec.get("field", "the metric"))
    hist_vals = np.array(history_values, dtype=float) if history_values else np.array([])
    fc_vals = np.array([p["predicted_value"] for p in forecast_periods], dtype=float) if forecast_periods else np.array([])

    historical_average = float(hist_vals.mean()) if len(hist_vals) else None
    forecast_average = float(fc_vals.mean()) if len(fc_vals) else None

    forecast_change_percent = None
    if len(fc_vals) >= 2 and fc_vals[0] != 0:
        forecast_change_percent = round(float((fc_vals[-1] - fc_vals[0]) / abs(fc_vals[0]) * 100), 2)

    historical_recent_change_percent = None
    recent_window = min(5, len(hist_vals) // 2) if len(hist_vals) >= 4 else 0
    if recent_window >= 2:
        recent = hist_vals[-recent_window:]
        older = hist_vals[-2 * recent_window:-recent_window]
        if len(older) and older.mean() != 0:
            historical_recent_change_percent = round(float((recent.mean() - older.mean()) / abs(older.mean()) * 100), 2)

    direction = _direction(forecast_change_percent if forecast_change_percent is not None else historical_recent_change_percent)

    peak_period = peak_value = lowest_period = lowest_value = None
    if forecast_periods:
        peak = max(forecast_periods, key=lambda p: p["predicted_value"])
        low = min(forecast_periods, key=lambda p: p["predicted_value"])
        peak_period, peak_value = peak["period"], peak["predicted_value"]
        lowest_period, lowest_value = low["period"], low["predicted_value"]

    cv = None
    if len(hist_vals) and hist_vals.mean() != 0:
        cv = round(float(hist_vals.std() / abs(hist_vals.mean())), 3)
    volatility = {"level": _volatility_level(cv), "coefficient_of_variation": cv}

    seasonality = _detect_seasonality(hist_vals) if len(hist_vals) >= 8 else {"detected": False, "period": None, "strength": None}
    anomalies = _detect_anomalies(hist_vals, [str(p) for p in history_periods], z_threshold=2.5)

    interval_widths = [
        (p["upper_bound"] - p["lower_bound"]) for p in forecast_periods
        if p.get("upper_bound") is not None and p.get("lower_bound") is not None
    ]
    interval_widens = len(interval_widths) >= 2 and interval_widths[-1] > interval_widths[0] * 1.05

    messages = []
    if direction == "increasing":
        messages.append(f"The forecast for {label} trends upward over the horizon.")
    elif direction == "decreasing":
        messages.append(f"The forecast for {label} trends downward over the horizon.")
    else:
        messages.append(f"The forecast for {label} is roughly stable over the horizon.")

    if forecast_change_percent is not None:
        messages.append(f"Forecasted values change by {forecast_change_percent:+.1f}% from the start to the end of the horizon.")

    if historical_average is not None and forecast_average is not None:
        cmp_word = "above" if forecast_average > historical_average else "below" if forecast_average < historical_average else "in line with"
        messages.append(f"The forecast average is {cmp_word} the historical average ({forecast_average:,.2f} vs {historical_average:,.2f}).")

    if historical_recent_change_percent is not None:
        recent_dir = "up" if historical_recent_change_percent > 0 else "down" if historical_recent_change_percent < 0 else "flat"
        if (recent_dir == "up" and direction == "decreasing") or (recent_dir == "down" and direction == "increasing"):
            messages.append("Recent historical momentum points the opposite direction from the forecast -- treat this as a signal the trend may be turning, not a contradiction to ignore.")

    if peak_period is not None:
        messages.append(f"Expected peak: {peak_period} ({peak_value:,.2f}). Expected low: {lowest_period} ({lowest_value:,.2f}).")

    messages.append(f"Historical volatility is {volatility['level']}"
                     + (f" (coefficient of variation {cv})." if cv is not None else "."))

    if seasonality["detected"]:
        messages.append(f"A repeating pattern of roughly every {seasonality['period']} periods was detected in the history "
                         f"(strength {seasonality['strength']}).")

    if anomalies:
        messages.append(f"{len(anomalies)} historical period(s) look like outliers relative to the rest of the series.")

    if interval_widens:
        messages.append("The forecast's uncertainty range widens further out in the horizon, as expected.")

    baseline_mae = quality.get("baseline_mae")
    model_mae = quality.get("model_mae")
    if baseline_mae is not None and model_mae is not None and model_mae >= baseline_mae:
        messages.append("This model does not clearly outperform a naive last-value baseline -- treat the forecast shape with extra caution.")

    if quality.get("quality_level") in ("low", "unreliable"):
        messages.append("Given limited data quality/quantity, treat this forecast as directional only, not a precise number to plan against.")

    if len(forecast_periods) > 0 and quality.get("history_periods", 0) and len(forecast_periods) > quality["history_periods"] * 0.5:
        messages.append("The forecast horizon is long relative to the available history -- confidence should be expected to degrade toward the far end of it.")

    return {
        "summary": messages[0] if messages else f"No forecast insight could be computed for {label}.",
        "direction": direction,
        "forecast_change_percent": forecast_change_percent,
        "historical_recent_change_percent": historical_recent_change_percent,
        "forecast_average": round(forecast_average, 2) if forecast_average is not None else None,
        "historical_average": round(historical_average, 2) if historical_average is not None else None,
        "peak_period": peak_period, "peak_value": peak_value,
        "lowest_period": lowest_period, "lowest_value": lowest_value,
        "volatility": volatility,
        "seasonality": seasonality,
        "anomalies": anomalies,
        "quality": quality,
        "insight_messages": messages,
    }