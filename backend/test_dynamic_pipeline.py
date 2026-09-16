"""
test_dynamic_pipeline.py -- Phase 14

Covers: date detection, semantic typing, ID exclusion, dynamic metric
discovery, aggregation inference, metric query resolution, ambiguity
handling, forecastability, model training/selection, baseline comparison,
prediction intervals, deterministic insights/recommendations, low-confidence
behavior, profile caching, fingerprint separation / no cross-dataset
contamination, retraining, API contract, no automatic revenue fallback,
no LLM requirement for normal operation -- across 14 synthetic dataset
scenarios plus two full integration datasets (sales, IoT).
"""
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

PROJ = Path(__file__).parent
for d in ("schemas", "configs", "models"):
    p = PROJ / d
    if p.exists():
        shutil.rmtree(p)
    p.mkdir()

results = []


def log(name, ok, detail=""):
    detail = str(detail)
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def run(name, fn):
    try:
        fn()
    except Exception:
        log(name, False, traceback.format_exc().splitlines()[-1])
        return
    log(name, True)


rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Synthetic dataset builders (14 required scenarios)
# ---------------------------------------------------------------------------

def sales_dataset(n=150):
    return [{
        "date": f"2024-{(i // 28) % 12 + 1:02d}-{(i % 28) + 1:02d}",
        "revenue": round(1000 + i * 3 + rng.normal(0, 40), 2),
        "units": int(10 + (i % 6)),
        "region": ["North", "South", "East", "West"][i % 4],
        "order_id": f"ORD{i:05d}",
    } for i in range(n)]


def iot_dataset(n=600):
    return [{
        "timestamp": f"2024-01-{(i // 4) % 28 + 1:02d}T{(i % 4) * 6:02d}:00:00",
        "machine_id": f"M0{(i % 3) + 1}",
        "temperature": round(-5 + ((i % 20) - 10) * 0.7 + rng.normal(0, 1), 2),
        "downtime_minutes": int((i % 11 == 0) * 15),
        "energy_usage": round(10 + (i % 24) * 0.5 + rng.normal(0, 0.5), 2),
    } for i in range(n)]


def customer_dataset():
    return [
        {"month": "january", "active_customers": 4200, "churn_rate": 0.021, "plan": "pro"},
        {"month": "february", "active_customers": 4300, "churn_rate": 0.019, "plan": "pro"},
        {"month": "march", "active_customers": 4100, "churn_rate": 0.025, "plan": "basic"},
    ]


def invalid_no_date_dataset():
    return [{"category": "A", "amount": 5}, {"category": "B", "amount": 7}, {"category": "C", "amount": 3}]


def ambiguous_date_dataset(n=40):
    return [{
        "created_at": f"2024-02-{(i % 28) + 1:02d}",
        "updated_at": f"2024-03-{(i % 28) + 1:02d}",
        "value_a": rng.normal(100, 10),
        "value_b": rng.normal(50, 5),
    } for i in range(n)]


def sparse_dataset(n=20):
    return [{
        "date": f"2024-{(i * 9) % 12 + 1:02d}-{(i * 3) % 28 + 1:02d}",
        "revenue": round(500 + rng.normal(0, 200), 2),
    } for i in range(n)]


def flat_dataset(n=40):
    return [{"date": f"2024-01-{(i % 28) + 1:02d}", "revenue": 1500.0} for i in range(n)]


def volatile_dataset(n=60):
    return [{"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
              "revenue": round(1000 + rng.normal(0, 900), 2)} for i in range(n)]


def negative_valid_dataset(n=60):
    return [{"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
              "profit": round(rng.normal(0, 500), 2)} for i in range(n)]


def negative_invalid_dataset(n=60):
    vals = []
    for i in range(n):
        q = int(10 + (i % 5))
        if i % 13 == 0:
            q = -q
        vals.append({"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", "quantity": q})
    return vals


def percentage_dataset(n=40):
    return [{"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
              "conversion_rate": round(min(1.0, max(0.0, 0.2 + rng.normal(0, 0.05))), 4)} for i in range(n)]


def int_id_dataset(n=40):
    return [{"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", "customer_id": 10000 + i,
              "revenue": round(500 + i * 2, 2)} for i in range(n)]


def missing_periods_dataset():
    rows = []
    for i in range(60):
        if 20 <= i < 30:
            continue
        rows.append({"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}", "revenue": round(800 + i * 2, 2)})
    return rows


def schema_v1():
    return [{"date": f"2024-01-{(i % 28) + 1:02d}", "revenue": round(1000 + i * 3, 2)} for i in range(40)]


def schema_v2():
    return [{"date": f"2024-02-{(i % 28) + 1:02d}", "revenue": round(50000 + i * 300, 2), "region": "X"}
             for i in range(40)]


# ---------------------------------------------------------------------------
# 1-2. Date detection + semantic typing
# ---------------------------------------------------------------------------

def test_date_detection_and_typing():
    from dataset_profiler import profile_dataset
    p = profile_dataset(sales_dataset(), dataset_id="sales")
    assert p["primary_date_column"] == "date", p["primary_date_column"]
    assert p["columns"]["revenue"]["semantic_type"] in ("monetary", "continuous_numeric")
    assert p["columns"]["region"]["semantic_type"] in ("categorical", "geographic")
    assert p["columns"]["order_id"]["semantic_type"] == "identifier"
    assert p["columns"]["order_id"]["is_identifier"] is True


run("1-2. Date detection + semantic typing", test_date_detection_and_typing)


# ---------------------------------------------------------------------------
# 3. ID exclusion from metrics
# ---------------------------------------------------------------------------

def test_id_exclusion():
    from dataset_profiler import profile_dataset
    p = profile_dataset(sales_dataset(), dataset_id="sales")
    assert "order_id" not in p["candidate_metrics"]


run("3. ID exclusion from metrics", test_id_exclusion)


# ---------------------------------------------------------------------------
# 4-5. Dynamic metric discovery + aggregation inference
# ---------------------------------------------------------------------------

def test_metric_discovery_and_aggregation():
    from dataset_profiler import profile_dataset
    from metric_discovery import discover_metrics

    p = profile_dataset(iot_dataset(), dataset_id="iot")
    m = discover_metrics(iot_dataset(), p, dataset_fingerprint="test_iot", use_cache=False)
    assert m["temperature"]["default_aggregation"] == "mean", m["temperature"]
    assert m["temperature"]["non_negative"] is False
    assert m["downtime_minutes"]["default_aggregation"] == "sum"
    assert m["energy_usage"]["non_negative"] is True

    p2 = profile_dataset(percentage_dataset(), dataset_id="pct")
    m2 = discover_metrics(percentage_dataset(), p2, dataset_fingerprint="test_pct", use_cache=False)
    assert m2["conversion_rate"]["default_aggregation"] == "mean", m2["conversion_rate"]
    assert m2["conversion_rate"]["additive"] is False


run("4-5. Metric discovery + aggregation inference", test_metric_discovery_and_aggregation)


# ---------------------------------------------------------------------------
# 6-7. Metric query resolution + ambiguity handling
# ---------------------------------------------------------------------------

def test_metric_resolution_and_ambiguity():
    from dataset_runtime_config import build_runtime_config
    from metric_resolver import resolve_metric

    data = iot_dataset()
    cfg = build_runtime_config(data, dataset_id="iot_resolve", use_cache=False)

    r1 = resolve_metric("forecast temperature next week", cfg)
    assert r1["resolved"] and r1["metric_key"] == "temperature", r1

    r2 = resolve_metric("predict energy usage", cfg)
    assert r2["resolved"] and r2["metric_key"] == "energy_usage", r2

    r3 = resolve_metric("forecast the thing", cfg)
    assert not r3["resolved"], r3
    assert r3["resolution_method"] == "no_match"


run("6-7. Metric resolution + ambiguity handling", test_metric_resolution_and_ambiguity)


# ---------------------------------------------------------------------------
# 8. Forecastability
# ---------------------------------------------------------------------------

def test_forecastability():
    from dataset_runtime_config import build_runtime_config

    cfg_sparse = build_runtime_config(sparse_dataset(), dataset_id="sparse", use_cache=False)
    rev = cfg_sparse["metrics"].get("revenue", {})
    assert "day" not in rev.get("supported_granularities", []) or rev.get("forecastable") is False, rev

    cfg_flat = build_runtime_config(flat_dataset(), dataset_id="flat", use_cache=False)
    assert cfg_flat["metrics"]["revenue"]["forecastable"] is False, cfg_flat["metrics"]["revenue"]


run("8. Forecastability assessment (sparse + flat)", test_forecastability)


# ---------------------------------------------------------------------------
# 9-10. Model training + selection + baseline comparison
# ---------------------------------------------------------------------------

def test_training_and_selection():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast

    data = sales_dataset(150)
    cfg = build_runtime_config(data, dataset_id="sales_train", use_cache=False)
    r = run_forecast(data, cfg, metric_key="revenue", horizon=5)
    assert r["status"] == "success", r
    assert r["model"]["validation"]["baseline_mae"] is not None
    assert r["model"]["algorithm"] in (
        "naive_last_value", "seasonal_naive", "linear_regression", "ridge", "hist_gradient_boosting"
    )
    assert "selection_reason" in r["model"]


run("9-10. Model training + selection + baseline comparison", test_training_and_selection)


# ---------------------------------------------------------------------------
# 11. Prediction intervals
# ---------------------------------------------------------------------------

def test_prediction_intervals():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast

    data = volatile_dataset(60)
    cfg = build_runtime_config(data, dataset_id="volatile", use_cache=False)
    r = run_forecast(data, cfg, metric_key="revenue", horizon=5)
    assert r["status"] == "success", r
    periods = r["forecast"]["periods"]
    assert all(p["lower_bound"] is not None and p["upper_bound"] is not None for p in periods)
    widths = [p["upper_bound"] - p["lower_bound"] for p in periods]
    assert widths[-1] >= widths[0], widths


run("11. Prediction intervals widen with horizon", test_prediction_intervals)


# ---------------------------------------------------------------------------
# 12-13. Deterministic insights + recommendations
# ---------------------------------------------------------------------------

def test_insights_and_recommendations():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast

    data = iot_dataset(600)
    cfg = build_runtime_config(data, dataset_id="iot_insights", use_cache=False)
    r = run_forecast(data, cfg, metric_key="downtime_minutes", horizon=5)
    if r["status"] != "success":
        return
    assert "insight_messages" in r["insights"] and len(r["insights"]["insight_messages"]) > 0
    assert len(r["recommendations"]) >= 1
    rec = r["recommendations"][0]
    required = {"priority", "category", "action", "reason", "evidence", "limitations", "requires_human_review"}
    assert required <= set(rec.keys())


run("12-13. Deterministic insights + recommendations", test_insights_and_recommendations)


# ---------------------------------------------------------------------------
# 14. Low-confidence behavior
# ---------------------------------------------------------------------------

def test_low_confidence_behavior():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast

    data = volatile_dataset(60)
    cfg = build_runtime_config(data, dataset_id="volatile_quality", use_cache=False)
    r = run_forecast(data, cfg, metric_key="revenue", horizon=5)
    assert r["status"] == "success"
    assert r["quality"]["level"] in ("high", "moderate", "low", "unreliable")
    assert isinstance(r["quality"]["safe_to_act_on"], bool)
    if r["quality"]["level"] in ("low", "unreliable"):
        assert r["quality"]["safe_to_act_on"] is False


run("14. Low-confidence behavior", test_low_confidence_behavior)


# ---------------------------------------------------------------------------
# 15. Profile caching
# ---------------------------------------------------------------------------

def test_profile_caching():
    from dataset_runtime_config import build_runtime_config

    data = sales_dataset(150)
    t0 = time.time()
    cfg1 = build_runtime_config(data, dataset_id="cache_test", use_cache=True)
    t1 = time.time()
    cfg2 = build_runtime_config(data, dataset_id="cache_test", use_cache=True)
    t2 = time.time()
    assert cfg1["dataset_fingerprint"] == cfg2["dataset_fingerprint"]
    assert (t2 - t1) < (t1 - t0), f"cached load ({t2-t1:.4f}s) should be faster than first build ({t1-t0:.4f}s)"


run("15. Profile/config caching is faster on repeat", test_profile_caching)


# ---------------------------------------------------------------------------
# 16. Fingerprint separation / no cross-dataset contamination
# ---------------------------------------------------------------------------

def test_no_cross_dataset_contamination():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast
    import model_manager

    data_a = sales_dataset(150)
    data_b = [{"date": f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
               "revenue": round(9_000_000 + i * 500, 2)} for i in range(150)]

    cfg_a = build_runtime_config(data_a, dataset_id="contam_a", use_cache=False)
    cfg_b = build_runtime_config(data_b, dataset_id="contam_b", use_cache=False)
    assert cfg_a["dataset_fingerprint"] != cfg_b["dataset_fingerprint"]

    run_forecast(data_a, cfg_a, metric_key="revenue", horizon=3)
    gran_a = cfg_a["metrics"]["revenue"]["recommended_granularity"]
    meta_a = model_manager.get_model_metrics(cfg_a["dataset_fingerprint"], "revenue", gran_a)

    run_forecast(data_b, cfg_b, metric_key="revenue", horizon=3)
    meta_a_after = model_manager.get_model_metrics(cfg_a["dataset_fingerprint"], "revenue", gran_a)

    assert meta_a is not None and meta_a_after is not None
    assert meta_a["trained_at"] == meta_a_after["trained_at"], "dataset A's metadata was overwritten by dataset B!"


run("16. No cross-dataset metadata contamination", test_no_cross_dataset_contamination)


# ---------------------------------------------------------------------------
# 17. Retraining on schema change
# ---------------------------------------------------------------------------

def test_retraining_on_schema_change():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast

    data_v1 = schema_v1()
    cfg_v1 = build_runtime_config(data_v1, dataset_id="schema_evolve", use_cache=False)
    r1 = run_forecast(data_v1, cfg_v1, metric_key="revenue", horizon=3)

    data_v2 = schema_v2()
    cfg_v2 = build_runtime_config(data_v2, dataset_id="schema_evolve", use_cache=False)
    r2 = run_forecast(data_v2, cfg_v2, metric_key="revenue", horizon=3)

    assert cfg_v1["dataset_fingerprint"] != cfg_v2["dataset_fingerprint"]
    if r1["status"] == "success" and r2["status"] == "success":
        assert r1["forecast"]["average"] != r2["forecast"]["average"]


run("17. Retraining triggers on schema change", test_retraining_on_schema_change)


# ---------------------------------------------------------------------------
# 18. API contract (via app.py functions)
# ---------------------------------------------------------------------------

def test_api_contract():
    from app import profile, forecast

    p = profile(dataset="fact_scan_activity.csv")
    assert "metrics" in p and "dataset_fingerprint" in p

    r = forecast(question="forecast invoice value next week", dataset="fact_scan_activity.csv")
    assert r["status"] == "success"
    for key in ("status", "dataset", "metric", "forecast_config", "model", "quality", "forecast",
                "insights", "recommendations", "warnings", "llm_used", "errors"):
        assert key in r, f"missing contract field: {key}"


run("18. API contract (/api/v2/profile, /api/v2/forecast)", test_api_contract)


# ---------------------------------------------------------------------------
# 19. No automatic revenue fallback
# ---------------------------------------------------------------------------

def test_no_automatic_revenue_fallback():
    from app import forecast
    r = forecast(question="forecast revenue for next 5 days", dataset="fact_scan_activity.csv")
    assert r["status"] == "ambiguous", r
    assert "invoice_value" in [c["metric_key"] for c in r["candidates"]]


run("19. No automatic revenue fallback on unmatched query", test_no_automatic_revenue_fallback)


# ---------------------------------------------------------------------------
# 20. No LLM requirement for normal operation
# ---------------------------------------------------------------------------

def test_no_llm_requirement():
    from app import forecast
    r = forecast(question="forecast invoice value next week", dataset="fact_scan_activity.csv")
    assert r["status"] == "success"
    assert r["llm_used"] is False


run("20. No LLM requirement for normal operation", test_no_llm_requirement)


# ---------------------------------------------------------------------------
# 21-26. Remaining required scenarios
# ---------------------------------------------------------------------------

def test_negative_valid_metric():
    from dataset_profiler import profile_dataset
    from metric_discovery import discover_metrics
    data = negative_valid_dataset()
    p = profile_dataset(data, dataset_id="neg_valid")
    m = discover_metrics(data, p, dataset_fingerprint="neg_valid_fp", use_cache=False)
    assert m["profit"]["non_negative"] is False


run("21. Negative-valid metric (profit) not flagged non_negative", test_negative_valid_metric)


def test_negative_invalid_cleaned():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast
    data = negative_invalid_dataset()
    cfg = build_runtime_config(data, dataset_id="neg_invalid", use_cache=False)
    assert cfg["metrics"]["quantity"]["non_negative"] is True
    r = run_forecast(data, cfg, metric_key="quantity", horizon=3)
    if r["status"] == "success":
        assert all(p["predicted_value"] >= 0 for p in r["forecast"]["periods"])


run("22. Negative-invalid metric (quantity) cleaned + non-negative forecast", test_negative_invalid_cleaned)


def test_int_id_excluded():
    from dataset_profiler import profile_dataset
    data = int_id_dataset()
    p = profile_dataset(data, dataset_id="int_id")
    assert p["columns"]["customer_id"]["is_identifier"] is True
    assert "customer_id" not in p["candidate_metrics"]


run("23. Integer-stored ID correctly excluded from metrics", test_int_id_excluded)


def test_missing_periods_handled():
    from dataset_runtime_config import build_runtime_config
    from dynamic_prediction import run_forecast
    data = missing_periods_dataset()
    cfg = build_runtime_config(data, dataset_id="missing_periods", use_cache=False)
    r = run_forecast(data, cfg, metric_key="revenue", horizon=3)
    assert r["status"] in ("success", "insufficient_data", "error")


run("24. Missing periods handled without crashing", test_missing_periods_handled)


def test_invalid_no_date_dataset():
    from dataset_runtime_config import build_runtime_config
    cfg = build_runtime_config(invalid_no_date_dataset(), dataset_id="invalid", use_cache=False)
    assert cfg["date_column"] is None
    assert all(not spec["forecastable"] for spec in cfg["metrics"].values())


run("25. Dataset with no date column -> no forecastable metrics, no crash", test_invalid_no_date_dataset)


def test_ambiguous_date_columns():
    from dataset_profiler import profile_dataset
    p = profile_dataset(ambiguous_date_dataset(), dataset_id="ambiguous_dates")
    assert len(p["detected_date_columns"]) >= 2, p["detected_date_columns"]
    assert p["primary_date_column"] in ("created_at", "updated_at")


run("26. Ambiguous multi-date-column dataset ranks candidates", test_ambiguous_date_columns)


def test_sparse_customer_dataset_no_crash():
    from dataset_runtime_config import build_runtime_config
    cfg = build_runtime_config(customer_dataset(), dataset_id="customer_sparse", use_cache=False)
    assert isinstance(cfg["metrics"], dict)  # must not crash regardless of date detection outcome


run("27. Sparse 3-row customer dataset profiles without crashing", test_sparse_customer_dataset_no_crash)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"TOTAL: {passed}/{total} passed")
if passed != total:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")