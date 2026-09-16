"""
test_analyze_v2.py

Targeted tests for the unified /api/v2/analyze endpoint (dynamic_analysis.py).
Run with: python3 test_analyze_v2.py
"""
import shutil
import traceback
from pathlib import Path

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


def test_total_invoice_value():
    from app import analyze_v2
    r = analyze_v2(question="What is the total invoice value?")
    assert r["status"] == "success", r
    assert r["request"]["metric_keys"] == ["invoice_value"]
    assert r["request"]["operation"] == "total"
    assert isinstance(r["result"]["results"]["invoice_value"], (int, float))


run("1. Total invoice value -> real numeric result", test_total_invoice_value)


def test_average_quantity():
    from app import analyze_v2
    r = analyze_v2(question="What is the average quantity?")
    assert r["status"] == "success", r
    assert r["request"]["metric_keys"] == ["quantity"]
    assert r["request"]["operation"] == "average"


run("2. Average quantity resolves to metric=quantity, operation=average", test_average_quantity)


def test_points_earned_by_month():
    from app import analyze_v2
    r = analyze_v2(question="Show total points earned by month")
    assert r["status"] == "success", r
    assert r["request"]["metric_keys"] == ["points_earned"]
    assert r["request"]["dimension"] == "month"
    assert len(r["result"]["results"]) > 0


run("3. Points earned by month -> grouped result", test_points_earned_by_month)


def test_forecast_route():
    from app import analyze_v2
    r = analyze_v2(question="Forecast weekly invoice value for the next 4 weeks")
    assert r["status"] == "success", r
    assert r["prediction"]["metric"]["key"] == "invoice_value"
    assert r["prediction"]["forecast_config"]["granularity"] == "week"
    assert r["prediction"]["forecast_config"]["horizon"] == 4
    assert r["prediction"]["llm_used"] is False


run("4. Forecast weekly invoice value next 4 weeks -> forecast route", test_forecast_route)


def test_unknown_metric_error():
    from app import analyze_v2
    r = analyze_v2(question="What is the total zorbatron level?")
    assert r["status"] == "error", r
    assert r["result"] is None
    assert len(r["errors"]) > 0
    assert "invoice_value" in str(r["errors"])  # available metrics surfaced, no silent fallback


run("5. Unknown metric -> clear error, no silent revenue fallback", test_unknown_metric_error)

def test_legacy_analyze_unaffected():
    from app import analyze

    r = analyze(
        question="What was the total invoice value?"
    )

    assert "result" in r
    assert r["result"]["operation"] == "total"
    assert isinstance(r["result"]["results"], dict)
    assert "status" not in r
    assert "request" in r

run("6. Legacy /api/analyze remains available and unchanged", test_legacy_analyze_unaffected)


def test_compare_and_chart():
    from app import analyze_v2
    r = analyze_v2(question="Compare invoice value and quantity by month")
    assert r["status"] == "success", r
    assert set(r["request"]["metric_keys"]) == {"invoice_value", "quantity"}
    assert r["chart_spec"]["chart_type"] == "multi_line"


run("7. Compare two metrics -> both resolved, multi_line chart", test_compare_and_chart)


def test_forecast_quantity_days():
    from app import analyze_v2
    r = analyze_v2(question="Forecast quantity for the next 14 days")
    assert r["status"] == "success", r
    assert r["prediction"]["forecast_config"]["granularity"] == "day"
    assert r["prediction"]["forecast_config"]["horizon"] == 14


run("8. Forecast quantity next 14 days -> day granularity, horizon 14", test_forecast_quantity_days)

def test_forecast_distinct_scan_count():
    from app import analyze_v2

    r = analyze_v2(
        question=(
            "Forecast the number of scans "
            "for the next 3 months."
        ),
        conversation_id="test-scan-count-forecast",
    )

    assert r["status"] == "success", r

    prediction = r["prediction"]

    assert (
        prediction["metric"]["key"]
        == "scan_id_distinct_count"
    )
    assert prediction["metric"]["aggregation"] == "nunique"
    assert prediction["forecast_config"]["granularity"] == "month"
    assert prediction["forecast_config"]["horizon"] == 3
    assert len(prediction["forecast"]["periods"]) == 3


run(
    "9. Forecast scan count -> preserve string IDs until nunique",
    test_forecast_distinct_scan_count,
)


print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"TOTAL: {passed}/{total} passed")
if passed != total:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")