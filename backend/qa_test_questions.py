"""
Zeapl AI Analytics - Expanded QA Test Suite

Run from backend:
    python qa_test_questions_v2.py

This suite intentionally tests different query shapes instead of repeating
the same campaign-count questions.

Categories:
1. Accuracy-validated scalar/grouped/filter/ranking tests
2. Time grouping + trend
3. Flat ranking / top-N / bottom-N
4. Nested ranking
5. Multi-intent capability
6. Forecast capability
7. Chart-generation behavior

Important:
- Expected numeric results are calculated independently from the CSV.
- Date-filter comparison treats date-only and timestamp representations as
  equivalent, so the QA validator does not create false failures.
- Nested ranking, multi-intent, and forecast tests are capability tests.
  Their raw API response is printed so failures can be investigated without
  inventing an expected response shape.
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "http://127.0.0.1:8000/api/v2/analyze"
CSV_PATH = Path(__file__).parent / "campaign master dump.csv"
QA_TODAY = pd.Timestamp("2026-08-18")


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------

def load_dataset():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    for column in ("created_at", "start_date", "end_date"):
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed",
            )

    return df


# ---------------------------------------------------------------------------
# INDEPENDENT CALCULATIONS
# ---------------------------------------------------------------------------

def campaign_count(df):
    return int(df["campaign_id"].nunique())


def agent_counts(df):
    grouped = (
        df.groupby("agentID")["campaign_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    return [
        {"agentID": str(agent), "value": int(value)}
        for agent, value in grouped.items()
    ]


def hsm_counts(df):
    grouped = (
        df.groupby("hsm_type")["campaign_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    return [
        {"hsm_type": str(value), "value": int(count)}
        for value, count in grouped.items()
    ]


def frequency_counts(df):
    grouped = (
        df.groupby("frequency")["campaign_id"]
        .nunique()
        .sort_index()
    )
    return [
        {
            "frequency": str(value),
            "campaign_id_distinct_count": int(count),
        }
        for value, count in grouped.items()
    ]


def yearly_counts(df):
    grouped = (
        df.assign(year=df["created_at"].dt.year)
        .dropna(subset=["year"])
        .groupby("year")["campaign_id"]
        .nunique()
        .sort_index()
    )
    return [
        {
            "year": int(year),
            "campaign_id_distinct_count": int(count),
        }
        for year, count in grouped.items()
    ]


def monthly_counts_2026(df):
    filtered = df[
        (df["created_at"] >= pd.Timestamp("2026-01-01"))
        & (df["created_at"] < pd.Timestamp("2027-01-01"))
    ].copy()

    filtered["month"] = filtered["created_at"].dt.to_period("M").astype(str)

    grouped = (
        filtered.groupby("month")["campaign_id"]
        .nunique()
        .sort_index()
    )

    return [
        {
            "period": month,
            "campaign_id_distinct_count": int(count),
        }
        for month, count in grouped.items()
    ]


def normalize_number(value):
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def numeric_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def extract_result_list(data):
    result = data.get("result")

    if not isinstance(result, dict):
        return None

    results = result.get("results")

    if isinstance(results, list):
        return results

    if isinstance(results, dict):
        for value in results.values():
            if isinstance(value, list):
                return value

    return None


def extract_scalar(data):
    result = data.get("result")

    if not isinstance(result, dict):
        return None

    results = result.get("results")

    if not isinstance(results, dict):
        return None

    # Campaign distinct count is the main metric in this dataset.
    if "campaign_id_distinct_count" in results:
        return results["campaign_id_distinct_count"]

    for value in results.values():
        if isinstance(value, (int, float)):
            return value

    return None


def normalize_date_value(value):
    if value is None:
        return None

    text = str(value)

    try:
        return pd.Timestamp(text).date().isoformat()
    except Exception:
        return text


def normalize_filters(filters):
    normalized = []

    for item in filters or []:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        operator = item.get("operator")
        value = item.get("value")

        if operator in {"date_from", "date_to"}:
            value = normalize_date_value(value)

        elif isinstance(value, list):
            value = tuple(
                normalize_date_value(v) if operator in {"date_from", "date_to"}
                else str(v)
                for v in value
            )
        else:
            value = str(value)

        normalized.append((field, operator, value))

    return sorted(normalized)


def filters_match(actual, expected):
    return normalize_filters(actual) == normalize_filters(expected)


# ---------------------------------------------------------------------------
# RESPONSE VALIDATORS
# ---------------------------------------------------------------------------

def validate_scalar(data, expected):
    actual = extract_scalar(data)
    return numeric_equal(actual, expected), actual


def validate_grouped(data, expected, group_key):
    rows = extract_result_list(data)

    if rows is None:
        return False, rows

    actual_map = {}

    for row in rows:
        if group_key not in row:
            return False, rows
        actual_map[str(row[group_key])] = row.get(
            "campaign_id_distinct_count"
        )

    expected_map = {
        str(row[group_key]): row["campaign_id_distinct_count"]
        for row in expected
    }

    if set(actual_map) != set(expected_map):
        return False, rows

    for key, expected_value in expected_map.items():
        if not numeric_equal(actual_map.get(key), expected_value):
            return False, rows

    return True, rows


def validate_ranking(data, expected, dimension, top_n=5):
    rows = extract_result_list(data)

    if rows is None:
        return False, rows

    expected_top = expected[:top_n]

    if len(rows) != len(expected_top):
        return False, rows

    for actual, wanted in zip(rows, expected_top):
        actual_dimension = str(actual.get("dimension"))
        wanted_dimension = str(wanted[dimension])

        if actual_dimension != wanted_dimension:
            return False, rows

        if not numeric_equal(
            actual.get("value"),
            wanted.get("value"),
        ):
            return False, rows

    return True, rows


def validate_plan_shape(data, expected_operation=None):
    request = data.get("request", {})

    if data.get("status") != "success":
        return False, {
            "reason": "API did not return success",
            "errors": data.get("errors"),
        }

    if expected_operation is not None:
        if request.get("operation") != expected_operation:
            return False, {
                "reason": "Operation mismatch",
                "actual_operation": request.get("operation"),
                "expected_operation": expected_operation,
            }

    return True, request


# ---------------------------------------------------------------------------
# TEST DEFINITIONS
# ---------------------------------------------------------------------------

def build_accuracy_tests(df):
    active_by_date = df[
        (df["start_date"] <= QA_TODAY)
        & (df["end_date"] >= QA_TODAY)
    ]

    created_2025 = df[
        (df["created_at"] >= pd.Timestamp("2025-01-01"))
        & (df["created_at"] < pd.Timestamp("2026-01-01"))
    ]

    whatsapp_2026 = df[
        (df["mode"].astype(str).str.casefold() == "whatsapp")
        & (df["created_at"] >= pd.Timestamp("2026-01-01"))
        & (df["created_at"] < pd.Timestamp("2027-01-01"))
    ]

    template_or_hsmbw = df[
        df["hsm_type"]
        .astype(str)
        .str.casefold()
        .isin({"template", "hsmbw"})
    ]

    return [
        {
            "id": "A01",
            "category": "Scalar",
            "question": "How many campaigns are there?",
            "kind": "scalar",
            "expected_operation": "distinct_count",
            "expected": campaign_count(df),
        },
        {
            "id": "A02",
            "category": "Grouped",
            "question": "How many campaigns are there for each frequency?",
            "kind": "grouped",
            "group_key": "frequency",
            "expected_operation": "distinct_count",
            "expected": frequency_counts(df),
        },
        {
            "id": "A03",
            "category": "Ranking",
            "question": "Which agents have created the most campaigns?",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected_operation": "top_users",
            "expected": agent_counts(df),
        },
        {
            "id": "A04",
            "category": "Bottom Ranking",
            "question": "Which agents have created the fewest campaigns?",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected_operation": "bottom_users",
            "expected": list(reversed(agent_counts(df))),
        },
        {
            "id": "A05",
            "category": "Top-N",
            "question": "Which are the top 3 agents by campaign count?",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 3,
            "expected_operation": "top_users",
            "expected": agent_counts(df),
        },
        {
            "id": "A06",
            "category": "Filter AND",
            "question": "How many WhatsApp campaigns were created in 2026?",
            "kind": "scalar",
            "expected_operation": "distinct_count",
            "expected": campaign_count(whatsapp_2026),
        },
        {
            "id": "A07",
            "category": "OR Filter",
            "question": "How many campaigns are TEMPLATE or HSMBW?",
            "kind": "scalar",
            "expected_operation": "distinct_count",
            "expected": campaign_count(template_or_hsmbw),
        },
        {
            "id": "A08",
            "category": "Date Range",
            "question": "How many campaigns were created in 2025?",
            "kind": "scalar",
            "expected_operation": "distinct_count",
            "expected": campaign_count(created_2025),
        },
        {
            "id": "A09",
            "category": "Date + Grouping",
            "question": "How many campaigns were created in 2025 for each agent?",
            "kind": "grouped",
            "group_key": "agentID",
            "expected_operation": "distinct_count",
            "expected": [
                {
                    "agentID": agent,
                    "campaign_id_distinct_count": count,
                }
                for agent, count in (
                    created_2025.groupby("agentID")["campaign_id"]
                    .nunique()
                    .sort_index()
                    .items()
                )
            ],
        },
        {
            "id": "A10",
            "category": "Current Date Window",
            "question": "How many campaigns are active today?",
            "kind": "scalar",
            "expected_operation": "distinct_count",
            "expected": campaign_count(active_by_date),
            "special": "current_active",
        },
        {
            "id": "A11",
            "category": "Year Grouping",
            "question": "How many campaigns were created each year?",
            "kind": "grouped",
            "group_key": "year",
            "expected_operation": "distinct_count",
            "expected": yearly_counts(df),
        },
        {
            "id": "A12",
            "category": "Monthly Grouping",
            "question": "How many campaigns were created each month in 2026?",
            "kind": "trend_like",
            "expected_operation": "distinct_count",
            "expected": monthly_counts_2026(df),
        },
    ]


CAPABILITY_TESTS = [
    {
        "id": "C01",
        "category": "Nested Ranking",
        "question": (
            "Which are the top 2 agents by campaign count, and within "
            "each agent which templates are used the most?"
        ),
        "check": "nested_ranking",
    },
    {
        "id": "C02",
        "category": "Nested Ranking + Filter",
        "question": (
            "Which are the top 2 agents by campaign count, and within "
            "each agent which templates are used most for campaigns "
            "created in 2026?"
        ),
        "check": "nested_ranking",
    },
    {
        "id": "C03",
        "category": "Multi-Intent",
        "question": (
            "How many campaigns are there, and which agents have the "
            "most campaigns?"
        ),
        "check": "multi_intent",
    },
    {
        "id": "C04",
        "category": "Multi-Intent",
        "question": (
            "How many WhatsApp campaigns are there and how many template "
            "campaigns are there?"
        ),
        "check": "multi_intent",
    },
    {
        "id": "C05",
        "category": "Trend",
        "question": "Show the campaign creation trend by month in 2026.",
        "check": "trend",
    },
    {
        "id": "C06",
        "category": "Chart",
        "question": "Show a bar chart of campaigns by agent.",
        "check": "chart",
    },
    {
        "id": "C07",
        "category": "Forecast",
        "question": "Forecast campaign creation for the next 3 months.",
        "check": "forecast",
    },
    {
        "id": "C08",
        "category": "Natural Language Ranking",
        "question": "Show me the 2 agents with the largest campaign workload.",
        "check": "top_n",
    },
]


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def call_api(question):
    params = urllib.parse.urlencode(
        {
            "question": question,
            "use_llm_fallback": "true",
        }
    )

    url = f"{BASE_URL}?{params}"

    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        status_code = response.status
        raw = response.read().decode("utf-8")
        return status_code, json.loads(raw)


# ---------------------------------------------------------------------------
# ACCURACY TESTS
# ---------------------------------------------------------------------------

def run_accuracy_test(test, df):
    try:
        http_status, data = call_api(test["question"])
        request = data.get("request", {})

        if data.get("status") != "success":
            return False, {
                "reason": "API returned non-success status",
                "errors": data.get("errors"),
            }, http_status, data

        if request.get("operation") != test["expected_operation"]:
            return False, {
                "reason": "Operation mismatch",
                "expected": test["expected_operation"],
                "actual": request.get("operation"),
            }, http_status, data

        if test["kind"] == "scalar":
            matched, actual = validate_scalar(
                data,
                test["expected"],
            )

        elif test["kind"] == "grouped":
            matched, actual = validate_grouped(
                data,
                test["expected"],
                test["group_key"],
            )

        elif test["kind"] == "ranking":
            matched, actual = validate_ranking(
                data,
                test["expected"],
                test["dimension"],
                test["top_n"],
            )

        elif test["kind"] == "trend_like":
            # For time-grouping tests, validate that the API did not add
            # the raw created_at field alongside the requested time bucket.
            group_by = request.get("group_by") or []
            granularity = request.get("granularity")

            matched = (
                "created_at" not in group_by
                and "month" in group_by
                and granularity == "month"
            )
            actual = {
                "group_by": group_by,
                "granularity": granularity,
                "result": data.get("result"),
            }

        else:
            return False, {
                "reason": f"Unknown test kind: {test['kind']}"
            }, http_status, data

        if not matched:
            return False, {
                "reason": "Independent result/schema validation failed",
                "expected": test["expected"],
                "actual": actual,
            }, http_status, data

        return True, {
            "reason": "Independent calculation and semantic request matched"
        }, http_status, data

    except Exception as error:
        return False, {
            "reason": "Test execution error",
            "error": str(error),
        }, None, {}


# ---------------------------------------------------------------------------
# CAPABILITY TESTS
# ---------------------------------------------------------------------------

def run_capability_test(test):
    try:
        http_status, data = call_api(test["question"])
        request = data.get("request", {})

        check = test["check"]

        if check == "nested_ranking":
            levels = request.get("ranking_levels") or []

            passed = (
                data.get("status") == "success"
                and request.get("operation") in {
                    "top_users",
                    "bottom_users",
                }
                and len(levels) == 2
                and levels[0].get("dimension") != levels[1].get("dimension")
                and all(
                    isinstance(level.get("top_n"), int)
                    and level.get("top_n") > 0
                    for level in levels
                )
            )

            details = {
                "ranking_levels": levels,
                "operation": request.get("operation"),
                "result": data.get("result"),
                "errors": data.get("errors"),
            }

        elif check == "multi_intent":
            # A genuine multi-intent response must expose more than one
            # independently addressable result/intent. We do not assume a
            # response format that has not been observed.
            result = data.get("result")

            possible_intent_containers = []

            if isinstance(result, list):
                possible_intent_containers = result

            elif isinstance(result, dict):
                for key in (
                    "intents",
                    "results",
                    "responses",
                    "sub_results",
                    "analyses",
                ):
                    value = result.get(key)
                    if isinstance(value, list):
                        possible_intent_containers = value
                        break

            passed = (
                data.get("status") == "success"
                and len(possible_intent_containers) >= 2
            )

            details = {
                "detected_independent_intents": len(
                    possible_intent_containers
                ),
                "request": request,
                "result": result,
                "errors": data.get("errors"),
            }

        elif check == "trend":
            passed = (
                data.get("status") == "success"
                and request.get("operation") == "trend"
                and request.get("granularity") == "month"
                and data.get("chart_data") is not None
            )

            details = {
                "operation": request.get("operation"),
                "granularity": request.get("granularity"),
                "chart_spec": data.get("chart_spec"),
                "chart_data": data.get("chart_data"),
                "errors": data.get("errors"),
            }

        elif check == "chart":
            passed = (
                data.get("status") == "success"
                and data.get("chart_spec") is not None
                and data.get("chart_data") is not None
            )

            details = {
                "operation": request.get("operation"),
                "chart_spec": data.get("chart_spec"),
                "chart_data_count": (
                    len(data.get("chart_data") or [])
                    if isinstance(data.get("chart_data"), list)
                    else None
                ),
                "errors": data.get("errors"),
            }

        elif check == "forecast":
            passed = (
                data.get("status") == "success"
                and request.get("operation") == "forecast"
                and data.get("prediction") is not None
            )

            details = {
                "operation": request.get("operation"),
                "horizon": request.get("horizon"),
                "granularity": request.get("granularity"),
                "prediction": data.get("prediction"),
                "errors": data.get("errors"),
                "warnings": data.get("warnings"),
            }

        elif check == "top_n":
            rows = extract_result_list(data)
            passed = (
                data.get("status") == "success"
                and request.get("operation") == "top_users"
                and request.get("top_n") == 2
                and rows is not None
                and len(rows) == 2
            )

            details = {
                "operation": request.get("operation"),
                "top_n": request.get("top_n"),
                "rows": rows,
                "errors": data.get("errors"),
            }

        else:
            passed = False
            details = {"reason": f"Unknown capability check: {check}"}

        return passed, details, http_status, data

    except Exception as error:
        return False, {
            "reason": "Capability test execution error",
            "error": str(error),
        }, None, {}


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def print_test_result(
    test_id,
    category,
    question,
    passed,
    http_status,
    request,
    details,
):
    print("=" * 110)
    print(
        f"{'PASS' if passed else 'FAIL'} "
        f"[{test_id}] [{category}] {question}"
    )
    print(f"HTTP: {http_status}")
    print(f"Operation: {request.get('operation')}")
    print(f"Metric keys: {request.get('metric_keys')}")
    print(f"Dimension: {request.get('dimension')}")
    print(f"Group by: {request.get('group_by')}")
    print(f"Granularity: {request.get('granularity')}")
    print(f"Top N: {request.get('top_n')}")
    print(f"Ranking levels: {request.get('ranking_levels')}")
    print(f"Filter logic: {request.get('filter_logic')}")
    print(f"Filters: {request.get('filters')}")
    print(f"Details: {details}")


def main():
    df = load_dataset()

    print()
    print("=" * 110)
    print("ZEAPL AI ANALYTICS - EXPANDED QA TEST SUITE")
    print("=" * 110)
    print(f"Dataset: {CSV_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Independent QA date: {QA_TODAY.date()}")
    print(f"API: {BASE_URL}")
    print("=" * 110)

    accuracy_passed = 0
    accuracy_failed = 0

    print("\n\n### ACCURACY-VALIDATED TESTS ###")

    for test in build_accuracy_tests(df):
        passed, details, http_status, data = run_accuracy_test(
            test,
            df,
        )

        request = data.get("request", {}) if data else {}

        print_test_result(
            test["id"],
            test["category"],
            test["question"],
            passed,
            http_status,
            request,
            details,
        )

        if passed:
            accuracy_passed += 1
        else:
            accuracy_failed += 1

    capability_passed = 0
    capability_failed = 0

    print("\n\n### CAPABILITY / ADVANCED TESTS ###")

    for test in CAPABILITY_TESTS:
        passed, details, http_status, data = run_capability_test(test)

        request = data.get("request", {}) if data else {}

        print_test_result(
            test["id"],
            test["category"],
            test["question"],
            passed,
            http_status,
            request,
            details,
        )

        if passed:
            capability_passed += 1
        else:
            capability_failed += 1

    print("\n\n" + "=" * 110)
    print("QA SUMMARY")
    print("=" * 110)

    print(
        f"Accuracy tests:     "
        f"{accuracy_passed} passed / {accuracy_failed} failed"
    )

    print(
        f"Capability tests:   "
        f"{capability_passed} passed / {capability_failed} failed"
    )

    print(
        f"Total tests:        "
        f"{accuracy_passed + accuracy_failed + capability_passed + capability_failed}"
    )

    print("=" * 110)

    if accuracy_failed or capability_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()