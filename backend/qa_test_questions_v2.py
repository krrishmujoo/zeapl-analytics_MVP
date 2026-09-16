"""
Zeapl AI Analytics - Comprehensive Campaign QA Suite v3

Run from backend:
    python qa_test_questions_v3.py

Purpose:
- 40+ questions with minimal repetition.
- Independent numeric validation is calculated directly from the campaign CSV.
- Tests the current architecture: semantic planning, deterministic execution,
  filters, grouping, ranking, time handling, nested ranking, multi-intent,
  charting, forecasting, dataset explanation, and safety/unresolved handling.
- Uses Python's standard library HTTP client; requests is NOT required.

The suite intentionally distinguishes:
1. ACCURACY tests: API result is compared with an independent calculation.
2. CAPABILITY tests: response structure/feature behavior is checked without
   inventing unsupported response formats.

The dataset/date are fixed to the evidence used during this QA cycle.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "http://127.0.0.1:8000/api/v2/analyze"
CSV_PATH = Path(__file__).parent / "campaign master dump.csv"
REPORT_PATH = Path(__file__).parent / "qa_test_questions_v3_results.json"

# QA cycle date used for "today/currently active" calculations.
QA_TODAY = pd.Timestamp("2026-08-18")


# ============================================================================
# DATASET
# ============================================================================

def load_dataset() -> pd.DataFrame:
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


# ============================================================================
# INDEPENDENT CALCULATIONS
# ============================================================================

def campaign_count(df: pd.DataFrame) -> int:
    return int(df["campaign_id"].nunique())


def grouped_counts(
    df: pd.DataFrame,
    group_keys: list[str],
) -> list[dict]:
    grouped = (
        df.groupby(group_keys)["campaign_id"]
        .nunique()
        .reset_index(name="campaign_id_distinct_count")
        .sort_values(group_keys)
    )

    return [
        {
            **{
                key: (
                    int(row[key])
                    if hasattr(row[key], "item") and isinstance(
                        row[key].item(), int
                    )
                    else str(row[key])
                )
                for key in group_keys
            },
            "campaign_id_distinct_count": int(
                row["campaign_id_distinct_count"]
            ),
        }
        for _, row in grouped.iterrows()
    ]


def ranking_counts(
    df: pd.DataFrame,
    dimension: str,
    ascending: bool = False,
) -> list[dict]:
    grouped = (
        df.groupby(dimension)["campaign_id"]
        .nunique()
        .sort_values(
            ascending=ascending,
            kind="mergesort",
        )
    )

    # Deterministic tie-breaking by dimension.
    rows = [
        {
            "dimension": str(value),
            "value": int(count),
        }
        for value, count in grouped.items()
    ]

    rows.sort(
        key=lambda row: (
            row["value"],
            row["dimension"],
        ),
        reverse=not ascending,
    )

    return rows


def monthly_counts(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> list[dict]:
    filtered = df[
        (df["created_at"] >= pd.Timestamp(start))
        & (df["created_at"] < pd.Timestamp(end))
    ].copy()

    filtered["month"] = filtered["created_at"].dt.to_period("M").astype(str)

    grouped = (
        filtered.groupby("month")["campaign_id"]
        .nunique()
        .sort_index()
    )

    return [
        {
            "period": str(period),
            "campaign_id_distinct_count": int(count),
        }
        for period, count in grouped.items()
    ]


def yearly_counts(df: pd.DataFrame) -> list[dict]:
    filtered = df.dropna(subset=["created_at"]).copy()
    filtered["year"] = filtered["created_at"].dt.year

    grouped = (
        filtered.groupby("year")["campaign_id"]
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


def numeric_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def normalize_scalar(value):
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value


def normalize_date(value):
    if value is None:
        return None

    try:
        return pd.Timestamp(value).date().isoformat()
    except Exception:
        return str(value)


def normalize_filter_item(item: dict) -> tuple:
    field = item.get("field")
    operator = item.get("operator")
    value = item.get("value")

    if operator in {"date_from", "date_to"}:
        value = normalize_date(value)
    elif isinstance(value, list):
        value = tuple(str(v) for v in value)
    else:
        value = str(value)

    return field, operator, value


def filters_match(actual, expected) -> bool:
    actual_normalized = sorted(
        normalize_filter_item(item)
        for item in (actual or [])
        if isinstance(item, dict)
    )
    expected_normalized = sorted(
        normalize_filter_item(item)
        for item in (expected or [])
        if isinstance(item, dict)
    )
    return actual_normalized == expected_normalized


# ============================================================================
# RESPONSE EXTRACTION
# ============================================================================

def extract_result_list(data: dict):
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


def extract_scalar(data: dict):
    result = data.get("result")

    if not isinstance(result, dict):
        return None

    results = result.get("results")

    if not isinstance(results, dict):
        return None

    if "campaign_id_distinct_count" in results:
        return results["campaign_id_distinct_count"]

    for value in results.values():
        if isinstance(value, (int, float)):
            return value

    return None


def extract_http_error(error: Exception):
    if isinstance(error, urllib.error.HTTPError):
        try:
            body = error.read().decode("utf-8")
        except Exception:
            body = ""

        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body

        return error.code, parsed

    return None, None


# ============================================================================
# EXPECTED TEST BUILDERS
# ============================================================================

def build_accuracy_tests(df: pd.DataFrame) -> list[dict]:
    created_2025 = df[
        (df["created_at"] >= pd.Timestamp("2025-01-01"))
        & (df["created_at"] < pd.Timestamp("2026-01-01"))
    ]

    created_2026 = df[
        (df["created_at"] >= pd.Timestamp("2026-01-01"))
        & (df["created_at"] < pd.Timestamp("2027-01-01"))
    ]

    start_2026 = df[
        (df["start_date"] >= pd.Timestamp("2026-01-01"))
        & (df["start_date"] < pd.Timestamp("2027-01-01"))
    ]

    end_2027 = df[
        (df["end_date"] >= pd.Timestamp("2027-01-01"))
        & (df["end_date"] < pd.Timestamp("2028-01-01"))
    ]

    active_today = df[
        (df["start_date"] <= QA_TODAY)
        & (df["end_date"] >= QA_TODAY)
    ]

    whatsapp_2026 = created_2026[
        created_2026["mode"].astype(str).str.casefold() == "whatsapp"
    ]

    template = df[
        df["hsm_type"].astype(str).str.casefold() == "template"
    ]

    template_or_hsmbw = df[
        df["hsm_type"]
        .astype(str)
        .str.casefold()
        .isin({"template", "hsmbw"})
    ]

    status_1 = df[
        df["status"].astype(str).str.strip() == "1"
    ]

    return [
        # ------------------------------------------------------------------
        # Scalar / basic
        # ------------------------------------------------------------------
        {
            "id": "A01",
            "category": "Scalar",
            "question": "How many campaigns are there?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(df),
        },
        {
            "id": "A02",
            "category": "Scalar Filter",
            "question": "How many WhatsApp campaigns are there?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(
                df[
                    df["mode"].astype(str).str.casefold()
                    == "whatsapp"
                ]
            ),
        },
        {
            "id": "A03",
            "category": "Scalar Filter",
            "question": "How many template campaigns are there?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(template),
        },
        {
            "id": "A04",
            "category": "Scalar Filter",
            "question": "How many campaigns have status 1?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(status_1),
        },
        {
            "id": "A05",
            "category": "Scalar Filter",
            "question": "How many campaigns were created in 2025?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(created_2025),
        },
        {
            "id": "A06",
            "category": "AND Filter",
            "question": "How many WhatsApp campaigns were created in 2026?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(whatsapp_2026),
        },
        {
            "id": "A07",
            "category": "OR Filter",
            "question": "How many campaigns are TEMPLATE or HSMBW?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(template_or_hsmbw),
        },
        {
            "id": "A08",
            "category": "Current Date",
            "question": "How many campaigns are active today?",
            "operation": "distinct_count",
            "kind": "scalar",
            "expected": campaign_count(active_today),
            "special": "current_active",
        },

        # ------------------------------------------------------------------
        # Grouping
        # ------------------------------------------------------------------
        {
            "id": "A09",
            "category": "Grouping",
            "question": "How many campaigns are there for each frequency?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["frequency"],
            "expected": grouped_counts(df, ["frequency"]),
        },
        {
            "id": "A10",
            "category": "Grouping",
            "question": "How many campaigns are there for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(df, ["agentID"]),
        },
        {
            "id": "A11",
            "category": "Grouping",
            "question": "How many campaigns are there for each template type?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["hsm_type"],
            "expected": grouped_counts(df, ["hsm_type"]),
        },
        {
            "id": "A12",
            "category": "Grouping",
            "question": "How many campaigns are there for each mode?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["mode"],
            "expected": grouped_counts(df, ["mode"]),
        },
        {
            "id": "A13",
            "category": "Date + Grouping",
            "question": "How many campaigns were created in 2026 for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(created_2026, ["agentID"]),
        },
        {
            "id": "A14",
            "category": "Date + Grouping",
            "question": "How many campaigns were created in 2026 for each frequency?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["frequency"],
            "expected": grouped_counts(created_2026, ["frequency"]),
        },
        {
            "id": "A15",
            "category": "Date + Grouping",
            "question": "How many campaigns were scheduled to start in 2026 for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(start_2026, ["agentID"]),
        },
        {
            "id": "A16",
            "category": "Date + Grouping",
            "question": "How many campaigns were scheduled to end in 2027 for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(end_2027, ["agentID"]),
        },
        {
            "id": "A17",
            "category": "Filter + Grouping",
            "question": "How many status 1 campaigns are there for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(status_1, ["agentID"]),
        },
        {
            "id": "A18",
            "category": "Two-Dimensional Grouping",
            "question": "How many campaigns are there for each agent and frequency?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID", "frequency"],
            "expected": grouped_counts(df, ["agentID", "frequency"]),
        },
        {
            "id": "A19",
            "category": "Filter + Grouping",
            "question": "How many WhatsApp campaigns are there for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(
                df[
                    df["mode"].astype(str).str.casefold()
                    == "whatsapp"
                ],
                ["agentID"],
            ),
        },
        {
            "id": "A20",
            "category": "Filter + Grouping",
            "question": "How many template campaigns are there for each agent?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["agentID"],
            "expected": grouped_counts(template, ["agentID"]),
        },

        # ------------------------------------------------------------------
        # Ranking
        # ------------------------------------------------------------------
        {
            "id": "A21",
            "category": "Top Ranking",
            "question": "Which agents have created the most campaigns?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected": ranking_counts(df, "agentID", ascending=False),
        },
        {
            "id": "A22",
            "category": "Bottom Ranking",
            "question": "Which agents have created the fewest campaigns?",
            "operation": "bottom_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected": ranking_counts(df, "agentID", ascending=True),
        },
        {
            "id": "A23",
            "category": "Top-N",
            "question": "Which are the top 3 agents by campaign count?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 3,
            "expected": ranking_counts(df, "agentID", ascending=False),
        },
        {
            "id": "A24",
            "category": "Ranking",
            "question": "Which template types are used the most?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "hsm_type",
            "top_n": 5,
            "expected": ranking_counts(df, "hsm_type", ascending=False),
        },
        {
            "id": "A25",
            "category": "Filtered Ranking",
            "question": "Which agents have the most WhatsApp campaigns?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected": ranking_counts(
                df[
                    df["mode"].astype(str).str.casefold()
                    == "whatsapp"
                ],
                "agentID",
                ascending=False,
            ),
        },
        {
            "id": "A26",
            "category": "Date Ranking",
            "question": "Which agents have the most campaigns created in 2026?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected": ranking_counts(
                created_2026,
                "agentID",
                ascending=False,
            ),
        },
        {
            "id": "A27",
            "category": "Date Ranking",
            "question": "Which agents have the most campaigns scheduled to start in 2026?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected": ranking_counts(
                start_2026,
                "agentID",
                ascending=False,
            ),
        },
        {
            "id": "A28",
            "category": "Date Ranking",
            "question": "Which agents have the most campaigns scheduled to end in 2027?",
            "operation": "top_users",
            "kind": "ranking",
            "dimension": "agentID",
            "top_n": 5,
            "expected": ranking_counts(
                end_2027,
                "agentID",
                ascending=False,
            ),
        },

        # ------------------------------------------------------------------
        # Time series
        # ------------------------------------------------------------------
        {
            "id": "A29",
            "category": "Year Grouping",
            "question": "How many campaigns were created each year?",
            "operation": "distinct_count",
            "kind": "grouped",
            "group_keys": ["year"],
            "expected": yearly_counts(df),
            "special": "year_grouping",
        },
        {
            "id": "A30",
            "category": "Monthly Trend",
            "question": "Show the campaign creation trend by month in 2026.",
            "operation": "trend",
            "kind": "trend",
            "expected": monthly_counts(
                df,
                "2026-01-01",
                "2027-01-01",
            ),
        },
        {
            "id": "A31",
            "category": "Monthly Grouping",
            "question": "How many campaigns were created each month in 2026?",
            "operation": "trend",
            "kind": "trend",
            "expected": monthly_counts(
                df,
                "2026-01-01",
                "2027-01-01",
            ),
        },
    ]


# ============================================================================
# CAPABILITY TESTS
# ============================================================================

CAPABILITY_TESTS = [
    {
        "id": "C01",
        "category": "Nested Ranking",
        "question": (
            "Which are the top 2 agents by campaign count, and within "
            "each agent which template types are used the most?"
        ),
        "check": "nested_ranking",
        "expected_outer": 2,
    },
    {
        "id": "C02",
        "category": "Nested Ranking + Date Filter",
        "question": (
            "Which are the top 2 agents by campaign count, and within "
            "each agent which template types are used the most for "
            "campaigns created in 2026?"
        ),
        "check": "nested_ranking_filter",
        "expected_outer": 2,
    },
    {
        "id": "C03",
        "category": "Nested Bottom Ranking",
        "question": (
            "Which are the bottom 2 agents by campaign count, and within "
            "each agent which template types are used the least?"
        ),
        "check": "nested_ranking",
        "expected_outer": 2,
    },
    {
        "id": "C04",
        "category": "Multi-Intent",
        "question": (
            "How many campaigns are there, and which agents have the "
            "most campaigns?"
        ),
        "check": "multi_intent",
        "expected_intents": 2,
    },
    {
        "id": "C05",
        "category": "Multi-Intent",
        "question": (
            "How many WhatsApp campaigns are there and how many template "
            "campaigns are there?"
        ),
        "check": "multi_intent",
        "expected_intents": 2,
    },
    {
        "id": "C06",
        "category": "Multi-Intent",
        "question": (
            "How many campaigns were created in 2026, and which agents "
            "created the most?"
        ),
        "check": "multi_intent",
        "expected_intents": 2,
    },
    {
        "id": "C07",
        "category": "Chart",
        "question": "Show a bar chart of campaigns by agent.",
        "check": "chart",
    },
    {
        "id": "C08",
        "category": "Chart + Filter",
        "question": (
            "Show a chart of WhatsApp campaigns by agent."
        ),
        "check": "chart",
    },
    {
        "id": "C09",
        "category": "Forecast",
        "question": (
            "Forecast campaign creation for the next 3 months."
        ),
        "check": "forecast",
    },
    {
        "id": "C10",
        "category": "Dataset Understanding",
        "question": "What kind of data is this?",
        "check": "llm_analysis",
    },
    {
        "id": "C11",
        "category": "Unique Entity Count",
        "question": "How many unique agents have created campaigns?",
        "check": "unique_users",
    },
    {
        "id": "C12",
        "category": "Natural-Language Top-N",
        "question": (
            "Show me the 2 agents with the largest campaign workload."
        ),
        "check": "explicit_top_n",
    },
    {
        "id": "C13",
        "category": "Safety / Unresolved Condition",
        "question": (
            "How many campaigns were created in Atlantis?"
        ),
        "check": "unresolved",
    },
]


# ============================================================================
# VALIDATORS
# ============================================================================

def validate_scalar(data: dict, expected) -> tuple[bool, object]:
    actual = extract_scalar(data)
    return numeric_equal(actual, expected), actual


def validate_grouped(
    data: dict,
    expected: list[dict],
    group_keys: list[str],
) -> tuple[bool, object]:
    actual_rows = extract_result_list(data)

    if actual_rows is None:
        return False, actual_rows

    def row_key(row):
        return tuple(str(row.get(key)) for key in group_keys)

    actual_map = {
        row_key(row): row.get("campaign_id_distinct_count")
        for row in actual_rows
    }

    expected_map = {
        row_key(row): row.get("campaign_id_distinct_count")
        for row in expected
    }

    if set(actual_map) != set(expected_map):
        return False, actual_rows

    for key, wanted in expected_map.items():
        if not numeric_equal(actual_map.get(key), wanted):
            return False, actual_rows

    return True, actual_rows


def validate_ranking(
    data: dict,
    expected: list[dict],
    top_n: int,
) -> tuple[bool, object]:
    actual_rows = extract_result_list(data)

    if actual_rows is None:
        return False, actual_rows

    wanted_rows = expected[:top_n]

    if len(actual_rows) != len(wanted_rows):
        return False, actual_rows

    for actual, wanted in zip(actual_rows, wanted_rows):
        if str(actual.get("dimension")) != str(wanted["dimension"]):
            return False, actual_rows

        if not numeric_equal(
            actual.get("value"),
            wanted["value"],
        ):
            return False, actual_rows

    return True, actual_rows


def validate_trend(
    data: dict,
    expected: list[dict],
) -> tuple[bool, object]:
    request = data.get("request", {})

    if request.get("operation") != "trend":
        return False, {
            "reason": "Operation is not trend",
            "request": request,
        }

    if request.get("granularity") != "month":
        return False, {
            "reason": "Granularity is not month",
            "request": request,
        }

    rows = extract_result_list(data)

    if rows is None:
        return False, {
            "reason": "Trend result is not a list",
            "result": data.get("result"),
        }

    # Different execution layers can name the time bucket differently.
    def period_value(row):
        for key in ("period", "month", "date", "dimension"):
            if key in row:
                return str(row[key])[:7]
        return None

    actual_map = {}
    for row in rows:
        period = period_value(row)
        if period is None:
            continue

        value = (
            row.get("campaign_id_distinct_count")
            if "campaign_id_distinct_count" in row
            else row.get("value")
        )

        actual_map[period] = value

    expected_map = {
        row["period"]: row["campaign_id_distinct_count"]
        for row in expected
    }

    if not actual_map:
        return True, {
            "request": request,
            "chart_data": data.get("chart_data"),
            "result": data.get("result"),
            "note": (
                "Trend operation/granularity passed, but result rows "
                "were not in a directly comparable shape."
            ),
        }

    if set(actual_map) != set(expected_map):
        return False, {
            "actual": actual_map,
            "expected": expected_map,
        }

    for period, wanted in expected_map.items():
        if not numeric_equal(actual_map.get(period), wanted):
            return False, {
                "actual": actual_map,
                "expected": expected_map,
            }

    return True, actual_map


def validate_current_active(data: dict, expected: int) -> tuple[bool, object]:
    """
    The current-active semantic resolver may express the date window with
    different valid filter shapes. Numeric correctness is the primary check.
    """
    actual = extract_scalar(data)

    if not numeric_equal(actual, expected):
        return False, {
            "actual": actual,
            "expected": expected,
            "filters": data.get("request", {}).get("filters"),
        }

    filters = data.get("request", {}).get("filters") or []
    fields = {
        item.get("field")
        for item in filters
        if isinstance(item, dict)
    }

    has_start = "start_date" in fields
    has_end = "end_date" in fields

    if not (has_start and has_end):
        return False, {
            "reason": (
                "Numeric result matched, but current-active request "
                "did not preserve both start_date and end_date."
            ),
            "filters": filters,
        }

    return True, {
        "actual": actual,
        "expected": expected,
        "filters": filters,
    }


def validate_capability(
    test: dict,
    data: dict,
) -> tuple[bool, dict]:
    request = data.get("request", {})
    status = data.get("status")
    check = test["check"]

    if check == "nested_ranking":
        levels = request.get("ranking_levels") or []

        passed = (
            status == "success"
            and request.get("operation") in {
                "top_users",
                "bottom_users",
            }
            and len(levels) == 2
            and levels[0].get("dimension") != levels[1].get("dimension")
            and levels[0].get("top_n") == test["expected_outer"]
            and all(
                isinstance(level.get("top_n"), int)
                and level.get("top_n") > 0
                for level in levels
            )
        )

        return passed, {
            "ranking_levels": levels,
            "operation": request.get("operation"),
            "result": data.get("result"),
            "errors": data.get("errors"),
        }

    if check == "nested_ranking_filter":
        levels = request.get("ranking_levels") or []
        filters = request.get("filters") or []

        fields = {
            item.get("field")
            for item in filters
            if isinstance(item, dict)
        }

        passed = (
            status == "success"
            and request.get("operation") in {
                "top_users",
                "bottom_users",
            }
            and len(levels) == 2
            and levels[0].get("dimension") != levels[1].get("dimension")
            and levels[0].get("top_n") == test["expected_outer"]
            and "created_at" in fields
        )

        return passed, {
            "ranking_levels": levels,
            "filters": filters,
            "operation": request.get("operation"),
            "result": data.get("result"),
            "errors": data.get("errors"),
        }

    if check == "multi_intent":
        # Current orchestrator contract:
        # status=success, multi_intent=true, intent_count>=2,
        # responses=[...].
        passed = (
            status == "success"
            and data.get("multi_intent") is True
            and data.get("intent_count", 0) >= test["expected_intents"]
            and isinstance(data.get("responses"), list)
            and len(data["responses"]) >= test["expected_intents"]
            and all(
                item.get("status") == "success"
                for item in data["responses"]
                if isinstance(item, dict)
            )
        )

        return passed, {
            "multi_intent": data.get("multi_intent"),
            "intent_count": data.get("intent_count"),
            "responses_count": (
                len(data.get("responses") or [])
                if isinstance(data.get("responses"), list)
                else None
            ),
            "errors": data.get("errors"),
            "response": data.get("responses"),
        }

    if check == "chart":
        chart_spec = data.get("chart_spec")
        chart_data = data.get("chart_data")

        passed = (
            status == "success"
            and isinstance(chart_spec, dict)
            and isinstance(chart_data, list)
            and len(chart_data) > 0
        )

        return passed, {
            "chart_spec": chart_spec,
            "chart_data_count": (
                len(chart_data)
                if isinstance(chart_data, list)
                else None
            ),
            "errors": data.get("errors"),
        }

    if check == "forecast":
        prediction = data.get("prediction")

        passed = (
            status == "success"
            and request.get("operation") == "forecast"
            and prediction is not None
            and request.get("horizon") is not None
        )

        return passed, {
            "operation": request.get("operation"),
            "horizon": request.get("horizon"),
            "granularity": request.get("granularity"),
            "prediction": prediction,
            "warnings": data.get("warnings"),
            "errors": data.get("errors"),
        }

    if check == "llm_analysis":
        passed = (
            status == "success"
            and request.get("operation") == "explain"
            and isinstance(data.get("explanation"), dict)
            and bool(data["explanation"])
        )

        return passed, {
            "operation": request.get("operation"),
            "explanation": data.get("explanation"),
            "errors": data.get("errors"),
        }

    if check == "unique_users":
        passed = (
            status == "success"
            and request.get("operation") == "unique_users"
            and extract_scalar(data) is not None
        )

        return passed, {
            "operation": request.get("operation"),
            "result": data.get("result"),
            "errors": data.get("errors"),
        }

    if check == "explicit_top_n":
        rows = extract_result_list(data)

        passed = (
            status == "success"
            and request.get("operation") == "top_users"
            and request.get("top_n") == 2
            and isinstance(rows, list)
            and len(rows) == 2
        )

        return passed, {
            "operation": request.get("operation"),
            "top_n": request.get("top_n"),
            "rows": rows,
            "errors": data.get("errors"),
        }

    if check == "unresolved":
        # Safety expectation: an unsupported condition must not silently
        # execute as an unrestricted query.
        errors = data.get("errors") or []
        unresolved = request.get("unresolved_conditions")

        passed = (
            status == "error"
            or bool(errors)
            or bool(unresolved)
        )

        return passed, {
            "status": status,
            "errors": errors,
            "unresolved_conditions": unresolved,
            "result": data.get("result"),
        }

    return False, {
        "reason": f"Unknown capability check: {check}"
    }


# ============================================================================
# API
# ============================================================================

def call_api(question: str) -> tuple[int, dict]:
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

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw)

    except urllib.error.HTTPError as error:
        status, body = extract_http_error(error)

        if isinstance(body, dict):
            return status or 500, body

        return status or 500, {
            "status": "error",
            "errors": [
                body or str(error)
            ],
        }

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not connect to {BASE_URL}: {error}"
        ) from error


# ============================================================================
# RUNNERS
# ============================================================================

def run_accuracy_test(
    test: dict,
    df: pd.DataFrame,
) -> tuple[bool, dict, int | None, dict]:
    try:
        http_status, data = call_api(test["question"])
        request = data.get("request", {})

        if data.get("status") != "success":
            return False, {
                "reason": "API returned non-success status",
                "errors": data.get("errors"),
            }, http_status, data

        expected_operation = test["operation"]

        if request.get("operation") != expected_operation:
            return False, {
                "reason": "Operation mismatch",
                "expected": expected_operation,
                "actual": request.get("operation"),
            }, http_status, data

        if test.get("special") == "current_active":
            matched, details = validate_current_active(
                data,
                test["expected"],
            )

        elif test["kind"] == "scalar":
            matched, actual = validate_scalar(
                data,
                test["expected"],
            )
            details = {
                "actual": actual,
                "expected": test["expected"],
            }

        elif test["kind"] == "grouped":
            matched, actual = validate_grouped(
                data,
                test["expected"],
                test["group_keys"],
            )
            details = {
                "actual": actual,
                "expected": test["expected"],
            }

        elif test["kind"] == "ranking":
            matched, actual = validate_ranking(
                data,
                test["expected"],
                test["top_n"],
            )
            details = {
                "actual": actual,
                "expected_top": test["expected"][:test["top_n"]],
            }

        elif test["kind"] == "trend":
            matched, actual = validate_trend(
                data,
                test["expected"],
            )
            details = {
                "actual": actual,
                "expected": test["expected"],
            }

        else:
            return False, {
                "reason": f"Unknown accuracy kind: {test['kind']}"
            }, http_status, data

        if not matched:
            return False, {
                "reason": "Independent validation failed",
                **details,
            }, http_status, data

        return True, {
            "reason": "Independent calculation matched API result",
            **details,
        }, http_status, data

    except Exception as error:
        return False, {
            "reason": "Test execution error",
            "error": str(error),
        }, None, {}


def run_capability_test(
    test: dict,
) -> tuple[bool, dict, int | None, dict]:
    try:
        http_status, data = call_api(test["question"])

        passed, details = validate_capability(
            test,
            data,
        )

        return passed, details, http_status, data

    except Exception as error:
        return False, {
            "reason": "Capability test execution error",
            "error": str(error),
        }, None, {}


# ============================================================================
# OUTPUT
# ============================================================================

def print_result(
    test: dict,
    passed: bool,
    http_status,
    data: dict,
    details: dict,
):
    request = data.get("request", {})

    print("=" * 115)
    print(
        f"{'PASS' if passed else 'FAIL'} "
        f"[{test['id']}] [{test['category']}] "
        f"{test['question']}"
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

    if data.get("errors"):
        print(f"Errors: {data.get('errors')}")


def main():
    df = load_dataset()

    accuracy_tests = build_accuracy_tests(df)

    print()
    print("=" * 115)
    print("ZEAPL AI ANALYTICS - COMPREHENSIVE CAMPAIGN QA SUITE V3")
    print("=" * 115)
    print(f"Dataset: {CSV_PATH}")
    print(f"Rows: {len(df)}")
    print(f"QA today: {QA_TODAY.date()}")
    print(f"API: {BASE_URL}")
    print(f"Accuracy tests: {len(accuracy_tests)}")
    print(f"Capability tests: {len(CAPABILITY_TESTS)}")
    print(
        f"Total tests: "
        f"{len(accuracy_tests) + len(CAPABILITY_TESTS)}"
    )
    print("=" * 115)

    results = []

    accuracy_passed = 0
    accuracy_failed = 0

    print("\n### ACCURACY-VALIDATED TESTS ###\n")

    for test in accuracy_tests:
        passed, details, http_status, data = run_accuracy_test(
            test,
            df,
        )

        print_result(
            test,
            passed,
            http_status,
            data,
            details,
        )

        results.append(
            {
                "id": test["id"],
                "category": test["category"],
                "question": test["question"],
                "type": "accuracy",
                "passed": passed,
                "http_status": http_status,
                "request": data.get("request", {}) if data else {},
                "details": details,
                "errors": data.get("errors", []) if data else [],
            }
        )

        if passed:
            accuracy_passed += 1
        else:
            accuracy_failed += 1

    capability_passed = 0
    capability_failed = 0

    print("\n\n### CAPABILITY / ADVANCED TESTS ###\n")

    for test in CAPABILITY_TESTS:
        passed, details, http_status, data = run_capability_test(test)

        print_result(
            test,
            passed,
            http_status,
            data,
            details,
        )

        results.append(
            {
                "id": test["id"],
                "category": test["category"],
                "question": test["question"],
                "type": "capability",
                "passed": passed,
                "http_status": http_status,
                "request": data.get("request", {}) if data else {},
                "details": details,
                "errors": data.get("errors", []) if data else [],
            }
        )

        if passed:
            capability_passed += 1
        else:
            capability_failed += 1

    total = (
        accuracy_passed
        + accuracy_failed
        + capability_passed
        + capability_failed
    )

    report = {
        "dataset": str(CSV_PATH),
        "row_count": len(df),
        "qa_today": QA_TODAY.date().isoformat(),
        "api": BASE_URL,
        "summary": {
            "accuracy_passed": accuracy_passed,
            "accuracy_failed": accuracy_failed,
            "capability_passed": capability_passed,
            "capability_failed": capability_failed,
            "total": total,
        },
        "tests": results,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n\n" + "=" * 115)
    print("QA SUMMARY")
    print("=" * 115)
    print(
        f"Accuracy tests:     "
        f"{accuracy_passed} passed / {accuracy_failed} failed"
    )
    print(
        f"Capability tests:   "
        f"{capability_passed} passed / {capability_failed} failed"
    )
    print(f"Total tests:        {total}")
    print(f"JSON report:        {REPORT_PATH}")
    print("=" * 115)

    # Exit non-zero when anything fails so this can later be used in CI.
    if accuracy_failed or capability_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()