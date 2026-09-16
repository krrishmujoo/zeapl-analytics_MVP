from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_sources import load_data
from dataset_runtime_config import build_runtime_config
from dynamic_analysis import (
    _extract_nested_ranking_request,
    analyze_dynamic,
)

DATASET = Path(__file__).parent / "fact_scan_activity.csv"
def _parse_nested_query(query):
    data = load_data(DATASET)

    config = build_runtime_config(
        data,
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
    )

    return _extract_nested_ranking_request(
        query=query,
        data=data,
        config=config,
    )



def _nested_plan(filters=None):
    return {
        "operation": "top_users",
        "metric_keys": ["invoice_value"],
        "field": None,
        "dimension": "user_id",
        "group_by": [],
        "filters": filters or [],
        "filter_logic": "and",
        "unresolved_conditions": [],
        "top_n": 5,
        "ranking_levels": [
            {
                "dimension": "scheme_id",
                "top_n": 5,
            },
            {
                "dimension": "user_id",
                "top_n": 5,
            },
        ],
        "granularity": None,
        "horizon": None,
        "requires_llm": False,
        "confidence": 1.0,
        "semantic_reason": "Nested ranking test plan.",
    }

def _generic_nested_plan(
    *,
    operation,
    metric_key,
    outer_dimension,
    outer_top_n,
    inner_dimension,
    inner_top_n,
    filters=None,
):
    return {
        "operation": operation,
        "metric_keys": [metric_key],
        "field": None,
        "dimension": inner_dimension,
        "group_by": [inner_dimension],
        "filters": filters or [],
        "filter_logic": "and",
        "unresolved_conditions": [],
        "top_n": inner_top_n,
        "ranking_levels": [
            {
                "dimension": outer_dimension,
                "top_n": outer_top_n,
            },
            {
                "dimension": inner_dimension,
                "top_n": inner_top_n,
            },
        ],
        "granularity": None,
        "horizon": None,
        "requires_llm": False,
        "confidence": 1.0,
        "semantic_reason": (
            "Generalized nested-ranking execution test."
        ),
    }

def _analyze(
    plan,
    query=(
        "Top 5 users in the top 5 schemes "
        "by invoice value"
    ),
):
    data = load_data(DATASET)

    return analyze_dynamic(
        data=data,
        query=query,
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=False,
        _plan_override=plan,
    )


def test_nested_ranking_contract():
    response = _analyze(_nested_plan())

    assert response["status"] == "success", response
    assert response["request"]["ranking_levels"] == [
        {
            "dimension": "scheme_id",
            "top_n": 5,
        },
        {
            "dimension": "user_id",
            "top_n": 5,
        },
    ]

    result = response["result"]

    assert result["operation"] == "nested_ranking"
    assert result["metric"] == "invoice_value"
    assert result["direction"] == "top"
    assert len(result["results"]) == 5

    for outer_index, outer_row in enumerate(
        result["results"],
        start=1,
    ):
        assert outer_row["rank"] == outer_index
        assert outer_row["dimension"] == "scheme_id"
        assert outer_row["scheme_id"] == outer_row["entity"]
        assert isinstance(outer_row["value"], (int, float))
        assert len(outer_row["children"]) <= 5

        for inner_index, child in enumerate(
            outer_row["children"],
            start=1,
        ):
            assert child["rank"] == inner_index
            assert child["dimension"] == "user_id"
            assert child["user_id"] == child["entity"]
            assert isinstance(child["value"], (int, float))


def test_nested_ranking_matches_dataset_aggregations():
    data = load_data(DATASET)
    frame = pd.DataFrame(data)

    frame["invoice_value"] = pd.to_numeric(
        frame["invoice_value"],
        errors="coerce",
    )

    expected_outer = (
        frame.groupby("scheme_id")["invoice_value"]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(5)
    )

    response = _analyze(_nested_plan())
    actual_outer = response["result"]["results"]

    assert [row["entity"] for row in actual_outer] == [
        str(value)
        for value in expected_outer.index
    ]

    for row, expected_value in zip(
        actual_outer,
        expected_outer.values,
    ):
        assert row["value"] == pytest.approx(
            float(expected_value)
        )

        expected_inner = (
            frame[
                frame["scheme_id"].astype(str)
                == str(row["entity"])
            ]
            .groupby("user_id")["invoice_value"]
            .sum(min_count=1)
            .sort_values(ascending=False)
            .head(5)
        )

        assert [
            child["entity"]
            for child in row["children"]
        ] == [
            str(value)
            for value in expected_inner.index
        ]

        for child, inner_value in zip(
            row["children"],
            expected_inner.values,
        ):
            assert child["value"] == pytest.approx(
                float(inner_value)
            )


def test_nested_ranking_respects_filters():
    plan = _nested_plan(
        filters=[
            {
                "field": "state",
                "operator": "equals",
                "value": "Rajasthan",
            }
        ]
    )

    response = _analyze(
        plan,
        query="Top 5 users by invoice value",
    )

    assert response["status"] == "success", response

    data = load_data(DATASET)
    frame = pd.DataFrame(data)
    frame = frame[frame["state"] == "Rajasthan"].copy()

    frame["invoice_value"] = pd.to_numeric(
        frame["invoice_value"],
        errors="coerce",
    )

    expected = (
        frame.groupby("scheme_id")["invoice_value"]
        .sum(min_count=1)
        .sort_values(ascending=False)
        .head(5)
    )

    actual = response["result"]["results"]

    assert [row["entity"] for row in actual] == [
        str(value)
        for value in expected.index
    ]

    for row, expected_value in zip(actual, expected.values):
        assert row["value"] == pytest.approx(
            float(expected_value)
        )


def test_flat_ranking_remains_backward_compatible():
    plan = _nested_plan()
    plan["ranking_levels"] = []
    plan["dimension"] = "user_id"

    response = _analyze(
        plan,
        query="Top 5 users by invoice value",
    )

    assert response["status"] == "success", response
    assert response["result"]["operation"] == "top_users"
    assert response["result"]["dimension"] == "user_id"
    assert isinstance(
        response["result"]["results"]["invoice_value"],
        list,
    )

@pytest.mark.parametrize(
    ("query", "operation", "outer", "outer_n", "inner", "inner_n"),
    [
        (
            "Top 5 users in the top 3 schemes by invoice value",
            "top_users",
            "scheme_id",
            3,
            "user_id",
            5,
        ),
        (
            "Highest 4 products within the highest 2 states by quantity",
            "top_users",
            "state",
            2,
            "product_id",
            4,
        ),
        (
            "5 best users inside the 3 best schemes by points earned",
            "top_users",
            "scheme_id",
            3,
            "user_id",
            5,
        ),
        (
            "Bottom 3 products among the bottom 2 cities by quantity",
            "bottom_users",
            "city",
            2,
            "product_id",
            3,
        ),
        (
            "2 lowest schemes inside the 4 lowest states by invoice value",
            "bottom_users",
            "state",
            4,
            "scheme_id",
            2,
        ),
        (
            "Largest 6 users within the largest 3 cities by points earned",
            "top_users",
            "city",
            3,
            "user_id",
            6,
        ),
    ],
)
def test_generalized_nested_ranking_language(
    query,
    operation,
    outer,
    outer_n,
    inner,
    inner_n,
):
    parsed = _parse_nested_query(query)

    assert parsed is not None
    assert parsed["operation"] == operation
    assert parsed["ranking_levels"] == [
        {
            "dimension": outer,
            "top_n": outer_n,
        },
        {
            "dimension": inner,
            "top_n": inner_n,
        },
    ]


@pytest.mark.parametrize(
    "query",
    [
        # Mixed ranking direction is not yet supported safely.
        "Top 5 users in the bottom 3 schemes by invoice value",

        # Unknown dimensions must never be invented.
        "Top 5 astronauts in the top 3 planets by invoice value",

        # Same dimension at both levels is not a meaningful nesting.
        "Top 5 users in the top 3 users by invoice value",

        # No explicit outer ranking limit: this means grouped ranking,
        # which requires a different execution contract.
        "Top 5 users per scheme by invoice value",

        # Ordinary one-level ranking must stay one-level.
        "Top 5 users by invoice value",
    ],
)
def test_unsupported_or_non_nested_language_is_not_guessed(
    query,
):
    assert _parse_nested_query(query) is None


def test_city_plural_resolves_to_city_dimension():
    parsed = _parse_nested_query(
        "Top 3 users in the top 2 cities by invoice value"
    )

    assert parsed is not None
    assert parsed["ranking_levels"][0]["dimension"] == "city"

@pytest.mark.parametrize(
    (
        "query",
        "operation",
        "metric_key",
        "outer_dimension",
        "outer_top_n",
        "inner_dimension",
        "inner_top_n",
    ),
    [
        (
            "Highest 4 products within the highest 2 states by quantity",
            "top_users",
            "quantity",
            "state",
            2,
            "product_id",
            4,
        ),
        (
            "Bottom 3 products among the bottom 2 cities by points earned",
            "bottom_users",
            "points_earned",
            "city",
            2,
            "product_id",
            3,
        ),
    ],
)
def test_generalized_nested_ranking_execution(
    query,
    operation,
    metric_key,
    outer_dimension,
    outer_top_n,
    inner_dimension,
    inner_top_n,
):
    plan = _generic_nested_plan(
        operation=operation,
        metric_key=metric_key,
        outer_dimension=outer_dimension,
        outer_top_n=outer_top_n,
        inner_dimension=inner_dimension,
        inner_top_n=inner_top_n,
    )

    response = _analyze(
        plan,
        query=query,
    )

    assert response["status"] == "success", response

    result = response["result"]

    assert result["operation"] == "nested_ranking"
    assert result["metric"] == metric_key
    assert result["direction"] == (
        "top"
        if operation == "top_users"
        else "bottom"
    )
    assert result["ranking_levels"] == [
        {
            "dimension": outer_dimension,
            "top_n": outer_top_n,
        },
        {
            "dimension": inner_dimension,
            "top_n": inner_top_n,
        },
    ]

    assert len(result["results"]) <= outer_top_n

    reverse = operation == "top_users"

    outer_values = [
        row["value"]
        for row in result["results"]
    ]

    assert outer_values == sorted(
        outer_values,
        reverse=reverse,
    )

    for outer_row in result["results"]:
        assert outer_row["dimension"] == outer_dimension
        assert len(outer_row["children"]) <= inner_top_n

        inner_values = [
            child["value"]
            for child in outer_row["children"]
        ]

        assert inner_values == sorted(
            inner_values,
            reverse=reverse,
        )

        for child in outer_row["children"]:
            assert child["dimension"] == inner_dimension

@pytest.mark.parametrize(
    "query",
    [
        (
            "Top 5 users in the bottom 3 schemes "
            "by invoice value"
        ),
        (
            "Top 5 astronauts in the top 3 planets "
            "by invoice value"
        ),
        (
            "Top 5 users in the top 3 users "
            "by invoice value"
        ),
    ],
)
def test_unsafe_nested_ranking_returns_error(query):
    data = load_data(DATASET)

    response = analyze_dynamic(
        data=data,
        query=query,
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=False,
    )

    assert response["status"] == "error"
    assert response["result"] is None
    assert response["errors"]
    assert "could not be resolved safely" in response["errors"][0]