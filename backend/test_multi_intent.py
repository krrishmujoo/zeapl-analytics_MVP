from __future__ import annotations

from pathlib import Path
from conversation_memory import ConversationMemory
from data_sources import load_data
from conversation_orchestrator import (
    deterministic_split,
    split_intents,
    analyze_conversation,
)


DATASET = Path(__file__).parent / "fact_scan_activity.csv"


def test_single_intent_is_not_split():
    question = (
        "What is the total invoice value for scans "
        "in Delhi or Rajasthan?"
    )

    intents = deterministic_split(question)

    assert len(intents) == 1


def test_between_is_not_split():
    question = (
        "What is the total invoice value for scans "
        "where quantity is between 2 and 3?"
    )

    intents = deterministic_split(question)

    assert len(intents) == 1


def test_multi_intent_deterministic_split():
    question = (
        "Give me the total invoice value across all scans "
        "and how many valid scans are there?"
    )

    intents = deterministic_split(question)

    assert len(intents) == 2


def test_semicolon_multi_intent():
    question = (
        "Give me the total invoice value across all scans; "
        "which 5 products have the highest invoice value?"
    )

    intents = deterministic_split(question)

    assert len(intents) == 2


def test_multi_intent_execution():
    data = load_data(DATASET)

    question = (
        "Give me the total invoice value across all scans "
        "and how many valid scans are there?"
    )

    response = analyze_conversation(
        data,
        question=question,
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
    )

    assert response["multi_intent"] is True
    assert response["intent_count"] == 2
    assert len(response["responses"]) == 2

    for item in response["responses"]:
        assert "intent_id" in item
        assert "question" in item
        assert "response" in item

def test_failed_single_intent_is_not_stored():
    memory = ConversationMemory()
    conversation_id = "failed-single-memory"

    memory.add_turn(
        conversation_id=conversation_id,
        question="What is the total zorbatron level?",
        response={
            "status": "error",
            "request": {
                "operation": "llm_analysis",
                "metric_keys": [],
                "filters": [],
                "error": "Unresolved zorbatron level",
            },
            "errors": ["Unresolved zorbatron level"],
        },
        dataset_id="test_dataset",
    )

    assert (
        memory.get_context(
            conversation_id,
            dataset_id="test_dataset",
        )
        == []
    )


def test_partial_success_stores_only_successful_intents():
    memory = ConversationMemory()
    conversation_id = "partial-success-memory"

    successful_request = {
        "operation": "distinct_count",
        "metric_keys": ["scan_id_distinct_count"],
        "metric_fields": ["scan_id"],
        "filters": [],
        "filter_logic": "and",
    }

    memory.add_turn(
        conversation_id=conversation_id,
        question=(
            "How many scans are there; "
            "what is the zorbatron level?"
        ),
        response={
            "status": "partial_success",
            "responses": [
                {
                    "status": "success",
                    "response": {
                        "status": "success",
                        "request": successful_request,
                    },
                },
                {
                    "status": "error",
                    "response": {
                        "status": "error",
                        "request": {
                            "operation": "llm_analysis",
                            "metric_keys": [],
                            "error": "Unresolved zorbatron level",
                        },
                    },
                },
            ],
        },
        dataset_id="test_dataset",
    )

    context = memory.get_context(
        conversation_id,
        dataset_id="test_dataset",
    )

    assert len(context) == 1
    assert context[0]["requests"] == [
        {
            "operation": "distinct_count",
            "metric_keys": ["scan_id_distinct_count"],
            "metric_fields": ["scan_id"],
            "dimension": None,
            "group_by": [],
            "filters": [],
            "filter_logic": "and",
            "granularity": None,
            "horizon": None,
            "top_n": None,
        }
    ]