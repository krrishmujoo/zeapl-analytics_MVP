from pathlib import Path

import dynamic_analysis
from data_sources import load_data

import json

import LLM
from dataset_runtime_config import build_runtime_config
from semantic_planner import build_recovery_plan


DATASET = Path(__file__).parent / "fact_scan_activity.csv"


def _failed_request():
    return {
        "query": "What is the total invoce value?",
        "operation": "total",
        "dimension": None,
        "filters": [],
        "error": "forced semantic resolution failure",
    }


def _repaired_plan():
    return {
        "operation": "total",
        "metric_keys": ["invoice_value"],
        "field": None,
        "dimension": None,
        "group_by": [],
        "filters": [],
        "filter_logic": "and",
        "unresolved_conditions": [],
        "validation_errors": [],
        "top_n": None,
        "granularity": None,
        "horizon": None,
        "requires_llm": False,
        "confidence": 0.9,
        "semantic_reason": "Recovered the misspelled invoice value metric.",
        "resolution_method": "llm_error_recovery",
        "error": None,
    }


def _install_failed_normal_resolver(monkeypatch):
    original_resolver = dynamic_analysis.resolve_semantic_request

    def fake_resolver(
        query,
        data,
        config,
        plan_override=None,
    ):
        if plan_override is None:
            return _failed_request()

        return original_resolver(
            query=query,
            data=data,
            config=config,
            plan_override=plan_override,
        )

    monkeypatch.setattr(
        dynamic_analysis,
        "resolve_semantic_request",
        fake_resolver,
    )


def test_recovery_is_not_called_when_disabled(monkeypatch):
    data = load_data(DATASET)
    _install_failed_normal_resolver(monkeypatch)

    def unexpected_recovery(**kwargs):
        raise AssertionError(
            "Recovery must not run when use_llm_fallback is false."
        )

    monkeypatch.setattr(
        dynamic_analysis,
        "build_recovery_plan",
        unexpected_recovery,
    )

    response = dynamic_analysis.analyze_dynamic(
        data=data,
        query="What is the total invoce value?",
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=False,
    )

    assert response["status"] == "error"
    assert response["result"] is None
    assert response["errors"] == [
        "forced semantic resolution failure"
    ]


def test_valid_recovery_uses_normal_execution_path(monkeypatch):
    data = load_data(DATASET)
    _install_failed_normal_resolver(monkeypatch)

    monkeypatch.setattr(
        dynamic_analysis,
        "build_recovery_plan",
        lambda **kwargs: _repaired_plan(),
    )

    response = dynamic_analysis.analyze_dynamic(
        data=data,
        query="What is the total invoce value?",
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=True,
    )

    assert response["status"] == "success"
    assert response["result"] is not None
    assert (
        response["request"]["metric_keys"]
        == ["invoice_value"]
    )
    assert (
        response["request"]["resolution_method"]
        == "llm_error_recovery"
    )


def test_failed_recovery_returns_original_error(monkeypatch):
    data = load_data(DATASET)
    _install_failed_normal_resolver(monkeypatch)

    monkeypatch.setattr(
        dynamic_analysis,
        "build_recovery_plan",
        lambda **kwargs: None,
    )

    response = dynamic_analysis.analyze_dynamic(
        data=data,
        query="What is the total invoce value?",
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=True,
    )

    assert response["status"] == "error"
    assert response["result"] is None
    assert response["errors"] == [
        "forced semantic resolution failure"
    ]

def _executable_request():
    return {
        "query": "What is the total invoice value?",
        "metric_keys": ["invoice_value"],
        "metric_fields": ["invoice_value"],
        "field": "invoice_value",
        "operation": "total",
        "top_n": None,
        "dimension": None,
        "group_by": [],
        "location": None,
        "filters": [],
        "filter_logic": "and",
        "requires_chart": False,
        "requires_forecast": False,
        "requires_llm": False,
        "requires_insights": False,
        "granularity": None,
        "horizon": None,
        "resolution_method": "test_normal_plan",
        "resolution_confidence": 1.0,
        "semantic_reason": "Test request.",
        "error": None,
    }


def _install_execution_failure_once(monkeypatch):
    original_resolver = dynamic_analysis.resolve_semantic_request
    original_execute = dynamic_analysis.execute_request
    calls = {"execute": 0}

    def fake_resolver(
        query,
        data,
        config,
        plan_override=None,
    ):
        if plan_override is None:
            return _executable_request()

        return original_resolver(
            query=query,
            data=data,
            config=config,
            plan_override=plan_override,
        )

    def fake_execute(data, config, resolved):
        calls["execute"] += 1

        if calls["execute"] == 1:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": ["forced execution failure"],
            }

        return original_execute(
            data,
            config,
            resolved,
        )

    monkeypatch.setattr(
        dynamic_analysis,
        "resolve_semantic_request",
        fake_resolver,
    )

    monkeypatch.setattr(
        dynamic_analysis,
        "execute_request",
        fake_execute,
    )

    return calls


def test_execution_failure_recovery_is_gated(monkeypatch):
    data = load_data(DATASET)
    calls = _install_execution_failure_once(monkeypatch)

    def unexpected_recovery(**kwargs):
        raise AssertionError(
            "Execution recovery must not run when fallback is disabled."
        )

    monkeypatch.setattr(
        dynamic_analysis,
        "build_recovery_plan",
        unexpected_recovery,
    )

    response = dynamic_analysis.analyze_dynamic(
        data=data,
        query="What is the total invoice value?",
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=False,
    )

    assert response["status"] == "error"
    assert response["result"] is None
    assert calls["execute"] == 1
    assert "forced execution failure" in response["errors"]


def test_execution_failure_uses_repaired_normal_path(monkeypatch):
    data = load_data(DATASET)
    calls = _install_execution_failure_once(monkeypatch)

    monkeypatch.setattr(
        dynamic_analysis,
        "build_recovery_plan",
        lambda **kwargs: _repaired_plan(),
    )

    response = dynamic_analysis.analyze_dynamic(
        data=data,
        query="What is the total invoice value?",
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
        use_llm_fallback=True,
    )

    assert response["status"] == "success"
    assert response["result"] is not None
    assert calls["execute"] == 2
    assert (
        response["request"]["resolution_method"]
        == "llm_error_recovery"
    )

class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, plan):
        self.content = [
            _FakeTextBlock(
                json.dumps(plan)
            )
        ]


class _FakeMessages:
    def __init__(self, plan):
        self.plan = plan

    def create(self, **kwargs):
        return _FakeResponse(self.plan)


class _FakeClient:
    def __init__(self, plan):
        self.messages = _FakeMessages(plan)


def _runtime_config(data):
    return build_runtime_config(
        data,
        dataset_id=DATASET.stem,
        source_path=str(DATASET),
    )


def test_recovery_rejects_invented_metric(monkeypatch):
    data = load_data(DATASET)
    config = _runtime_config(data)

    invented_plan = {
        "operation": "total",
        "metric_keys": ["invented_revenue"],
        "field": None,
        "dimension": None,
        "group_by": [],
        "unresolved_conditions": [],
        "filter_logic": "and",
        "filters": [],
        "top_n": None,
        "granularity": None,
        "horizon": None,
        "requires_llm": False,
        "confidence": 1.0,
        "semantic_reason": "Invented metric.",
    }

    monkeypatch.setattr(
        LLM,
        "get_anthropic_client",
        lambda: _FakeClient(invented_plan),
    )

    repaired = build_recovery_plan(
        query="What is the total zorbatron level?",
        data=data,
        config=config,
        failed_plan=_failed_request(),
        error_message="unknown metric",
    )

    assert repaired is None


def test_recovery_rejects_removed_filter(monkeypatch):
    data = load_data(DATASET)
    config = _runtime_config(data)

    filter_dropping_plan = {
        "operation": "total",
        "metric_keys": ["invoice_value"],
        "field": None,
        "dimension": None,
        "group_by": [],
        "unresolved_conditions": [],
        "filter_logic": "and",
        "filters": [],
        "top_n": None,
        "granularity": None,
        "horizon": None,
        "requires_llm": False,
        "confidence": 1.0,
        "semantic_reason": "Dropped the original filter.",
    }

    monkeypatch.setattr(
        LLM,
        "get_anthropic_client",
        lambda: _FakeClient(filter_dropping_plan),
    )

    failed_plan = _failed_request()
    failed_plan["filters"] = [
        {
            "field": "state",
            "operator": "equals",
            "value": "Rajasthan",
        }
    ]

    repaired = build_recovery_plan(
        query="What is invoice value in Rajasthan?",
        data=data,
        config=config,
        failed_plan=failed_plan,
        error_message="execution failure",
    )

    assert repaired is None