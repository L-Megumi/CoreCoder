from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from openai import APIConnectionError
from corecoder.llm import LLM, LLMResponse, ScriptedLLM
from researchcoder.runtime.protocol import RuntimeLimits
from researchcoder.runtime.tracing import TraceSink
from researchcoder.runtime.budget import BudgetLedger, BudgetedModel, BudgetExceeded


def ledger(**kwargs):
    return BudgetLedger(RuntimeLimits(**kwargs), TraceSink("j", "t", "a"))


def test_reservation_settlement_duplicates_and_unknown_liability():
    budget = ledger(max_context_tokens=100, max_output_tokens=10, max_total_tokens=100)
    first = budget.reserve_model(20)
    budget.settle(first, prompt=18, completion=5)
    assert budget.charged_tokens == 23
    with pytest.raises(ValueError):
        budget.settle(first, prompt=1, completion=1)
    second = budget.reserve_model(20)
    budget.settle(second, status="error")
    assert budget.unknown_requests == 1 and budget.charged_tokens == 53
    assert budget.usage("live").prompt_tokens is None
    assert budget.snapshot()["settlements"][1]["prompt_tokens"] is None


@pytest.mark.parametrize("field", ["max_model_calls", "max_tool_calls", "max_rounds"])
def test_counter_caps_deny_before_execution(field):
    budget = ledger(**{field: 1})
    action = {"max_model_calls": lambda: budget.reserve_model(1), "max_tool_calls": budget.tool, "max_rounds": budget.round}[
        field
    ]
    action()
    with pytest.raises(BudgetExceeded):
        action()
    assert budget.stop_kind == "budget_exhausted"


def test_context_output_and_total_reservation_caps():
    budget = ledger(max_context_tokens=20, max_output_tokens=10, max_total_tokens=20)
    with pytest.raises(BudgetExceeded):
        budget.reserve_model(11)
    budget = ledger(max_context_tokens=30, max_output_tokens=10, max_total_tokens=20)
    reservation = budget.reserve_model(10)
    with pytest.raises(BudgetExceeded):
        budget.reserve_model(0)
    budget.settle(reservation, prompt=25, completion=2)
    assert budget.charged_tokens == 27 and budget.stop_kind == "budget_exhausted"


def test_failure_calls_and_explicit_zero_usage_have_distinct_meaning():
    budget = ledger()
    model = BudgetedModel(ScriptedLLM([LLMResponse(content="ok", usage_complete=True)]), budget, budget.trace)
    model.chat([{"role": "user", "content": "hi"}])
    assert budget.usage("injected").usage_complete and budget.prompt_tokens == 0
    with pytest.raises(RuntimeError):
        model.chat([])
    assert budget.model_calls == 2 and budget.unknown_requests == 1 and budget.charged_tokens > 0


def test_missing_usage_is_not_zero_and_large_output_stops_before_tools():
    budget = ledger(max_output_tokens=10)
    model = BudgetedModel(ScriptedLLM([LLMResponse(content="x" * 90)]), budget, budget.trace)
    with pytest.raises(BudgetExceeded):
        model.chat([])
    assert budget.unknown_requests == 1 and budget.charged_tokens >= 10


def test_retry_transport_attempts_are_counted_without_real_network(monkeypatch):
    model = LLM("fake-model", "fake-key")
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=2)
    chunk = SimpleNamespace(usage=usage, choices=[])
    model.client.chat.completions.create = Mock(side_effect=[APIConnectionError(request=None), iter([chunk])])
    monkeypatch.setattr("corecoder.llm.time.sleep", lambda _: None)
    budget = ledger()
    response = BudgetedModel(model, budget, budget.trace).chat([{"role": "user", "content": "hi"}])
    assert response.prompt_tokens == 7 and budget.model_calls == 2
    assert budget.unknown_requests == 1 and budget.prompt_tokens == 7
    assert [e["status"] for e in budget.trace.events if e["event_type"] == "model.transport.end"] == ["error", "stream_opened"]


def test_retry_limit_prevents_next_transport_attempt(monkeypatch):
    model = LLM("fake-model", "fake-key")
    model.client.chat.completions.create = Mock(side_effect=APIConnectionError(request=None))
    monkeypatch.setattr("corecoder.llm.time.sleep", lambda _: None)
    budget = ledger(max_model_calls=1)
    with pytest.raises(BudgetExceeded):
        BudgetedModel(model, budget, budget.trace).chat([])
    assert model.client.chat.completions.create.call_count == 1
    assert budget.model_calls == 1 and budget.unknown_requests == 1


@pytest.mark.parametrize("value", [-1, True, "10"])
def test_invalid_estimate_does_not_allocate(value):
    budget = ledger()
    with pytest.raises(ValueError):
        budget.reserve_model(value)
    assert budget.model_calls == 0


def test_context_summary_request_uses_the_same_budget_gateway():
    from corecoder.context import ContextManager

    budget = ledger()
    model = BudgetedModel(ScriptedLLM([LLMResponse(content="summary")]), budget, budget.trace)
    assert ContextManager()._get_summary([{"role": "user", "content": "Earlier investigation"}], model) == "summary"
    assert budget.model_calls == 1 and budget.unknown_requests == 1
