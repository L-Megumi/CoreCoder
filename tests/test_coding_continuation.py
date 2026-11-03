"""The new opt-in only preserves work; it cannot award success or more budget."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from researchcoder.runtime.budget import BudgetLedger, BudgetExceeded
from researchcoder.runtime.coding import CodingContinuationPolicy, known_local_stop
from researchcoder.runtime.harness import ExecutionConfig
from researchcoder.runtime.protocol import RuntimeLimits, WorkerRequest
from researchcoder.runtime.tracing import TraceSink
from researchcoder.orchestration.budget import JobBudget, JobLimits
from tests.test_runtime_worker import request


def test_typed_continuation_roundtrip_and_default_wire_compatibility(tmp_path):
    policy = CodingContinuationPolicy()
    assert CodingContinuationPolicy.from_json(policy.to_json()) == policy
    with pytest.raises(ValueError):
        CodingContinuationPolicy.from_dict(dict(mode="unlimited"))
    req = request(tmp_path)
    assert "coding_continuation" not in req.to_dict()
    assert WorkerRequest.from_json(req.to_json()) == req
    with pytest.raises(ValueError, match="coding task"):
        replace(req, coding_continuation=policy)
    with pytest.raises(TypeError, match="typed Coding"):
        ExecutionConfig(artifact_root=str(tmp_path / "out"), allowed_workspaces=(), coding_continuation={})


def ledger():
    return BudgetLedger(RuntimeLimits(max_model_calls=1), TraceSink("job", "task", "attempt"))


def test_opt_in_stop_detail_does_not_reset_accounting_or_existing_wire():
    b = ledger()
    reservation = b.reserve_model(10)
    b.settle(reservation, prompt=8, completion=3)
    with pytest.raises(BudgetExceeded):
        b.reserve_model(10)
    assert "local_limit_detail" not in b.snapshot()
    b.coding_continuation = CodingContinuationPolicy()
    assert b.snapshot()["local_limit_detail"] == "model_calls"
    assert b.snapshot()["charged_tokens"] == 11 and b.model_calls == 1
    hooks = SimpleNamespace(ledger=b, permission_stop=False)
    assert known_local_stop(hooks)


@pytest.mark.parametrize("change", ["pending", "unknown", "permission", "tool_error", "model_error", "timeout", "tool_calls", "missing_detail"])
def test_unsafe_or_unproven_stop_never_qualifies(change):
    b = ledger()
    b.stop_kind = "budget_exhausted"
    b.local_limit_detail = "rounds"
    hooks = SimpleNamespace(ledger=b, permission_stop=False)
    if change == "pending": b.pending["reservation"] = 100
    elif change == "unknown": b.unknown_requests = 1
    elif change == "permission": hooks.permission_stop = True
    elif change == "tool_calls": b.local_limit_detail = "tool_calls"
    elif change == "missing_detail": b.local_limit_detail = None
    elif change == "timeout": b.stop_kind = "timeout"
    else: b.last_failure_kind = change
    assert not known_local_stop(hooks)


def test_job_keeps_unknown_liability_for_unsettled_worker_request():
    job = JobBudget(JobLimits(max_model_calls=10, max_tool_calls=10, max_total_tokens=100000))
    caps = RuntimeLimits(max_model_calls=2, max_tool_calls=3, max_total_tokens=50000)
    job.reserve("first", caps)
    settled = job.settle("first", dict(budget=dict(model_calls=1, tool_calls=1,
        charged_tokens=10, unknown_requests=0, pending={"unsettled": 10000})))
    assert settled["usage_unknown"] and not settled["accounting_complete"]
    assert settled["charged"] == settled["reserved"]
    assert job.snapshot()["charged"]["total_tokens"] == 50000
    with pytest.raises(ValueError):
        job.settle("first", {})


def test_unknown_job_settlement_cannot_erase_reported_overrun():
    job = JobBudget()
    limits = RuntimeLimits(max_total_tokens=10000)
    job.reserve("overrun", limits)
    settled = job.settle("overrun", dict(budget=dict(model_calls=1, tool_calls=0,
        charged_tokens=12000, unknown_requests=0, pending={"unsettled": 100})))
    assert settled["usage_unknown"] and settled["charged"]["total_tokens"] == 12000


def test_real_command_then_budget_hook_failure_is_unknown_never_not_run(tmp_path):
    import sys
    from corecoder.llm import ScriptedLLM, LLMResponse
    from researchcoder.contracts import VerificationSpec
    from researchcoder.orchestration.baseline import run_single_agent
    from researchcoder.runtime.budget import BudgetedModel
    from tests.test_coding_context import setup
    manager, _, hooks, io = setup(tmp_path)
    marker = tmp_path / "executed.txt"
    command = VerificationSpec(argv=(sys.executable, "-B", "-c",
        "from pathlib import Path;Path(" + repr(str(marker)) + ").write_text('executed')"))
    spec = replace(manager.request.spec, verification=command)
    req = replace(manager.request, spec=spec, permitted_verifications=(command,),
        coding_continuation=CodingContinuationPolicy())
    hooks.context_engine = None
    def after_command(*_):
        hooks.ledger.stop("tool_error")
    hooks.after_verification = after_command
    factory = lambda: BudgetedModel(ScriptedLLM([LLMResponse(content="Check now", prompt_tokens=3,
        completion_tokens=2, usage_complete=True)]), hooks.ledger, hooks.trace)
    with pytest.raises(BudgetExceeded):
        run_single_agent(spec, req, io, factory, hooks, lambda: None)
    assert marker.read_text() == "executed"
    assert len(hooks.coding_partial_refs) == 1
    raw = io.read_json(hooks.coding_partial_refs[0])
    assert raw["verification"] is None
    assert raw["verification_execution"] == "started_outcome_unknown"
    assert hooks.ledger.stop_kind == "tool_error" and not known_local_stop(hooks)
