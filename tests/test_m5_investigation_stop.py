"""Typed no-progress guards preserve old behavior unless v2 is explicitly selected."""
from dataclasses import replace

import pytest

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.workflow import RepairAgent, RepairInvestigationLimit, project_directory
from researchcoder.orchestration.baseline import run_single_agent
from researchcoder.runtime.budget import BudgetedModel, BudgetLedger, RuntimeHooks
from researchcoder.runtime.coding import CodingContinuationPolicy, investigation_stop_proof
from researchcoder.runtime.protocol import RuntimeLimits
from researchcoder.runtime.tracing import TraceSink
from researchcoder.evaluation.projects.handoff import HandoffPolicy, _valid_investigation_window
from tests.test_coding_context import setup


def read_model(count=9):
    return ScriptedLLM([LLMResponse(tool_calls=[ToolCall(str(i), "read_file", {"file_path": "module.py"})],
        prompt_tokens=10, completion_tokens=5, usage_complete=True) for i in range(count)])


def stopped_agent(tmp_path, limit=3, typed=True):
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n")
    trace = TraceSink("job", "task", "attempt")
    ledger = BudgetLedger(RuntimeLimits(max_model_calls=12, max_rounds=16), trace)
    hooks = RuntimeHooks(ledger, trace, ("read_file",))
    with project_directory(root):
        agent = RepairAgent(BudgetedModel(read_model(), ledger, trace),
            {"project_root": str(root), "file_snapshots": {}},
            allowed_tool_names=("read_file",), max_investigation_rounds=limit,
            **({} if typed is None else {"typed_investigation_stops": typed}))
        agent.execution_hooks = hooks
        with pytest.raises(RuntimeError) as stopped:
            agent.chat("Investigate the public source")
    return agent, stopped.value, hooks


def test_exact_guard_is_typed_and_retains_original_threshold_and_message(tmp_path):
    agent, error, hooks = stopped_agent(tmp_path)
    assert isinstance(error, RuntimeError)
    assert str(error) == "Repair stopped: consecutive investigation rounds exhausted without an effective file edit"
    assert agent.investigation_rounds == agent.max_investigation_rounds == 3
    assert len(agent.events) == hooks.ledger.model_calls == hooks.ledger.tool_calls == 3
    proof = investigation_stop_proof(agent, error, hooks, CodingContinuationPolicy(mode="local-stop-v2"))
    assert proof["window_event_ends"] == [1, 2, 3]
    assert proof["event_count"] == 3 and proof["effective_file_edits"] == 0
    assert _valid_investigation_window(proof, agent.events)
    assert hooks.ledger.stop_kind is None and hooks.ledger.charged_tokens == 45


@pytest.mark.parametrize("typed", [None, False])
def test_default_and_explicit_legacy_guard_keep_exact_runtime_error(tmp_path, typed):
    agent, error, hooks = stopped_agent(tmp_path, typed=typed)
    assert type(error) is RuntimeError
    assert investigation_stop_proof(agent, error, hooks, CodingContinuationPolicy(mode="local-stop-v2")) is None


@pytest.mark.parametrize("value", [1, "true", None])
def test_typed_stop_flag_requires_trusted_boolean(tmp_path, value):
    with pytest.raises(ValueError, match="trusted boolean"):
        RepairAgent(ScriptedLLM([]), {"project_root": str(tmp_path), "file_snapshots": {}},
            typed_investigation_stops=value)


@pytest.mark.parametrize("mode", [None, "local-stop-v1"])
def test_legacy_policy_does_not_opt_into_typed_guard(tmp_path, mode):
    agent, error, hooks = stopped_agent(tmp_path)
    policy = CodingContinuationPolicy(mode=mode) if mode else None
    assert investigation_stop_proof(agent, error, hooks, policy) is None


@pytest.mark.parametrize("kind", ["same_message", "forged_typed", "subclass"])
def test_error_string_or_unowned_exception_cannot_authorize_continuation(tmp_path, kind):
    agent, error, hooks = stopped_agent(tmp_path)
    class Subtype(RepairInvestigationLimit):
        pass
    cls = {"same_message": RuntimeError, "forged_typed": RepairInvestigationLimit, "subclass": Subtype}[kind]
    assert investigation_stop_proof(agent, cls(str(error)), hooks,
        CodingContinuationPolicy(mode="local-stop-v2")) is None


@pytest.mark.parametrize("change", ["pending", "unknown", "permission", "tool_error", "model_error",
    "timeout", "budget_stop", "changed_events", "changed_mutations", "wrong_rounds"])
def test_unsafe_or_stale_runtime_state_cannot_qualify(tmp_path, change):
    agent, error, hooks = stopped_agent(tmp_path)
    if change == "pending": hooks.ledger.pending["unfinished"] = 200
    elif change == "unknown": hooks.ledger.unknown_requests = 1
    elif change == "permission": hooks.permission_stop = True
    elif change in ("tool_error", "model_error"): hooks.ledger.last_failure_kind = change
    elif change in ("timeout", "budget_stop"): hooks.ledger.stop_kind = change
    elif change == "changed_events": agent.events.append(dict(tool="read_file"))
    elif change == "changed_mutations": agent.mutation_count += 1
    else: agent.investigation_rounds -= 1
    assert investigation_stop_proof(agent, error, hooks, CodingContinuationPolicy(mode="local-stop-v2")) is None


def test_effective_edit_restarts_the_observed_investigation_window(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n")
    responses = [LLMResponse(tool_calls=[ToolCall("first", "read_file", {"file_path": "module.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "module.py",
            "old_string": "value = 1", "new_string": "value = 2"})])]
    responses += [LLMResponse(tool_calls=[ToolCall("last-" + str(i), "read_file", {"file_path": "module.py"})]) for i in range(3)]
    with project_directory(root):
        agent = RepairAgent(ScriptedLLM(responses), {"project_root": str(root), "file_snapshots": {}},
            allowed_tool_names=("read_file", "edit_file"), max_investigation_rounds=3,
            typed_investigation_stops=True)
        with pytest.raises(RepairInvestigationLimit):
            agent.chat("Read, edit and read again")
    assert agent._investigation_window_start == 2
    assert agent._investigation_window_ends == [3, 4, 5]
    assert agent.mutation_count == 1


@pytest.mark.parametrize("mode", ["local-stop-v1", "local-stop-v2"])
def test_versioned_policy_roundtrip(mode):
    for value in (CodingContinuationPolicy(mode=mode), HandoffPolicy(mode=mode)):
        assert type(value).from_json(value.to_json()) == value


def baseline_setup(tmp_path, mode, *, source=None):
    engine, _, hooks, io = setup(tmp_path, source=source)
    hooks.context_engine = None
    # The ninth loop entry must reach the existing eight-investigation guard;
    # the shared setup's default eight-round task would stop earlier in Agent.chat.
    spec = replace(engine.request.spec, budget=replace(engine.request.spec.budget, max_rounds=12))
    request = replace(engine.request, spec=spec, coding_continuation=CodingContinuationPolicy(mode=mode))
    hooks.ledger.coding_continuation = request.coding_continuation
    factory = lambda: BudgetedModel(read_model(), hooks.ledger, hooks.trace)
    return request, hooks, io, factory


def test_v2_baseline_archives_incomplete_and_runs_real_failed_public_check(tmp_path):
    request, hooks, io, factory = baseline_setup(tmp_path, "local-stop-v2")
    result = run_single_agent(request.spec, request, io, factory, hooks, lambda: None)
    assert result.status == "incomplete" and result.failure_kind == "budget_exhausted"
    assert not result.workflow_completed and result.verification.status == "failed"
    assert result.verification.exit_code == 1 and result.verification.checked_workspace_revision
    raw = io.read_json(result.outputs[0])
    proof = raw["coding_stop_receipt"]
    assert proof["policy_mode"] == "local-stop-v2" and proof["local_limit_detail"] == "investigation_limit"
    assert proof["investigation_stop"]["limit"] == 8
    assert len(raw["events"]) == 8 and raw["file_changes"] == []
    assert hooks.ledger.model_calls == 8 and hooks.ledger.tool_calls == 9
    assert hooks.ledger.charged_tokens == 120 and not hooks.ledger.pending
    assert [e["event_type"] for e in hooks.trace.events].count("coding.investigation_limit") == 1


def test_v1_baseline_keeps_original_exception_path_and_no_verifier(tmp_path):
    request, hooks, io, factory = baseline_setup(tmp_path, "local-stop-v1")
    with pytest.raises(RuntimeError) as stopped:
        run_single_agent(request.spec, request, io, factory, hooks, lambda: None)
    assert type(stopped.value) is RuntimeError
    assert hooks.ledger.model_calls == hooks.ledger.tool_calls == 8
    assert hooks.ledger.stop_kind is None
    assert not any(e["event_type"] == "verification.command.start" for e in hooks.trace.events)


def test_passing_public_check_does_not_promote_an_unfinished_workflow(tmp_path):
    request, hooks, io, factory = baseline_setup(tmp_path, "local-stop-v2",
        source="def increment(value):\n    return value + 1\n")
    result = run_single_agent(request.spec, request, io, factory, hooks, lambda: None)
    assert result.verification.status == "passed"
    assert result.status == "incomplete" and not result.workflow_completed
    assert result.failure_kind == "budget_exhausted"
    assert io.read_json(result.outputs[0])["coding_stop_receipt"]["verification_status"] == "passed"


def test_generic_runtime_error_remains_unhandled_by_v2(tmp_path, monkeypatch):
    request, hooks, io, factory = baseline_setup(tmp_path, "local-stop-v2")
    def broken_chat(*_):
        raise RuntimeError("Repair stopped: consecutive investigation rounds exhausted without an effective file edit")
    monkeypatch.setattr(RepairAgent, "chat", broken_chat)
    with pytest.raises(RuntimeError):
        run_single_agent(request.spec, request, io, factory, hooks, lambda: None)
    assert hooks.ledger.stop_kind is None and hooks.ledger.tool_calls == 0


@pytest.mark.parametrize("change", ["extra", "bool", "wrong_count", "short_rounds", "duplicate_end", "write_in_window", "bad_source"])
def test_controller_rejects_invalid_investigation_window(tmp_path, change):
    agent, error, hooks = stopped_agent(tmp_path)
    proof = investigation_stop_proof(agent, error, hooks, CodingContinuationPolicy(mode="local-stop-v2"))
    if change == "extra": proof["untrusted"] = True
    elif change == "bool": proof["rounds"] = True
    elif change == "wrong_count": proof["event_count"] += 1
    elif change == "short_rounds": proof["window_event_ends"].pop()
    elif change == "duplicate_end": proof["window_event_ends"][1] = 1
    elif change == "write_in_window": agent.events[1]["tool"] = "write_file"
    else: proof["source"] = "model_said_investigation_limit"
    assert not _valid_investigation_window(proof, agent.events)
