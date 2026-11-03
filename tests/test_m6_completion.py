"""Explicit coding closure reserves existing resources; a passed check is not completion."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from corecoder.context import estimate_tokens
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.workflow import RepairAgent, project_directory, workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import BudgetSpec, ExecutionPolicy, TaskInput, VerificationSpec, WorkspaceRef
from researchcoder.orchestration.baseline import run_single_agent
from researchcoder.runtime.budget import BudgetExceeded, BudgetedModel, BudgetLedger, RuntimeHooks
from researchcoder.runtime.coding import (CODING_FINAL_REPORT, CODING_FINAL_REPORT_SHA256,
    CodingContinuationPolicy, validate_coding_final_report)
from researchcoder.runtime.harness import ExecutionConfig, ResearchHarness
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.protocol import (CodingTaskSpec, ModelSpec, RuntimeLimits,
    ScriptedCall, ScriptedTurn, WorkerRequest)
from researchcoder.runtime.tracing import TraceSink


def response(content="", calls=(), finish_reason="stop", known=True):
    return LLMResponse(content=content, tool_calls=list(calls), finish_reason=finish_reason,
                       prompt_tokens=20 if known else 0, completion_tokens=10 if known else 0,
                       usage_complete=known)


def repair_turns(final=None):
    return [response(calls=[ToolCall("read", "read_file", {"file_path": "module.py"})]),
        response(calls=[ToolCall("edit", "edit_file", {"file_path": "module.py",
            "old_string": "value = 0", "new_string": "value = 1"})]),
        final or response(json.dumps(dict(status="completed", summary="Changed value; verification remains pending.")))]


class CaptureModel(ScriptedLLM):
    def __init__(self, turns):
        super().__init__(turns)
        self.inputs = []

    def chat(self, messages, tools=None, on_token=None):
        self.inputs.append(json.loads(json.dumps(dict(messages=messages, tools=tools))))
        return super().chat(messages, tools, on_token)


def setup(tmp_path, *, explicit=True, limits=None, source="value = 0\n", prefix=""):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text(source)
    io = ArtifactIO(tmp_path / "artifacts")
    verify = VerificationSpec(argv=(sys.executable, "-B", "-c", "from module import value; assert value == 1"),
                              timeout_seconds=2)
    limits = limits or RuntimeLimits(max_model_calls=3, max_tool_calls=3, max_rounds=6,
        max_runtime_seconds=30, max_context_tokens=16000, max_output_tokens=512, max_total_tokens=100000)
    opt = RuntimeOptimization(version="explicit-report-v1", compact_reports=True, report_reserve_calls=2) if explicit else None
    spec = CodingTaskSpec(job_id="job", task_id="coding", objective="Set value to one, retaining other behavior.",
        workspace=WorkspaceRef(workspace_id="candidate", root=str(root), source_manifest_sha256=workspace_revision(root)),
        inputs=(TaskInput(role="raw_log", reference=io.put_bytes("job", b"AssertionError: expected one", media_type="text/plain")),),
        policy=ExecutionPolicy(allowed_tool_names=("read_file", "edit_file"), allowed_read_roots=(str(root),),
            writable_paths=("module.py",), allow_verification=True), verification=verify,
        budget=BudgetSpec(max_rounds=limits.max_rounds, max_tool_calls=limits.max_tool_calls,
                         max_context_tokens=limits.max_context_tokens))
    request = WorkerRequest(spec=spec, attempt_id="attempt", artifact_root=str(io.root),
        trace_path=str(tmp_path / "trace.jsonl"), parent_event_id="parent", limits=limits,
        permitted_verifications=(verify,), optimization=opt, system_instructions=prefix,
        coding_continuation=CodingContinuationPolicy(mode="local-stop-v2") if explicit else None)
    trace = TraceSink("job", "coding", "attempt")
    ledger = BudgetLedger(limits, trace)
    ledger.optimization, ledger.coding_continuation = opt, request.coding_continuation
    hooks = RuntimeHooks(ledger, trace, spec.policy.allowed_tool_names)
    hooks.optimization = opt
    return request, io, hooks


def execute(request, io, hooks, turns):
    model = CaptureModel(turns)
    result = run_single_agent(request.spec, request, io,
        lambda: BudgetedModel(model, hooks.ledger, hooks.trace, system_instructions=request.system_instructions),
        hooks, lambda: None)
    raw = io.read_json(result.outputs[0])
    return result, raw, model


def test_repair_then_explicit_closing_response_and_real_public_verification(tmp_path):
    request, io, hooks = setup(tmp_path)
    result, raw, model = execute(request, io, hooks, repair_turns())
    assert result.status == "succeeded" and result.workflow_completed
    assert result.verification.status == "passed" and result.verification.exit_code == 0
    assert result.verification.checked_workspace_revision == workspace_revision(request.spec.workspace.root)
    assert hooks.ledger.model_calls == 3 and hooks.ledger.tool_calls == 3
    assert hooks.ledger.stop_kind is None and not hooks.ledger.pending and not hooks.ledger.unknown_requests
    assert model.inputs[-1]["tools"] == []
    assert model.inputs[-1]["messages"][-1] == dict(role="user", content=CODING_FINAL_REPORT)
    assert all(x["tools"] for x in model.inputs[:-1])
    receipt = raw["coding_final_report"]
    assert receipt["prompt_sha256"] == CODING_FINAL_REPORT_SHA256
    assert receipt["declared_status"] == "completed" and receipt["failure_kind"] is None
    assert json.loads(receipt["original_response"])["status"] == "completed"
    assert any(e["event_type"] == "verification.command.end" and e["status"] == "passed" for e in hooks.trace.events)


@pytest.mark.parametrize("final,kind", [
    (response("<｜DSML｜calls>read_file</｜DSML｜calls>"), "invalid_json"),
    (response('{"status":"completed","summary":"done"', finish_reason="length"), "response_length_exceeded"),
    (response('{"status":"completed","summary":"done","extra":true}'), "invalid_report"),
    (response('{"status":"incomplete","summary":"Further work is required"}'), "unknown"),
])
def test_real_passed_check_never_promotes_bad_or_incomplete_closing_report(tmp_path, final, kind):
    request, io, hooks = setup(tmp_path)
    result, raw, _ = execute(request, io, hooks, repair_turns(final))
    assert result.status == "incomplete" and not result.workflow_completed
    assert result.failure_kind == kind and result.verification.status == "passed"
    assert raw["coding_final_report"]["original_response"] == final.content
    assert hooks.ledger.model_calls == 3  # no writer retry or budget reset


def test_native_tool_call_after_finalization_is_not_executed(tmp_path):
    request, io, hooks = setup(tmp_path)
    final = response(calls=[ToolCall("bad", "edit_file", {"file_path": "module.py",
        "old_string": "value = 1", "new_string": "value = 99"})])
    result, raw, _ = execute(request, io, hooks, repair_turns(final))
    assert result.status == "incomplete" and result.failure_kind == "invalid_report"
    assert result.verification.status == "passed"
    assert Path(request.spec.workspace.root, "module.py").read_text() == "value = 1\n"
    assert len(raw["events"]) == 2
    assert raw["coding_final_report"]["original_message"]["tool_calls"][0]["id"] == "bad"


def test_legacy_and_natural_early_completion_keep_plain_text_path(tmp_path):
    request, io, hooks = setup(tmp_path, explicit=False)
    result, raw, model = execute(request, io, hooks, repair_turns(response("Edited; controller may verify.")))
    assert result.status == "succeeded" and result.summary == "Edited; controller may verify."
    assert "coding_final_report" not in raw and model.inputs[-1]["tools"]
    other = tmp_path / "explicit"
    other.mkdir()
    request, io, hooks = setup(other, limits=RuntimeLimits(max_model_calls=8, max_rounds=8,
        max_tool_calls=10, max_runtime_seconds=30, max_output_tokens=512))
    result, raw, _ = execute(request, io, hooks, repair_turns(response("Edited; controller may verify.")))
    assert result.status == "succeeded" and "coding_final_report" not in raw


def test_public_verification_failure_is_not_overridden_by_completed_declaration(tmp_path):
    request, io, hooks = setup(tmp_path)
    turns = repair_turns()
    turns[1] = response(calls=[ToolCall("edit", "edit_file", {"file_path":"module.py",
        "old_string":"value = 0", "new_string":"value = 2"})])
    result, raw, _ = execute(request, io, hooks, turns)
    assert result.status == "failed" and result.failure_kind == "verification_failed"
    assert result.workflow_completed and result.verification.status == "failed"
    assert raw["coding_final_report"]["declared_status"] == "completed"


def test_last_tool_slot_is_reserved_even_for_multiple_calls_in_one_response(tmp_path):
    request, io, hooks = setup(tmp_path, source="value = 1\n")
    turns = [response(calls=[ToolCall(str(i), "read_file", {"file_path":"module.py"}) for i in range(3)])]
    result, raw, model = execute(request, io, hooks, turns)
    assert result.status == "incomplete" and result.failure_kind == "budget_exhausted"
    assert result.verification.status == "passed" and not result.workflow_completed
    assert hooks.ledger.local_limit_detail == "verification_reserve"
    assert hooks.ledger.tool_calls == 3 and hooks.ledger.model_calls == 1
    assert len(raw["events"]) == 2 and len(model.inputs) == 1


@pytest.mark.parametrize("unsafe", ["unknown", "pending", "permission", "effect_error", "timeout"])
def test_report_repair_and_verification_reserve_cannot_bypass_unknown_or_unsafe_execution(tmp_path, unsafe):
    request, io, hooks = setup(tmp_path)
    hooks._coding_completion = True
    if unsafe == "unknown": hooks.ledger.unknown_requests = 1
    elif unsafe == "pending": hooks.ledger.pending["unsettled"] = 50
    elif unsafe == "permission": hooks.permission_stop = True
    elif unsafe == "effect_error": hooks.ledger.last_failure_kind = "tool_error"
    else: hooks.ledger.started -= 40
    assert hooks.report_repair_allowed() is False
    with pytest.raises(BudgetExceeded): hooks.before_verification()
    assert hooks.ledger.tool_calls == 0


def test_report_repair_admission_reserves_no_extra_call_or_credit(tmp_path):
    request, io, hooks = setup(tmp_path)
    assert hooks.report_repair_allowed()
    assert hooks.ledger.model_calls == 0 and hooks.ledger.pending == {}
    hooks.ledger.model_calls = request.limits.max_model_calls
    assert not hooks.report_repair_allowed()


def test_second_investigation_usage_unknown_blocks_third_model_request(tmp_path):
    request, io, hooks = setup(tmp_path)
    model = CaptureModel([response("first observation"), response("second observation", known=False),
                          response("must not be requested")])
    gateway = BudgetedModel(model, hooks.ledger, hooks.trace)
    messages = [dict(role="user", content="Investigate public evidence")]
    gateway.chat(messages)
    gateway.chat(messages)
    liability = hooks.ledger.charged_tokens
    with pytest.raises(BudgetExceeded, match="unknown"):
        gateway.chat(messages)
    assert len(model.inputs) == hooks.ledger.model_calls == 2
    assert hooks.ledger.unknown_requests == 1 and hooks.ledger.charged_tokens == liability
    assert hooks.ledger.stop_kind == "unknown" and hooks.ledger.local_limit_detail == "unknown_usage"
    assert hooks.ledger.pending == {}


def test_explicit_transport_retry_cannot_replay_a_pending_request(tmp_path):
    request, io, hooks = setup(tmp_path)
    first = hooks.ledger.reserve_model(100)
    pending = dict(hooks.ledger.pending)
    with pytest.raises(BudgetExceeded, match="unknown"):
        hooks.ledger.reserve_model(100)
    assert hooks.ledger.model_calls == 1 and hooks.ledger.pending == pending
    hooks.ledger.settle(first, status="error")
    assert hooks.ledger.unknown_requests == 1 and hooks.ledger.charged_tokens == 100 + request.limits.max_output_tokens


def test_unknown_usage_in_final_declaration_keeps_liability_and_blocks_public_verification(tmp_path):
    request, io, hooks = setup(tmp_path)
    final = response('{"status":"completed","summary":"Changed value; verification pending."}', known=False)
    result, raw, model = execute(request, io, hooks, repair_turns(final))
    assert result.status == "incomplete" and result.failure_kind == "unknown"
    assert not result.workflow_completed and result.verification.status == "not_run"
    assert raw["coding_final_report"]["declared_status"] == "completed"
    assert raw["coding_final_report"]["original_response"] == final.content
    assert Path(request.spec.workspace.root, "module.py").read_text() == "value = 1\n"
    assert hooks.ledger.model_calls == len(model.inputs) == 3 and hooks.ledger.tool_calls == 2
    assert hooks.ledger.unknown_requests == 1 and hooks.ledger.pending == {}
    assert hooks.ledger.charged_tokens == 60 + hooks.ledger.settlements[-1]["reserved"]
    assert hooks.ledger.stop_kind == "unknown" and hooks.ledger.local_limit_detail == "unknown_usage"
    assert not any(e["event_type"] == "verification.command.start" for e in hooks.trace.events)


def test_output_and_gateway_prefix_are_counted_in_existing_repair_context(tmp_path):
    limits = RuntimeLimits(max_model_calls=3, max_tool_calls=3, max_rounds=6, max_runtime_seconds=30,
        max_context_tokens=6000, max_output_tokens=1024, max_total_tokens=100000)
    source = "# prior source detail\n" * 500 + "value = 0\n"
    request, io, hooks = setup(tmp_path, limits=limits, source=source, prefix="Trusted task rules. " * 80)
    result, _, model = execute(request, io, hooks, repair_turns())
    assert result.status == "succeeded"
    for item in model.inputs:
        estimate = estimate_tokens(item["messages"]) + len(json.dumps(item["tools"] or [], ensure_ascii=False)) // 3
        assert estimate + limits.max_output_tokens <= limits.max_context_tokens
    assert hooks._coding_fixed_prefix_tokens >= len(request.system_instructions) // 3


@pytest.mark.parametrize("text,kind", [
    ('{"status":"completed","status":"incomplete","summary":"x"}', "invalid_json"),
    ('{"status":"completed","summary":NaN}', "invalid_json"),
    ('{"status":"completed","summary":" "}', "invalid_report"),
    ('{"status":[],"summary":"x"}', "invalid_report"),
])
def test_closing_contract_rejects_ambiguous_or_invalid_values(text, kind):
    assert validate_coding_final_report(text)["failure_kind"] == kind


def test_real_worker_uses_same_closure_budget_and_public_verifier(tmp_path):
    request, io, _ = setup(tmp_path)
    turns = tuple(ScriptedTurn(content=t.content, calls=tuple(ScriptedCall(call_id=c.id,
        name=c.name, arguments_json=json.dumps(c.arguments)) for c in t.tool_calls),
        prompt_tokens=20, completion_tokens=10) for t in repair_turns())
    model = ModelSpec(provider="scripted", model="offline-closure", turns=turns)
    config = ExecutionConfig(artifact_root=str(io.root), allowed_workspaces=(request.spec.workspace.root,),
        max_policy=request.spec.policy, permitted_verifications=request.permitted_verifications,
        limits=request.limits, model=model, optimization=request.optimization,
        coding_continuation=request.coding_continuation, record_requests=True)
    harness = ResearchHarness()
    result = harness.execute(request.spec, config)
    assert result.status == "succeeded" and result.verification.status == "passed"
    assert result.usage.llm_calls == 3 and result.usage.tool_calls == 3
    record = harness.records[result.attempt_id]
    assert record["budget"]["pending"] == {} and record["budget"]["unknown_requests"] == 0
    assert record["worker_exit_code"] == 0
