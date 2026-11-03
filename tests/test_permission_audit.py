"""Typed pre-execution denials remain auditable without storing model arguments."""

import json
from types import SimpleNamespace

import pytest

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.permission import PermissionAuditError, PermissionDenied, validate_permission_receipt
from corecoder.workflow import RepairAgent, project_directory


def denied(*, correction=False, tool="read_file"):
    return PermissionDenied(
        reason_code="public_read_path_mismatch" if correction else "outside_workspace",
        target_category="public_check" if correction else "outside_workspace",
        tool=tool, target_sha256="a" * 64,
        action="correct_parameters" if correction else "stop",
        correction_index=1 if correction else 0,
    )


def agent_for(root, *, guard=None, turns=()):
    root.mkdir(exist_ok=True)
    return RepairAgent(
        ScriptedLLM(list(turns)), {"project_root": str(root.resolve()), "file_snapshots": {}},
        tool_guard=guard, allowed_tool_names=("read_file", "write_file", "edit_file"),
    )


def hooks(callback):
    return SimpleNamespace(
        permission_denied=callback, before_round=lambda: None,
        before_tool=lambda tool: tool, after_tool=lambda token, status: None,
    )


def test_permission_receipt_roundtrip_preserves_identity_without_aliasing():
    exc = denied(correction=True)
    receipt = json.loads(json.dumps(exc.receipt))
    restored = PermissionDenied.from_receipt(receipt)
    assert restored.receipt == receipt
    assert restored.feedback == exc.feedback
    assert isinstance(restored, ValueError)
    receipt["tool"] = "edit_file"
    assert restored.receipt["tool"] == "read_file"
    exported = restored.to_dict()
    exported["action"] = "stop"
    assert restored.receipt["action"] == "correct_parameters"


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("denial_id", "../../private"),
    ("reason_code", "secret: password"), ("target_category", "secret: path"),
    ("tool", "read_file; cat .env"), ("action", "allow"),
    ("target_sha256", "/private/secret.txt"), ("target_sha256", "A" * 64),
    ("correction_index", True), ("correction_index", 2), ("tool_executed", True),
    ("tool_executed", 0),
])
def test_permission_receipt_rejects_unknown_or_unsafe_values(field, value):
    receipt = denied().receipt
    receipt[field] = value
    with pytest.raises(ValueError):
        validate_permission_receipt(receipt)
    with pytest.raises(ValueError):
        PermissionDenied.from_receipt(receipt)


@pytest.mark.parametrize("mutation", ["extra", "missing", "nonobject"])
def test_permission_receipt_requires_exact_schema(mutation):
    receipt = denied().receipt
    if mutation == "extra":
        receipt["arguments"] = {"content": "secret"}
    elif mutation == "missing":
        receipt.pop("target_sha256")
    else:
        receipt = list(receipt.items())
    with pytest.raises(ValueError):
        PermissionDenied.from_receipt(receipt)


@pytest.mark.parametrize("field,value", [
    ("reason_code", "guard_rejected"), ("target_category", "workspace_file"),
    ("tool", "write_file"), ("correction_index", 0), ("target_sha256", None),
])
def test_correction_cannot_authorize_writes_or_unclassified_targets(field, value):
    receipt = denied(correction=True).receipt
    receipt[field] = value
    with pytest.raises(ValueError):
        PermissionDenied.from_receipt(receipt)


def test_terminal_denial_cannot_claim_correction_credit():
    receipt = denied().receipt
    receipt["correction_index"] = 1
    with pytest.raises(ValueError):
        validate_permission_receipt(receipt)


def test_guard_receipt_is_archived_before_event_or_tool_and_redacts_arguments(tmp_path, monkeypatch):
    exc = denied(tool="write_file")
    def guard(name, arguments):
        raise exc
    agent = agent_for(tmp_path, guard=guard)
    executed, archived = [], []
    monkeypatch.setattr(agent._tool_by_name["write_file"], "execute",
                        lambda file_path, content: executed.append(file_path) or "written")
    def archive(receipt):
        assert not executed and not agent.events
        archived.append(json.loads(json.dumps(receipt)))
        receipt["tool"] = "unknown"  # An observer must not mutate the local event.
    agent.execution_hooks = hooks(archive)
    call = ToolCall("opaque-call", "write_file", {"file_path": "private-secret.py", "content": "API_KEY=secret-value"})
    with pytest.raises(PermissionDenied) as raised:
        agent._exec_tool(call)
    assert raised.value is exc
    assert not executed and not (tmp_path / "private-secret.py").exists()
    assert archived == [exc.receipt]
    event = agent.events[0]
    assert event["permission_denial"] == archived[0]
    assert event["arguments"] == {} and event["status"] == "error"
    assert event["permission_denial"]["tool_executed"] is False
    text = json.dumps({"events": agent.events, "archive": archived})
    assert "API_KEY" not in text and "secret-value" not in text and "private-secret.py" not in text
    assert "opaque-call" not in text


def test_correct_parameters_uses_original_loop_for_real_candidate_read_and_edit(tmp_path):
    source = tmp_path / "module.py"
    source.write_text("value = 1\n")
    receipt_path = tmp_path / "receipt.json"
    archived = []
    def guard(name, arguments):
        if arguments.get("file_path") == "/controller/check.py":
            raise denied(correction=True)
    turns = [
        LLMResponse(tool_calls=[ToolCall("denied", "read_file", {"file_path": "/controller/check.py"})]),
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "module.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "module.py", "old_string": "value = 1", "new_string": "value = 2"})]),
        LLMResponse(content="Candidate edited; independent checks remain the controller's responsibility."),
    ]
    agent = agent_for(tmp_path, guard=guard, turns=turns)
    def archive(receipt):
        receipt_path.write_text(json.dumps(receipt))
        archived.append(receipt)
    agent.execution_hooks = hooks(archive)
    with project_directory(tmp_path):
        summary = agent.chat("Read candidate source and make the authorized edit.")
    assert "Candidate edited" in summary
    assert source.read_text() == "value = 2\n"
    assert len(archived) == 1
    assert json.loads(receipt_path.read_text()) == agent.events[0]["permission_denial"]
    assert [event["status"] for event in agent.events] == ["error", "ok", "ok"]
    assert agent.mutation_count == 1 and not agent.llm._turns
    denied_reply = next(message for message in agent.messages if message.get("tool_call_id") == "denied")
    assert "not executed" in denied_reply["content"] and "candidate source" in denied_reply["content"]
    assert "/controller/check.py" not in denied_reply["content"]


def test_terminal_denial_stops_original_loop_before_second_request(tmp_path):
    exc = denied()
    def guard(name, arguments):
        raise exc
    agent = agent_for(tmp_path, guard=guard, turns=[
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "/private/path"})]),
        LLMResponse(content="This response must remain unrequested."),
    ])
    archived = []
    agent.execution_hooks = hooks(archived.append)
    with project_directory(tmp_path), pytest.raises(PermissionDenied):
        agent.chat("Inspect candidate.")
    assert len(agent.llm._turns) == 1
    assert archived == [exc.receipt]
    assert len(agent.events) == 1 and agent.events[0]["status"] == "error"


def test_persisted_receipt_survives_later_model_failure_without_final_report(tmp_path):
    archive_path = tmp_path / "denial.json"
    def guard(name, arguments):
        raise denied(correction=True)
    agent = agent_for(tmp_path, guard=guard, turns=[
        LLMResponse(tool_calls=[ToolCall("denied", "read_file", {"file_path": "/controller/check.py"})]),
    ])
    agent.execution_hooks = hooks(lambda receipt: archive_path.write_text(json.dumps(receipt)))
    with project_directory(tmp_path), pytest.raises(RuntimeError, match="ran out of turns"):
        agent.chat("Inspect candidate.")
    receipt = json.loads(archive_path.read_text())
    assert validate_permission_receipt(receipt) == agent.events[0]["permission_denial"]
    assert receipt["action"] == "correct_parameters" and receipt["tool_executed"] is False


def test_failed_audit_is_terminal_and_does_not_leak_observer_exception(tmp_path):
    def guard(name, arguments):
        raise denied(correction=True)
    agent = agent_for(tmp_path, guard=guard, turns=[
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "/controller/check.py"})]),
        LLMResponse(content="Must not run."),
    ])
    def broken_archive(receipt):
        raise OSError("private-secret-env-value")
    agent.execution_hooks = hooks(broken_archive)
    with project_directory(tmp_path), pytest.raises(PermissionAuditError) as raised:
        agent.chat("Inspect candidate.")
    assert len(agent.llm._turns) == 1
    assert agent.events[0]["permission_audit_failed"] is True
    assert agent.events[0]["arguments"] == {}
    assert "private-secret-env-value" not in str(raised.value)
    assert "private-secret-env-value" not in json.dumps(agent.events)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_audit_interruption_preserves_safe_local_receipt_then_propagates(tmp_path, interruption):
    def guard(name, arguments):
        raise denied(correction=True)
    agent = agent_for(tmp_path, guard=guard)
    def interrupted_archive(receipt):
        raise interruption()
    agent.execution_hooks = hooks(interrupted_archive)
    with pytest.raises(interruption):
        agent._exec_tool(ToolCall("read", "read_file", {"file_path": "/controller/check.py"}))
    assert len(agent.events) == 1
    assert agent.events[0]["permission_denial"]["tool_executed"] is False
    assert agent.events[0]["arguments"] == {}


def test_tool_raised_denial_cannot_forge_pre_execution_proof(tmp_path, monkeypatch):
    agent = agent_for(tmp_path, guard=lambda name, args: None)
    side_effects, archived = [], []
    def malicious_tool(file_path, content):
        side_effects.append(content)
        raise denied(correction=True)
    monkeypatch.setattr(agent._tool_by_name["write_file"], "execute", malicious_tool)
    agent.execution_hooks = hooks(archived.append)
    with project_directory(tmp_path), pytest.raises(PermissionAuditError):
        agent._exec_tool(ToolCall("write", "write_file", {"file_path": "module.py", "content": "secret"}))
    assert side_effects == ["secret"] and not archived
    assert "permission_denial" not in agent.events[0]
    assert agent.events[0]["permission_audit_failed"] is True
    assert "secret" not in json.dumps(agent.events)


def test_legacy_valueerror_guard_keeps_original_feedback_and_audit(tmp_path):
    def guard(name, arguments):
        raise ValueError("protected fixture")
    agent = agent_for(tmp_path, guard=guard)
    arguments = {"file_path": "module.py", "content": "value = 1\n"}
    result = agent._exec_tool(ToolCall("write", "write_file", arguments))
    assert result == "Error: protected fixture"
    assert agent.events[0]["arguments"] == arguments
    assert "permission_denial" not in agent.events[0]
    assert not (tmp_path / "module.py").exists()


def test_optional_hook_is_not_required_by_legacy_normal_execution(tmp_path):
    agent = agent_for(tmp_path)
    with project_directory(tmp_path):
        result = agent._exec_tool(ToolCall("write", "write_file", {"file_path": "module.py", "content": "value = 1\n"}))
    assert not result.startswith("Error")
    assert (tmp_path / "module.py").read_text() == "value = 1\n"
    assert agent.events[0]["status"] == "ok"
    assert "permission_denial" not in agent.events[0]


def baseline_request(tmp_path):
    import sys
    from corecoder.workflow import workspace_revision
    from researchcoder.adapters.artifacts import ArtifactIO
    from researchcoder.contracts import ExecutionPolicy, TaskInput, VerificationSpec, WorkspaceRef
    from researchcoder.runtime.budget import BudgetLedger, RuntimeHooks
    from researchcoder.runtime.protocol import CodingTaskSpec, RuntimeLimits, WorkerRequest
    from researchcoder.runtime.tracing import TraceSink

    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("value = 1\n")
    public = tmp_path / "public_check.py"
    public.write_text("print('controller-only fixture')\n")
    marker = tmp_path / "verification-ran"
    command = VerificationSpec(argv=(sys.executable, "-B", "-c",
        "from pathlib import Path; Path(" + repr(str(marker)) + ").write_text('executed')"))
    io = ArtifactIO(tmp_path / "artifacts")
    log = io.put_bytes("job", b"Expected value=2", media_type="text/plain")
    spec = CodingTaskSpec(job_id="job", task_id="coding", objective="Correct the candidate value.",
        workspace=WorkspaceRef(workspace_id="candidate", root=str(root), source_manifest_sha256=workspace_revision(root)),
        inputs=(TaskInput(role="raw_log", reference=log),),
        policy=ExecutionPolicy(allowed_tool_names=("read_file", "edit_file"), writable_paths=("module.py",), allow_verification=True),
        verification=command)
    request = WorkerRequest(spec=spec, attempt_id="attempt", artifact_root=str(io.root),
        trace_path=str(tmp_path / "trace.jsonl"), parent_event_id="parent", permitted_verifications=(command,))
    trace = TraceSink("job", "coding", "attempt")
    ledger = BudgetLedger(RuntimeLimits(), trace)
    runtime = RuntimeHooks(ledger, trace, spec.policy.allowed_tool_names)
    return root, public, marker, io, request, runtime


def test_baseline_keeps_correction_and_terminal_events_with_legal_patch(tmp_path):
    from researchcoder.orchestration.baseline import run_single_agent
    from researchcoder.runtime.permissions import PermissionRecoverySession, PermissionRecoverySpec

    root, public, marker, io, request, runtime = baseline_request(tmp_path)
    runtime.permission_recovery_session = PermissionRecoverySession(PermissionRecoverySpec(
        mode="bounded-public-read-v1", public_check_paths=(str(public),), remaining_corrections=1))
    runtime.permission_artifacts = io
    runtime.permission_workspace_revision = request.spec.workspace.source_manifest_sha256
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("denied-1", "read_file", {"file_path": str(public)})]),
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "module.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "module.py", "old_string": "value = 1", "new_string": "value = 2"})]),
        LLMResponse(tool_calls=[ToolCall("denied-2", "read_file", {"file_path": str(public)})]),
        LLMResponse(content="This success claim must never be requested."),
    ])
    result = run_single_agent(request.spec, request, io, lambda: model, runtime, lambda: None)
    assert result.status == "incomplete" and result.failure_kind == "tool_error"
    assert not result.workflow_completed and not result.summary
    assert result.verification.status == "not_run"
    assert result.verification.exit_code is None and result.verification.checked_workspace_revision is None
    assert not marker.exists() and len(model._turns) == 1
    assert (root / "module.py").read_text() == "value = 2\n"
    raw = io.read_json(result.outputs[0])
    assert raw["permission_stopped"] is True
    assert raw["verification"]["blocked_reason"] == "permission_denied"
    assert [event["status"] for event in raw["events"]] == ["error", "ok", "ok", "error"]
    receipts = [event["permission_denial"] for event in raw["events"] if "permission_denial" in event]
    assert [receipt["action"] for receipt in receipts] == ["correct_parameters", "stop"]
    assert len(set(receipt["denial_id"] for receipt in receipts)) == 2
    assert [io.read_json(ref)["receipt"] for ref in runtime.permission_refs] == receipts
    assert raw["file_changes"][0]["before_sha256"] != raw["file_changes"][0]["after_sha256"]
    assert runtime.ledger.stop_kind == "tool_error"


def test_baseline_archives_hook_controlled_permission_stop_without_verification(tmp_path):
    from researchcoder.orchestration.baseline import run_single_agent
    from researchcoder.runtime.budget import BudgetExceeded

    _, _, marker, io, request, runtime = baseline_request(tmp_path)
    class HookStoppedModel(ScriptedLLM):
        def chat(self, *args, **kwargs):
            runtime.permission_stop = True
            raise BudgetExceeded("Permission refusal requires controller review")
    result = run_single_agent(request.spec, request, io, lambda: HookStoppedModel([]), runtime, lambda: None)
    assert result.status == "incomplete" and result.failure_kind == "tool_error"
    assert result.verification.status == "not_run" and not marker.exists()
    assert io.read_json(result.outputs[0])["events"] == []


def test_baseline_does_not_relabel_unrelated_runtime_failure_as_permission_stop(tmp_path):
    from researchcoder.orchestration.baseline import run_single_agent

    _, _, marker, io, request, runtime = baseline_request(tmp_path)
    with pytest.raises(RuntimeError, match="ran out of turns"):
        run_single_agent(request.spec, request, io, lambda: ScriptedLLM([]), runtime, lambda: None)
    assert not marker.exists() and not runtime.permission_stop


@pytest.mark.parametrize("failure_kind", ["runtime", "budget", "interrupt", "verifier"])
def test_baseline_partial_archive_preserves_receipt_and_original_exception(tmp_path, monkeypatch, failure_kind):
    from researchcoder.orchestration.baseline import run_single_agent
    from researchcoder.runtime.budget import BudgetExceeded
    from researchcoder.runtime.permissions import PermissionRecoverySession, PermissionRecoverySpec

    _, public, marker, io, request, runtime = baseline_request(tmp_path)
    runtime.permission_recovery_session = PermissionRecoverySession(PermissionRecoverySpec(
        mode="bounded-public-read-v1", public_check_paths=(str(public),), remaining_corrections=1))
    runtime.permission_artifacts = io
    runtime.permission_workspace_revision = request.spec.workspace.source_manifest_sha256
    failure = (KeyboardInterrupt() if failure_kind == "interrupt" else
               BudgetExceeded("budget_exhausted") if failure_kind == "budget" else RuntimeError("original failure"))
    class InterruptedModel(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if not self._turns:
                raise failure
            return super().chat(*args, **kwargs)
    turns = [LLMResponse(tool_calls=[ToolCall("denied", "read_file", {"file_path": str(public)})])]
    if failure_kind == "verifier":
        turns.append(LLMResponse(content="No successful validation is claimed."))
        def interrupted_verifier(*args, **kwargs):
            raise failure
        monkeypatch.setattr("researchcoder.orchestration.baseline.run_verification", interrupted_verifier)
    model = InterruptedModel(turns)
    with pytest.raises(type(failure)) as raised:
        run_single_agent(request.spec, request, io, lambda: model, runtime, lambda: None)
    assert raised.value is failure
    assert not marker.exists() and not runtime.permission_stop
    assert len(runtime.permission_refs) == 1 and len(runtime.permission_partial_refs) == 1
    raw = io.read_json(runtime.permission_partial_refs[0])
    assert raw["partial_execution"] is True and raw["verification"] is None
    assert raw["baseline_hypothesis"] == "" and raw["file_changes"] == []
    assert len(raw["events"]) == 1
    assert raw["events"][0]["permission_denial"] == io.read_json(runtime.permission_refs[0])["receipt"]
