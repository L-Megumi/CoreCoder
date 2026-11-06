"""Guard decisions travel intact from the coding loop to the persisted denial."""

from dataclasses import replace
import json

import pytest

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.permission import PermissionDenied
from researchcoder.orchestration.baseline import run_single_agent
from researchcoder.runtime.permissions import PermissionRecoverySession, PermissionRecoverySpec
from tests.test_permission_audit import agent_for, baseline_request


@pytest.mark.parametrize("scenario,reason,category", [
    ("unread", "current_read_required", "workspace_file"),
    ("stale", "source_changed_since_read", "workspace_file"),
    ("protected", "write_not_allowed", "protected_source"),
    ("outside", "outside_workspace", "outside_workspace"),
    ("missing", "source_digest_unavailable", "workspace_file"),
])
def test_coding_write_denial_survives_guard_trace_and_artifact(tmp_path, scenario, reason, category):
    root, public, marker, io, request, runtime = baseline_request(tmp_path)
    target = {
        "outside": public, "protected": root / "LICENSE", "missing": root / "absent.py",
    }.get(scenario, root / "module.py")
    policy = replace(request.spec.policy, profile="project-expert-reread-v1",
                     writable_paths=("module.py", "absent.py"))
    request = replace(request, spec=replace(request.spec, policy=policy))
    runtime.permission_recovery_session = PermissionRecoverySession(PermissionRecoverySpec(
        mode="bounded-public-read-v1", remaining_corrections=1))
    runtime.permission_artifacts = io
    runtime.permission_workspace_revision = request.spec.workspace.source_manifest_sha256
    candidate = root / "module.py"
    unchanged_public = public.read_bytes()

    class ChangingSource(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            if scenario == "stale" and any(
                message.get("tool_call_id") == "read" and message["role"] == "tool"
                for message in messages
            ):
                # A competing source change after the real read, before admission.
                candidate.write_text("value = 3\n")
            return super().chat(messages, tools=tools, on_token=on_token)

    turns = []
    if scenario == "stale":
        turns.append(LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": str(candidate)})]))
    turns.append(LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {
        "file_path": str(target), "old_string": "value = 1", "new_string": "UNTRUSTED_PAYLOAD",
    })]))
    result = run_single_agent(request.spec, request, io, lambda: ChangingSource(turns), runtime, lambda: None)
    assert result.status == "incomplete" and result.verification.status == "not_run"
    assert not marker.exists() and runtime.permission_recovery_session.corrections_used == 0
    event = io.read_json(result.outputs[0])["events"][-1]
    receipt = event["permission_denial"]
    assert receipt == io.read_json(runtime.permission_refs[0])["receipt"]
    assert receipt["reason_code"] == reason and receipt["target_category"] == category
    assert receipt["action"] == "stop" and receipt["tool_executed"] is False
    assert event["arguments"] == {} and "UNTRUSTED_PAYLOAD" not in json.dumps(event)
    assert str(target) not in json.dumps(event) and not (root / "absent.py").exists()
    assert not (root / "LICENSE").exists() and public.read_bytes() == unchanged_public
    assert candidate.read_text() == ("value = 3\n" if scenario == "stale" else "value = 1\n")
    traced = [row for row in runtime.trace.events if row["event_type"] == "permission.denied"]
    assert [row["reason"] for row in traced] == [reason]


def test_untyped_guard_text_cannot_forge_freshness_classification(tmp_path):
    from researchcoder.runtime.budget import BudgetLedger, RuntimeHooks
    from researchcoder.runtime.protocol import RuntimeLimits
    from researchcoder.runtime.tracing import TraceSink

    def unrelated_guard(name, args):
        raise ValueError("source_changed_since_read PRIVATE_MARKER")

    agent = agent_for(tmp_path / "candidate", guard=unrelated_guard)
    trace = TraceSink("job", "task", "attempt")
    hooks = RuntimeHooks(BudgetLedger(RuntimeLimits(), trace), trace)
    hooks.permission_recovery_session = PermissionRecoverySession(PermissionRecoverySpec())
    hooks.bind_agent(agent)
    with pytest.raises(PermissionDenied) as caught:
        agent.tool_guard("edit_file", {"file_path": "/candidate/module.py"})
    assert caught.value.receipt["reason_code"] == "guard_rejected"
    assert caught.value.receipt["action"] == "stop"
    assert "PRIVATE_MARKER" not in str(caught.value)


@pytest.mark.parametrize("tool", ["edit_file", "write_file"])
def test_sensitive_target_keeps_redaction_for_all_write_subtypes(tool):
    session = PermissionRecoverySession(PermissionRecoverySpec())
    denial = session.deny(tool, "/work/.env", reason_code="source_changed_since_read")
    assert denial["reason_code"] == "sensitive_target" and denial["target_category"] == "sensitive"
    assert denial["action"] == "stop" and denial["tool_executed"] is False
    assert ".env" not in json.dumps(denial)
