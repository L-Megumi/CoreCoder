"""Real-Worker controls for denied-read feedback and immutable denial receipts.

Scripted repairs exercise engineering behavior, not model repair quality. The
public checker contains private-style branches deliberately: its contents must
remain inaccessible even when one parameter correction is permitted.
"""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.permission_audit import permission_records, guard_violations
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from researchcoder.runtime.harness import ExecutionConfig, ResearchHarness
from researchcoder.runtime.permissions import PermissionRecoverySpec, normalized_target_sha256
from researchcoder.runtime.protocol import CodingTaskResult, CodingTaskSpec, ModelSpec, ProtectedFile, ScriptedTurn, WorkerRequest
from tests.test_project_runner import BAD, GOOD, SECRET, call, fixture, policy, requests, scripted


MODE = "bounded-public-read-v1"


def with_public_read(public, model=None):
    model = scripted() if model is None else model
    return replace(model, turns=(call("read_file", {"file_path": str(public)}, "read-public"), *model.turns))


def run_case(tmp_path, model_factory=with_public_read, *, mode=MODE, settings=None, feedback_factory=None):
    source, task, checks, private = fixture(tmp_path)
    public = Path(checks["public"].protected_files[0].path)
    model = model_factory(public)
    feedback = () if feedback_factory is None else (feedback_factory(public),)
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        model, policy=settings or policy(), feedback_models=feedback, permission_recovery_mode=mode)
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    return result, raw, target, io, public, private, checks


def payloads(result, io):
    values = result.outputs if isinstance(result, CodingTaskResult) else tuple(ArtifactRef.from_dict(v) for v in result["outputs"])
    return [(ref, io.read_json(ref)) for ref in values]


@pytest.fixture(scope="module")
def completed_case(tmp_path_factory):
    return run_case(tmp_path_factory.mktemp("permission-success"))


def test_one_denied_public_read_corrects_parameters_and_binds_original_receipt(completed_case):
    result, raw, target, io, public, private, checks = completed_case
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"]
    assert result["model_requests"] == 4 and raw["permission_corrections"] == 1
    assert len(raw["rounds"]) == 1 and len(raw["permission_records"]) == 1
    row = raw["rounds"][0]
    record = raw["permission_records"][0]
    assert row["record"]["worker_pid"] and row["record"]["worker_exit_code"] == 0
    assert (record["job_id"], record["task_id"], record["attempt_id"]) == (
        raw["job_id"], row["result"]["task_id"], row["result"]["attempt_id"])
    receipt = record["receipt"]
    assert receipt["action"] == "correct_parameters" and receipt["tool_executed"] is False
    assert receipt["target_sha256"] == normalized_target_sha256(str(public))
    immutable = io.read_json(ArtifactRef.from_dict(record["reference"]))
    assert immutable["receipt"] == receipt
    baseline = next(p for _, p in payloads(row["result"], io) if "baseline_hypothesis" in p)
    denied = [e for e in baseline["events"] if "permission_denial" in e]
    assert len(denied) == 1 and denied[0]["permission_denial"] == receipt
    assert denied[0]["status"] == "error" and denied[0]["arguments"] == {}
    trace = io.read_json(ArtifactRef.from_dict(row["record"]["trace"]))["events"]
    events = [e for e in trace if e["event_type"] == "permission.denied"]
    assert len(events) == 1 and events[0]["parent_id"]
    assert (target / "candidate" / "module.py").read_text() == GOOD
    assert (target.parent / "source" / "module.py").read_text() == BAD
    assert hashlib.sha256(public.read_bytes()).hexdigest() == checks["public"].protected_files[0].sha256
    serialized = json.dumps(requests(raw, io))
    assert "One parameter correction is permitted" in serialized
    assert "private-output" not in serialized and "preserve-identity" not in serialized
    assert SECRET not in serialized and private.read_text() not in serialized
    assert recompute(target) == result


@pytest.mark.parametrize("same_batch", [False, True])
def test_second_public_read_stops_before_future_write_in_same_or_next_round(tmp_path, same_batch):
    def model(public):
        first = call("read_file", {"file_path": str(public)}, "denied-first")
        second = call("read_file", {"file_path": str(public)}, "denied-second")
        prefix = (replace(first, calls=first.calls + second.calls),) if same_batch else (first, second)
        return replace(scripted(), turns=prefix + scripted().turns)
    result, raw, target, io, _, _, _ = run_case(tmp_path, model)
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert not result["end_to_end_success"] and not raw["final_checks"]
    assert raw["permission_corrections"] == 1 and len(raw["permission_records"]) == 2
    assert [r["receipt"]["action"] for r in raw["permission_records"]] == ["correct_parameters", "stop"]
    assert raw["permission_records"][1]["receipt"]["reason_code"] == "correction_exhausted"
    assert (target / "candidate" / "module.py").read_text() == BAD
    assert result["model_requests"] == (1 if same_batch else 2)
    assert SECRET not in json.dumps(requests(raw, io))


def test_feedback_attempt_carries_consumed_job_correction_and_original_failure(tmp_path):
    result, raw, target, io, _, _, _ = run_case(tmp_path,
        lambda public: with_public_read(public, scripted(None)),
        settings=policy(max_feedback_rounds=1), feedback_factory=with_public_read)
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert len(raw["rounds"]) == 2 and raw["permission_corrections"] == 1
    first, second = raw["rounds"]
    assert first["result"]["failure_kind"] == "verification_failed"
    assert first["public_receipt"]["passed"] is False
    assert second["result"]["workflow_completed"] is False
    assert first["result"]["job_id"] == second["result"]["job_id"] == raw["job_id"]
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert [r["receipt"]["action"] for r in raw["permission_records"]] == ["correct_parameters", "stop"]
    assert raw["permission_records"][1]["receipt"]["reason_code"] == "correction_exhausted"
    assert raw["budget"]["charged"]["model_calls"] == 4
    assert set(raw["budget"]["settlements"]) == {"coding-0", "coding-1"}
    assert not raw["budget"]["pending"] and not raw["final_checks"]
    assert (target / "candidate" / "module.py").read_text() == BAD
    assert SECRET not in json.dumps(requests(raw, io))


@pytest.mark.parametrize("target_kind", ["unknown", "private", "write-public"])
def test_other_targets_or_writes_stop_without_disclosure_or_following_edit(tmp_path, target_kind):
    def model(public):
        target = str(public.parent / ("private-check.py" if target_kind == "private" else "unknown.py"))
        first = (call("edit_file", {"file_path": str(public), "old_string": "import json", "new_string": "# changed"}, "write-public")
                 if target_kind == "write-public" else call("read_file", {"file_path": target}, "unsafe-read"))
        return replace(scripted(), turns=(first, *scripted().turns))
    result, raw, target, io, public, _, checks = run_case(tmp_path, model)
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert not result["end_to_end_success"] and not raw["final_checks"]
    assert raw["permission_corrections"] == 0 and len(raw["permission_records"]) == 1
    assert raw["permission_records"][0]["receipt"]["action"] == "stop"
    assert result["model_requests"] == 1
    assert (target / "candidate" / "module.py").read_text() == BAD
    assert hashlib.sha256(public.read_bytes()).hexdigest() == checks["public"].protected_files[0].sha256
    assert SECRET not in json.dumps(requests(raw, io))


@pytest.mark.parametrize("target_kind", ["none", "list", "nul", "symlink-loop"])
def test_unresolvable_read_target_stops_real_worker_before_any_following_tool(tmp_path, target_kind):
    def model(public):
        if target_kind == "symlink-loop":
            loop = public.parent / "self-loop.py"
            loop.symlink_to(loop)
            target = str(loop)
        else:
            target = {"none": None, "list": ["module.py"], "nul": str(public.parent) + "/invalid\x00.py"}[target_kind]
        return replace(scripted(), turns=(call("read_file", {"file_path": target}, "invalid-target"), *scripted().turns))
    result, raw, target, io, _, _, _ = run_case(tmp_path, model)
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert result["model_requests"] == 1 and not raw["final_checks"]
    assert raw["permission_corrections"] == 0 and len(raw["permission_records"]) == 1
    denial = raw["permission_records"][0]["receipt"]
    assert denial["action"] == "stop" and denial["reason_code"] == "path_unavailable"
    assert denial["tool_executed"] is False
    if target_kind != "symlink-loop":
        assert denial["target_sha256"] is None
    assert (target / "candidate" / "module.py").read_text() == BAD
    assert SECRET not in json.dumps(requests(raw, io))


def test_off_mode_preserves_old_prompt_denial_path_and_has_no_correction(tmp_path):
    result, raw, target, _, _, _, _ = run_case(tmp_path,
        lambda public: scripted(first_read=public), mode="off")
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert "permission_corrections" not in raw and "permission_records" not in raw
    assert raw["prompt_version"] == "project-single-v1"
    # Historical mode finishes its original loop before controller audit. This
    # negative control preserves that path rather than silently rewriting it.
    assert result["model_requests"] == 3 and not raw["final_checks"]
    assert (target / "candidate" / "module.py").read_text() == GOOD


@pytest.mark.parametrize("mutation", [
    "foreign-job", "foreign-task", "foreign-attempt", "policy-hash", "source-hash", "target-hash",
    "missing-receipt", "missing-event", "event-identity", "event-tool", "event-arguments",
    "duplicate-receipt", "artifact-bytes", "audit-failed-without-receipt",
])
def test_forged_or_incomplete_permission_artifacts_never_exempt_guard(completed_case, mutation):
    _, raw, target, io, public, _, _ = completed_case
    result = CodingTaskResult.from_dict(raw["rounds"][0]["result"])
    task = CodingTaskSpec.from_json((target / "coding-0.input.json").read_text())
    recovery = PermissionRecoverySpec(mode=MODE, public_check_paths=(str(public),), remaining_corrections=1)
    outputs = list(result.outputs)
    values = payloads(result, io)
    receipt_index, receipt_ref, original = next((i, ref, value) for i, (ref, value) in enumerate(values)
                                               if value.get("record_type") == "permission_denial")
    event_index, _, baseline = next((i, ref, value) for i, (ref, value) in enumerate(values)
                                    if "baseline_hypothesis" in value)
    changed = copy.deepcopy(original)
    if mutation.startswith("foreign-"):
        changed[mutation.removeprefix("foreign-") + "_id"] = "foreign-identity"
    elif mutation in ("policy-hash", "source-hash"):
        changed["policy_sha256" if mutation == "policy-hash" else "workspace_revision"] = "0" * 64
    elif mutation == "target-hash":
        changed["receipt"]["target_sha256"] = normalized_target_sha256("/private/unknown.py")
    elif mutation == "missing-receipt":
        outputs.pop(receipt_index)
    elif mutation == "duplicate-receipt":
        outputs.append(receipt_ref)
    elif mutation == "artifact-bytes":
        fresh = io.put_json(result.job_id, {**changed, "workspace_revision": "1" * 64})
        (io.root / fresh.storage_key).write_text("{}")  # New isolated corrupt fixture, never the original Artifact.
        outputs[receipt_index] = fresh
    elif mutation.startswith("event-") or mutation in ("missing-event", "audit-failed-without-receipt"):
        altered = copy.deepcopy(baseline)
        denied = next(e for e in altered["events"] if "permission_denial" in e)
        if mutation == "missing-event":
            altered["events"].remove(denied)
        elif mutation == "audit-failed-without-receipt":
            denied.pop("permission_denial")
            denied["permission_audit_failed"] = True
            denied["result"] = "Error: permission denial was raised outside the trusted guard"
        elif mutation == "event-identity":
            denied["permission_denial"]["denial_id"] = "denial-" + "0" * 32
        elif mutation == "event-tool":
            denied["tool"] = "edit_file"
        else:
            denied["arguments"] = {"file_path": str(public)}
        outputs[event_index] = io.put_json(result.job_id, altered)
    if mutation in ("foreign-job", "foreign-task", "foreign-attempt", "policy-hash", "source-hash", "target-hash"):
        outputs[receipt_index] = io.put_json(result.job_id, changed)
    forged = replace(result, outputs=tuple(outputs))
    with pytest.raises(ValueError):
        records = permission_records(forged, io, recovery, task.workspace.source_manifest_sha256)
        guard_violations(forged, io, records)


def test_runner_rejects_missing_receipt_before_private_scoring(tmp_path, monkeypatch):
    execute = ResearchHarness.execute
    def lose_receipt(self, spec, config):
        result = execute(self, spec, config)
        if spec.kind == "coding":
            io = ArtifactIO(config.artifact_root)
            kept = tuple(ref for ref, data in payloads(result, io) if data.get("record_type") != "permission_denial")
            result = replace(result, outputs=kept)
        return result
    monkeypatch.setattr(ResearchHarness, "execute", lose_receipt)
    result, raw, _, _, _, _, _ = run_case(tmp_path)
    assert not result["end_to_end_success"] and result["failure_kind"] == "setup_or_integrity_error"
    assert not raw["policy_intact"] and not raw["final_checks"]
    assert len(raw["attempt_results"]) == 1 and raw["attempt_results"][0]["status"] == "succeeded"
    assert raw["budget"]["charged"]["model_calls"] == 4


@pytest.mark.parametrize("protection", ["neither", "protected-only", "argv-only"])
def test_worker_protocol_rejects_paths_not_in_both_trusted_command_and_protection(completed_case, protection):
    _, _, target, io, public, private, checks = completed_case
    task = CodingTaskSpec.from_json((target / "coding-0.input.json").read_text())
    recovery = PermissionRecoverySpec(mode=MODE, public_check_paths=(str(private),), remaining_corrections=1)
    command = task.verification
    protected = checks["public"].protected_files
    if protection == "protected-only":
        protected += checks["f2p"].protected_files
    if protection == "argv-only":
        command = replace(command, argv=command.argv + (str(private),))
    with pytest.raises(ValueError, match="Recovery targets"):
        WorkerRequest(spec=task, attempt_id="unauthorized-attempt", artifact_root=str(io.root), trace_path="/unused",
            parent_event_id="parent", permitted_verifications=(command,), protected_files=protected,
            permission_recovery=recovery)


def test_extra_permitted_private_command_does_not_expand_current_public_recovery_scope(completed_case):
    _, _, target, io, _, private, checks = completed_case
    task = CodingTaskSpec.from_json((target / "coding-0.input.json").read_text())
    private_command = checks["f2p"].command(Path(task.workspace.root))
    # A private verifier may be authorized for another controller phase. It is
    # not the current Coding Task's public verifier and grants no correction.
    recovery = PermissionRecoverySpec(mode=MODE, public_check_paths=(str(private),), remaining_corrections=1)
    with pytest.raises(ValueError, match="Recovery targets"):
        WorkerRequest(spec=task, attempt_id="foreign-checker", artifact_root=str(io.root), trace_path="/unused",
            parent_event_id="parent", permitted_verifications=(task.verification, private_command),
            protected_files=checks["public"].protected_files + checks["f2p"].protected_files,
            permission_recovery=recovery)


def test_execution_config_rejects_untyped_policy_and_harness_never_starts_unauthorized_worker(completed_case):
    _, _, target, io, _, private, checks = completed_case
    task = CodingTaskSpec.from_json((target / "coding-0.input.json").read_text())
    task = replace(task, task_id="unauthorized-config")
    kwargs = dict(artifact_root=str(io.root), allowed_workspaces=(task.workspace.root,), max_policy=task.policy,
                  permitted_verifications=(task.verification,), protected_files=checks["public"].protected_files)
    with pytest.raises(TypeError, match="permission recovery"):
        ExecutionConfig(**kwargs, permission_recovery={"public_check_paths": [str(private)]})
    config = ExecutionConfig(**kwargs, permission_recovery=PermissionRecoverySpec(
        mode=MODE, public_check_paths=(str(private),), remaining_corrections=1))
    harness = ResearchHarness()
    result = harness.execute(task, config)
    assert result.status == "incomplete" and result.failure_kind == "input_invalid"
    assert harness.records[result.attempt_id]["worker_pid"] is None
    harness.close_job(task.job_id)


def test_unknown_recovery_mode_is_rejected_before_materialization(tmp_path):
    source, task, checks, _ = fixture(tmp_path)
    target = tmp_path / "trial"
    with pytest.raises(ValueError, match="Unknown permission recovery"):
        ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
            scripted(), permission_recovery_mode="read-any-absolute-path")
    assert not target.exists()
