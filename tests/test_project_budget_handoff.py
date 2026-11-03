"""Known local stops use real Workers and retain the original incomplete Attempt."""
from dataclasses import replace
import copy
import json

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.handoff import HandoffPolicy, LocalStopReceipt, assess_handoff, validate_handoff
from researchcoder.evaluation.projects.routing import ExpertPolicy, RULES_SHA256, V2_RULES_SHA256, decide, extract_public_features
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner
from researchcoder.evaluation.projects.contracts import RepositoryManifest
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import CodingTaskResult, ModelSpec
from tests.test_project_consultation import expert_model
from tests.test_project_runner import fixture, call, scripted, policy, BAD, GOOD, PARTIAL, SECRET, requests


def local_stop_model():
    return ModelSpec(turns=(
        call("read_file", {"file_path": "module.py"}, "read-original"),
        call("edit_file", {"file_path": "module.py", "old_string": BAD, "new_string": PARTIAL}, "partial-edit"),
        call("read_file", {"file_path": "module.py"}, "read-partial"),
        call("read_file", {"file_path": "module.py"}, "not-admitted")))


def run_handoff(tmp_path, *, enabled=True, expert_limit=1, feedback=None, feedback_rounds=1):
    source, task, checks, private = fixture(tmp_path)
    settings = policy(max_feedback_rounds=feedback_rounds,
        task_limits=replace(policy().task_limits, max_model_calls=3),
        job_limits=JobLimits(max_model_calls=24, max_tool_calls=60, max_total_tokens=400000, max_runtime_seconds=180))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source, checks["public"], checks["f2p"], checks["p2p"],
        local_stop_model(), policy=settings,
        feedback_models=(feedback or scripted(GOOD, PARTIAL),) if feedback_rounds else (),
        expert_policy=ExpertPolicy(mode="public-failure-v1", max_episodes=expert_limit,
            schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": expert_model()},
        handoff_policy=HandoffPolicy(mode="local-stop-v1") if enabled else None)
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    return result, raw, directory, io, task


def test_real_local_stop_handoff_preserves_original_failure_and_same_job(tmp_path):
    result, raw, directory, io, task = run_handoff(tmp_path)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["rounds"]) == 2 and len(raw["consultations"]) == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "incomplete" and not first["result"]["workflow_completed"]
    assert first["result"]["failure_kind"] == "budget_exhausted"
    assert first["result"]["verification"]["status"] == "failed"
    assert second["result"]["status"] == "succeeded"
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert {r["job_id"] for r in raw["records"]} == {raw["job_id"]}
    assert raw["budget"]["charged"]["model_calls"] == 9
    assert len(raw["budget"]["settlements"]) == 3 and not raw["budget"]["pending"]
    assert raw["handoffs"][0]["dispatched"] and raw["handoffs"][0]["receipt"]["eligible"]
    route = raw["routing_decisions"][0]
    assert not route["features"]["workflow_completed"]
    assert route["features"]["failure_kind"] == "budget_exhausted"
    assert route["decision"]["rules_sha256"] == V2_RULES_SHA256 != RULES_SHA256
    assert (directory/"candidate"/"module.py").read_text() == GOOD
    assert SECRET not in json.dumps(requests(raw, io))


def test_default_path_keeps_original_budget_stop(tmp_path):
    result, raw, _, _, _ = run_handoff(tmp_path, enabled=False)
    assert not result["end_to_end_success"] and result["failure_kind"] == "budget_exhausted"
    assert len(raw["rounds"]) == 1 and not raw["consultations"]
    assert "handoffs" not in raw and "handoff_policy" not in raw


def test_no_remaining_writer_retry_cannot_consult(tmp_path):
    result, raw, _, _, _ = run_handoff(tmp_path, feedback_rounds=0)
    assert not result["end_to_end_success"] and not raw["consultations"]
    assert raw["routing_decisions"][0]["decision"]["reason_codes"] == ["retry_limit"]


@pytest.fixture(scope="module")
def audit_sample(tmp_path_factory):
    return run_handoff(tmp_path_factory.mktemp("handoff-audit"), expert_limit=0)


def reassess(sample, *, mutate_record=None, mutate_report=None, mutate_result=None, **changes):
    _, raw, directory, io, _ = sample
    receipt = LocalStopReceipt.from_dict(raw["handoffs"][0]["receipt"])
    record = copy.deepcopy(io.read_json(receipt.record_ref))
    result = CodingTaskResult.from_dict(io.read_json(receipt.result_ref))
    if mutate_record:
        mutate_record(record)
    if mutate_report:
        outputs = []
        for ref in result.outputs:
            report = io.read_json(ref)
            if isinstance(report, dict) and "baseline_hypothesis" in report:
                mutate_report(report)
                ref = io.put_json(result.job_id, report)
            outputs.append(ref)
        result = replace(result, outputs=tuple(outputs))
    if mutate_result:
        result = mutate_result(result)
    return assess_handoff(result_ref=io.put_json(result.job_id, result.to_dict()),
        record_ref=io.put_json(result.job_id, record), before=receipt.before,
        after=RepositoryManifest.from_dict(raw["rounds"][0]["after"]),
        writable_paths=receipt.writable_paths, candidate=directory/"candidate", io=io,
        policy=changes.pop("policy", receipt.policy), **dict(
            handoffs_used=changes.pop("handoffs_used", 0),
            context_relief_available=changes.pop("context_relief_available", False)), **changes)


@pytest.mark.parametrize("mutation", [
    lambda r: r["budget"].update(pending={"unsettled": 10}),
    lambda r: r["budget"].update(unknown_requests=1),
    lambda r: r["budget"].pop("settlements"),
    lambda r: r["budget"]["settlements"].append(r["budget"]["settlements"][0]),
    lambda r: r["budget"].update(charged_tokens=r["budget"]["charged_tokens"]+1),
    lambda r: r["budget"].update(last_failure_kind="model_error"),
    lambda r: r.update(worker_exit_code=-9),
])
def test_unknown_or_incomplete_accounting_never_authorizes_handoff(audit_sample, mutation):
    receipt = reassess(audit_sample, mutate_record=mutation)
    assert not receipt.eligible and receipt.reason_codes == ("accounting_unknown",)


@pytest.mark.parametrize("detail", [None, "tool_calls", "timeout", "invented"])
def test_unreviewed_local_limits_never_reset_a_worker(audit_sample, detail):
    receipt = reassess(audit_sample, mutate_record=lambda r: r["budget"].update(local_limit_detail=detail))
    assert not receipt.eligible and receipt.reason_codes == ("unsupported_local_limit",)


def test_context_limit_needs_an_actual_context_relief_policy(audit_sample):
    receipt = reassess(audit_sample, mutate_record=lambda r: r["budget"].update(local_limit_detail="context_tokens"))
    assert not receipt.eligible and receipt.reason_codes == ("context_relief_unavailable",)


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop("coding_stop_receipt"),
    lambda r: r["coding_stop_receipt"].update(attempt_id="forged"),
    lambda r: r["coding_stop_receipt"].update(before_manifest_sha256="0"*64),
    lambda r: r["coding_stop_receipt"].update(after_manifest_sha256="0"*64),
    lambda r: r["coding_stop_receipt"].update(events_complete=False),
    lambda r: r["coding_stop_receipt"].update(accounting_settled=False),
    lambda r: r.update(model_error_type="APIError"),
    lambda r: r.update(permission_stopped=True),
    lambda r: r.update(file_changes=[]),
    lambda r: r["file_changes"][0].update(after_sha256="0"*64),
    lambda r: r["events"][1].update(status="error"),
    lambda r: r["verification"].update(exit_code=0),
])
def test_proof_tampering_and_unknown_writes_never_authorize_handoff(audit_sample, mutation):
    receipt = reassess(audit_sample, mutate_report=mutation)
    assert not receipt.eligible


@pytest.mark.parametrize("mutation", [
    lambda r: replace(r, status="succeeded", workflow_completed=True, failure_kind=None,
        verification=replace(r.verification, status="passed", exit_code=0)),
    lambda r: replace(r, verification=None),
    lambda r: replace(r, verification=replace(r.verification, status="passed", exit_code=0)),
    lambda r: replace(r, verification=replace(r.verification,
        stdout=replace(r.verification.stdout, truncated=True))),
])
def test_missing_or_nonfailure_public_check_is_not_a_handoff(audit_sample, mutation):
    assert not reassess(audit_sample, mutate_result=mutation).eligible


def test_handoff_policy_roundtrip_and_one_episode_limit(audit_sample):
    policy_value = HandoffPolicy(mode="local-stop-v1")
    assert HandoffPolicy.from_json(policy_value.to_json()) == policy_value
    with pytest.raises(ValueError):
        HandoffPolicy(max_handoffs=2)
    assert reassess(audit_sample, handoffs_used=1).reason_codes == ("handoff_limit",)
    assert reassess(audit_sample, policy=HandoffPolicy()).reason_codes == ("handoff_disabled",)


def test_consultation_revalidates_original_proof_identity(audit_sample):
    _, raw, directory, io, task = audit_sample
    receipt = LocalStopReceipt.from_dict(raw["handoffs"][0]["receipt"])
    result = CodingTaskResult.from_dict(raw["rounds"][0]["result"])
    args = dict(candidate=directory/"candidate", manifest=RepositoryManifest.from_dict(raw["rounds"][0]["after"]),
        public_failure=result.verification, public_task=task, io=io)
    assert validate_handoff(receipt, **args) == receipt
    for changed in (replace(receipt, attempt_id="forged"), replace(receipt, after_sha256="0"*64),
                    replace(receipt, verification_sha256="0"*64)):
        with pytest.raises(ValueError):
            validate_handoff(changed, **args)


@pytest.mark.parametrize("change", ["empty", "identity", "unpaired", "duplicate", "wrong_kind"])
def test_incomplete_or_foreign_execution_trace_cannot_authorize_handoff(audit_sample, change):
    _, raw, _, io, _ = audit_sample
    def mutate(record):
        trace = io.read_json(ArtifactRef.from_dict(record["trace"]))
        if change == "empty":
            trace["events"] = []
        elif change == "identity":
            trace["events"][0]["attempt_id"] = "foreign-attempt"
        elif change == "unpaired":
            trace["events"] = [e for e in trace["events"] if e["event_type"] != "tool.call.end"]
        elif change == "duplicate":
            trace["events"].append(next(e for e in trace["events"] if e["event_type"] == "tool.call.end"))
        else:
            next(e for e in trace["events"] if e["event_type"] == "tool.call.end")["event_type"] = "verification.command.end"
        record["trace"] = io.put_json(raw["job_id"], trace).to_dict()
    receipt = reassess(audit_sample, mutate_record=mutate)
    assert not receipt.eligible and receipt.reason_codes == ("incomplete_execution_trace",)
