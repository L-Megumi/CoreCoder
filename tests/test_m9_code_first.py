"""Explicit Code-first cooperation through real Workers and private checks.

Scripted actions validate execution/provenance, not real-model effectiveness.
Negative receipt controls reuse one archived real preflight without replaying it.
"""
from dataclasses import replace
import copy
import json

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import ArtifactRef, TaskResult, VerificationResult
from researchcoder.evaluation.projects.consultation import InitialConsultationReceipt, validate_initial_consultation
from researchcoder.evaluation.projects.contracts import RepositoryManifest
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy, PublicRoutingFeatures, decide
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import CodingTaskSpec
from tests.test_project_consultation import expert_model
from tests.test_project_runner import fixture, policy, scripted, BAD, GOOD, SECRET, requests


VERSION = "code-first-v1"


def run_initial(tmp_path, *, enabled=True, known=True):
    source, task, checks, _ = fixture(tmp_path)
    settings = policy(completion_version="explicit-v1", progress_version="progress-v1",
        max_feedback_rounds=1,
        task_limits=replace(policy().task_limits, max_model_calls=12, max_rounds=10,
                            max_tool_calls=24, max_total_tokens=200000, max_output_tokens=2048),
        job_limits=JobLimits(max_model_calls=24, max_tool_calls=60,
                            max_total_tokens=400000, max_runtime_seconds=180))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source,
        checks["public"], checks["f2p"], checks["p2p"], scripted(), policy=settings,
        expert_policy=ExpertPolicy(mode="public-failure-v1", initial_mode=VERSION if enabled else "off",
            schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": expert_model(known_usage=known)},
        writer_context_policy=ContextPolicy(role="repair", strategy="evidence", allow_recovery=True,
            coding_source_mode="observed-source-v2", working_set_version="active-v1"),
        handoff_policy=HandoffPolicy(mode="local-stop-v3", closure_version="reserve-incomplete-v1",
                                    request_version="explicit-request-v1"))
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    return result, raw, directory, io, task


def specialist_requests(raw, io):
    values = []
    for consultation in raw["consultations"]:
        for result in consultation["run"]["results"]:
            for reference in result["outputs"]:
                data = io.read_json(ArtifactRef.from_dict(reference))
                for item in data.get("request_records", []):
                    values.append(io.read_json(ArtifactRef.from_dict(item)))
    return values


def test_initial_mode_preserves_historical_policy_wire_identity():
    old = ExpertPolicy(mode="public-failure-v1")
    assert "initial_mode" not in old.to_dict()
    assert ExpertPolicy.from_json(old.to_json()) == old
    new = replace(old, initial_mode=VERSION)
    assert new.to_dict()["initial_mode"] == VERSION
    assert ExpertPolicy.from_json(new.to_json()) == new
    assert new.sha256 != old.sha256
    with pytest.raises((ValueError, TypeError)):
        replace(old, initial_mode="model-can-start-any-worker")


def test_real_initial_code_worker_evidence_and_first_writer_complete_one_job(tmp_path):
    from researchcoder.evaluation.projects.routing import INITIAL_RULES_SHA256
    result, raw, directory, io, _ = run_initial(tmp_path)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["consultations"]) == len(raw["rounds"]) == 1
    first = raw["rounds"][0]
    assert first["generation"] == 0 and first["result"]["task_id"] == "coding-0"
    assert first["result"]["status"] == "succeeded" and first["result"]["workflow_completed"]
    assert raw["handoffs"] == []  # No fictional stopped writer authorizes initial cooperation.
    initial = InitialConsultationReceipt.from_dict(raw["initial_consultation"]["receipt"])
    assert initial.task_id == "preflight-public"
    original = TaskResult.from_dict(io.read_json(initial.result_ref))
    assert original.kind == "verify" and original.status == "failed"
    assert original.workflow_completed is False  # Preserve the actual verifier's old semantics.
    assert original.verification.status == "failed" and original.verification.exit_code == 1
    assert raw["routing_decisions"][0]["phase"] == "initial_public_failure"
    route = raw["routing_decisions"][0]
    assert route["decision"]["roles"] == ["code_analysis"]
    assert route["decision"]["rules_sha256"] == INITIAL_RULES_SHA256
    assert route["features"]["workflow_completed"] is False
    assert route["features"]["failure_kind"] == original.failure_kind
    assert len(raw["records"]) == 2
    assert raw["records"][0]["role"] == "code_analysis"
    assert raw["records"][1]["task_id"] == "coding-0"
    assert len({r["worker_pid"] for r in raw["records"]}) == 2
    assert {r["job_id"] for r in raw["records"]} == {raw["job_id"]}
    traces = [io.read_json(ArtifactRef.from_dict(r["trace"]))["events"] for r in raw["records"]]
    expert_ended = max(e["monotonic_ns"] for e in traces[0] if e["event_type"] == "model.request.end")
    writer_started = min(e["monotonic_ns"] for e in traces[1] if e["event_type"] == "model.request.start")
    assert expert_ended < writer_started
    assert raw["budget"]["charged"]["model_calls"] == 6
    assert len(raw["budget"]["settlements"]) == 2 and not raw["budget"]["pending"]
    assert not any(s["usage_unknown"] for s in raw["budget"]["settlements"].values())
    consultation = raw["consultations"][0]
    assert consultation["safe_to_continue"] and consultation["selected_producers"]
    context = io.read_json(ArtifactRef.from_dict(consultation["context_ref"]))
    assert context["observations"] and context["candidate_source_paths"]
    before = RepositoryManifest.from_dict(raw["source_before"])
    assert context["complete_source_sha256"] == initial.source_manifest_sha256 == before.digest
    source_sha = next(entry.sha256 for entry in before.entries if entry.path == "module.py")
    assert any(p["candidate_relative_path"] == "module.py" and p["sha256"] == source_sha
               for p in context["candidate_source_paths"])
    writer_spec = CodingTaskSpec.from_json((directory / "coding-0.input.json").read_text())
    assert writer_spec.policy.profile == "project-expert-reread-v1"
    writer_inputs = requests(raw, io)
    visible = json.dumps(writer_inputs)
    assert consultation["selected_producers"][0] in visible
    assert "increment" in visible and "module.py" in visible
    assert "[Controller cooperation option: explicit-request-v1]" not in visible
    assert SECRET not in json.dumps(writer_inputs + specialist_requests(raw, io))
    reports = [io.read_json(ArtifactRef.from_dict(ref)) for ref in first["result"]["outputs"]]
    report = next(v for v in reports if "baseline_hypothesis" in v)
    actual = [e["tool"] for e in report["events"]]
    assert actual.index("read_file") < actual.index("edit_file")
    assert "coding_request_receipt" not in report and "coding_closure_receipt" not in report
    assert (directory / "candidate/module.py").read_text() == GOOD
    assert (tmp_path / "source/module.py").read_text() == BAD
    assert recompute(directory) == result


def test_default_off_keeps_direct_success_without_initial_expert(tmp_path):
    result, raw, directory, _, _ = run_initial(tmp_path, enabled=False)
    assert result["end_to_end_success"] and len(raw["rounds"]) == 1
    assert not raw["consultations"] and "initial_consultation" not in raw
    assert "initial_mode" not in raw["expert_policy"]
    assert result["model_requests"] == 3
    assert recompute(directory) == result


@pytest.fixture(scope="module")
def initial_sample(tmp_path_factory):
    sample = run_initial(tmp_path_factory.mktemp("initial-audit"), known=False)
    result, raw, directory, _, _ = sample
    assert result["status"] == "needs_review" and result["failure_kind"] == "accounting_unknown"
    assert not result["end_to_end_success"] and not raw["rounds"]
    assert len(raw["consultations"]) == 1 and len(raw["budget"]["settlements"]) == 1
    assert any(s["usage_unknown"] for s in raw["budget"]["settlements"].values())
    assert raw["budget"]["charged"]["total_tokens"] > 0 and not raw["budget"]["pending"]
    assert (directory / "candidate/module.py").read_text() == BAD
    return sample


def initial_arguments(sample):
    _, raw, directory, io, _ = sample
    receipt = InitialConsultationReceipt.from_dict(raw["initial_consultation"]["receipt"])
    failure = VerificationResult.from_dict(raw["preflight"]["checks"]["public"]["result"]["verification"])
    return receipt, dict(candidate=directory / "candidate", manifest=RepositoryManifest.from_dict(raw["source_before"]),
        public_failure=failure, io=io, policy=ExpertPolicy.from_dict(raw["expert_policy"]),
        features=PublicRoutingFeatures.from_dict(raw["routing_decisions"][0]["features"]))


def test_initial_receipt_revalidates_real_public_verifier_not_writer(initial_sample):
    receipt, args = initial_arguments(initial_sample)
    validate_initial_consultation(receipt, **args)
    assert InitialConsultationReceipt.from_json(receipt.to_json()) == receipt


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(task_id="coding-pretend"),
    lambda r: r.update(attempt_id="foreign-attempt"),
    lambda r: r.update(kind="coding"),
    lambda r: r.update(workflow_completed=True),
    lambda r: r.update(status="succeeded", failure_kind=None),
    lambda r: r["verification"].update(exit_code=0, status="passed"),
    lambda r: r["verification"].update(checked_workspace_revision="0" * 64),
])
def test_initial_result_forgery_cannot_authorize_expert(initial_sample, mutation):
    receipt, args = initial_arguments(initial_sample)
    value = copy.deepcopy(args["io"].read_json(receipt.result_ref))
    mutation(value)
    with pytest.raises((ValueError, TypeError)):
        ref = args["io"].put_json(receipt.job_id, value)
        validate_initial_consultation(replace(receipt, result_ref=ref), **args)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(worker_exit_code=-9),
    lambda r: r.update(attempt_id="foreign-attempt"),
    lambda r: r["budget"].update(unknown_requests=1),
    lambda r: r["budget"].update(pending={"unsettled": 1}),
    lambda r: r.pop("trace"),
])
def test_initial_worker_accounting_or_identity_forgery_is_rejected(initial_sample, mutation):
    receipt, args = initial_arguments(initial_sample)
    value = copy.deepcopy(args["io"].read_json(receipt.record_ref))
    mutation(value)
    ref = args["io"].put_json(receipt.job_id, value)
    with pytest.raises((ValueError, TypeError)):
        validate_initial_consultation(replace(receipt, record_ref=ref), **args)


def test_initial_source_artifact_and_protected_command_are_bound(initial_sample):
    receipt, args = initial_arguments(initial_sample)
    variants = (
        replace(receipt, source_manifest_sha256="0" * 64),
        replace(receipt, result_ref=replace(receipt.result_ref, sha256="0" * 64)),
        replace(receipt, check=replace(receipt.check, argv_template=receipt.check.argv_template[:-1] + ("f2p",))),
    )
    for forged in variants:
        with pytest.raises((ValueError, TypeError)):
            validate_initial_consultation(forged, **args)
    with pytest.raises((ValueError, TypeError)):
        validate_initial_consultation(receipt, **dict(args, policy=replace(args["policy"], initial_mode="off")))


@pytest.mark.parametrize("event_type", ["verification.command.start", "verification.command.end", "worker.result"])
def test_initial_trace_cannot_omit_actual_execution(initial_sample, event_type):
    receipt, args = initial_arguments(initial_sample)
    record = copy.deepcopy(args["io"].read_json(receipt.record_ref))
    trace = args["io"].read_json(ArtifactRef.from_dict(record["trace"]))
    assert any(e["event_type"] == event_type for e in trace["events"])
    trace["events"] = [e for e in trace["events"] if e["event_type"] != event_type]
    record["trace"] = args["io"].put_json(receipt.job_id, trace).to_dict()
    ref = args["io"].put_json(receipt.job_id, record)
    with pytest.raises((ValueError, TypeError)):
        validate_initial_consultation(replace(receipt, record_ref=ref), **args)


@pytest.mark.parametrize("changes", [
    dict(remaining_model_calls=0), dict(remaining_tool_calls=0), dict(remaining_total_tokens=0),
    dict(remaining_seconds=0), dict(remaining_retries=0), dict(episodes_used=1),
    dict(usage_unknown=True), dict(budget_pending=True), dict(policy_intact=False),
    dict(source_intact=False), dict(cancelled=True), dict(verification_status="passed"),
])
def test_initial_route_cannot_override_resource_and_safety_limits(initial_sample, changes):
    _, args = initial_arguments(initial_sample)
    features = replace(args["features"], **changes)
    assert decide(features, args["policy"], initial=True).action == "stop"


def test_later_public_failure_cannot_start_a_second_expert_episode(initial_sample):
    _, args = initial_arguments(initial_sample)
    features = replace(args["features"], workflow_completed=True, failure_kind="verification_failed",
                       episodes_used=1, remaining_retries=1)
    assert decide(features, args["policy"]).action != "consult_experts"
