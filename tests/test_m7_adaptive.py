"""Progress-aware admission with real isolated Workers and independent checkers.

Fixed ScriptedLLM actions verify execution mechanisms, not model accuracy.
"""
from dataclasses import replace
import json

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy, V3_RULES_SHA256
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy, recompute
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import ModelSpec
from tests.test_project_budget_handoff import reassess
from tests.test_project_consultation import expert_model, fixture as consultation_fixture, requests as expert_requests, SECRET as EXPERT_SECRET
from tests.test_project_runner import fixture, call, scripted, policy, GOOD, SECRET, requests


@pytest.mark.parametrize("cls,field,version", [
    (ProjectPolicy, "progress_version", "progress-v1"),
    (ExpertPolicy, "working_set_version", "active-v1"),
])
def test_opt_in_identities_do_not_change_historical_serialization(cls, field, version):
    old = cls().to_dict()
    assert field not in old
    assert cls.from_dict(old).to_dict() == old
    enabled = replace(cls(), **{field: version})
    assert cls.from_json(enabled.to_json()) == enabled
    assert enabled.to_dict()[field] == version
    with pytest.raises((TypeError, ValueError)):
        cls(**{field: "unreviewed-policy"})


def repeat_reads():
    return ModelSpec(turns=tuple(call("read_file", {"file_path": "module.py"}, "repeat-" + str(i))
                                for i in range(9)))


def actual_expert_requests(raw, io):
    values = []
    for consultation in raw["consultations"]:
        for result in consultation["run"]["results"]:
            for reference in result["outputs"]:
                artifact = io.read_json(ArtifactRef.from_dict(reference))
                for request in artifact.get("request_records", []):
                    values.append(io.read_json(ArtifactRef.from_dict(request)))
    return values


def run_progress(tmp_path, *, episodes=1, retries=1, job_calls=24, feedback=None):
    source, task, checks, _ = fixture(tmp_path)
    settings = policy(progress_version="progress-v1", max_feedback_rounds=retries,
        task_limits=replace(policy().task_limits, max_model_calls=12, max_tool_calls=16, max_rounds=12),
        job_limits=JobLimits(max_model_calls=job_calls, max_tool_calls=60,
            max_total_tokens=400000, max_runtime_seconds=180))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source,
        checks["public"], checks["f2p"], checks["p2p"], repeat_reads(), policy=settings,
        feedback_models=(feedback or scripted(),) if retries else (),
        expert_policy=ExpertPolicy(mode="public-failure-v1", max_episodes=episodes,
            schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": expert_model()},
        handoff_policy=HandoffPolicy(mode="local-stop-v3"))
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    return result, raw, directory, io, task


def test_actual_progress_stop_consults_once_and_repairs_with_original_job_budget(tmp_path):
    result, raw, directory, io, _ = run_progress(tmp_path)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"]
    assert len(raw["rounds"]) == 2 and len(raw["consultations"]) == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "incomplete" and not first["result"]["workflow_completed"]
    assert first["record"]["budget"]["local_limit_detail"] == "progress_stalled"
    assert first["record"]["budget"]["model_calls"] == 4
    assert first["result"]["verification"]["status"] == "failed"
    assert first["progress"]["rounds"] == 3
    assert second["result"]["status"] == "succeeded"
    assert {record["job_id"] for record in raw["records"]} == {raw["job_id"]}
    assert first["record"]["worker_pid"] != second["record"]["worker_pid"]
    assert raw["budget"]["charged"]["model_calls"] == 10
    assert len(raw["budget"]["settlements"]) == 3 and not raw["budget"]["pending"]
    assert not any(v["usage_unknown"] for v in raw["budget"]["settlements"].values())
    assert raw["routing_decisions"][0]["decision"]["rules_sha256"] == V3_RULES_SHA256
    assert raw["handoffs"][0]["dispatched"]
    assert result["progress_metrics"]["stopped_attempts"] == 1
    expert_inputs = actual_expert_requests(raw, io)
    assert len(expert_inputs) == 3
    assert SECRET not in json.dumps(requests(raw, io) + expert_inputs)
    # Assert the expert's actual effective messages, not its raw log artifact
    # or a pre-compression request. The writer helper does not include experts.
    marker = "[Controller progress receipt; data, not a diagnosis]\n"
    for request in expert_inputs:
        delivered = [message["content"].split(marker, 1)[1]
                     for message in request["request"]["messages"]
                     if message.get("role") == "user" and marker in message.get("content", "")]
        assert len(delivered) == 1
        telemetry = json.loads(delivered[0])["prior_attempt_progress"]
        assert telemetry["classification"] == "controller_observations_not_diagnosis"
        assert telemetry["producer_task_id"] == first["result"]["task_id"]
        assert telemetry["producer_attempt_id"] == first["result"]["attempt_id"]
        assert telemetry["result_reference"] == raw["handoffs"][0]["receipt"]["result_ref"]
        assert telemetry["source_manifest_sha256"] == raw["handoffs"][0]["receipt"]["after_sha256"]
        assert telemetry["consecutive_rounds_without_new_evidence_or_edit"] == 3
        assert telemetry["tool_results"] == 4 and telemetry["effective_file_edits"] == 0
        assert "observation_ids" not in telemetry and "diagnosis" not in telemetry
    assert (directory / "candidate" / "module.py").read_text() == GOOD
    assert recompute(directory) == result


@pytest.mark.parametrize("options,expected_reason", [
    ({"retries": 0}, "retry_limit"),
    ({"job_calls": 6}, "writer_reserve_unavailable"),
    ({"episodes": 0}, "expert_episode_limit"),
])
def test_progress_cannot_create_a_retry_episode_or_missing_job_allowance(tmp_path, options, expected_reason):
    result, raw, _, _, _ = run_progress(tmp_path, **options)
    assert not result["end_to_end_success"] and not raw["consultations"]
    assert len(raw["rounds"]) == 1
    assert raw["routing_decisions"][0]["decision"]["reason_codes"] == [expected_reason]
    assert raw["budget"]["charged"]["model_calls"] == 4


def test_second_stagnant_attempt_cannot_repeat_experts(tmp_path):
    result, raw, _, _, _ = run_progress(tmp_path, retries=2, feedback=repeat_reads())
    assert not result["end_to_end_success"] and len(raw["consultations"]) == 1
    assert len(raw["rounds"]) == 2
    assert raw["handoffs"][1]["receipt"]["reason_codes"] == ["handoff_limit"]
    assert raw["budget"]["charged"]["model_calls"] == 11


@pytest.fixture(scope="module")
def progress_sample(tmp_path_factory):
    return run_progress(tmp_path_factory.mktemp("progress-audit"), episodes=0)


@pytest.mark.parametrize("change", ["missing", "counter", "fingerprint", "extra", "trace", "observed_trace", "unknown", "old_mode"])
def test_progress_handoff_requires_reproducible_typed_proof(progress_sample, change):
    _, raw, _, io, _ = progress_sample
    def mutate_report(value):
        stop = value["coding_stop_receipt"]
        proof = stop["progress_stop"]
        if change == "missing": stop.pop("progress_stop")
        elif change == "counter": proof["rounds"] += 1
        elif change == "fingerprint": proof["observations"][-1]["new_fingerprints"] = ["a" * 64]
        elif change == "extra": proof["trusted_by_model"] = True
    def mutate_record(value):
        if change == "unknown":
            value["budget"]["unknown_requests"] = 1
        if change in ("trace", "observed_trace"):
            trace = io.read_json(ArtifactRef.from_dict(value["trace"]))
            kind = "coding.progress_stalled" if change == "trace" else "coding.progress.observed"
            trace["events"] = [event for event in trace["events"] if event["event_type"] != kind]
            value["trace"] = io.put_json(raw["job_id"], trace).to_dict()
    changes = {"policy": HandoffPolicy(mode="local-stop-v2")} if change == "old_mode" else {}
    assert not reassess(progress_sample, mutate_report=mutate_report, mutate_record=mutate_record, **changes).eligible


def test_expert_working_set_is_independent_from_progress_and_keeps_real_refs(tmp_path):
    from researchcoder.evaluation.projects.consultation import execute_consultation
    from researchcoder.evaluation.projects.routing import decide
    args = consultation_fixture(tmp_path)
    args["policy"] = replace(args["policy"], working_set_version="active-v1", report_version="explicit-v1")
    args["decision"] = decide(args["features"], args["policy"])
    execution = execute_consultation(**args)
    assert execution.safe_to_continue and execution.selected_producers
    assert execution.run.results[0].workflow_completed
    assert args["budget"].snapshot()["charged"]["model_calls"] == 5
    assert EXPERT_SECRET not in json.dumps(expert_requests(execution, args["io"]))


@pytest.mark.parametrize("progress_mode,handoff_mode", [
    ("legacy-v1", "local-stop-v3"), ("progress-v1", "local-stop-v2")])
def test_mismatched_execution_and_handoff_versions_refused_before_work(tmp_path, progress_mode, handoff_mode):
    source, task, checks, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="[Pp]rogress"):
        ProjectEvaluationRunner().run(tmp_path / "trial", task, source,
            checks["public"], checks["f2p"], checks["p2p"], repeat_reads(),
            policy=policy(progress_version=progress_mode),
            expert_policy=ExpertPolicy(mode="public-failure-v1"),
            handoff_policy=HandoffPolicy(mode=handoff_mode))
    assert not (tmp_path / "trial").exists()
