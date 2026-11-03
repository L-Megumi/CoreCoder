"""Actual Worker/Verifier/Exchange coverage for the versioned investigation handoff.

These are fixed-action engineering tests. They do not measure LLM accuracy.
"""
from dataclasses import replace
import json

import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import ModelSpec
from tests.test_project_budget_handoff import reassess
from tests.test_project_consultation import expert_model
from tests.test_project_runner import fixture, call, scripted, policy, BAD, GOOD, PARTIAL, SECRET, requests


def investigation_model(*, prior_edit=False):
    turns = []
    if prior_edit:
        turns = [call("read_file", {"file_path": "module.py"}, "before-edit"),
            call("edit_file", {"file_path": "module.py", "old_string": BAD, "new_string": PARTIAL}, "partial-edit")]
    turns.extend(call("read_file", {"file_path": "module.py"}, "read-" + str(i)) for i in range(9))
    return ModelSpec(turns=tuple(turns))


def run_investigation(tmp_path, *, mode="local-stop-v2", expert_limit=1, feedback_rounds=1,
                      feedback=None, prior_edit=False, job_calls=24):
    source, task, checks, private = fixture(tmp_path)
    settings = policy(max_feedback_rounds=feedback_rounds,
        task_limits=replace(policy().task_limits, max_model_calls=12, max_tool_calls=16, max_rounds=12),
        job_limits=JobLimits(max_model_calls=job_calls, max_tool_calls=60,
            max_total_tokens=400000, max_runtime_seconds=180))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source, checks["public"], checks["f2p"], checks["p2p"],
        investigation_model(prior_edit=prior_edit), policy=settings,
        feedback_models=(feedback or scripted(GOOD, PARTIAL if prior_edit else BAD),) if feedback_rounds else (),
        expert_policy=ExpertPolicy(mode="public-failure-v1", max_episodes=expert_limit,
            schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": expert_model()}, handoff_policy=HandoffPolicy(mode=mode))
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    return result, raw, directory, io, task


@pytest.mark.parametrize("prior_edit", [False, True])
def test_v2_real_workers_continue_only_after_trusted_failed_verification(tmp_path, prior_edit):
    result, raw, directory, io, _ = run_investigation(tmp_path, prior_edit=prior_edit)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["rounds"]) == 2 and len(raw["consultations"]) == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "incomplete" and not first["result"]["workflow_completed"]
    assert first["result"]["failure_kind"] == "budget_exhausted"
    assert first["record"]["budget"]["local_limit_detail"] == "investigation_limit"
    assert first["result"]["verification"]["status"] == "failed"
    assert second["result"]["status"] == "succeeded"
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert first["record"]["worker_pid"] != second["record"]["worker_pid"]
    assert {r["job_id"] for r in raw["records"]} == {raw["job_id"]}
    assert raw["budget"]["charged"]["model_calls"] == (16 if prior_edit else 14)
    assert len(raw["budget"]["settlements"]) == 3 and not raw["budget"]["pending"]
    assert raw["handoffs"][0]["dispatched"] and raw["handoffs"][0]["receipt"]["eligible"]
    assert (directory / "candidate" / "module.py").read_text() == GOOD
    assert SECRET not in json.dumps(requests(raw, io))


def test_v1_real_worker_preserves_historical_tool_error_and_stops(tmp_path):
    result, raw, _, io, _ = run_investigation(tmp_path, mode="local-stop-v1")
    assert not result["end_to_end_success"] and result["failure_kind"] == "tool_error"
    assert len(raw["rounds"]) == 1 and not raw["consultations"] and not raw["handoffs"]
    assert raw["rounds"][0]["result"]["verification"] is None
    assert raw["budget"]["charged"]["model_calls"] == 8
    trace = io.read_json(ArtifactRef.from_dict(raw["rounds"][0]["record"]["trace"]))
    assert next(e for e in trace["events"] if e["event_type"] == "worker.execution_error")["reason"] == "RuntimeError"


def test_investigation_handoff_cannot_reset_job_budget_or_create_missing_retry(tmp_path):
    result, raw, _, _, _ = run_investigation(tmp_path, feedback_rounds=0)
    assert not result["end_to_end_success"] and not raw["consultations"]
    assert len(raw["rounds"]) == 1
    assert raw["routing_decisions"][0]["decision"]["reason_codes"] == ["retry_limit"]
    assert raw["budget"]["charged"]["model_calls"] == 8
    assert not raw["budget"]["pending"]


def test_insufficient_job_remainder_refuses_experts_and_writer_restart(tmp_path):
    result, raw, _, _, _ = run_investigation(tmp_path, job_calls=9)
    assert not result["end_to_end_success"] and not raw["consultations"]
    assert len(raw["rounds"]) == 1 and not raw["handoffs"][0]["dispatched"]
    assert raw["routing_decisions"][0]["decision"]["reason_codes"] == ["writer_reserve_unavailable"]
    assert raw["budget"]["charged"]["model_calls"] == 8
    assert not raw["budget"]["pending"]


def test_second_investigation_stop_cannot_repeat_the_bounded_handoff(tmp_path):
    result, raw, _, _, _ = run_investigation(tmp_path,
        feedback_rounds=2, feedback=investigation_model())
    assert not result["end_to_end_success"] and len(raw["rounds"]) == 2
    assert len(raw["consultations"]) == 1
    assert raw["handoffs"][0]["dispatched"]
    assert not raw["handoffs"][1]["dispatched"]
    assert raw["handoffs"][1]["receipt"]["reason_codes"] == ["handoff_limit"]
    assert raw["budget"]["charged"]["model_calls"] == 19
    assert not raw["budget"]["pending"]


@pytest.fixture(scope="module")
def investigation_sample(tmp_path_factory):
    return run_investigation(tmp_path_factory.mktemp("m5-investigation-audit"), expert_limit=0)


@pytest.mark.parametrize("change", ["missing", "foreign_source", "bad_limit", "wrong_event_count", "duplicate_window", "extra_field"])
def test_controller_refuses_modified_investigation_proof(investigation_sample, change):
    def mutate(report):
        stop = report["coding_stop_receipt"]
        proof = stop["investigation_stop"]
        if change == "missing": stop.pop("investigation_stop")
        elif change == "foreign_source": proof["source"] = "model-invented-limit"
        elif change == "bad_limit": proof["limit"] = 0
        elif change == "wrong_event_count": proof["event_count"] += 1
        elif change == "duplicate_window": proof["window_event_ends"][1] = 1
        else: proof["untrusted_claim"] = True
    receipt = reassess(investigation_sample, mutate_report=mutate)
    assert not receipt.eligible and receipt.reason_codes == ("invalid_investigation_proof",)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_count", "wrong_limit", "wrong_source", "after_verifier"])
def test_controller_requires_actual_guard_trace_before_verifier(investigation_sample, change):
    _, raw, _, io, _ = investigation_sample
    def mutate(record):
        trace = io.read_json(ArtifactRef.from_dict(record["trace"]))
        events = trace["events"]
        event = next(e for e in events if e["event_type"] == "coding.investigation_limit")
        if change == "missing": events.remove(event)
        elif change == "duplicate": events.append(dict(event))
        elif change == "wrong_count": event["count"] += 1
        elif change == "wrong_limit": event["limit"] += 1
        elif change == "wrong_source": event["reason"] = "model_claim"
        else:
            events.remove(event)
            events.append(event)
        record["trace"] = io.put_json(raw["job_id"], trace).to_dict()
    receipt = reassess(investigation_sample, mutate_record=mutate)
    assert not receipt.eligible and receipt.reason_codes == ("invalid_investigation_proof",)


def test_new_receipt_cannot_be_replayed_under_old_handoff_policy(investigation_sample):
    value = reassess(investigation_sample, policy=HandoffPolicy(mode="local-stop-v1"))
    assert not value.eligible and value.reason_codes == ("unsupported_local_limit",)


@pytest.mark.parametrize("change", ["pending", "unknown", "crash", "foreign_attempt", "source_digest"])
def test_existing_identity_accounting_and_source_guards_remain_strict(investigation_sample, change):
    def record(value):
        if change == "pending": value["budget"]["pending"] = {"unfinished": 1}
        elif change == "unknown": value["budget"]["unknown_requests"] = 1
        elif change == "crash": value["worker_exit_code"] = -9
        elif change == "foreign_attempt": value["attempt_id"] = "foreign-attempt"
    def report(value):
        if change == "source_digest": value["coding_stop_receipt"]["after_manifest_sha256"] = "0" * 64
    assert not reassess(investigation_sample, mutate_record=record, mutate_report=report).eligible
