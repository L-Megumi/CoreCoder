"""Bounded closing-phase handoff over real Workers and independent checks.

Scripted actions exercise execution and provenance, not real-model accuracy.
The first writer's valid incomplete report remains an incomplete Attempt.
"""
from dataclasses import replace
import json
import sys

import pytest

from corecoder.llm import ToolCall
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.handoff import HandoffPolicy, LocalStopReceipt
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.coding import CodingContinuationPolicy
from researchcoder.runtime.protocol import ModelSpec, RuntimeLimits, ScriptedTurn, WorkerRequest
from tests.test_m6_completion import setup, execute, response
from tests.test_project_budget_handoff import reassess
from tests.test_project_consultation import expert_model
from tests.test_project_runner import fixture, call, scripted, policy, BAD, GOOD, SECRET, requests


VERSION = "reserve-incomplete-v1"
INCOMPLETE = json.dumps(dict(status="incomplete", summary="No modification was made; further source investigation is required."))


def closing_model(final=INCOMPLETE):
    # Distinct source windows avoid intentionally triggering progress-stalled.
    turns = [call("read_file", dict(file_path="module.py", offset=line, limit=1), "read-" + str(line))
             for line in (1, 2, 4, 5)]
    return ModelSpec(turns=tuple(turns) + (ScriptedTurn(content=final, prompt_tokens=10, completion_tokens=5),))


def run_closure(tmp_path, *, enabled=True, episodes=1, retries=1, job_calls=24,
                final=INCOMPLETE, boundary="round_reserve", model=None, feedback=None,
                writer_context_policy=None, worker_command=None):
    source, task, checks, _ = fixture(tmp_path)
    limits = replace(policy().task_limits, max_model_calls=10 if boundary == "round_reserve" else 5,
        max_rounds=6 if boundary == "round_reserve" else 10, max_tool_calls=16,
        max_total_tokens=100000, max_runtime_seconds=30)
    settings = policy(completion_version="explicit-v1", progress_version="progress-v1",
        max_feedback_rounds=retries, task_limits=limits,
        job_limits=JobLimits(max_model_calls=job_calls, max_tool_calls=60,
                            max_total_tokens=400000, max_runtime_seconds=180))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source,
        checks["public"], checks["f2p"], checks["p2p"], model or closing_model(final), policy=settings,
        feedback_models=(feedback or scripted(),) if retries else (),
        expert_policy=ExpertPolicy(mode="public-failure-v1", max_episodes=episodes,
            schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": expert_model()},
        writer_context_policy=writer_context_policy, worker_command=worker_command,
        handoff_policy=HandoffPolicy(mode="local-stop-v3", closure_version=VERSION if enabled else "off"))
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    return result, raw, directory, io, task


def first_report(raw, io):
    values = [io.read_json(ArtifactRef.from_dict(ref)) for ref in raw["rounds"][0]["result"]["outputs"]]
    return next(value for value in values if isinstance(value, dict) and "baseline_hypothesis" in value)


@pytest.mark.parametrize("contract", [HandoffPolicy, CodingContinuationPolicy])
def test_new_closure_opt_in_preserves_old_serialization(contract):
    old = contract(mode="local-stop-v3")
    assert "closure_version" not in old.to_dict()
    assert contract.from_dict(old.to_dict()).to_dict() == old.to_dict()
    enabled = replace(old, closure_version=VERSION)
    assert contract.from_json(enabled.to_json()) == enabled
    assert enabled.to_dict()["closure_version"] == VERSION
    with pytest.raises((TypeError, ValueError)):
        contract(mode="local-stop-v3", closure_version="accept-any-unknown")


def test_nested_worker_request_keeps_default_off_wire_identity(tmp_path):
    request, _, _ = setup(tmp_path)
    old = replace(request, coding_continuation=CodingContinuationPolicy(mode="local-stop-v3"))
    assert "closure_version" not in old.to_dict()["coding_continuation"]
    assert WorkerRequest.from_json(old.to_json()) == old
    enabled = replace(old, coding_continuation=replace(old.coding_continuation, closure_version=VERSION))
    assert enabled.to_dict()["coding_continuation"]["closure_version"] == VERSION
    assert WorkerRequest.from_json(enabled.to_json()) == enabled


@pytest.mark.parametrize("boundary", ["round_reserve", "model_call_reserve"])
def test_real_closure_expert_evidence_repair_keeps_original_failure_and_job(tmp_path, boundary):
    from researchcoder.evaluation.projects.routing import CLOSURE_RULES_SHA256, V3_RULES_SHA256
    result, raw, directory, io, _ = run_closure(tmp_path, boundary=boundary)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["rounds"]) == 2 and len(raw["consultations"]) == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "incomplete" and not first["result"]["workflow_completed"]
    assert first["result"]["failure_kind"] == "unknown"
    assert first["result"]["verification"]["status"] == "failed"
    assert first["before"] == first["after"]
    assert first["record"]["budget"]["stop_kind"] is None
    report = first_report(raw, io)
    assert report["coding_final_report"]["original_response"] == INCOMPLETE
    assert report["coding_final_report"]["reason"] == "model_declared_incomplete"
    assert report["coding_final_report"]["trigger"] == boundary
    assert "coding_closure_receipt" in report and "coding_stop_receipt" not in report
    assert report["file_changes"] == []
    assert second["result"]["status"] == "succeeded" and second["result"]["workflow_completed"]
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert first["record"]["worker_pid"] != second["record"]["worker_pid"]
    assert {record["job_id"] for record in raw["records"]} == {raw["job_id"]}
    assert raw["budget"]["charged"]["model_calls"] == 11
    assert len(raw["budget"]["settlements"]) == 3 and raw["budget"]["pending"] == {}
    assert not any(row["usage_unknown"] for row in raw["budget"]["settlements"].values())
    handoff = raw["handoffs"][0]
    assert handoff["dispatched"] and handoff["receipt"]["reason_codes"] == ["known_closure_stop"]
    assert raw["routing_decisions"][0]["decision"]["rules_sha256"] == CLOSURE_RULES_SHA256 != V3_RULES_SHA256
    consultation = raw["consultations"][0]
    assert consultation["selected_producers"] and consultation["safe_to_continue"]
    context = io.read_json(ArtifactRef.from_dict(consultation["context_ref"]))
    assert context["observations"] and context["candidate_source_paths"]
    recorded = requests(raw, io)
    later = json.dumps([value for value in recorded if value["task_id"] == second["result"]["task_id"]])
    assert consultation["selected_producers"][0] in later
    assert "increment" in later and SECRET not in json.dumps(recorded)
    assert (directory / "candidate" / "module.py").read_text() == GOOD
    assert (tmp_path / "source" / "module.py").read_text() == BAD
    assert recompute(directory) == result


def test_default_off_keeps_valid_incomplete_unfinished_without_experts(tmp_path):
    result, raw, _, io, _ = run_closure(tmp_path, enabled=False)
    assert not result["end_to_end_success"] and len(raw["rounds"]) == 1
    assert not raw["consultations"] and not raw["handoffs"]
    assert raw["rounds"][0]["result"]["failure_kind"] == "unknown"
    assert "coding_closure_receipt" not in first_report(raw, io)
    assert raw["budget"]["charged"]["model_calls"] == 5


def test_actual_source_restore_can_continue_through_expert_handoff_and_repair(tmp_path):
    from researchcoder.context.engine import RESTORE
    from researchcoder.context.policy import ContextPolicy
    from tests.test_coding_recovery_delivery_integration import passive_worker
    # This existing observer resolves only a source ID actually exposed to the
    # scripted model. Real restore, accounting, Worker and validator run intact.
    wrapper, audit = passive_worker(tmp_path)
    turns = (
        call("read_file", dict(file_path="module.py"), "read-source"),
        call(RESTORE, dict(reference_id="$SOURCE", start_line=2, max_lines=1), "restore-source"),
        call("read_file", dict(file_path="module.py", offset=1, limit=1), "read-header"),
        call("read_file", dict(file_path="module.py", offset=4, limit=1), "read-identity"),
        ScriptedTurn(content=INCOMPLETE, prompt_tokens=10, completion_tokens=5),
    )
    result, raw, _, io, _ = run_closure(tmp_path, model=ModelSpec(turns=turns),
        writer_context_policy=ContextPolicy(role="repair", strategy="evidence", allow_recovery=True,
            coding_source_mode="observed-source-v2", working_set_version="active-v1"),
        worker_command=(sys.executable, "-B", str(wrapper)))
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["consultations"]) == 1 and raw["handoffs"][0]["dispatched"]
    first = raw["rounds"][0]
    report = first_report(raw, io)
    proof = report["coding_closure_receipt"]["closure_stop"]
    # Recovery is intercepted by the Context Engine, so Agent events omit it;
    # the complete tool Trace and ledger must still account for that execution.
    assert len(report["events"]) == proof["event_count"] == 3
    assert proof["tool_count"] == 4 and first["record"]["budget"]["tool_calls"] == 5
    assert first["result"]["status"] == "incomplete" and first["result"]["failure_kind"] == "unknown"
    observed = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    restored = [row["result"] for row in observed if row["kind"] == "restore_return"]
    assert len(restored) == 1 and "return value - 1" in restored[0]["content"]
    provider = [row for row in observed if row["kind"] == "provider_input"]
    actual_reply = next(message for message in provider[2]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") == "restore-source")
    assert json.loads(actual_reply["content"]) == restored[0]
    assert raw["budget"]["charged"]["model_calls"] == 11 and not raw["budget"]["pending"]


@pytest.mark.parametrize("options,reason", [
    ({"retries": 0}, "retry_limit"),
    ({"episodes": 0}, "expert_episode_limit"),
    ({"job_calls": 7}, "writer_reserve_unavailable"),
])
def test_closure_cannot_create_retry_episode_or_job_allowance(tmp_path, options, reason):
    result, raw, _, _, _ = run_closure(tmp_path, **options)
    assert not result["end_to_end_success"] and not raw["consultations"]
    assert len(raw["rounds"]) == 1
    assert raw["routing_decisions"][0]["decision"]["reason_codes"] == [reason]
    assert raw["budget"]["charged"]["model_calls"] == 5


@pytest.mark.parametrize("final,kind", [
    ('{"status":"incomplete","summary":"cut', "invalid_json"),
    ('{"status":"incomplete","summary":"missing evidence","allow_retry":true}', "invalid_report"),
])
def test_malformed_final_report_never_buys_an_expert(tmp_path, final, kind):
    result, raw, _, io, _ = run_closure(tmp_path, final=final)
    assert not result["end_to_end_success"] and not raw["consultations"]
    assert len(raw["rounds"]) == 1 and raw["rounds"][0]["result"]["failure_kind"] == kind
    assert "coding_closure_receipt" not in first_report(raw, io)
    assert first_report(raw, io)["coding_final_report"]["original_response"] == final


@pytest.fixture(scope="module")
def closure_sample(tmp_path_factory):
    return run_closure(tmp_path_factory.mktemp("closure-audit"), episodes=0)


@pytest.mark.parametrize("mutation", [
    lambda row: row["budget"].update(pending={"unsettled": 10}),
    lambda row: row["budget"].update(unknown_requests=1),
    lambda row: row["budget"].update(last_failure_kind="model_error"),
    lambda row: row["budget"].update(stop_kind="budget_exhausted"),
    lambda row: row["budget"]["settlements"].append(row["budget"]["settlements"][0]),
    lambda row: row.update(worker_exit_code=-9),
    lambda row: row.update(attempt_id="foreign-attempt"),
])
def test_unknown_accounting_or_foreign_worker_record_cannot_authorize_closure(closure_sample, mutation):
    assert not reassess(closure_sample, mutate_record=mutation).eligible


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("coding_closure_receipt"),
    lambda value: value["coding_closure_receipt"].update(attempt_id="foreign-attempt"),
    lambda value: value["coding_closure_receipt"].update(before_manifest_sha256="0" * 64),
    lambda value: value["coding_closure_receipt"]["closure_stop"].update(rounds=999),
    lambda value: value["coding_closure_receipt"]["closure_stop"].update(report_sha256="0" * 64),
    lambda value: value["coding_closure_receipt"]["closure_stop"].update(effective_file_edits=1),
    lambda value: value["coding_closure_receipt"]["closure_stop"].update(trusted_by_model=True),
    lambda value: value["coding_final_report"].update(original_response='{"status":"completed","summary":"done"}'),
    lambda value: value["coding_final_report"].update(trigger="invented"),
    lambda value: value["coding_final_report"].update(reason="free-form-incomplete"),
    lambda value: value.update(permission_stopped=True),
    lambda value: value.update(model_error_type="APIError"),
    lambda value: value["events"][0].update(permission_audit_failed=True),
    lambda value: value["events"][0].update(tool="edit_file"),
    lambda value: value["verification"].update(exit_code=0),
])
def test_forged_or_unsafe_closure_proof_is_rejected(closure_sample, mutation):
    assert not reassess(closure_sample, mutate_report=mutation).eligible


@pytest.mark.parametrize("event_type", ["coding.report_phase", "coding.report_received", "coding.closure.archived", "tool.call.end"])
def test_missing_controller_finalization_or_tool_pair_is_rejected(closure_sample, event_type):
    _, raw, _, io, _ = closure_sample
    def mutate(record):
        trace = io.read_json(ArtifactRef.from_dict(record["trace"]))
        assert any(event["event_type"] == event_type for event in trace["events"])
        trace["events"] = [event for event in trace["events"] if event["event_type"] != event_type]
        record["trace"] = io.put_json(raw["job_id"], trace).to_dict()
    assert not reassess(closure_sample, mutate_record=mutate).eligible


@pytest.mark.parametrize("mutation", [
    lambda value: replace(value, verification=None),
    lambda value: replace(value, verification=replace(value.verification, status="passed", exit_code=0)),
    lambda value: replace(value, verification=replace(value.verification,
        stdout=replace(value.verification.stdout, truncated=True))),
    lambda value: replace(value, failure_kind="model_error"),
])
def test_missing_public_failure_or_wrong_termination_is_rejected(closure_sample, mutation):
    assert not reassess(closure_sample, mutate_result=mutation).eligible


def test_closure_cannot_be_replayed_under_old_policy_or_twice(closure_sample):
    assert not reassess(closure_sample, policy=HandoffPolicy(mode="local-stop-v3")).eligible
    assert reassess(closure_sample, handoffs_used=1).reason_codes == ("handoff_limit",)


def test_nested_receipt_retains_legacy_policy_wire_identity(closure_sample):
    receipt = LocalStopReceipt.from_dict(closure_sample[1]["handoffs"][0]["receipt"])
    assert receipt.to_dict()["policy"]["closure_version"] == VERSION
    assert LocalStopReceipt.from_json(receipt.to_json()) == receipt
    old_shape = replace(receipt, policy=HandoffPolicy(mode="local-stop-v3"), reason_codes=("known_local_stop",))
    assert "closure_version" not in old_shape.to_dict()["policy"]
    assert LocalStopReceipt.from_json(old_shape.to_json()) == old_shape


def test_changed_workspace_invalidates_a_previously_valid_closure(tmp_path):
    sample = run_closure(tmp_path, episodes=0)
    source = sample[2] / "candidate" / "module.py"
    source.write_text(source.read_text() + "\n# independent change\n")
    receipt = reassess(sample)
    assert not receipt.eligible and receipt.reason_codes == ("source_changed",)


@pytest.mark.parametrize("kind", ["length", "unknown", "no_op_write", "write_then_revert"])
def test_runtime_closure_receipt_excludes_incomplete_transport_and_any_writes(tmp_path, kind):
    limits = RuntimeLimits(max_model_calls=10, max_rounds=6, max_tool_calls=16,
        max_runtime_seconds=30, max_context_tokens=16000, max_output_tokens=512)
    request, io, hooks = setup(tmp_path, limits=limits, source="value = 0\n# a\n# b\n# c\n")
    request = replace(request, coding_continuation=CodingContinuationPolicy(mode="local-stop-v3", closure_version=VERSION))
    hooks.ledger.coding_continuation = request.coding_continuation
    turns = [response(calls=[ToolCall("read-" + str(line), "read_file",
        dict(file_path="module.py", offset=line, limit=1))]) for line in (1, 2, 3, 4)]
    if kind == "no_op_write":
        turns[1] = response(calls=[ToolCall("noop", "edit_file",
            dict(file_path="module.py", old_string="value = 0", new_string="value = 0"))])
    if kind == "write_then_revert":
        turns[1] = response(calls=[ToolCall("edit", "edit_file",
            dict(file_path="module.py", old_string="value = 0", new_string="value = 2"))])
        turns[2] = response(calls=[ToolCall("revert", "edit_file",
            dict(file_path="module.py", old_string="value = 2", new_string="value = 0"))])
    turns.append(response(INCOMPLETE, finish_reason="length" if kind == "length" else "stop",
                          known=kind != "unknown"))
    result, raw, model = execute(request, io, hooks, turns)
    assert len(model.inputs) == 5 and not result.workflow_completed
    assert result.status == "incomplete" and "coding_closure_receipt" not in raw
    assert raw["coding_final_report"]["original_response"] == INCOMPLETE
    if kind == "length":
        assert result.failure_kind == "response_length_exceeded"
    elif kind == "unknown":
        assert hooks.ledger.unknown_requests == 1 and result.verification.status == "not_run"
    else:
        assert any(event["tool"] == "edit_file" for event in raw["events"])
        assert result.verification.status == "failed"
