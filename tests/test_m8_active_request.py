"""Active expert requests over real Workers; scripted results are not accuracy.

The original unfinished writer and its failed public check remain archived.
Proof mutation tests reuse one real execution without replaying model actions.
"""
from dataclasses import replace
import json

import pytest

from corecoder.llm import ToolCall
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.handoff import HandoffPolicy, LocalStopReceipt, validate_handoff
from researchcoder.evaluation.projects.contracts import RepositoryManifest
from researchcoder.evaluation.projects.routing import ExpertPolicy, PublicRoutingFeatures, decide
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, recompute
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.coding import CodingContinuationPolicy, validate_expert_request
from researchcoder.runtime.protocol import CodingTaskResult, ModelSpec, RuntimeLimits, ScriptedTurn, WorkerRequest
from tests.test_m6_completion import setup, execute, response
from tests.test_project_budget_handoff import reassess
from tests.test_project_consultation import expert_model
from tests.test_project_runner import fixture, call, scripted, policy, BAD, GOOD, SECRET, requests


VERSION = "explicit-request-v1"
QUESTION = "Check whether the increment operation satisfies the public contract while preserving identity."
REQUEST = dict(action="request_expert", requested_role="code_analysis", question=QUESTION,
               evidence_paths=["module.py"])
REQUEST_TEXT = json.dumps(REQUEST)


def request_model(text=REQUEST_TEXT):
    return ModelSpec(turns=(call("read_file", dict(file_path="module.py"), "read-source"),
        ScriptedTurn(content=text, prompt_tokens=10, completion_tokens=5)))


def run_request(tmp_path, *, model=None, enabled=True, expert_report=True, retries=1):
    source, task, checks, _ = fixture(tmp_path)
    settings = policy(completion_version="explicit-v1", progress_version="progress-v1",
        max_feedback_rounds=retries,
        task_limits=replace(policy().task_limits, max_model_calls=12, max_rounds=10,
                            max_tool_calls=24, max_total_tokens=200000, max_output_tokens=2048),
        job_limits=JobLimits(max_model_calls=24, max_tool_calls=60,
                            max_total_tokens=400000, max_runtime_seconds=180))
    directory = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(directory, task, source,
        checks["public"], checks["f2p"], checks["p2p"], model or request_model(), policy=settings,
        feedback_models=(scripted(),) if retries else (),
        expert_policy=ExpertPolicy(mode="public-failure-v1", schedule_version="serial-parallel-v2", max_workers=1),
        expert_models={"code_analysis": expert_model(report=expert_report)},
        handoff_policy=HandoffPolicy(mode="local-stop-v3", closure_version="reserve-incomplete-v1",
                                    request_version=VERSION if enabled else "off"))
    io = ArtifactIO(directory / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((directory / "raw-reference.json").read_text()))
    return result, raw, directory, io, task


def first_report(raw, io):
    values = [io.read_json(ArtifactRef.from_dict(ref)) for ref in raw["rounds"][0]["result"]["outputs"]]
    return next(value for value in values if isinstance(value, dict) and "baseline_hypothesis" in value)


def expert_requests(raw, io):
    values = []
    for consultation in raw["consultations"]:
        for result in consultation["run"]["results"]:
            for reference in result["outputs"]:
                value = io.read_json(ArtifactRef.from_dict(reference))
                for item in value.get("request_records", []):
                    values.append(io.read_json(ArtifactRef.from_dict(item)))
    return values


@pytest.mark.parametrize("contract", [CodingContinuationPolicy, HandoffPolicy])
def test_active_request_opt_in_keeps_historical_wire_identity(contract):
    old = contract(mode="local-stop-v3", closure_version="reserve-incomplete-v1")
    assert "request_version" not in old.to_dict()
    assert contract.from_json(old.to_json()) == old
    enabled = replace(old, request_version=VERSION)
    assert enabled.to_dict()["request_version"] == VERSION
    assert contract.from_json(enabled.to_json()) == enabled
    with pytest.raises((ValueError, TypeError)):
        contract(mode="local-stop-v3", request_version="accept-any-request")


def test_nested_worker_request_roundtrip_keeps_default_off_omitted(tmp_path):
    request, _, _ = setup(tmp_path)
    old = replace(request, coding_continuation=CodingContinuationPolicy(mode="local-stop-v3"))
    assert "request_version" not in old.to_dict()["coding_continuation"]
    assert WorkerRequest.from_json(old.to_json()) == old
    enabled = replace(old, coding_continuation=replace(old.coding_continuation, request_version=VERSION))
    assert WorkerRequest.from_json(enabled.to_json()) == enabled


def test_request_parser_preserves_original_and_normal_completion():
    value = validate_expert_request(REQUEST_TEXT)
    assert value["failure_kind"] is None and value["request"] == REQUEST
    assert value["original_response"] == REQUEST_TEXT and value["version"] == VERSION
    assert validate_expert_request("Applied the repair; Controller verification remains pending.") is None
    assert validate_expert_request(json.dumps(dict(status="completed", summary="changed source"))) is None
    assert validate_expert_request(REQUEST_TEXT, "length")["failure_kind"] == "response_length_exceeded"


@pytest.mark.parametrize("changes", [
    {"requested_role": "shell_executor"}, {"question": " "}, {"question": "x" * 2001},
    {"evidence_paths": []}, {"evidence_paths": ["module.py", "module.py"]},
    {"evidence_paths": ["../private-check.py"]}, {"evidence_paths": ["/etc/passwd"]},
    {"evidence_paths": ["a\\module.py"]}, {"extra_permissions": "write_anywhere"},
])
def test_request_parser_rejects_invalid_contract(changes):
    value = validate_expert_request(json.dumps(dict(REQUEST, **changes)))
    assert value is not None and value["failure_kind"] == "invalid_report"


@pytest.mark.parametrize("text", [
    '{"action":"request_expert",',
    '{"action":"request_expert","action":"request_expert","requested_role":"code_analysis",'
    '"question":"Investigate source", "evidence_paths":["module.py"]}',
])
def test_request_parser_does_not_repair_broken_or_duplicate_json(text):
    value = validate_expert_request(text)
    assert value is not None and value["failure_kind"] == "invalid_json"
    assert value["original_response"] == text


@pytest.mark.parametrize("text,finish_reason,kind", [
    (json.dumps(dict(REQUEST, action="request_unknown")), None, "invalid_report"),
    ('{"action":"request_ex', None, "invalid_json"),
    ('{"action":"request_ex', "length", "response_length_exceeded"),
])
def test_request_parser_rejects_unknown_or_truncated_action(text, finish_reason, kind):
    value = validate_expert_request(text, finish_reason)
    assert value is not None and value["failure_kind"] == kind
    assert value["original_response"] == text and value["request"] is None


def test_real_worker_request_expert_evidence_and_repair_use_one_job(tmp_path):
    result, raw, directory, io, _ = run_request(tmp_path)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"], raw
    assert len(raw["rounds"]) == 2 and len(raw["consultations"]) == 1
    first, second = raw["rounds"]
    assert first["result"]["status"] == "incomplete" and not first["result"]["workflow_completed"]
    assert first["result"]["failure_kind"] == "unknown"
    assert first["result"]["verification"]["status"] == "failed"
    assert first["before"] == first["after"] and first["changed_paths"] == []
    assert first["record"]["budget"]["stop_kind"] is None
    assert first["record"]["budget"]["local_limit_detail"] is None
    report = first_report(raw, io)
    assert report["coding_expert_request"]["original_response"] == REQUEST_TEXT
    assert report["coding_request_receipt"]["local_limit_detail"] == "expert_request"
    assert "coding_stop_receipt" not in report and "coding_closure_receipt" not in report
    assert second["result"]["status"] == "succeeded" and second["result"]["workflow_completed"]
    assert first["result"]["attempt_id"] != second["result"]["attempt_id"]
    assert first["record"]["worker_pid"] != second["record"]["worker_pid"]
    assert {record["job_id"] for record in raw["records"]} == {raw["job_id"]}
    assert raw["budget"]["charged"]["model_calls"] == 8
    assert len(raw["budget"]["settlements"]) == 3 and raw["budget"]["pending"] == {}
    assert not any(row["usage_unknown"] for row in raw["budget"]["settlements"].values())
    assert raw["handoffs"][0]["dispatched"]
    assert raw["handoffs"][0]["receipt"]["reason_codes"] == ["known_explicit_request"]
    assert raw["routing_decisions"][0]["decision"]["roles"] == ["code_analysis"]
    consultation = raw["consultations"][0]
    assert consultation["selected_producers"] and consultation["safe_to_continue"]
    context = io.read_json(ArtifactRef.from_dict(consultation["context_ref"]))
    assert context["observations"] and context["candidate_source_paths"]
    writer_inputs = requests(raw, io)
    later = json.dumps([item for item in writer_inputs if item["task_id"] == second["result"]["task_id"]])
    assert consultation["selected_producers"][0] in later and "increment" in later
    specialist_inputs = expert_requests(raw, io)
    assert specialist_inputs and QUESTION in json.dumps(specialist_inputs)
    assert SECRET not in json.dumps(writer_inputs + specialist_inputs)
    assert (directory / "candidate" / "module.py").read_text() == GOOD
    assert (tmp_path / "source" / "module.py").read_text() == BAD
    assert recompute(directory) == result


def test_enabled_request_does_not_force_expert_on_direct_success(tmp_path):
    result, raw, _, io, _ = run_request(tmp_path, model=scripted())
    assert result["end_to_end_success"] and len(raw["rounds"]) == 1
    assert not raw["consultations"] and not raw["handoffs"]
    assert "coding_request_receipt" not in first_report(raw, io)
    assert raw["budget"]["charged"]["model_calls"] == 3


@pytest.fixture(scope="module")
def request_sample(tmp_path_factory):
    # The expert's intentionally invalid report leaves candidate source unchanged,
    # allowing reassessment of the initial legal request without repeated Workers.
    sample = run_request(tmp_path_factory.mktemp("active-request-audit"), expert_report=False)
    assert len(sample[1]["rounds"]) == 1 and sample[1]["handoffs"][0]["dispatched"]
    assert (sample[2] / "candidate" / "module.py").read_text() == BAD
    return sample


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("coding_request_receipt"),
    lambda r: r.update(coding_request_rejection="invalid_report"),
    lambda r: r["coding_request_receipt"].update(attempt_id="foreign-attempt"),
    lambda r: r["coding_request_receipt"].update(before_manifest_sha256="0" * 64),
    lambda r: r["coding_request_receipt"].update(accounting_settled=False),
    lambda r: r["coding_request_receipt"]["request_stop"].update(trusted_by_model=True),
    lambda r: r["coding_request_receipt"]["request_stop"]["source_bindings"][0].update(file_sha256="0" * 64),
    lambda r: r["coding_request_receipt"]["request_stop"]["source_bindings"][0].update(event_sequences=[]),
    lambda r: r["coding_request_receipt"]["request_stop"].update(request_sha256="0" * 64),
    lambda r: r["coding_expert_request"].update(original_response=REQUEST_TEXT.replace("module.py", "LICENSE")),
    lambda r: r["events"][0].update(tool="edit_file"),
    lambda r: r["events"][0].update(status="error"),
    lambda r: r["events"][0]["arguments"].update(file_path="LICENSE"),
])
def test_request_proof_tampering_cannot_authorize_expert(request_sample, mutate):
    assert not reassess(request_sample, mutate_report=mutate).eligible


@pytest.mark.parametrize("mutation", [
    lambda r: r["budget"].update(unknown_requests=1),
    lambda r: r["budget"].update(pending={"unsettled": 20}),
    lambda r: r.update(attempt_id="foreign-attempt"),
])
def test_unknown_accounting_and_producer_identity_cannot_authorize_request(request_sample, mutation):
    assert not reassess(request_sample, mutate_record=mutation).eligible


@pytest.mark.parametrize("event_type", ["coding.expert_request", "coding.request.archived", "tool.call.end"])
def test_missing_request_trace_or_tool_pair_rejects_admission(request_sample, event_type):
    _, raw, _, io, _ = request_sample
    def mutate(record):
        trace = io.read_json(ArtifactRef.from_dict(record["trace"]))
        assert any(event["event_type"] == event_type for event in trace["events"])
        trace["events"] = [event for event in trace["events"] if event["event_type"] != event_type]
        record["trace"] = io.put_json(raw["job_id"], trace).to_dict()
    assert not reassess(request_sample, mutate_record=mutate).eligible


def test_request_cannot_cross_policy_identity_source_or_artifact_boundaries(request_sample):
    _, raw, directory, io, task = request_sample
    receipt = LocalStopReceipt.from_dict(raw["handoffs"][0]["receipt"])
    result = CodingTaskResult.from_dict(raw["rounds"][0]["result"])
    args = dict(candidate=directory / "candidate", manifest=RepositoryManifest.from_dict(raw["rounds"][0]["after"]),
                public_failure=result.verification, public_task=task, io=io)
    assert validate_handoff(receipt, **args) == receipt
    assert not reassess(request_sample, policy=HandoffPolicy(mode="local-stop-v3")).eligible
    assert reassess(request_sample, handoffs_used=1).reason_codes == ("handoff_limit",)
    for changed in (replace(receipt, attempt_id="foreign-attempt"),
                    replace(receipt, result_ref=replace(receipt.result_ref, sha256="0" * 64))):
        with pytest.raises(ValueError):
            validate_handoff(changed, **args)
    source = directory / "candidate" / "module.py"
    content = source.read_bytes()
    try:
        source.write_bytes(content + b"\n# external change\n")
        assert reassess(request_sample).reason_codes == ("source_changed",)
    finally:
        source.write_bytes(content)


@pytest.mark.parametrize("changes,reason", [
    ({"remaining_retries": 0}, "retry_limit"),
    ({"episodes_used": 1}, "expert_episode_limit"),
    ({"remaining_model_calls": 3}, "expert_allocation_unavailable"),
    ({"usage_unknown": True}, "accounting_unknown"),
])
def test_active_request_cannot_override_global_admission(request_sample, changes, reason):
    _, raw, _, _, _ = request_sample
    receipt = LocalStopReceipt.from_dict(raw["handoffs"][0]["receipt"])
    features = PublicRoutingFeatures.from_dict(raw["routing_decisions"][0]["features"])
    selected = decide(replace(features, **changes),
        ExpertPolicy(mode="public-failure-v1", schedule_version="serial-parallel-v2", max_workers=1), handoff=receipt)
    assert selected.action == "stop" and selected.reason_codes == (reason,)


@pytest.mark.parametrize("kind", ["no_op_write", "unread_path", "unknown_usage", "disabled"])
def test_runtime_request_requires_read_only_observed_source_and_known_usage(tmp_path, kind):
    limits = RuntimeLimits(max_model_calls=10, max_rounds=10, max_tool_calls=16,
        max_runtime_seconds=30, max_context_tokens=16000, max_output_tokens=512)
    request, io, hooks = setup(tmp_path, limits=limits)
    request = replace(request, coding_continuation=CodingContinuationPolicy(mode="local-stop-v3",
        request_version="off" if kind == "disabled" else VERSION))
    hooks.ledger.coding_continuation = request.coding_continuation
    turns = [response(calls=[ToolCall("read", "read_file", dict(file_path="module.py"))])]
    if kind == "no_op_write":
        turns.append(response(calls=[ToolCall("noop", "edit_file",
            dict(file_path="module.py", old_string="value = 0", new_string="value = 0"))]))
    text = json.dumps(dict(REQUEST, evidence_paths=["unread.py"])) if kind == "unread_path" else REQUEST_TEXT
    turns.append(response(text, known=kind != "unknown_usage"))
    result, raw, model = execute(request, io, hooks, turns)
    assert "coding_request_receipt" not in raw and result.status != "succeeded"
    if kind == "disabled":
        assert "coding_expert_request" not in raw
        assert all(VERSION not in json.dumps(item) for item in model.inputs)
    else:
        assert not result.workflow_completed
        assert raw["coding_expert_request"]["original_response"] == text
        if kind == "unknown_usage":
            assert hooks.ledger.unknown_requests == 1 and result.verification.status == "not_run"
        else:
            assert result.verification.status == "failed"
