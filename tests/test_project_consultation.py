"""Real isolated expert Workers; fixed actions are engineering fixtures, not accuracy."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest

from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef, OutputSlice, VerificationResult, VerificationSpec
from researchcoder.evaluation.projects.consultation import execute_consultation
from researchcoder.evaluation.projects.contracts import ProjectRunInput
from researchcoder.evaluation.projects.routing import ExpertPolicy, decide, extract_public_features
from researchcoder.evaluation.projects.workspace import capture_manifest
from researchcoder.orchestration.budget import JobBudget, JobLimits
from researchcoder.orchestration.scheduler import DAGScheduler
from researchcoder.runtime.protocol import ModelSpec, RuntimeLimits, ScriptedCall, ScriptedTurn


SECRET = "private-oracle-never-in-expert-request-6ac42c"
LOG = "AssertionError: increment(3) returned 2, expected 4"
CHAINED = '''Traceback (most recent call last):
  File "module.py", line 2, in increment
    return value - 1
AssertionError: increment(3) returned 2, expected 4

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "entry.py", line 1, in run
    increment(3)
RuntimeError: public reproduction failed
'''


def expert_model(path="module.py", *, known_usage=True, report=True):
    """Reusable fixed extract/read/strict-report fixture over actual observed IDs."""
    usage = dict(prompt_tokens=10 if known_usage else 0, completion_tokens=5 if known_usage else 0)
    turns = [ScriptedTurn(calls=(ScriptedCall(call_id="extract", name="extract_log_evidence"),), **usage),
             ScriptedTurn(calls=(ScriptedCall(call_id="read", name="read_project_file",
                 arguments_json=json.dumps(dict(path=path))),), **usage)]
    value = dict(summary="The observed source should be compared with the public failure.",
        historical_failure=dict(description="The public log records an assertion failure.", evidence_ids=["$E0"]),
        current_state=dict(assessment="The current implementation was read.", observation_ids=["$O0"]),
        hypotheses=[dict(explanation="The operation may violate the public increment contract.",
            evidence_ids=["$E0"], observation_ids=["$O0"], verification="Run the authorized public check after repair.")],
        recommended_actions=["Re-read candidate source and make a minimal repair if supported."],
        limitations=["Fixed offline actions; semantic diagnosis accuracy is not established."])
    turns.append(ScriptedTurn(content=json.dumps(value) if report else "{incomplete JSON", **usage))
    return ModelSpec(turns=tuple(turns))


def fixture(tmp_path, *, chained=False, known=True):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "module.py").write_text("def increment(value):\n    return value - 1\n")
    (root / ".projectrc").write_text("mode = stable\n")
    (root / "README.rst").write_text("Public increment contract\n")
    (tmp_path / "private-oracle.py").write_text(SECRET)
    manifest = capture_manifest(root)
    public_log = CHAINED if chained else LOG
    task = ProjectRunInput(project_id="fixture", source_revision="frozen-v1", manifest=manifest,
        objective="Fix increment to increase its input by one without changing configuration.",
        public_log=public_log, writable_paths=("module.py",))
    failure = VerificationResult(job_id="job", task_id="coding-0", attempt_id="attempt-0",
        verification_id="public-verification-0", status="failed", exit_code=1,
        spec=VerificationSpec(argv=(sys.executable, "public-check.py")),
        stdout=OutputSlice(text="public check failed", original_chars=19),
        checked_workspace_revision=workspace_revision(root))
    budget = JobBudget(JobLimits(max_model_calls=24, max_tool_calls=60,
                                 max_total_tokens=400000, max_runtime_seconds=180))
    # Preserve actual first-attempt responsibility in the same object.
    first = RuntimeLimits(max_model_calls=4, max_tool_calls=4, max_total_tokens=10000)
    budget.reserve("coding-0", first)
    budget.settle("coding-0", dict(budget=dict(model_calls=2, tool_calls=2, charged_tokens=30,
        unknown_requests=0, known_prompt_tokens=20, known_completion_tokens=10, estimated_input_tokens=1000)))
    policy = ExpertPolicy(mode="public-failure-v1")
    features = extract_public_features(public_log=public_log, source_files=tuple(e.path for e in manifest.entries),
        verification_status="failed", workflow_completed=True, failure_kind="verification_failed",
        verification_stdout=failure.stdout.text, verification_stderr=failure.stderr.text,
        remaining_model_calls=22, remaining_tool_calls=58, remaining_total_tokens=399970,
        remaining_seconds=120, remaining_retries=1)
    return dict(job_id="job", candidate=root, manifest=manifest, public_task=task,
        public_failure=failure, io=ArtifactIO(tmp_path/"artifacts"), budget=budget, model=expert_model(known_usage=known),
        decision=decide(features, policy), policy=policy, features=features, remaining_seconds=120)


def requests(value, io):
    result = []
    for task in value.run.results:
        for reference in task.outputs:
            raw = io.read_json(reference)
            for request in raw.get("request_records", []):
                result.append(io.read_json(ArtifactRef.from_dict(request)))
    return result


def test_actual_expert_worker_preserves_same_job_budget_and_legal_handoff(tmp_path):
    args = fixture(tmp_path)
    value = execute_consultation(**args)
    assert value.safe_to_continue and value.stop_reason is None, value.to_dict()
    assert value.run.status == "succeeded" and len(value.records) == 1
    assert value.records[0]["worker_pid"] and value.records[0]["worker_exit_code"] == 0
    assert value.run.results[0].job_id == "job" and value.run.results[0].kind == "diagnose"
    assert value.run.results[0].verification is None
    assert value.budget_before["charged"]["model_calls"] == 2
    assert value.budget_after["charged"]["model_calls"] == 5
    assert len(value.budget_after["settlements"]) == 2 and not value.budget_after["pending"]
    assert value.budget_after == args["budget"].snapshot()
    data = json.loads(value.context)
    assert data["observations"] and data["hypotheses"]
    assert all(item["classification"] == "unverified_hypothesis" for item in data["hypotheses"])
    assert data["candidate_source_paths"][0]["candidate_relative_path"] == "module.py"
    assert value.bundle.items[0].reference.producer_task_id in value.selected_producers
    assert args["io"].read_json(value.context_ref) == data
    assert capture_manifest(args["candidate"]) == args["manifest"]
    captured = json.dumps(requests(value, args["io"]))
    assert SECRET not in captured and "private-oracle.py" not in captured
    assert "edit_file" not in captured and "write_file" not in captured
    assert json.loads(json.dumps(value.to_dict()))["report_semantics"] == "expert_hypotheses_not_single_validated_diagnosis"


def test_code_success_can_be_used_without_relabelling_failed_log_task(tmp_path):
    args = fixture(tmp_path, chained=True)
    value = execute_consultation(**args, role_models={"log_analysis": expert_model(report=False)})
    assert value.safe_to_continue and value.run.status != "succeeded", value.to_dict()
    assert len(value.run.results) == 2 and len({r["worker_pid"] for r in value.records}) == 2
    code = next(result for result in value.run.results if result.task_id.endswith("expert-code"))
    log = next(result for result in value.run.results if result.task_id.endswith("expert-log"))
    assert code.status == "succeeded" and log.status == "incomplete"
    assert not log.workflow_completed and log.failure_kind == "invalid_json"
    assert value.selected_producers == (code.task_id,)
    assert {item.reference.producer_task_id for item in value.bundle.items} == {code.task_id}
    assert len(value.budget_after["settlements"]) == 3


def test_all_failed_reports_are_not_promoted_to_a_successful_handoff(tmp_path):
    args = fixture(tmp_path)
    args["model"] = expert_model(report=False)
    value = execute_consultation(**args)
    assert not value.safe_to_continue and value.stop_reason == "consultation_no_evidence"
    assert value.run.results[0].status == "incomplete" and not value.selected_producers
    assert value.bundle is None and not value.context


def test_log_local_tool_exhaustion_does_not_discard_successful_code_evidence(tmp_path):
    args = fixture(tmp_path, chained=True)
    model = expert_model()
    burst = ScriptedTurn(calls=tuple(ScriptedCall(call_id="read-"+str(index), name="read_project_file",
        arguments_json=json.dumps(dict(path="module.py"))) for index in range(7)),
        prompt_tokens=10, completion_tokens=5)
    log = replace(model, turns=(model.turns[0], burst, model.turns[-1]))
    value = execute_consultation(**args, role_models={"log_analysis": log})
    assert value.safe_to_continue and value.run.status == "incomplete", value.to_dict()
    failed = next(result for result in value.run.results if result.task_id.endswith("expert-log"))
    assert failed.status == "incomplete" and failed.failure_kind == "local_budget_exhausted"
    assert not failed.workflow_completed and failed.evidence_refs
    assert len(value.selected_producers) == 1 and value.selected_producers[0].endswith("expert-code")
    assert all(item.reference.producer_task_id in value.selected_producers for item in value.bundle.items)
    record = next(record for record in value.records if record["role"] == "log_analysis")
    assert record["budget"]["local_limit_detail"] == "tool_calls"
    assert not any(s["usage_unknown"] for s in value.budget_after["settlements"].values())
    assert value.budget_after["charged"]["model_calls"] == (
        value.budget_before["charged"]["model_calls"] + sum(r["budget"]["model_calls"] for r in value.records))


def test_unknown_provider_usage_stops_consultation_and_preserves_debt(tmp_path):
    args = fixture(tmp_path, known=False)
    value = execute_consultation(**args)
    assert not value.safe_to_continue and value.stop_reason == "accounting_unknown"
    assert any(s["usage_unknown"] for s in value.budget_after["settlements"].values())
    assert value.budget_after["charged"]["total_tokens"] > value.budget_before["charged"]["total_tokens"]


@pytest.mark.parametrize("violation", ["private_read", "write_tool"])
def test_actual_worker_refuses_unauthorized_expert_actions(tmp_path, violation):
    args = fixture(tmp_path)
    if violation == "private_read":
        args["model"] = expert_model(str(tmp_path/"private-oracle.py"))
    else:
        model = expert_model()
        bad = ScriptedTurn(calls=(ScriptedCall(call_id="forbidden", name="edit_file",
            arguments_json=json.dumps(dict(file_path="module.py", old_string="value - 1", new_string="value + 1"))),),
            prompt_tokens=10, completion_tokens=5)
        args["model"] = replace(model, turns=(bad,)+model.turns)
    value = execute_consultation(**args)
    assert not value.safe_to_continue and value.stop_reason == "policy_violation", value.to_dict()
    assert capture_manifest(args["candidate"]) == args["manifest"]
    assert SECRET not in json.dumps(requests(value, args["io"]))


@pytest.mark.parametrize("change", ["source", "stale_verification", "pending", "charged", "seconds", "cancel"])
def test_admission_rechecks_real_source_budget_time_and_cancellation(tmp_path, change):
    args = fixture(tmp_path)
    reason = "source_changed"
    if change == "source":
        (args["candidate"]/"README.rst").write_text("changed\n")
    elif change == "stale_verification":
        args["public_failure"] = replace(args["public_failure"], checked_workspace_revision="a"*64)
    elif change == "pending":
        args["budget"].reserve("other", RuntimeLimits(max_model_calls=1, max_tool_calls=1, max_total_tokens=10000))
        reason = "accounting_unknown"
    elif change == "charged":
        caps = RuntimeLimits(max_model_calls=20, max_tool_calls=4, max_total_tokens=10000)
        args["budget"].reserve("other", caps)
        args["budget"].settle("other", dict(budget=dict(model_calls=20, tool_calls=0, charged_tokens=20, unknown_requests=0)))
        reason = "budget_exhausted"
    elif change == "seconds":
        args["remaining_seconds"] = 30
        reason = "budget_exhausted"
    else:
        event = threading.Event()
        event.set()
        args["cancel_event"] = event
        reason = "interrupted"
    before = args["budget"].snapshot()
    value = execute_consultation(**args)
    assert value.stop_reason == reason and not value.safe_to_continue and value.run is None
    assert value.budget_after == before


def test_mutation_after_real_expert_run_invalidates_handoff(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = DAGScheduler.execute
    def mutate(self, *pos, **kw):
        run = original(self, *pos, **kw)
        (args["candidate"]/".projectrc").write_text("changed\n")
        return run
    monkeypatch.setattr(DAGScheduler, "execute", mutate)
    value = execute_consultation(**args)
    assert value.run.results[0].status == "succeeded"
    assert value.stop_reason == "source_changed" and not value.safe_to_continue
    assert len(value.records) == 1 and value.budget_after["charged"]["model_calls"] == 5


def test_tampered_original_evidence_is_not_rendered_as_trusted_handoff(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    original = DAGScheduler.execute
    def corrupt(self, *pos, **kw):
        run = original(self, *pos, **kw)
        reference = run.results[0].evidence_refs[0].artifact
        args["io"].resolve(reference).write_text("{}")
        return run
    monkeypatch.setattr(DAGScheduler, "execute", corrupt)
    value = execute_consultation(**args)
    assert not value.safe_to_continue and value.stop_reason == "source_changed"
    assert not value.context


def test_second_episode_cannot_reset_one_job_limit_with_a_new_revision_name(tmp_path):
    args = fixture(tmp_path)
    first = execute_consultation(**args)
    assert first.safe_to_continue
    second = execute_consultation(**args, episode_id="different-name")
    assert second.stop_reason == "consultation_no_evidence" and second.run is None
    assert second.budget_after == first.budget_after


def test_trusted_worker_command_is_used_without_adding_agent_capabilities(tmp_path):
    args = fixture(tmp_path)
    wrapper = tmp_path / "observed-worker.py"
    marker = tmp_path / "worker-was-observed"
    wrapper.write_text("from pathlib import Path\nPath("+repr(str(marker))+").write_text('started')\n"
                       "from researchcoder.runtime.worker import main\nraise SystemExit(main())\n")
    value = execute_consultation(**args, worker_command=(sys.executable, str(wrapper)))
    assert value.safe_to_continue and marker.read_text() == "started", value.to_dict()
    assert value.records[0]["worker_pid"]


def test_collected_source_failure_cancels_real_running_peer(tmp_path):
    args = fixture(tmp_path, chained=True)
    marker = tmp_path / "peer-started"
    wrapper = tmp_path / "fault-worker.py"
    # Actual supervised processes: Code runs the original model/tools, then a
    # controlled collection-time fault changes only its completion state. Log
    # waits in its real process, which must be terminated and reaped promptly.
    wrapper.write_text("\n".join((
        "import json, sys, time",
        "from pathlib import Path",
        "from dataclasses import replace",
        "import researchcoder.runtime.worker as worker",
        "request = json.loads(Path(sys.argv[1]).read_text())",
        "marker = Path("+repr(str(marker))+")",
        "if request['spec']['task_id'].endswith('expert-log'):",
        "    marker.write_text('started')",
        "    time.sleep(30)",
        "else:",
        "    started = time.monotonic()",
        "    while not marker.exists() and time.monotonic()-started < 10: time.sleep(.02)",
        "    original = worker.execute_request",
        "    def fault(request):",
        "        response = original(request)",
        "        return replace(response, result=replace(response.result, status='incomplete',",
        "            failure_kind='tool_error', workflow_completed=False))",
        "    worker.execute_request = fault",
        "raise SystemExit(worker.main())",
    ))+"\n")
    started = time.monotonic()
    value = execute_consultation(**args, worker_command=(sys.executable, str(wrapper)))
    assert marker.exists() and time.monotonic()-started < 20, value.to_dict()
    assert value.stop_reason == "source_changed" and not value.safe_to_continue
    assert len(value.records) == 2
    assert next(r for r in value.run.results if r.task_id.endswith("expert-code")).failure_kind == "tool_error"
    assert next(r for r in value.run.results if r.task_id.endswith("expert-log")).status == "cancelled"
    for record in value.records:
        with pytest.raises(ProcessLookupError):
            os.kill(record["worker_pid"], 0)


def test_untrusted_decision_public_identity_and_live_model_switch_are_rejected(tmp_path):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match="matching affirmative"):
        execute_consultation(**dict(args, decision=replace(args["decision"], policy_sha256="a"*64)))
    with pytest.raises(ValueError, match="public evidence"):
        execute_consultation(**dict(args, public_log="ERROR different public input"))
    with pytest.raises(ValueError, match="ScriptedLLM"):
        execute_consultation(**dict(args, model=ModelSpec(provider="openai", model="no-network")),
                             role_models={"code_analysis": expert_model()})
    assert not (tmp_path/"snapshot-experts-1").exists()
