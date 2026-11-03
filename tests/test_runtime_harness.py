from dataclasses import replace
import json
import sys
import threading
import pytest

from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import TaskSpec, TaskInput, WorkspaceRef, ExecutionPolicy, VerificationSpec
from researchcoder.runtime.harness import ResearchHarness, ExecutionConfig
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, ScriptedCall, RuntimeLimits


def case(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "training.py").write_text("loss = 1\n")
    store = tmp_path / "artifacts"
    io = ArtifactIO(store)
    log = io.put_bytes("job", b"WARNING: missing backbone gradients\n", media_type="text/plain")
    command = VerificationSpec(argv=(sys.executable, "-c", "print('safe verification')"))
    workspace = WorkspaceRef(workspace_id="p", root=str(root), source_manifest_sha256=workspace_revision(root))
    policy = ExecutionPolicy(allow_verification=True)
    spec = TaskSpec(
        job_id="job",
        task_id="verify",
        kind="verify",
        objective="Verify",
        workspace=workspace,
        policy=policy,
        verification=command,
    )
    config = ExecutionConfig(
        artifact_root=str(store), allowed_workspaces=(str(root),), max_policy=policy, permitted_verifications=(command,)
    )
    return spec, config, io, log


def test_harness_uses_real_adapter_archives_receipt_and_closes_states(tmp_path):
    spec, config, io, _ = case(tmp_path)
    harness = ResearchHarness()
    result = harness.execute(spec, config)
    assert result.status == "succeeded" and result.verification.status == "passed"
    assert harness.repository.task("job", "verify").status == "succeeded"
    for artifact in result.outputs:
        io.read_bytes(artifact)
    record = harness.records[result.attempt_id]
    assert record["worker_pid"] and record["worker_exit_code"] == 0
    from researchcoder.contracts import ArtifactRef

    trace = io.read_json(ArtifactRef.from_dict(record["trace"]))["events"]
    assert trace[-1]["event_type"] == "task.execution.end"
    assert {"job.created", "worker.lifecycle.start", "worker.ready", "worker.result", "task.execution.end"} <= {
        e["event_type"] for e in trace
    }
    with pytest.raises(ValueError):
        harness.execute(spec, config)
    second = harness.execute(replace(spec, task_id="second"), config)
    assert second.status == "succeeded" and result.attempt_id != second.attempt_id
    assert record["worker_pid"] != harness.records[second.attempt_id]["worker_pid"]
    harness.close_job("job")


@pytest.mark.parametrize(
    "change", [{"allowed_workspaces": ()}, {"permitted_verifications": ()}, {"max_policy": ExecutionPolicy()}]
)
def test_controller_policy_rejects_before_worker(tmp_path, change):
    spec, config, _, _ = case(tmp_path)
    harness = ResearchHarness()
    result = harness.execute(spec, replace(config, **change))
    assert result.status == "incomplete" and result.failure_kind == "input_invalid"
    assert harness.records[result.attempt_id]["worker_pid"] is None


@pytest.mark.parametrize(
    "limits",
    [
        RuntimeLimits(max_model_calls=1),
        RuntimeLimits(max_tool_calls=1),
        RuntimeLimits(max_rounds=1),
        RuntimeLimits(max_total_tokens=1),
    ],
)
def test_actual_diagnostic_loop_stops_on_each_budget(tmp_path, limits):
    spec, config, io, log = case(tmp_path)
    spec = replace(
        spec, task_id="diagnose", kind="diagnose", verification=None, inputs=(TaskInput(role="raw_log", reference=log),)
    )
    model = ModelSpec(
        turns=(
            ScriptedTurn(calls=(ScriptedCall(call_id="e", name="extract_log_evidence"),)),
            ScriptedTurn(calls=(ScriptedCall(call_id="r", name="read_project_file", arguments_json='{"path":"training.py"}'),)),
            ScriptedTurn(content="No final report"),
        )
    )
    harness = ResearchHarness()
    result = harness.execute(spec, replace(config, model=model, limits=limits))
    assert result.status == "incomplete" and result.failure_kind == "budget_exhausted"
    record = harness.records[result.attempt_id]
    assert record["budget"]["stop_kind"] == "budget_exhausted"
    assert result.outputs and result.usage.prompt_tokens is None
    assert record["budget"]["model_calls"] <= limits.max_model_calls
    assert record["budget"]["tool_calls"] <= limits.max_tool_calls


def test_model_cannot_expand_diagnostic_permissions(tmp_path):
    spec, config, io, log = case(tmp_path)
    spec = replace(
        spec, task_id="diagnose", kind="diagnose", verification=None, inputs=(TaskInput(role="raw_log", reference=log),)
    )
    model = ModelSpec(turns=(ScriptedTurn(calls=(ScriptedCall(call_id="e", name="extract_log_evidence"),)),))
    harness = ResearchHarness()
    result = harness.execute(spec, replace(config, model=model, diagnostic_tools=()))
    assert result.status == "incomplete" and result.usage.tool_calls == 0


@pytest.mark.parametrize("mode,expected", [("timeout", "timeout"), ("cancel", "interrupted"), ("crash", "tool_error")])
def test_runtime_terminal_states_for_supervisor_failures(tmp_path, mode, expected):
    spec, config, _, _ = case(tmp_path)
    cancel = threading.Event()
    if mode == "cancel":
        cancel.set()
    program = "import os;os._exit(7)" if mode == "crash" else "import time;time.sleep(30)"
    harness = ResearchHarness()
    result = harness.execute(
        spec,
        replace(
            config,
            cancel_event=cancel,
            limits=RuntimeLimits(max_runtime_seconds=0.2),
            worker_command=(sys.executable, "-c", program),
        ),
    )
    assert result.failure_kind == expected
    assert result.status == ("cancelled" if mode == "cancel" else "failed")
    assert harness.repository.task("job", "verify").status == result.status
    assert harness.records[result.attempt_id]["budget"]["usage"] == "unknown"


def test_partial_repair_after_model_budget_stop_still_has_independent_verification(tmp_path):
    from corecoder.workflow_demo import prepare_demo

    _, root, report, command, _ = prepare_demo(tmp_path / "case")
    io = ArtifactIO(tmp_path / "artifacts")
    original = io.put_bytes("job", report.read_bytes())
    workspace = WorkspaceRef(workspace_id="p", root=str(root), source_manifest_sha256=workspace_revision(root))
    verify = VerificationSpec(argv=tuple(command))
    policy = ExecutionPolicy(writable_paths=("ratio.py",), allow_verification=True)
    spec = TaskSpec(
        job_id="job",
        task_id="repair",
        kind="repair",
        objective="Fix ratio",
        workspace=workspace,
        inputs=(TaskInput(role="diagnosis_run", reference=original),),
        policy=policy,
        verification=verify,
    )
    model = ModelSpec(
        turns=(
            ScriptedTurn(calls=(ScriptedCall(call_id="r", name="read_file", arguments_json='{"file_path":"ratio.py"}'),)),
            ScriptedTurn(
                calls=(
                    ScriptedCall(
                        call_id="e",
                        name="edit_file",
                        arguments_json=json.dumps({"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"}),
                    ),
                )
            ),
        )
    )
    harness = ResearchHarness()
    result = harness.execute(
        spec,
        ExecutionConfig(
            artifact_root=str(io.root),
            allowed_workspaces=(str(root),),
            max_policy=policy,
            permitted_verifications=(verify,),
            model=model,
            limits=RuntimeLimits(max_model_calls=2),
        ),
    )
    assert result.status == "incomplete" and result.failure_kind == "budget_exhausted"
    assert result.verification.status == "passed" and result.verification.trigger == "partial_file_changes"
    assert result.usage.tool_calls == 3
    assert not result.workflow_completed


def test_real_autograd_runs_through_two_isolated_workers(tmp_path):
    from examples.harness_autograd_demo import run_demo

    result = run_demo(tmp_path / "harness-demo")
    assert result["initial_status"] == "failed" and result["provider_requests"] == 0
    assert result["diagnosis"]["status"] == "succeeded" and result["repair"]["verification"]["status"] == "passed"
    assert result["closed_job"] == "closed"
    assert result["records"][0]["worker_pid"] != result["records"][1]["worker_pid"]
    assert all(record["budget"]["unknown_requests"] > 0 for record in result["records"])


def test_controller_archive_error_still_finishes_state_and_preserves_legal_verification(tmp_path, monkeypatch):
    spec, config, _, _ = case(tmp_path)

    def unavailable(*args, **kwargs):
        raise OSError("PRIVATE_DISK_ERROR_CANARY")

    monkeypatch.setattr(ArtifactIO, "put_json", unavailable)
    harness = ResearchHarness()
    result = harness.execute(spec, config)
    assert result.status == "incomplete" and result.failure_kind == "tool_error"
    assert result.verification.status == "passed"
    assert harness.repository.task("job", "verify").status == "incomplete"
    assert "PRIVATE_DISK_ERROR_CANARY" not in json.dumps(harness.records[result.attempt_id])


def test_inner_model_exception_retains_failed_request_accounting_and_result(tmp_path):
    spec, config, io, log = case(tmp_path)
    spec = replace(
        spec, task_id="diagnose", kind="diagnose", verification=None, inputs=(TaskInput(role="raw_log", reference=log),)
    )
    harness = ResearchHarness()
    result = harness.execute(spec, config)
    assert result.status == "incomplete" and result.failure_kind == "model_error"
    assert result.usage.llm_calls == 1
    assert harness.records[result.attempt_id]["budget"]["unknown_requests"] == 1
    assert io.read_json(result.outputs[0])["status"] == "incomplete"


def test_report_correction_is_a_counted_request_not_a_new_agent_round(tmp_path):
    spec, config, io, log = case(tmp_path)
    spec = replace(
        spec, task_id="diagnose", kind="diagnose", verification=None, inputs=(TaskInput(role="raw_log", reference=log),)
    )
    report = {
        "summary": "Scripted report correction",
        "historical_failure": {"description": "A warning was logged", "evidence_ids": ["$E0"]},
        "current_state": {"assessment": "training.py was actually read", "observation_ids": ["$O0"]},
        "hypotheses": [
            {
                "explanation": "Further investigation is required",
                "evidence_ids": ["$E0"],
                "observation_ids": ["$O0"],
                "verification": "Check the independent oracle",
            }
        ],
        "recommended_actions": ["Investigate gradients"],
        "limitations": ["Scripted structural test only"],
    }
    model = ModelSpec(
        turns=(
            ScriptedTurn(calls=(ScriptedCall(call_id="e", name="extract_log_evidence"),)),
            ScriptedTurn(calls=(ScriptedCall(call_id="r", name="read_project_file", arguments_json='{"path":"training.py"}'),)),
            ScriptedTurn(content="invalid report"),
            ScriptedTurn(content=json.dumps(report)),
        )
    )
    harness = ResearchHarness()
    result = harness.execute(spec, replace(config, model=model))
    assert result.status == "succeeded"
    assert result.usage.llm_calls == 4
    assert harness.records[result.attempt_id]["budget"]["rounds"] == 3
    assert io.read_json(result.outputs[0])["metrics"]["report_repairs"] == 1
