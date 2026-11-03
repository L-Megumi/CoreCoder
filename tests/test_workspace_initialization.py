"""Workspace metadata must agree with the existing guarded tool execution scope."""

from pathlib import Path
import pytest

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.tools.bash import _local
from corecoder.workflow import load_handoff, run_repair, workspace_revision
from corecoder.workflow_demo import prepare_demo
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ExecutionPolicy, TaskInput, VerificationSpec, WorkspaceRef
from researchcoder.orchestration.baseline import run_single_agent
from researchcoder.runtime.budget import BudgetLedger, RuntimeHooks
from researchcoder.runtime.protocol import CodingTaskSpec, RuntimeLimits, WorkerRequest
from researchcoder.runtime.tracing import TraceSink


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    _, project, report, command, _ = prepare_demo(tmp_path / "case")
    worker = tmp_path / "worker"
    worker.mkdir()
    outside = worker / "outside.py"
    outside.write_text("outside = 1\n")
    monkeypatch.chdir(worker)
    monkeypatch.setattr(_local, "cwd", str(worker), raising=False)
    return project, report, command, worker, outside


def execute(mode, prepared, tmp_path, model):
    project, report, command, _, _ = prepared
    if mode == "repair":
        return run_repair(
            load_handoff(report, project),
            "Fix ratio and keep the original check unchanged.",
            model,
            allowed_tool_names=("read_file", "edit_file"),
            verify_command=command,
        )
    io = ArtifactIO(tmp_path / "artifacts")
    log = io.put_bytes("job", b"WARNING: ratio(5, 2) should be 2.5\n", media_type="text/plain")
    verification = VerificationSpec(argv=tuple(command))
    spec = CodingTaskSpec(
        job_id="job", task_id="coding", objective="Fix ratio and preserve the original check.",
        workspace=WorkspaceRef(workspace_id="candidate", root=str(project), source_manifest_sha256=workspace_revision(project)),
        inputs=(TaskInput(role="raw_log", reference=log),),
        policy=ExecutionPolicy(allowed_tool_names=("read_file", "edit_file"), writable_paths=("ratio.py",), allow_verification=True),
        verification=verification,
    )
    request = WorkerRequest(spec=spec, attempt_id="attempt", artifact_root=str(io.root),
                            trace_path=str(tmp_path / "trace"), parent_event_id="parent",
                            permitted_verifications=(verification,))
    trace = TraceSink("job", "coding", "attempt")
    hooks = RuntimeHooks(BudgetLedger(RuntimeLimits(), trace), trace, spec.policy.allowed_tool_names)
    result = run_single_agent(spec, request, io, lambda: model, hooks, lambda: None)
    return {"task_result": result, "workflow": io.read_json(result.outputs[0])}


@pytest.mark.parametrize("mode", ["single", "repair"])
@pytest.mark.parametrize("path_mode", ["relative", "absolute"])
def test_declared_workspace_supports_authorized_paths_and_preserves_guard(mode, path_mode, prepared, tmp_path):
    project, _, _, worker, outside = prepared
    target = "ratio.py" if path_mode == "relative" else str(project / "ratio.py")
    prompts = []

    class Model(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            prompts.append(messages[0]["content"])
            assert Path.cwd() == project and _local.cwd == str(project)
            return super().chat(messages, tools, on_token)

    model = Model([
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": target})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": target, "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(tool_calls=[ToolCall("outside-read", "read_file", {"file_path": str(outside)})]),
        LLMResponse(tool_calls=[ToolCall("outside-edit", "edit_file", {"file_path": str(outside), "old_string": "outside = 1", "new_string": "outside = 2"})]),
        LLMResponse(content="Updated ratio; the controller owns verification."),
    ])
    result = execute(mode, prepared, tmp_path, model)
    assert prompts and all("Working directory: " + str(project) in p for p in prompts)
    assert all("Working directory: " + str(worker) not in p for p in prompts)
    assert Path.cwd() == worker and _local.cwd == str(worker)
    assert outside.read_text() == "outside = 1\n"
    assert (project / "ratio.py").read_text() == "def ratio(a, b):\n    return a / b\n"
    if mode == "single":
        assert result["task_result"].status == "succeeded"
        events = result["workflow"]["events"]
    else:
        assert result["status"] == "verified"
        events = result["tool_events"]
    assert [e["status"] for e in events] == ["ok", "ok", "error", "error"]
    assert events[-2]["result"] == "Error: File target is outside the repair project"
    expected_write_error = "Error: Unauthorized baseline write" if mode == "single" else "Error: File target is outside the repair project"
    assert events[-1]["result"] == expected_write_error


@pytest.mark.parametrize("mode", ["single", "repair"])
def test_constructor_failure_restores_process_and_shell_workspace(mode, prepared, tmp_path, monkeypatch):
    project, _, _, worker, _ = prepared
    constructed_in = []

    def broken(*args, **kwargs):
        constructed_in.append(Path.cwd())
        raise ValueError("constructor marker")

    monkeypatch.setattr("researchcoder.orchestration.baseline.RepairAgent" if mode == "single" else "corecoder.workflow.RepairAgent", broken)
    with pytest.raises(ValueError, match="constructor marker"):
        execute(mode, prepared, tmp_path, ScriptedLLM([]))
    assert constructed_in == [project]
    assert Path.cwd() == worker and _local.cwd == str(worker)


@pytest.mark.parametrize("mode", ["single", "repair"])
def test_model_failure_preserves_existing_error_semantics_and_restores_workspace(mode, prepared, tmp_path):
    project, _, _, worker, _ = prepared
    prompts = []

    class BrokenModel(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            prompts.append(messages[0]["content"])
            raise RuntimeError("chat marker")

    if mode == "single":
        with pytest.raises(RuntimeError, match="chat marker"):
            execute(mode, prepared, tmp_path, BrokenModel([]))
    else:
        result = execute(mode, prepared, tmp_path, BrokenModel([]))
        assert result["status"] == "incomplete" and "chat marker" in result["failure_reason"]
        assert result["verification"]["status"] == "not_run"
    assert prompts and "Working directory: " + str(project) in prompts[0]
    assert Path.cwd() == worker and _local.cwd == str(worker)
