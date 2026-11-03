"""Provider failures preserve paid-attempt accounting and legal partial repairs."""

from dataclasses import replace
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from openai import APIConnectionError, APITimeoutError

from corecoder.llm import LLM
from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ExecutionPolicy, TaskInput, VerificationSpec, WorkspaceRef
from researchcoder.orchestration.baseline import run_single_agent
from researchcoder.runtime.budget import BudgetedModel, BudgetLedger, RuntimeHooks
from researchcoder.runtime.protocol import CodingTaskSpec, ModelSpec, RuntimeLimits, WorkerRequest
from researchcoder.runtime.tracing import TraceSink


def setup_request(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    (root / "training.py").write_text("value = 1\n")
    io = ArtifactIO(tmp_path / "artifacts")
    log = io.put_bytes("job", b"WARNING: expected value=2\n", media_type="text/plain")
    command = VerificationSpec(argv=(sys.executable, "-B", "-c", "import training; assert training.value == 2"))
    spec = CodingTaskSpec(job_id="job", task_id="coding", objective="Correct the value and preserve verification.",
        workspace=WorkspaceRef(workspace_id="p", root=str(root), source_manifest_sha256=workspace_revision(root)),
        inputs=(TaskInput(role="raw_log", reference=log),),
        policy=ExecutionPolicy(allowed_tool_names=("read_file", "edit_file"), writable_paths=("training.py",), allow_verification=True),
        verification=command)
    request = WorkerRequest(spec=spec, attempt_id="attempt", artifact_root=str(io.root),
        trace_path=str(tmp_path / "trace.jsonl"), parent_event_id="parent", permitted_verifications=(command,))
    return root, io, request


def call_stream(name, arguments, prompt, completion):
    call = SimpleNamespace(index=0, id=name, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return iter([
        SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[call]))]),
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion), choices=[]),
    ])


@pytest.mark.parametrize("error_class", [APIConnectionError, APITimeoutError])
def test_broken_final_stream_keeps_patch_verification_and_unknown_usage(tmp_path, monkeypatch, error_class):
    root, io, request = setup_request(tmp_path)
    monkeypatch.setattr("socket.socket.connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No network")))
    model = LLM("offline-model", "offline-placeholder-only")

    def broken_stream():
        yield SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content="partial response", tool_calls=None))])
        raise error_class(request=None)

    model.client.chat.completions.create = Mock(side_effect=[
        call_stream("read_file", {"file_path": "training.py"}, 8, 3),
        call_stream("edit_file", {"file_path": "training.py", "old_string": "value = 1", "new_string": "value = 2"}, 9, 4),
        broken_stream(),
    ])
    trace = TraceSink("job", "coding", "attempt")
    ledger = BudgetLedger(RuntimeLimits(), trace)
    hooks = RuntimeHooks(ledger, trace, request.spec.policy.allowed_tool_names)
    result = run_single_agent(request.spec, request, io, lambda: BudgetedModel(model, ledger, trace), hooks, lambda: None)
    assert result.status == "incomplete" and result.failure_kind == "model_error"
    assert not result.workflow_completed
    assert result.verification.status == "passed" and (root / "training.py").read_text() == "value = 2\n"
    raw = io.read_json(result.outputs[0])
    assert raw["model_error_type"] == error_class.__name__
    assert [e["status"] for e in raw["events"]] == ["ok", "ok"]
    assert raw["file_changes"][0]["before_sha256"] != raw["file_changes"][0]["after_sha256"]
    assert ledger.model_calls == 3 and ledger.unknown_requests == 1 and not ledger.pending
    assert ledger.charged_tokens > 24 and ledger.usage("live").prompt_tokens is None
    assert model.client.chat.completions.create.call_count == 3


@pytest.mark.parametrize("error_class", [APIConnectionError, APITimeoutError])
def test_worker_converts_provider_factory_failure_without_losing_accounting(tmp_path, monkeypatch, error_class):
    import researchcoder.runtime.worker as worker
    _, io, request = setup_request(tmp_path)
    request = replace(request, model=ModelSpec(provider="openai", model="deepseek-flash",
        base_url="https://api.deepseek.com", credential_env="DEEPSEEK_API_KEY"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-placeholder-only")

    def unavailable(**kwargs):
        raise error_class(request=None)

    monkeypatch.setattr(worker, "LLM", unavailable)
    response = worker.execute_request(request)
    assert response.result.status == "incomplete" and response.result.failure_kind == "model_error"
    assert not response.result.workflow_completed
    budget = json.loads(response.budget_json)
    assert budget["model_calls"] == 0 and budget["pending"] == {}
    assert "offline-placeholder-only" not in (tmp_path / "trace.jsonl").read_text()
