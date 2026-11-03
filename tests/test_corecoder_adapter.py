import sys
from dataclasses import replace

import pytest
from openai import APIError

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.workflow import workspace_revision
from corecoder.workflow_demo import DiagnosisDemoLLM, prepare_demo
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.adapters.corecoder import AdapterContext, CoreCoderAdapter
from researchcoder.contracts import (
    DiagnosticOptions,
    ExecutionPolicy,
    TaskInput,
    TaskResult,
    TaskSpec,
    VerificationSpec,
    WorkspaceRef,
)


@pytest.fixture
def case(tmp_path):
    directory, root, report, command, _ = prepare_demo(tmp_path / "case")
    io = ArtifactIO(tmp_path / "artifacts")
    log = io.put_bytes("job", (directory / "failure.log").read_bytes(), media_type="text/plain")
    diagnosis = io.put_bytes("job", report.read_bytes())
    workspace = WorkspaceRef(workspace_id="project", root=str(root), source_manifest_sha256=workspace_revision(root))
    return io, root, workspace, log, diagnosis, VerificationSpec(argv=tuple(command))


def spec_for(case, kind="repair", verify=True):
    _io, _root, workspace, log, diagnosis, command = case
    return TaskSpec(
        job_id="job",
        task_id=kind,
        kind=kind,
        objective="Inspect or repair ratio semantics",
        workspace=workspace,
        inputs=(TaskInput(role="raw_log", reference=log),)
        if kind == "diagnose"
        else (TaskInput(role="diagnosis_run", reference=diagnosis),)
        if kind == "repair"
        else (),
        diagnostic_options=DiagnosticOptions(source_hints=("ratio.py",)),
        policy=ExecutionPolicy(writable_paths=("ratio.py",), allow_verification=verify),
        verification=command if verify else None,
    )


def repaired_model():
    return ScriptedLLM(
        [
            LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "ratio.py"})]),
            LLMResponse(
                tool_calls=[
                    ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})
                ]
            ),
            LLMResponse(content="Edit complete; explicit command will check it."),
        ]
    )


def test_real_diagnose_preserves_complete_raw_events_and_references(case):
    io, _root, _workspace, _, _, _ = case
    spec = spec_for(case, "diagnose", False)
    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a1", artifacts=io, model_factory=DiagnosisDemoLLM))
    assert result.status == "succeeded" and result.kind == "diagnose"
    raw = io.read_json(result.outputs[0])
    assert raw["validation"]["semantic_correctness"] == "not_automatically_verified"
    for key in ["evidence", "observations", "tool_events", "context_events", "selection_events", "request_metrics"]:
        assert key in raw
    assert result.evidence_refs
    assert all(io.read_record(r)["id"] == r.origin_id for r in result.evidence_refs)
    assert result.usage.prompt_tokens is None and not result.usage.usage_complete
    assert TaskResult.from_json(result.to_json()) == result


def test_real_repair_uses_handoff_guard_and_common_verifier(case):
    io, root, _, _, _, command = case
    spec = spec_for(case)
    result = CoreCoderAdapter().execute(
        spec, AdapterContext(attempt_id="a1", artifacts=io, model_factory=repaired_model, permitted_verifications=(command,))
    )
    assert result.status == "succeeded" and result.verification.status == "passed"
    raw = io.read_json(result.outputs[0])
    assert raw["status"] == "verified"
    assert raw["file_changes"] == io.read_json(result.file_changes_ref)
    assert "a / b" in (root / "ratio.py").read_text()
    assert result.verification.checked_workspace_revision == workspace_revision(root)


def test_verification_requires_trusted_context_allowance_before_edit(case):
    io, root, *_ = case

    def forbidden():
        pytest.fail("Model must not start")

    result = CoreCoderAdapter().execute(spec_for(case), AdapterContext(attempt_id="a1", artifacts=io, model_factory=forbidden))
    assert result.status == "incomplete" and result.failure_kind == "input_invalid"
    assert "a // b" in (root / "ratio.py").read_text()


def test_changed_workspace_or_artifact_blocks_before_model(case):
    io, root, *_ = case
    spec = spec_for(case, verify=False)
    (root / "ratio.py").write_text("changed\n")
    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a1", artifacts=io))
    assert result.failure_kind == "reference_stale"


def test_input_digest_rechecked(case):
    io, _root, _, _, diagnosis, _ = case
    spec = spec_for(case, verify=False)
    io.resolve(diagnosis).write_text("{}")
    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a1", artifacts=io))
    assert result.failure_kind == "reference_stale"


def test_effective_edit_then_model_failure_and_passing_verification_is_incomplete(case):
    io, _root, _, _, _, command = case

    class FailingModel(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if not self._turns:
                raise APIError("DO_NOT_PERSIST_PROVIDER_SECRET", request=None, body={})
            return super().chat(*args, **kwargs)

    def model():
        return FailingModel(repaired_model()._turns[:-1])

    result = CoreCoderAdapter().execute(
        spec_for(case), AdapterContext(attempt_id="a1", artifacts=io, model_factory=model, permitted_verifications=(command,))
    )
    assert result.status == "incomplete" and result.failure_kind == "model_error"
    assert not result.workflow_completed and result.verification.status == "passed"
    assert result.verification.trigger == "partial_file_changes"
    assert "DO_NOT_PERSIST_PROVIDER_SECRET" not in io.read_bytes(result.outputs[0]).decode()


def test_unverified_completion_is_explicit(case):
    io, *_ = case
    result = CoreCoderAdapter().execute(
        spec_for(case, verify=False),
        AdapterContext(
            attempt_id="a1", artifacts=io, model_factory=lambda: ScriptedLLM([LLMResponse(content="No command was run.")])
        ),
    )
    assert result.status == "succeeded" and result.verification.status == "not_run"


def test_forbidden_file_edit_is_blocked_by_adapter_and_original_guard(case):
    io, root, *_ = case
    before = (root / "check_ratio.py").read_bytes()

    def model():
        return ScriptedLLM(
            [
                LLMResponse(tool_calls=[ToolCall("bad", "write_file", {"file_path": "check_ratio.py", "content": "pass\n"})]),
                LLMResponse(content="Unable to edit the protected test."),
            ]
        )

    result = CoreCoderAdapter().execute(
        spec_for(case, verify=False), AdapterContext(attempt_id="a1", artifacts=io, model_factory=model)
    )
    assert (root / "check_ratio.py").read_bytes() == before
    assert io.read_json(result.outputs[0])["tool_events"][0]["status"] == "error"


@pytest.mark.parametrize("code,status,reason", [(0, "succeeded", None), (5, "failed", "verification_failed")])
def test_standalone_verify_uses_no_model_and_preserves_state(case, code, status, reason):
    io, *_ = case
    command = VerificationSpec(argv=(sys.executable, "-c", f"import sys;sys.exit({code})"))
    spec = replace(spec_for(case, "verify"), verification=command)
    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a1", artifacts=io, permitted_verifications=(command,)))
    assert result.status == status and result.failure_kind == reason and result.verification.exit_code == code


def test_interrupt_and_factory_error_are_not_retried_or_leaked(case):
    io, *_ = case
    spec = spec_for(case, verify=False)

    def interrupted():
        raise KeyboardInterrupt

    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a1", artifacts=io, model_factory=interrupted))
    assert result.status == "cancelled" and result.failure_kind == "interrupted"

    def error():
        raise RuntimeError("secret credential")

    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a2", artifacts=io, model_factory=error))
    assert result.status == "incomplete"
    assert "secret credential" not in io.read_bytes(result.outputs[0]).decode()


def test_delegate_arguments_and_factory_are_attempt_local(case, monkeypatch):
    io, *_ = case
    spec = spec_for(case, "diagnose", False)
    calls = []
    created = []

    def factory():
        value = object()
        created.append(value)
        return value

    def diagnose(log, root, model, **kwargs):
        calls.append((model, kwargs))
        return {
            "status": "incomplete",
            "evidence": [],
            "observations": [],
            "metrics": {},
            "tool_events": [],
            "context_events": [],
            "selection_events": [],
        }

    monkeypatch.setattr("researchcoder.adapters.corecoder.run_diagnosis", diagnose)
    for attempt in ["a1", "a2"]:
        result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id=attempt, artifacts=io, model_factory=factory))
        assert result.status == "incomplete"
    assert len(calls) == 2 and created[0] is not created[1]
    assert calls[0][1]["tool_selection"] == "guided" and calls[0][1]["context_mode"] == "evidence"
    assert calls[0][1]["max_tool_calls"] == spec.budget.max_tool_calls


@pytest.mark.parametrize(
    "command,expected",
    [
        (VerificationSpec(argv=("/missing-phase1-adapter-executable",)), "launch_error"),
        (VerificationSpec(argv=(sys.executable, "-c", "import time;time.sleep(5)"), timeout_seconds=1), "timeout"),
    ],
)
def test_standalone_error_categories(case, command, expected):
    io, *_ = case
    spec = replace(spec_for(case, "verify"), verification=command)
    result = CoreCoderAdapter().execute(spec, AdapterContext(attempt_id="a1", artifacts=io, permitted_verifications=(command,)))
    assert result.status == "failed" and result.failure_kind == expected
    assert result.verification.status == expected and result.verification.exit_code is None


def test_verification_guard_is_checked_again_after_an_edit(case):
    io, root, _, _, _, command = case
    calls = []

    def guard(argv, project):
        calls.append((argv, (root / "ratio.py").read_text()))

    result = CoreCoderAdapter().execute(
        spec_for(case),
        AdapterContext(
            attempt_id="a1",
            artifacts=io,
            model_factory=repaired_model,
            permitted_verifications=(command,),
            verification_guard=guard,
        ),
    )
    assert result.status == "succeeded" and len(calls) == 2
    assert "a // b" in calls[0][1] and "a / b" in calls[1][1]
