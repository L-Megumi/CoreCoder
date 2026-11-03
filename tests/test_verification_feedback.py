"""Failure feedback provenance, fresh read-only diagnosis and bounded retry behavior."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
from openai import APIError

from corecoder.diagnose import run_diagnosis
from corecoder.feedback_demo import run_demo
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.routed_session import RoutedSession
from corecoder.verification_feedback import create_feedback, validate_context, verify_feedback
from corecoder.workflow import _bounded_output, load_handoff, run_repair
from corecoder.workflow_demo import DiagnosisDemoLLM, prepare_demo


@pytest.fixture
def case(tmp_path):
    return prepare_demo(tmp_path / "case")


def quiet():
    return ScriptedLLM([LLMResponse(content="Model claims it is fixed.")])


def failed_artifact(case, **kwargs):
    workspace, root, report, command, _ = case
    result = run_repair(load_handoff(report, root), "Inspect without editing", quiet(), verify_command=command, **kwargs)
    path = workspace / "failed-repair.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return result, path


def test_failed_command_becomes_new_evidence_with_original_ancestry(case):
    workspace, root, report, _, _ = case
    result, repair = failed_artifact(case)
    feedback = create_feedback(repair, workspace / "feedback", root)
    metadata = feedback["metadata"]
    assert metadata["repair_sha256"] == hashlib.sha256(repair.read_bytes()).hexdigest()
    assert metadata["previous_report"] == str(report)
    assert metadata["previous_report_sha256"] == result["handoff"]["report_sha256"]
    assert metadata["verification"]["exit_code"] != 0
    assert "AssertionError" in Path(feedback["log_file"]).read_text()
    assert metadata["previous_hypotheses"] and metadata["source_hints"]
    verify_feedback(feedback)


def test_failed_verification_automatically_diagnoses_but_does_not_edit_again(case):
    _, root, report, command, _ = case
    session = RoutedSession(root, report_file=report, verify_command=command,
                            coding_factory=quiet, diagnosis_factory=DiagnosisDemoLLM, execution_mode="offline_scripted")
    before = (root / "ratio.py").read_bytes()
    response = session.handle("修复比例计算")
    assert response["status"] == "verification_failed"
    record = session.feedback_records[0]
    assert record["status"] == "completed", record
    assert session.feedback_attempts == 1 and session.report_file is not None
    diagnosis = json.loads(Path(record["diagnosis_artifact"]).read_text())
    assert diagnosis["feedback_context"]["feedback_id"] == record["feedback_id"]
    assert diagnosis["feedback_context"]["previous_report"] == str(report)
    assert diagnosis["log_file"] != json.loads(report.read_text())["log_file"]
    assert all(item["source"] == diagnosis["log_file"] for item in diagnosis["evidence"])
    assert all(event["tool"] in {"extract_log_evidence", "locate_source", "read_project_file", "inspect_paths", "search_project_source"}
               for event in diagnosis["tool_events"])
    assert (root / "ratio.py").read_bytes() == before
    assert len(session.turns) == 1  # No synthetic repair task was scheduled.


def test_fresh_report_can_be_used_for_a_new_explicit_successful_repair(case):
    _, root, report, command, _ = case
    models = iter([quiet(), ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(content="Changed division; awaiting command verification."),
    ])])
    session = RoutedSession(root, report_file=report, verify_command=command,
                            coding_factory=lambda: next(models), diagnosis_factory=DiagnosisDemoLLM, execution_mode="offline_scripted")
    assert session.handle("修复比例计算")["status"] == "verification_failed"
    fresh_report = session.report_file
    assert fresh_report != report
    response = session.handle("修复新诊断发现的问题")
    assert response["status"] == "verified"
    repair = json.loads(Path(response["artifact"]).read_text())
    assert repair["handoff"]["feedback_context"] is not None
    assert repair["handoff"]["report_path"] == str(fresh_report)
    assert session.feedback_attempts == 1
    assert session.report_file is None and session.active_feedback is None


def test_feedback_uses_current_file_hashes_and_records_failed_edit(case):
    _, root, report, command, _ = case
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a // b + 1"})]),
        LLMResponse(content="Done."),
    ])
    session = RoutedSession(root, report_file=report, verify_command=command,
                            coding_factory=lambda: model, diagnosis_factory=DiagnosisDemoLLM)
    response = session.handle("修复比例计算")
    assert response["status"] == "verification_failed"
    metadata = json.loads(Path(session.feedback_records[0]["metadata_file"]).read_text())
    assert metadata["file_changes"][0]["before_sha256"] != metadata["file_changes"][0]["after_sha256"]
    current_hash = hashlib.sha256((root / "ratio.py").read_bytes()).hexdigest()
    handoff = load_handoff(session.report_file, root)
    assert handoff["file_snapshots"][str(root / "ratio.py")] == current_hash


def test_feedback_budget_stops_model_calls_while_preserving_all_failure_outputs(case):
    _, root, report, command, _ = case
    calls = []
    def factory():
        calls.append(1)
        return DiagnosisDemoLLM()
    session = RoutedSession(root, report_file=report, verify_command=command, coding_factory=quiet,
                            diagnosis_factory=factory, max_feedback_rounds=1)
    session.handle("修复比例计算")
    assert session.report_file is not None
    session.handle("修复比例计算")
    assert len(calls) == 1 and session.feedback_attempts == 1
    assert session.feedback_records[-1]["status"] == "budget_exhausted"
    metadata = json.loads(Path(session.feedback_records[-1]["metadata_file"]).read_text())
    assert "feedback_context" not in metadata  # Ancestors are references, not recursively nested attempts.
    assert metadata["previous_report"] == session.feedback_records[0]["diagnosis_artifact"]
    assert session.report_file is None
    assert Path(session.feedback_records[-1]["metadata_file"]).exists()
    assert session.handle("排查验证失败")["status"] == "incomplete"
    assert len(calls) == 1


def test_zero_budget_captures_output_without_building_diagnosis_model(case):
    _, root, report, command, _ = case
    session = RoutedSession(root, report_file=report, verify_command=command, coding_factory=quiet,
                            diagnosis_factory=lambda: pytest.fail("No feedback calls allowed"), max_feedback_rounds=0)
    assert session.handle("修复比例计算")["status"] == "verification_failed"
    assert session.feedback_records[0]["status"] == "budget_exhausted"
    assert session.feedback_attempts == 0


def test_feedback_provider_failure_preserves_parent_failure_and_consumes_budget(case):
    _, root, report, command, _ = case
    def fail():
        raise APIError("secret-provider-body", request=None, body={"key": "secret-provider-body"})
    session = RoutedSession(root, report_file=report, verify_command=command, coding_factory=quiet, diagnosis_factory=fail)
    response = session.handle("修复比例计算")
    assert response["status"] == "verification_failed" and session.feedback_attempts == 1
    assert session.feedback_records[0]["status"] == "incomplete" and session.report_file is None
    assert "secret-provider-body" not in (session.directory / "session.json").read_text()
    assert Path(session.feedback_records[0]["metadata_file"]).exists()


def test_incomplete_feedback_diagnosis_is_not_a_valid_handoff(case):
    _, root, report, command, _ = case
    session = RoutedSession(root, report_file=report, verify_command=command, coding_factory=quiet,
                            diagnosis_factory=lambda: ScriptedLLM([LLMResponse(content="finished") for _ in range(8)]))
    assert session.handle("修复比例计算")["status"] == "verification_failed"
    assert session.feedback_records[0]["status"] == "incomplete" and session.report_file is None
    assert Path(session.feedback_records[0]["diagnosis_artifact"]).exists()


def test_timeout_retains_partial_process_output_and_no_fake_exit_code(case):
    workspace, root, report, _, _ = case
    command = [sys.executable, "-c", "import sys,time; print('loss=nan',flush=True); print('waiting',file=sys.stderr,flush=True); time.sleep(3)"]
    result = run_repair(load_handoff(report, root), "Inspect", quiet(), verify_command=command, verification_timeout=1)
    assert result["verification"]["status"] == "timeout"
    assert "loss=nan" in result["verification"]["stdout"]["text"]
    assert "waiting" in result["verification"]["stderr"]["text"]
    assert "exit_code" not in result["verification"]
    path = workspace / "timeout.json"
    path.write_text(json.dumps(result))
    feedback = create_feedback(path, workspace / "timeout-feedback", root)
    assert "exit_code=unavailable" in Path(feedback["log_file"]).read_text()


def test_missing_verification_executable_is_distinguished_from_application_failure(case):
    workspace, root, report, _, _ = case
    result = run_repair(load_handoff(report, root), "Inspect", quiet(), verify_command=[str(root / "missing-executable-xyz")])
    assert result["status"] == "incomplete" and result["verification"]["status"] == "launch_error"
    path = workspace / "launch.json"
    path.write_text(json.dumps(result))
    feedback = create_feedback(path, workspace / "launch-feedback", root)
    assert "command did not start" in Path(feedback["log_file"]).read_text()


def test_output_truncation_keeps_head_tail_and_original_size(case):
    workspace, root, _, _, _ = case
    result, _ = failed_artifact(case)
    result["verification"]["stdout"] = _bounded_output("head" + "x" * 20000 + "tail")
    path = workspace / "large-output.json"
    path.write_text(json.dumps(result))
    feedback = create_feedback(path, workspace / "large-feedback", root)
    output = feedback["metadata"]["outputs"]["stdout"]
    assert output["text"].startswith("head") and output["text"].endswith("tail")
    assert output["truncated"] and output["omitted_chars"] == 8008
    assert "truncated=True" in Path(feedback["log_file"]).read_text()


@pytest.mark.parametrize("status", ["passed", "not_run"])
def test_success_or_unverified_result_creates_no_failure_feedback(case, status):
    workspace, root, _, _, _ = case
    result, path = failed_artifact(case)
    result["verification"]["status"] = status
    path.write_text(json.dumps(result))
    assert create_feedback(path, workspace / "none", root) is None
    assert not (workspace / "none").exists()


@pytest.mark.parametrize("target", ["log", "repair"])
def test_tampered_feedback_input_rejected_before_diagnosis(case, target):
    workspace, root, _, _, _ = case
    _, path = failed_artifact(case)
    feedback = create_feedback(path, workspace / "feedback", root)
    (Path(feedback["log_file"]) if target == "log" else path).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_feedback(feedback)


def test_oversized_context_and_wrong_project_rejected(case, tmp_path):
    workspace, _, _, _, _ = case
    _, path = failed_artifact(case)
    with pytest.raises(ValueError, match="different project"):
        create_feedback(path, workspace / "wrong", tmp_path)
    with pytest.raises(ValueError, match="20000"):
        validate_context({"text": "x" * 20001})
    with pytest.raises(ValueError, match="20000"):
        run_diagnosis(workspace / "failure.log", case[1], quiet(), feedback_context={"text": "x" * 20001})


def test_new_log_clears_active_feedback_but_preserves_attempt_history(case):
    workspace, root, report, command, _ = case
    session = RoutedSession(root, report_file=report, verify_command=command, coding_factory=quiet, diagnosis_factory=DiagnosisDemoLLM)
    session.handle("修复比例计算")
    session.bind_log(workspace / "failure.log")
    assert session.active_feedback is None and session.report_file is None
    assert session.feedback_attempts == 1 and len(session.feedback_records) == 1


@pytest.mark.parametrize("budget", [-1, True, 6])
def test_invalid_feedback_budget_rejected(case, budget):
    with pytest.raises(ValueError, match="max_feedback_rounds"):
        RoutedSession(case[1], coding_factory=quiet, diagnosis_factory=DiagnosisDemoLLM, max_feedback_rounds=budget)


def test_complete_failed_edit_feedback_and_second_explicit_repair(tmp_path):
    result = run_demo(tmp_path / "demo")
    assert result["first_repair_status"] == "verification_failed"
    assert result["feedback_status"] == "completed"
    assert result["second_repair_status"] == "verified"
    assert result["feedback_attempts"] == 1
