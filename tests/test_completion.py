"""Completion coverage, restricted configuration resolution and bounded continuation."""

import json

import pytest

from corecoder.completion import CompletionChecklist, dataset_validation_path
from corecoder.diagnose import run_diagnosis, validate_diagnosis
from corecoder.diagnosis_demo import DemoLLM, create_demo_case, run_demo
from corecoder.diagnostic_tools import Investigation
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall


@pytest.fixture
def case(tmp_path):
    return create_demo_case(tmp_path / "case")


def test_read_free_checklist_requires_current_and_historical_literal_paths(case):
    workspace, root, log, current, previous = case
    state = Investigation(log, root, allowed_path_roots=[workspace])
    plan = CompletionChecklist(state)
    assert plan.status()["pending_count"] == 1
    state.execute("extract_log_evidence", {})
    state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    state.execute("inspect_paths", {"paths": [str(previous)]})
    pending = [item for item in plan.status()["checks"] if item["status"] == "pending"]
    assert len(pending) == 1
    assert pending[0]["kind"] == "current_validation_path"
    assert pending[0]["subject"] == str(current)
    assert len(state.events) == 3  # status calculation did not call tools or stat the current directory.
    state.execute("inspect_paths", {"paths": [str(current)]})
    assert plan.status()["ready_for_report"]


def test_external_path_is_blocked_unknown_and_can_be_reported(case, tmp_path):
    _, root, log, _, _ = case
    outside = tmp_path / "outside.yaml"
    log.write_text(f"FileNotFoundError: Dataset '{outside}' images not found, missing path '{tmp_path / 'other'}'\n")
    state = Investigation(log, root)
    state.execute("extract_log_evidence", {})
    state.execute("read_project_file", {"path": str(outside)})
    state.execute("inspect_paths", {"paths": [str(tmp_path / 'other')]})
    status = CompletionChecklist(state).status()
    assert status["ready_for_report"]
    assert status["blocked_count"] == 2
    assert all(item["status"] == "blocked" for item in status["checks"])


@pytest.mark.parametrize("text", [
    "path: relative\nval: images/val", "path: /root\nval: [a,b]", "path: /root\nval: ${VAL}",
    "path: /root\nval: !include other", "path: /root\nval: a\nval: b", "path: /root\nval: &anchor a",
])
def test_unsupported_yaml_does_not_guess_a_current_dataset_path(text):
    resolved, reason = dataset_validation_path(text)
    assert resolved is None and reason


def test_scalar_absolute_root_and_quoted_hash_are_resolved():
    resolved, reason = dataset_validation_path("path: '/root/data #1'\nval: \"images/val\" # comment")
    assert resolved.endswith("/root/data #1/images/val") and reason is None


def test_source_hint_is_an_input_check_not_a_gold_answer(case):
    workspace, root, log, _, _ = case
    state = Investigation(log, root, allowed_path_roots=[workspace])
    plan = CompletionChecklist(state, ["train.py"])
    hint = next(check for check in plan.status()["checks"] if check["kind"] == "source_hint_read")
    assert hint["status"] == "pending"
    state.execute("read_project_file", {"path": "train.py"})
    assert next(check for check in plan.status()["checks"] if check["kind"] == "source_hint_read")["status"] == "observed"


def test_early_report_continues_with_tools_within_original_round_budget(case):
    workspace, root, log, current, previous = case
    good = run_diagnosis(log, root, DemoLLM(root, current, previous), allowed_path_roots=[workspace])
    script = []
    for index, event in enumerate(good["tool_events"]):
        if event["tool"] == "inspect_paths":
            script.append(LLMResponse(content=json.dumps(good["diagnosis"])))
        script.append(LLMResponse(tool_calls=[ToolCall(str(index), event["tool"], event["arguments"])]))
    script.append(LLMResponse(content=json.dumps(good["diagnosis"])))
    result = run_diagnosis(log, root, ScriptedLLM(script), allowed_path_roots=[workspace], completion_mode="checklist")
    assert result["status"] == "completed", result["failure_reason"]
    assert result["investigation_checklist"]["ready_for_report"]
    assert result["metrics"]["investigation_continuations"] == 1
    assert result["metrics"]["llm_calls"] == 6
    assert len(result["selection_events"]) <= 8


def test_model_refusing_required_checks_is_incomplete_without_unbounded_retry(case):
    _, root, log, _, _ = case
    model = ScriptedLLM([LLMResponse(content="finished") for _ in range(8)])
    result = run_diagnosis(log, root, model, completion_mode="checklist", max_rounds=4)
    assert result["status"] == "incomplete"
    assert "completion requirements" in result["failure_reason"]
    assert result["metrics"]["llm_calls"] <= 4
    assert result["investigation_checklist"]["pending_count"] > 0


def test_duplicate_report_keys_are_rejected(case):
    workspace, root, log, current, previous = case
    report = run_diagnosis(log, root, DemoLLM(root, current, previous), allowed_path_roots=[workspace])
    state = type("State", (), {"evidence": report["evidence"], "observations": {item["id"]: item for item in report["observations"]}})()
    text = json.dumps(report["diagnosis"])
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        validate_diagnosis(text[:-1] + ', "summary": "other"}', state)


def test_unexposed_critical_page_is_not_silently_complete(case):
    _, root, log, _, _ = case
    state = Investigation(log, root)
    state.execute("extract_log_evidence", {"limit": 1})
    pending = CompletionChecklist(state).guidance()["next_checks"]
    assert any(check["kind"] == "evidence_page" for check in pending)


@pytest.mark.parametrize("selection,context", [("all", "baseline"), ("guided", "baseline"),
                                               ("all", "evidence"), ("guided", "evidence")])
def test_completion_composes_with_existing_optimization_switches(tmp_path, selection, context):
    result = run_demo(directory=tmp_path / "demo", tool_selection=selection, context_mode=context,
                       completion_mode="checklist")
    assert result["status"] == "completed", result["failure_reason"]
    assert result["investigation_checklist"]["ready_for_report"]
    assert result["metrics"]["tool_calls"] == 4


def test_configuration_page_beyond_preview_remains_pending_and_combines_same_hash(case):
    workspace, root, log, current, _ = case
    config = root / "data.yaml"
    config.write_text(f"path: {root / 'dataset'}\n" + "# comment\n" * 70 + "val: images/val\n")
    state = Investigation(log, root, allowed_path_roots=[workspace])
    state.execute("extract_log_evidence", {})
    state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    plan = CompletionChecklist(state)
    missing = next(item for item in plan.status()["checks"] if item["kind"] == "config_resolution")
    assert missing["status"] == "pending"
    state.execute("read_project_file", missing["next_action"]["arguments"])
    resolved = next(item for item in plan.status()["checks"] if item["kind"] == "current_validation_path")
    assert resolved["subject"] == str(current)


def test_unsupported_scalar_is_blocked_even_when_excerpt_has_more_lines(case):
    workspace, root, log, _, _ = case
    (root / "data.yaml").write_text("path: /root\nval: [a,b]\n" + "# comment\n" * 100)
    state = Investigation(log, root, allowed_path_roots=[workspace])
    state.execute("extract_log_evidence", {})
    state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    check = next(item for item in CompletionChecklist(state).status()["checks"] if item["kind"] == "config_resolution")
    assert check["status"] == "blocked"
