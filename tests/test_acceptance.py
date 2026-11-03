"""Acceptance regressions: usable citations, visible source, budgets and audit metrics."""

import copy
import json
from types import SimpleNamespace

import pytest

from corecoder.completion import CompletionChecklist
from corecoder.diagnose import run_diagnosis, validate_diagnosis
from corecoder.diagnosis_demo import DemoLLM, create_demo_case, run_demo
from corecoder.diagnostic_tools import Investigation
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.report_findings import validate_findings


def test_current_state_cannot_cite_only_an_unchecked_external_path(tmp_path):
    result = run_demo(directory=tmp_path / "demo")
    state = SimpleNamespace(evidence=result["evidence"], observations={item["id"]: item for item in result["observations"]})
    state.observations["obs_blocked"] = {
        "id": "obs_blocked", "kind": "path_status", "evidence_ids": [],
        "data": {"status": "outside_allowed_roots", "requested_path": "/outside"},
    }
    report = copy.deepcopy(result["diagnosis"])
    report["current_state"]["observation_ids"] = ["obs_blocked"]
    with pytest.raises(ValueError, match="cite.*usable current"):
        validate_diagnosis(json.dumps(report), state)


@pytest.mark.parametrize("via_locator", [False, True])
def test_primary_location_cannot_point_past_character_truncated_content(tmp_path, via_locator):
    root = tmp_path / "project"
    root.mkdir()
    source = root / "entry.py"
    source.write_text("# " + "x" * 13000 + "\nraise RuntimeError('bad')\n", encoding="utf-8")
    log = tmp_path / "failure.log"
    log.write_text(f'Traceback (most recent call last):\n  File "{source}", line 2, in <module>\n'
                   "    raise RuntimeError('bad')\nRuntimeError: bad\n", encoding="utf-8")
    state = Investigation(log, root)
    state.execute("extract_log_evidence", {})
    if via_locator:
        response = json.loads(state.execute("locate_source", {"evidence_ids": [state.evidence[0]["id"]]}))
    else:
        response = json.loads(state.execute("read_project_file", {"path": "entry.py"}))
    observation = response["observations"][0]
    prediction = {"failure_category": "unknown", "cause_code": "unknown",
                  "primary_locations": [{"observation_id": observation["id"], "line_no": 2}]}
    with pytest.raises(ValueError, match="observed source line"):
        validate_findings(prediction, state)


def test_excerpt_exposes_character_truncation_separately_from_pagination(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "entry.py").write_text("# " + "x" * 13000 + "\nvalue = 1\n", encoding="utf-8")
    log = tmp_path / "failure.log"
    log.write_text("RuntimeError: bad\n", encoding="utf-8")
    state = Investigation(log, root)
    data = json.loads(state.execute("read_project_file", {"path": "entry.py"}))["observations"][0]["data"]
    assert data["char_truncated"] is True
    assert data["visible_end_line"] == 0  # The first source line itself is incomplete.


def test_report_validation_failure_does_not_discard_complete_provider_usage(tmp_path):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")

    class InvalidReportWithUsage(DemoLLM):
        def chat(self, *args, **kwargs):
            response = super().chat(*args, **kwargs)
            response.prompt_tokens, response.completion_tokens = 10, 2
            if not response.tool_calls:
                response.content = "invalid report"
            return response

    result = run_diagnosis(log, root, InvalidReportWithUsage(root, current, previous),
                           allowed_path_roots=[workspace], max_report_repairs=0, mode="live")
    assert result["status"] == "incomplete"
    assert result["metrics"]["prompt_tokens"] == 50
    assert result["metrics"]["provider_usage_complete"] is True


@pytest.mark.parametrize("arguments", [None, [], ["bad"], 7])
def test_malformed_tool_arguments_still_consume_budget_and_leave_audit(tmp_path, arguments):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("invalid", "extract_log_evidence", arguments)]),
        LLMResponse(tool_calls=[ToolCall("valid", "extract_log_evidence", {})]),
        LLMResponse(content="No report"),
    ])
    result = run_diagnosis(log, root, model, max_tool_calls=1, max_report_repairs=0)
    assert result["status"] == "incomplete"
    assert result["metrics"]["tool_calls"] == len(result["tool_events"]) == 2
    assert result["metrics"]["tool_errors"] == 2
    assert result["tool_events"][0]["arguments"] == arguments
    assert not result["evidence"]


def test_completion_guidance_reuses_observation_references_for_finished_checks(tmp_path):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")
    state = Investigation(log, root, allowed_path_roots=[workspace])
    state.execute("extract_log_evidence", {})
    state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    state.execute("inspect_paths", {"paths": [str(previous), str(current)]})
    events_before = len(state.events)
    guidance = CompletionChecklist(state).guidance()
    assert guidance["ready_for_report"]
    completed = guidance["completed_checks"]
    assert any(item["kind"] == "config_read" for item in completed)
    assert all(item["observation_ids"] for item in completed)
    assert len(state.events) == events_before


def test_failed_calls_with_missing_usage_remain_incomplete_usage(tmp_path):
    _, root, log, current, previous = create_demo_case(tmp_path / "case")
    result = run_diagnosis(log, root, DemoLLM(root, current, previous), max_rounds=1, mode="live")
    assert result["status"] == "incomplete"
    assert result["metrics"]["provider_usage_complete"] is False


def test_hidden_source_can_be_recovered_by_reading_from_the_requested_line(tmp_path):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")
    (root / "entry.py").write_text("# " + "x" * 13000 + "\nraise RuntimeError('bad')\n", encoding="utf-8")
    state = Investigation(log, root)
    state.execute("extract_log_evidence", {})
    response = json.loads(state.execute("read_project_file", {"path": "entry.py", "start_line": 2}))
    observation = response["observations"][0]
    validate_findings({"failure_category": "unknown", "cause_code": "unknown", "primary_locations": [
        {"observation_id": observation["id"], "line_no": 2},
    ]}, state)
    assert observation["data"]["visible_end_line"] == 2


@pytest.mark.parametrize("path", [None, 123, False, ["data.yaml"]])
def test_nonstring_read_path_cannot_poison_completion_state(tmp_path, path):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")
    state = Investigation(log, root)
    response = json.loads(state.execute("read_project_file", {"path": path}))
    assert response["error"]["type"] == "ValueError"
    assert not state.observations
    status = CompletionChecklist(state, ["data.yaml"]).status()
    assert status["pending_count"] == 2


def test_hash_without_any_source_excerpt_cannot_support_current_state(tmp_path):
    result = run_demo(directory=tmp_path / "demo")
    state = SimpleNamespace(evidence=result["evidence"], observations={item["id"]: item for item in result["observations"]})
    state.observations["obs_no_excerpt"] = {
        "id": "obs_no_excerpt", "kind": "source_location", "evidence_ids": [],
        "data": {"status": "line_out_of_range", "file_sha256": "known-hash", "snippet": None},
    }
    report = copy.deepcopy(result["diagnosis"])
    report["current_state"]["observation_ids"] = ["obs_no_excerpt"]
    with pytest.raises(ValueError, match="cite.*usable current"):
        validate_diagnosis(json.dumps(report), state)


def test_unreadable_long_config_line_stops_repeating_the_same_page(tmp_path):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")
    (root / "data.yaml").write_text(f"path: {root / 'dataset'}\n# " + "x" * 13000 + "\nval: images/val\n", encoding="utf-8")
    state = Investigation(log, root)
    state.execute("extract_log_evidence", {})
    state.execute("locate_source", {"evidence_ids": [item["id"] for item in state.evidence]})
    plan = CompletionChecklist(state)
    pending = next(item for item in plan.status()["checks"] if item["kind"] == "config_resolution")
    assert pending["status"] == "pending"
    state.execute("read_project_file", pending["next_action"]["arguments"])
    check = next(item for item in plan.status()["checks"] if item["kind"] == "config_resolution")
    assert check["status"] == "blocked"
    assert "character limit" in check["blocked_reason"]
    assert check["next_action"] is None
