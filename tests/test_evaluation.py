"""Annotation isolation, exact scoring, denominators and optional report contract."""

import copy
import json
from types import SimpleNamespace

import pytest

from corecoder.diagnose import run_diagnosis, validate_diagnosis
from corecoder.diagnostic_cases import GoldReplayLLM, create_cases
from corecoder.diagnostic_tools import Investigation
from corecoder.evaluation import aggregate, run_suite, score_report


@pytest.fixture
def sample(tmp_path):
    case = create_cases(tmp_path / "fixtures")[2]
    report = run_diagnosis(case["log"], case["project"], GoldReplayLLM(case), findings=True,
                           source_hints=case["source_hints"], completion_mode="checklist")
    assert report["status"] == "completed", report["failure_reason"]
    return case, report


def test_annotations_outside_project_cannot_be_read_by_diagnosis_tool(tmp_path):
    cases = create_cases(tmp_path / "fixtures")
    assert len(cases) == 7
    assert len({case["expected_category"] for case in cases}) == 5
    for case in cases:
        state = Investigation(case["log"], case["project"])
        response = json.loads(state.execute("read_project_file", {"path": "../annotation.json"}))
        assert response["observations"][0]["data"]["status"] == "outside_project"
        assert "expected_category" not in json.dumps(response)


def test_offline_suite_tests_grader_without_claiming_model_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder.evaluation.build_deepseek_llm", lambda **kwargs: pytest.fail("Unexpected API builder"))
    result = run_suite(tmp_path)
    assert len(result["runs"]) == 14
    assert result["fixture_validation_passed"]
    assert result["summary"] is None
    assert all(not row["metrics"]["provider_usage_complete"] for row in result["runs"])
    assert all(row["score"]["required_literal_path_checks_completed"] for row in result["runs"])


@pytest.mark.parametrize("field,value,metric", [
    ("failure_category", "unknown", "category_correct"),
    ("cause_code", "unknown", "annotated_cause_code_correct"),
    ("primary_locations", [], "annotated_source_line_correct"),
])
def test_grader_rejects_incorrect_predictions_even_with_valid_report(sample, field, value, metric):
    case, report = sample
    damaged = copy.deepcopy(report)
    damaged["diagnosis"][field] = value
    assert not score_report(damaged, case)[metric]


def test_failed_reports_remain_in_denominator(sample):
    case, report = sample
    bad = copy.deepcopy(report)
    bad["status"] = "incomplete"
    rows = [{"arm": "baseline", "status": item["status"], "metrics": item["metrics"], "score": score_report(item, case)}
            for item in (report, bad)]
    summary = aggregate(rows)["baseline"]
    assert summary["runs"] == 2
    assert summary["category_correct"] == 0.5
    assert summary["annotated_source_line_correct"] == 0.5


@pytest.mark.parametrize("change", ["unknown_id", "outside_span", "bool_line", "unknown_category", "unknown_cause"])
def test_structured_prediction_contract_checks_actual_source_ranges(sample, change):
    _, report = sample
    prediction = copy.deepcopy(report["diagnosis"])
    if change == "unknown_id":
        prediction["primary_locations"][0]["observation_id"] = "obs_invented"
    elif change == "outside_span":
        prediction["primary_locations"][0]["line_no"] = 10000
    elif change == "bool_line":
        prediction["primary_locations"][0]["line_no"] = True
    elif change == "unknown_category":
        prediction["failure_category"] = "invented_category"
    else:
        prediction["cause_code"] = "invented_cause"
    state = SimpleNamespace(evidence=report["evidence"], observations={item["id"]: item for item in report["observations"]})
    with pytest.raises(ValueError):
        validate_diagnosis(json.dumps(prediction), state, findings=True)


def test_structured_predictions_are_not_silently_accepted_in_legacy_profile(sample):
    _, report = sample
    state = SimpleNamespace(evidence=report["evidence"], observations={item["id"]: item for item in report["observations"]})
    with pytest.raises(ValueError, match="exactly"):
        validate_diagnosis(json.dumps(report["diagnosis"]), state)
