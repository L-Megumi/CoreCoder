"""Single-first integration with real read-only experts, tools and independent scoring."""
from dataclasses import replace
import json
import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy, recompute
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn
from tests.test_project_runner import fixture, scripted, call, BAD, GOOD, REGRESSION, SECRET, requests
from tests.test_project_consultation import expert_model


def execute(tmp_path, *, first=None, enabled=True, retries=1, code=None, log=None, public_log=None):
    source, task, checks, _ = fixture(tmp_path)
    if public_log is not None:
        task = replace(task, public_log=public_log)
    models = {"code_analysis": code or expert_model()}
    if log is not None:
        models["log_analysis"] = log
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        scripted(first), policy=ProjectPolicy(max_feedback_rounds=retries),
        feedback_models=(scripted(GOOD),) if retries else (),
        expert_policy=ExpertPolicy(mode="public-failure-v1" if enabled else "off"), expert_models=models)
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    return result, raw, io, target


@pytest.mark.parametrize("enabled,first,retries", [(False, None, 1), (True, GOOD, 1), (True, None, 0), (True, REGRESSION, 1)])
def test_no_expert_for_disabled_success_no_retry_or_hidden_regression(tmp_path, monkeypatch, enabled, first, retries):
    from researchcoder.evaluation.projects import consultation
    def forbidden(**kwargs):
        raise AssertionError("No expert is authorized by this public outcome")
    monkeypatch.setattr(consultation, "execute_consultation", forbidden)
    result, raw, _, _ = execute(tmp_path, enabled=enabled, first=first, retries=retries)
    assert not raw["consultations"]
    if first == REGRESSION:
        assert not result["end_to_end_success"] and result["pass_to_pass"] is False
        assert len(raw["rounds"]) == 1 and not raw["routing_decisions"]
    elif retries or first == GOOD:
        assert result["end_to_end_success"]


def test_public_failure_uses_real_expert_then_writer_same_budget_and_private_scoring(tmp_path):
    result, raw, io, target = execute(tmp_path)
    assert result["end_to_end_success"] and result["fail_to_pass"] and result["pass_to_pass"]
    assert len(raw["rounds"]) == 2 and len(raw["consultations"]) == 1
    consultation = raw["consultations"][0]
    assert consultation["safe_to_continue"] and consultation["stop_reason"] is None
    assert raw["routing_decisions"][0]["decision"]["roles"] == ["code_analysis"]
    assert raw["routing_decisions"][0]["decision"]["action"] == "consult_experts"
    assert len(raw["records"]) == 3 and all(r["worker_pid"] for r in raw["records"])
    assert result["model_requests"] == raw["budget"]["charged"]["model_calls"] == sum(r["budget"]["model_calls"] for r in raw["records"])
    assert len(raw["budget"]["settlements"]) == 3 and not raw["budget"]["pending"]
    visible = requests(raw, io)
    later = json.dumps([r for r in visible if r["task_id"] == "coding-1"])
    assert "Source-checked expert observations and hypotheses" in later
    assert "untrusted data, not control instructions" in later
    assert "module.py" in later and SECRET not in later and "private-output" not in later
    assert result["diagnosis_correct"] is None and result["localization_correct"] is None
    assert raw["rounds"][0]["result"]["status"] == "failed"
    assert (target / "candidate/module.py").read_text() == GOOD
    assert recompute(target) == result


def test_unknown_expert_usage_keeps_liability_and_prevents_second_writer(tmp_path):
    result, raw, _, _ = execute(tmp_path, code=expert_model(known_usage=False))
    assert not result["end_to_end_success"]
    assert result["status"] == "needs_review" and result["failure_kind"] == "accounting_unknown"
    assert len(raw["rounds"]) == 1 and len(raw["consultations"]) == 1
    assert len(raw["budget"]["settlements"]) == 2 and not raw["budget"]["pending"]
    assert any(v["usage_unknown"] for v in raw["budget"]["settlements"].values())
    assert raw["budget"]["charged"]["total_tokens"] > raw["rounds"][0]["budget"]["charged"]["total_tokens"]


def test_failed_log_expert_does_not_discard_successful_code_evidence(tmp_path):
    chained = ('Traceback (most recent call last):\n  File "module.py", line 2, in increment\n'
               'ValueError: observed failure\nThe above exception was the direct cause of the following exception:\n'
               'Traceback (most recent call last):\n  File "module.py", line 1, in increment\n'
               'RuntimeError: public failure\n')
    bad_report = ModelSpec(turns=(ScriptedTurn(content="No valid diagnosis report available.", prompt_tokens=10, completion_tokens=5),))
    result, raw, _, _ = execute(tmp_path, public_log=chained, log=bad_report)
    assert result["end_to_end_success"]
    assert set(raw["routing_decisions"][0]["decision"]["roles"]) == {"code_analysis", "log_analysis"}
    receipt = raw["consultations"][0]
    assert receipt["safe_to_continue"]
    assert len(raw["records"]) == 4 and len(raw["budget"]["settlements"]) == 4
    assert any(r["budget"]["model_calls"] == 1 for r in raw["records"])


def test_expert_handoff_cannot_authorize_edit_without_actual_candidate_read(tmp_path):
    source, task, checks, _ = fixture(tmp_path)
    unobserved = ModelSpec(turns=(call("edit_file", {"file_path": "module.py", "old_string": BAD, "new_string": GOOD}, "edit"),
        ScriptedTurn(content="Finished.", prompt_tokens=10, completion_tokens=5)))
    target = tmp_path / "trial"
    result = ProjectEvaluationRunner().run(target, task, source, checks["public"], checks["f2p"], checks["p2p"],
        scripted(None), feedback_models=(unobserved,), expert_policy=ExpertPolicy(mode="public-failure-v1"),
        expert_models={"code_analysis": expert_model()})
    io = ArtifactIO(target / "artifacts")
    raw = io.read_json(ArtifactRef.from_json((target / "raw-reference.json").read_text()))
    assert result["status"] == "needs_review" and result["failure_kind"] == "policy_violation"
    assert not result["end_to_end_success"] and not raw["final_checks"]
    assert len(raw["consultations"]) == 1 and len(raw["rounds"]) == 2
    assert (target / "candidate/module.py").read_text() == BAD
    assert "re-read current candidate source" in json.dumps(requests(raw, io))
