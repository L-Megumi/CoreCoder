from dataclasses import replace
from types import SimpleNamespace
import pytest
from researchcoder.evaluation.admission import BatchAdmission
from researchcoder.evaluation.contracts import EvaluationConfig


def settle(admission, status="incomplete", kind="unknown", *, complete=True, intact=True):
    result = SimpleNamespace(status=status, metrics=SimpleNamespace(usage_complete=complete))
    raw = dict(failure_kind=kind, policy_intact=intact,
               budget=dict(charged=dict(model_calls=1, total_tokens=100)))
    admission.settle(result, raw)


def test_business_failures_continue_and_consecutive_api_failures_stop():
    config = EvaluationConfig(stop_after_consecutive_api_failures=2, stop_on_safety_violation=True,
                              stop_on_unknown_usage=True)
    assert EvaluationConfig.from_json(config.to_json()) == config
    admission = BatchAdmission(config)
    settle(admission)
    assert admission.admit() is None and admission.failures == 1
    settle(admission, kind="model_error")
    assert admission.admit() is None
    settle(admission, status="succeeded", kind=None)
    assert admission.consecutive_api_failures == 0
    settle(admission, kind="model_error")
    settle(admission, kind="model_error")
    assert admission.admit() == "batch_consecutive_api_failures"
    assert admission.model_charge == 5 and admission.token_charge == 500


@pytest.mark.parametrize("kwargs,reason", [
    (dict(complete=False), "batch_usage_unknown"),
    (dict(intact=False), "batch_safety_violation"),
    (dict(status="needs_review"), "batch_safety_violation"),
])
def test_new_protocol_stops_on_unknown_and_safety(kwargs, reason):
    admission = BatchAdmission(EvaluationConfig(stop_on_unknown_usage=True, stop_on_safety_violation=True))
    settle(admission, **kwargs)
    assert admission.admit() == reason


def test_legacy_stop_policy_and_invalid_limit():
    old = EvaluationConfig().to_dict()
    assert "stop_after_consecutive_api_failures" not in old and "stop_on_safety_violation" not in old
    assert EvaluationConfig.from_dict(old).to_dict() == old
    admission = BatchAdmission(EvaluationConfig(stop_after_failures=1))
    settle(admission)
    assert admission.admit() == "batch_failure_limit"
    with pytest.raises(ValueError):
        replace(EvaluationConfig(), stop_after_consecutive_api_failures=0)


def test_denied_tool_stops_even_when_candidate_files_are_unchanged():
    admission = BatchAdmission(EvaluationConfig(stop_on_safety_violation=True))
    result = SimpleNamespace(status="incomplete", metrics=SimpleNamespace(usage_complete=True))
    raw = dict(policy_intact=True, policy_violations=[dict(event_type="tool.rejected")],
               budget=dict(charged=dict(model_calls=1, total_tokens=100)))
    admission.settle(result, raw)
    assert admission.admit() == "batch_safety_violation"
    assert admission.token_charge == 100


def test_real_worker_denial_trace_reaches_batch_admission(tmp_path, monkeypatch):
    import json
    import os
    from researchcoder.evaluation.runner import EvaluationRunner
    from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, ScriptedCall
    from researchcoder.adapters.artifacts import ArtifactIO
    from researchcoder.contracts import ArtifactRef

    # The Log role cannot search; this request must be denied before dispatch.
    # Real Worker/Trace/Artifact/Runner plumbing, not an invented mock raw flag.
    model = ModelSpec(turns=(ScriptedTurn(prompt_tokens=10, completion_tokens=1,
        calls=(ScriptedCall(call_id="denied", name="search_project_source", arguments_json='{"query":"shape"}'),)),
        ScriptedTurn(content="Unable to complete", prompt_tokens=10, completion_tokens=1)))
    monkeypatch.setattr("researchcoder.evaluation.runner.offline_models", lambda *args: (model, ModelSpec()))
    config = EvaluationConfig(case_ids=("autograd_detach", "shape_transpose"), modes=("evidence",),
                              stop_on_safety_violation=True, max_trials=2)
    target = tmp_path / "run"
    EvaluationRunner().run(target, config, torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"])
    outcomes = []
    for ordinal in range(2):
        trial_id = f"trial-{ordinal:04d}"
        io = ArtifactIO(target / "trials" / trial_id / "artifacts")
        result = io.read_json(ArtifactRef.from_json((target / "records" / (trial_id + ".json")).read_text()))
        outcomes.append((result, io.read_json(ArtifactRef.from_dict(result["raw"]))))
    first, raw = outcomes[0]
    assert first["status"] == "incomplete" and raw["policy_intact"]
    assert raw["policy_violations"][0]["tool"] == "search_project_source"
    assert outcomes[1][0]["status"] == "cancelled" and not outcomes[1][1]["records"]
    assert json.loads((target / "batch-admission.json").read_text())["stop_reason"] == "batch_safety_violation"
