from dataclasses import replace
import json
import os
import pytest
from researchcoder.runtime.protocol import (
    ModelSpec,
    ModelRequestOptions,
    model_request_kwargs,
    WorkerRequest,
)
from researchcoder.evaluation.dataset import load_manifest
from researchcoder.evaluation.runner import build_workflow, offline_models
from researchcoder.evaluation.contracts import EvaluationConfig, TrialSpec


def test_explicit_non_thinking_option_and_no_arbitrary_extra_body():
    live = ModelSpec(
        provider="openai",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        credential_env="DEEPSEEK_API_KEY",
    )
    assert model_request_kwargs(live, None) == {}
    assert model_request_kwargs(live, ModelRequestOptions()) == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert (
        ModelRequestOptions.from_json(ModelRequestOptions().to_json())
        == ModelRequestOptions()
    )
    with pytest.raises(ValueError):
        model_request_kwargs(ModelSpec(), ModelRequestOptions())
    with pytest.raises(ValueError):
        model_request_kwargs(
            replace(live, base_url="https://example.org"), ModelRequestOptions()
        )
    with pytest.raises(ValueError):
        ModelRequestOptions.from_dict(
            dict(schema_version=1, thinking="enabled", extra_body={"secret": 1})
        )


def test_real_worker_factory_routes_options_through_budgeted_model_offline(
    tmp_path, monkeypatch
):
    import researchcoder.runtime.worker as worker

    case = load_manifest().cases[0]
    live = ModelSpec(
        provider="openai",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        credential_env="DEEPSEEK_API_KEY",
    )
    config = EvaluationConfig(
        model=live, model_options=ModelRequestOptions(), allow_live=True
    )
    trial = TrialSpec(
        trial_id="t", case_id=case.case_id, mode="parallel", repeat=0, ordinal=0
    )
    built = build_workflow(
        case,
        trial,
        tmp_path / "trial",
        config,
        os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
    )
    template = built["config"].templates["diagnosis"]
    captured = []

    class Client:
        def with_options(self, **kwargs):
            captured.append(kwargs)
            return self

    def fake_llm(**kwargs):
        captured.append(kwargs)
        model = worker.ProtocolScript(offline_models(case, "correct")[0])
        model.client = Client()
        return model

    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-placeholder-credential")
    monkeypatch.setattr(worker, "LLM", fake_llm)
    request = WorkerRequest(
        spec=template.spec,
        attempt_id="a",
        artifact_root=str(built["io"].root),
        trace_path=str(tmp_path / "trace"),
        parent_event_id="root",
        limits=config.task_limits,
        model=live,
        model_options=config.model_options,
    )
    response = worker.execute_request(request)
    assert response.result.status == "succeeded"
    budget = json.loads(response.budget_json)
    assert budget["model_calls"] >= 3 and budget["unknown_requests"] > 0
    assert (
        captured[0]["extra_body"] == {"thinking": {"type": "disabled"}}
        and captured[1]["max_retries"] == 0
    )


def test_checkpoint_authority_omits_default_option_and_binds_nondefault(tmp_path):
    from tests.test_orchestration_static import setup_graph
    from researchcoder.durable.recovery import authority
    from researchcoder.durable.store import digest

    plan, cfg, io = setup_graph(tmp_path)
    # Capture the exact canonical legacy material, not a hash recomputed with new fields.
    import researchcoder.durable.recovery as module

    saved = []
    original = module.digest
    module.digest = lambda value: (saved.append(value), original(value))[1]
    try:
        authority(cfg)
    finally:
        module.digest = original
    assert all("model_options" not in t for t in saved[0]["templates"].values())
    assert authority(cfg) == digest(saved[0])
    template = cfg.templates["read"]
    live = ModelSpec(
        provider="openai", model="deepseek-flash", base_url="https://api.deepseek.com"
    )
    changed = replace(
        cfg,
        templates=dict(
            cfg.templates,
            read=replace(
                template,
                config=replace(
                    template.config, model=live, model_options=ModelRequestOptions()
                ),
            ),
        ),
    )
    assert authority(changed) != authority(cfg)


def test_pilot_proposal_never_executes_and_uses_planning_only_prices():
    from researchcoder.evaluation.pilot import proposal

    p = proposal()
    assert not p["executed"] and p["provider_requests_executed"] == 0
    config = EvaluationConfig.from_dict(p["config"])
    assert (
        config.price.billing_basis == "planning_only"
        and config.stop_on_unknown_usage
        and config.max_trials == 12
    )
    assert config.model_options.thinking == "disabled"
