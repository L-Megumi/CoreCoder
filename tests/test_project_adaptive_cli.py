"""Trusted CLI strategy selection is distinct from model-facing public input."""
import json
import pytest
from researchcoder.evaluation.projects.__main__ import load_config
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.runtime.protocol import ModelSpec
from tests.test_project_runner import fixture, scripted


def configuration(tmp_path):
    source, task, checks, _ = fixture(tmp_path)
    return dict(schema_version=1, task=task.to_dict(), source_root=str(source),
        public_check=checks["public"].to_dict(), f2p_check=checks["f2p"].to_dict(), p2p_check=checks["p2p"].to_dict(),
        model=scripted().to_dict(), policy=ProjectPolicy().to_dict())


def load(tmp_path, value):
    path = tmp_path / "trusted.json"
    path.write_text(json.dumps(value))
    return load_config(path)


def test_old_cli_config_defaults_to_no_experts(tmp_path):
    parsed = load(tmp_path, configuration(tmp_path))
    assert parsed["expert_policy"].mode == "off" and parsed["expert_models"] == {}


def test_cli_explicit_versioned_strategy_and_scripted_expert_fixture(tmp_path):
    value = configuration(tmp_path)
    value["expert_policy"] = ExpertPolicy(mode="public-failure-v1").to_dict()
    value["expert_models"] = {"code_analysis": scripted().to_dict()}
    parsed = load(tmp_path, value)
    assert parsed["expert_policy"].mode == "public-failure-v1"
    assert parsed["expert_models"]["code_analysis"].provider == "scripted"


@pytest.mark.parametrize("extra", [dict(expert_models=[]), dict(expert_models={"planner": {}}),
    dict(expert_policy={"mode": "invented"}), dict(worker_command=["untrusted-program"]), dict(api_key="forbidden")])
def test_cli_cannot_expand_expert_roles_permissions_or_worker_program(tmp_path, extra):
    value = configuration(tmp_path)
    value.update(extra)
    with pytest.raises((ValueError, TypeError)):
        load(tmp_path, value)


def test_live_execution_cannot_switch_models_via_expert_fixture(tmp_path):
    value = configuration(tmp_path)
    value["model"] = ModelSpec(provider="openai", model="not-called", base_url="https://api.deepseek.com").to_dict()
    value["expert_models"] = {"code_analysis": scripted().to_dict()}
    parsed = load(tmp_path, value)
    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="offline fixtures"):
        ProjectEvaluationRunner().run(target, **parsed, allow_live=True)
    assert not target.exists()
