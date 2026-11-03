"""M4 strategies come only from versioned trusted execution configuration."""

import pytest

from researchcoder.context.policy import ContextPolicy
from researchcoder.evaluation.projects.handoff import HandoffPolicy
from researchcoder.evaluation.projects.routing import ExpertPolicy
from tests.test_project_adaptive_cli import configuration, load


def test_new_project_strategies_default_off(tmp_path):
    parsed = load(tmp_path, configuration(tmp_path))
    assert parsed["handoff_policy"] is None and parsed["writer_context_policy"] is None


@pytest.mark.parametrize("strategy", ["original", "evidence"])
def test_trusted_project_context_and_schedule_roundtrip(tmp_path, strategy):
    value = configuration(tmp_path)
    value["handoff_policy"] = HandoffPolicy(mode="local-stop-v1").to_dict()
    value["writer_context_policy"] = ContextPolicy(strategy=strategy, role="repair", allow_recovery=True,
        coding_source_mode="observed-source-v1").to_dict()
    value["expert_policy"] = ExpertPolicy(mode="public-failure-v1", schedule_version="serial-parallel-v2", max_workers=1).to_dict()
    parsed = load(tmp_path, value)
    assert parsed["writer_context_policy"].strategy == strategy
    assert parsed["handoff_policy"].mode == "local-stop-v1"
    assert parsed["expert_policy"].max_workers == 1


@pytest.mark.parametrize("field,value", [
    ("handoff_policy", {"mode": "unbounded"}),
    ("writer_context_policy", {"coding_source_mode": "trust-arbitrary-paths"}),
    ("expert_policy", {"schedule_version": "serial-parallel-v2", "max_workers": 99}),
])
def test_cli_rejects_unversioned_or_expansive_controls(tmp_path, field, value):
    config = configuration(tmp_path)
    config[field] = value
    with pytest.raises((ValueError, TypeError)):
        load(tmp_path, config)
