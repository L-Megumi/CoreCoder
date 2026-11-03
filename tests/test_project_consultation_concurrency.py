"""Serial/parallel expert arms vary only existing Scheduler concurrency."""
from dataclasses import replace
import pytest

from researchcoder.evaluation.projects.consultation import execute_consultation
from researchcoder.evaluation.projects.routing import ExpertPolicy, RULES_SHA256, V2_RULES_SHA256, RoutingDecision, decide
from researchcoder.orchestration.scheduler import DAGScheduler
from tests.test_project_consultation import fixture


def test_legacy_policy_and_decision_roundtrip_keep_original_identity():
    policy = ExpertPolicy(mode="public-failure-v1")
    raw = policy.to_dict()
    assert "schedule_version" not in raw and "max_workers" not in raw
    assert ExpertPolicy.from_dict(raw) == policy
    old = RoutingDecision(action="keep_single", roles=(), reason_codes=("legacy",),
        policy_sha256=policy.sha256, features_sha256="a"*64)
    assert old.rules_sha256 == RULES_SHA256
    assert RoutingDecision.from_json(old.to_json()) == old


@pytest.mark.parametrize("workers", [0, 3, True, None])
def test_v2_requires_reviewed_explicit_concurrency(workers):
    with pytest.raises((ValueError, TypeError)):
        ExpertPolicy(schedule_version="serial-parallel-v2", max_workers=workers)


def test_v2_serial_and_parallel_share_role_selection_and_conservative_time(tmp_path):
    args = fixture(tmp_path, chained=True)
    features = args["features"]
    for seconds, expected in ((104, ("code_analysis",)), (105, ("code_analysis", "log_analysis"))):
        decisions = []
        for workers in (1, 2):
            policy = replace(args["policy"], schedule_version="serial-parallel-v2", max_workers=workers)
            result = decide(replace(features, remaining_seconds=seconds), policy)
            assert result.roles == expected and result.rules_sha256 == V2_RULES_SHA256
            decisions.append(result)
        assert decisions[0].roles == decisions[1].roles
        assert decisions[0].reason_codes == decisions[1].reason_codes
        assert decisions[0].features_sha256 == decisions[1].features_sha256
    assert decide(replace(features, remaining_seconds=60), args["policy"]).roles == ("code_analysis", "log_analysis")


@pytest.mark.parametrize("workers", [1, 2])
def test_actual_workers_use_same_roles_caps_and_no_artificial_dependencies(tmp_path, monkeypatch, workers):
    args = fixture(tmp_path, chained=True)
    args["policy"] = replace(args["policy"], schedule_version="serial-parallel-v2", max_workers=workers)
    args["decision"] = decide(args["features"], args["policy"])
    original = DAGScheduler.execute
    captured = []
    def observe(self, plan, config, **kw):
        captured.append((plan, config, kw["budget"]))
        return original(self, plan, config, **kw)
    monkeypatch.setattr(DAGScheduler, "execute", observe)
    value = execute_consultation(**args)
    assert value.safe_to_continue, value.to_dict()
    assert len(value.records) == 2 and len({r["worker_pid"] for r in value.records}) == 2
    plan, config, budget = captured[0]
    assert config.max_workers == workers and budget is args["budget"]
    assert tuple(task.role for task in plan.tasks) == ("code_analysis", "log_analysis")
    assert all(not task.dependencies for task in plan.tasks)
    assert all(template.config.limits == args["policy"].expert_limits for template in config.templates.values())
    assert value.schedule["max_workers"] == workers and value.schedule["effective_workers"] == workers
    assert value.schedule["admitted_expert_seconds"] == 90
    assert value.budget_after["charged"]["model_calls"] == 8


def test_code_only_is_not_reported_as_two_worker_parallelism(tmp_path):
    args = fixture(tmp_path)
    args["policy"] = replace(args["policy"], schedule_version="serial-parallel-v2", max_workers=2)
    args["decision"] = decide(args["features"], args["policy"])
    value = execute_consultation(**args)
    assert value.safe_to_continue and len(value.records) == 1
    assert value.schedule["max_workers"] == 2 and value.schedule["effective_workers"] == 1
