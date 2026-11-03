from dataclasses import replace
import json
import pytest
from researchcoder.runtime.optimization import RuntimeOptimization, runtime_optimization
from researchcoder.runtime.protocol import WorkerRequest,ModelSpec,RuntimeLimits
from researchcoder.evaluation.contracts import EvaluationConfig
from researchcoder.orchestration.profiles import RoleBudgetOverride
from researchcoder.orchestration.remediation import RemediationPolicy
from researchcoder.durable.recovery import authority
from tests.test_runtime_worker import request
from tests.test_orchestration_static import setup_graph


def test_old_worker_and_evaluation_canonical_json_omit_unused_strategy(tmp_path):
    old=request(tmp_path)
    assert 'optimization' not in old.to_dict()
    assert WorkerRequest.from_json(old.to_json())==old
    assert 'harness_strategy' not in EvaluationConfig().to_dict()
    for name in ('budget-v1','compact-v1','constraints-v1','combined-v1'):
        cfg=EvaluationConfig(harness_strategy=name)
        assert EvaluationConfig.from_json(cfg.to_json())==cfg
        optimized=replace(old,optimization=runtime_optimization(name))
        assert WorkerRequest.from_json(optimized.to_json())==optimized


@pytest.mark.parametrize('kwargs',[dict(report_reserve_calls=0),dict(report_reserve_calls=3),dict(version='untrusted')])
def test_runtime_optimization_is_typed_and_bounded(kwargs):
    with pytest.raises((ValueError,TypeError)):RuntimeOptimization(**kwargs)


def test_unknown_flags_cannot_expand_worker_permissions(tmp_path):
    old=request(tmp_path);wire=old.to_dict();wire['optimization']={'schema_version':1,'allow_shell':True}
    with pytest.raises(ValueError):WorkerRequest.from_dict(wire)
    assert runtime_optimization('fixed-v1') is None
    with pytest.raises(ValueError):runtime_optimization('arbitrary')


@pytest.mark.parametrize('kwargs',[dict(max_revisions=3),dict(max_expert_retries=5),dict(extra_tool_calls=5)])
def test_controller_remedy_policy_cannot_be_unbounded(kwargs):
    with pytest.raises(ValueError):RemediationPolicy(**kwargs)


def test_new_policy_and_role_caps_bind_durable_authority(tmp_path):
    _,cfg,_=setup_graph(tmp_path)
    old=authority(cfg)
    templates={k:replace(t,config=replace(t.config,optimization=RuntimeOptimization(compact_reports=True))) for k,t in cfg.templates.items()}
    assert authority(replace(cfg,templates=templates))!=old
    assert authority(replace(cfg,role_overrides=(RoleBudgetOverride(role='log_analysis',max_model_calls=8,max_tool_calls=10),)))!=old
    assert authority(cfg)==old
