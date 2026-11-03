from dataclasses import replace
import json,os
import pytest
from researchcoder.evaluation import runner
from researchcoder.evaluation.complexity.protocol import offline_config
from researchcoder.evaluation.runner import EvaluationRunner,recompute
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.runtime.optimization import STRATEGIES
from researchcoder.orchestration.budget import JobLimits
from benchmarks.failure_aware.fixtures import decorate,SCENARIOS


def execute(tmp_path,monkeypatch,scenario,strategy,mode='parallel'):
    original=runner.build_workflow
    def build(*args,**kwargs):return decorate(original(*args,**kwargs),scenario)
    monkeypatch.setattr(runner,'build_workflow',build)
    config=offline_config(case_ids=(SCENARIOS[scenario],),modes=(mode,),harness_strategy=strategy,max_feedback_rounds=1,
        job_limits=JobLimits(max_model_calls=48,max_tool_calls=128,max_total_tokens=1000000,max_runtime_seconds=240))
    path=tmp_path/'evaluation'
    report=EvaluationRunner().run(path,config,torch_python=os.environ['RESEARCHCODER_EXPERIMENT_PYTHON'])
    assert recompute(path)==report
    io=ArtifactIO(path/'trials/trial-0000/artifacts')
    result=io.read_json(ArtifactRef.from_json((path/'records/trial-0000.json').read_text()))
    raw=io.read_json(ArtifactRef.from_dict(result['raw']))
    from benchmarks.failure_aware.run_offline import summarize
    saved=summarize(path)
    assert saved['metrics']==result['metrics'] and saved['budget']==raw['budget']
    assert config.job_limits.to_dict()==raw['budget']['limits'] and not raw['budget']['pending']
    assert raw['budget']['charged']['model_calls']==sum(v['charged']['model_calls'] for v in raw['budget']['settlements'].values())
    return result,raw,io,path


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_local_expert_quota_ablation_uses_real_workers_and_original_oracles(tmp_path,monkeypatch,strategy):
    result,raw,io,_=execute(tmp_path,monkeypatch,'role_quota',strategy)
    expected=strategy in ('budget-v1','compact-v1','combined-v1')
    assert result['metrics']['end_to_end_success'] is expected,(result,raw)
    if strategy=='budget-v1':
        info=raw['harness_optimization'];assert info['remediation_count']==1
        original=next(t for t in raw['task_results'] if t['task_id']=='r0.log')
        assert original['status']=='incomplete' and original['failure_kind']=='local_budget_exhausted'
        assert any(t['task_id']=='r0.remedy1.log' and t['status']=='succeeded' for t in raw['task_results'])
        assert len({r['attempt_id'] for r in raw['records']})==len(raw['records'])
        assert any(s['status']=='reused' and s['task_id']=='code' for s in io.read_json(ArtifactRef.from_dict(info['remediation_refs'][0]))['revisions'][1]['states'])
    if expected:assert result['metrics']['fail_to_pass'] and result['metrics']['pass_to_pass']


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_compact_format_ablation_keeps_strict_validator_and_usage_charge(tmp_path,monkeypatch,strategy):
    result,raw,io,_=execute(tmp_path,monkeypatch,'report_length',strategy)
    expected=strategy in ('compact-v1','combined-v1')
    assert result['metrics']['end_to_end_success'] is expected,(result,raw)
    assert result['metrics']['cost'] is None  # synthetic usage is never billed as real API cost
    if not expected:
        assert any(t['status']=='incomplete' for t in raw['task_results'])
        assert raw['budget']['charged']['model_calls']>0


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_public_constraint_ablation_feedback_is_public_and_budget_is_shared(tmp_path,monkeypatch,strategy):
    result,raw,io,path=execute(tmp_path,monkeypatch,'ownership',strategy)
    expected=strategy in ('constraints-v1','combined-v1')
    assert result['metrics']['end_to_end_success'] is expected,(result,raw)
    if expected:
        assert len(raw['feedback_rounds'])==2
        assert raw['feedback_rounds'][0]['verification']['spec']['oracle_profile']=='public-joint-training-ownership-v1'
        assert raw['feedback_rounds'][0]['verification']['status']=='failed'
        assert raw['feedback_rounds'][1]['verification']['status']=='passed'
        assert raw['feedback_rounds'][1]['budget']['charged']['model_calls']>raw['feedback_rounds'][0]['budget']['charged']['model_calls']
        assert any(t['task_id']=='r0.public_contract' and t['status']=='failed' for t in raw['task_results'])
    else:
        assert result['metrics']['fail_to_pass'] is True and result['metrics']['pass_to_pass'] is False
        assert len(raw['feedback_rounds'])==1


@pytest.mark.parametrize('scenario',['unknown_usage','unknown_after_edit','invalid_citation','permission_denied'])
@pytest.mark.parametrize('strategy',['fixed-v1','combined-v1'])
def test_negative_controls_are_not_replayed_or_promoted(tmp_path,monkeypatch,scenario,strategy):
    result,raw,io,_=execute(tmp_path,monkeypatch,scenario,strategy)
    assert result['status']!='succeeded' and not result['metrics']['end_to_end_success']
    assert raw['budget']['charged']['model_calls']>0 and len(raw['feedback_rounds'])==1
    if scenario.startswith('unknown'):
        assert any(s['usage_unknown'] for s in raw['budget']['settlements'].values())
        assert raw['budget']['charged']['total_tokens']>0 and result['metrics']['cost'] is None
    if scenario=='unknown_after_edit':
        assert result['metrics']['fail_to_pass'] and result['metrics']['pass_to_pass']
    if scenario=='permission_denied':assert raw['policy_violations']
    if strategy=='combined-v1':assert raw['harness_optimization']['remediation_count']==0


def test_combined_strategy_preserves_evidence_context_exchange(tmp_path,monkeypatch):
    result,raw,io,_=execute(tmp_path,monkeypatch,'role_quota','combined-v1',mode='evidence')
    assert result['status']=='succeeded'
    diag=raw['complexity_diagnostics']
    assert diag['recoverable_evidence_refs']==diag['evidence_refs'] and diag['evidence_refs']>0
