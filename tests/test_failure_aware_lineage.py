"""Actual Workers: expert remediation followed by public-check feedback."""
from dataclasses import replace
import json,os
from researchcoder.evaluation import runner
from researchcoder.evaluation.complexity.protocol import offline_config
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import ScriptedTurn,ScriptedCall
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from benchmarks.failure_aware.fixtures import decorate


def test_feedback_descends_from_actual_remediated_revision_without_budget_reset(tmp_path,monkeypatch):
    original=runner.build_workflow
    def build(*args,**kwargs):
        built=decorate(original(*args,**kwargs),'ownership')
        templates=dict(built['config'].templates);template=templates['log_analysis']
        turns=template.config.model.turns
        # One model response crosses the local tool cap before the soft report
        # reserve can act. The trusted retry cap allows these identical reads.
        burst=ScriptedTurn(calls=tuple(ScriptedCall(call_id='read-'+str(i),name='read_project_file',
            arguments_json=json.dumps({'path':'entry.py'})) for i in range(8)),prompt_tokens=100,completion_tokens=10)
        model=replace(template.config.model,turns=(turns[0],burst,turns[-1]))
        templates['log_analysis']=replace(template,config=replace(template.config,model=model))
        built['config']=replace(built['config'],templates=templates)
        return built
    monkeypatch.setattr(runner,'build_workflow',build)
    config=offline_config(case_ids=('cpx_04',),modes=('parallel',),harness_strategy='combined-v1',max_feedback_rounds=1,
        job_limits=JobLimits(max_model_calls=48,max_tool_calls=128,max_total_tokens=1000000,max_runtime_seconds=240))
    root=tmp_path/'evaluation'
    report=runner.EvaluationRunner().run(root,config,torch_python=os.environ['RESEARCHCODER_EXPERIMENT_PYTHON'])
    assert runner.recompute(root)==report
    io=ArtifactIO(root/'trials/trial-0000/artifacts')
    result=io.read_json(ArtifactRef.from_json((root/'records/trial-0000.json').read_text()))
    raw=io.read_json(ArtifactRef.from_dict(result['raw']))
    assert result['metrics']['end_to_end_success'],(result,raw)
    assert raw['harness_optimization']['remediation_count']==2
    assert [r['revision_id'] for r in raw['feedback_rounds']]==['r0.remedy1','r1.remedy1']
    assert [r['plan_generation'] for r in raw['feedback_rounds']]==[1,3]
    assert raw['feedback_rounds'][0]['verification']['status']=='failed'
    assert raw['feedback_rounds'][1]['verification']['status']=='passed'
    summaries=[io.read_json(ArtifactRef.from_dict(r)) for r in raw['harness_optimization']['remediation_refs']]
    plans=[io.read_json(ArtifactRef.from_dict(rev['summary_ref']))['plan'] for s in summaries for rev in s['revisions']]
    assert [p['generation'] for p in plans]==[0,1,2,3]
    assert [p['parent_revision_id'] for p in plans]==[None,'r0','r0.remedy1','r1']
    assert all(t['retry_of']=='r0.remedy1.'+t['task_id'] for t in plans[2]['tasks'])
    assert len({r['attempt_id'] for r in raw['records']})==len(raw['records'])
    assert raw['budget']['limits']==config.job_limits.to_dict() and not raw['budget']['pending']
    assert raw['budget']['charged']['model_calls']<=48
    assert raw['budget']['charged']['model_calls']==sum(r['budget']['model_calls'] for r in raw['records'])
    assert sum(t['failure_kind']=='local_budget_exhausted' for t in raw['task_results'])==2
    assert any(t['task_id']=='r0.remedy1.public_contract' and t['status']=='failed' for t in raw['task_results'])
