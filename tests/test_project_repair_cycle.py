"""Bounded correction and phase controls over real isolated Workers; no paid API."""
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner, ProjectPolicy, recompute
from researchcoder.evaluation.projects.routing import ExpertPolicy, decide
from researchcoder.evaluation.projects.repair_cycle import task_limits, provider_accounting_unknown
from researchcoder.orchestration.budget import JobLimits
from researchcoder.runtime.protocol import RuntimeLimits, ModelSpec, ScriptedTurn
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.budget import BudgetExceeded, BudgetLedger, RuntimeHooks
from researchcoder.runtime.tracing import TraceSink
from tests.test_project_routing import features
from tests.test_project_runner import fixture, scripted, call, BAD, GOOD, PARTIAL, SECRET, requests
from tests.test_project_consultation import expert_model

VERSION='repair-cycle-v1'


def test_opt_in_contracts_preserve_legacy_json_and_reject_unbounded_modes():
    assert 'repair_cycle_version' not in ProjectPolicy().to_dict()
    assert 'coding_phase_version' not in RuntimeOptimization().to_dict()
    settings=ProjectPolicy(completion_version='explicit-v1',progress_version='progress-v1',repair_cycle_version=VERSION)
    assert ProjectPolicy.from_json(settings.to_json())==settings
    with pytest.raises(ValueError):replace(settings,max_feedback_rounds=2)
    with pytest.raises(ValueError):RuntimeOptimization(coding_phase_version=VERSION)


@pytest.mark.parametrize('remaining',[4,7,11])
def test_short_correction_uses_remaining_job_instead_of_full_writer_floor(remaining):
    observed=features(remaining_model_calls=remaining,remaining_seconds=100)
    observed=replace(observed,episodes_used=1)
    policy=ExpertPolicy(mode='public-failure-v1',writer_min_model_calls=12)
    assert decide(observed,policy).reason_codes==('writer_reserve_unavailable',)
    new=decide(observed,policy,repair_cycle_version=VERSION)
    assert new.action=='keep_single' and new.roles==() and new.reason_codes==('bounded_public_correction',)


@pytest.mark.parametrize('changes',[
    dict(usage_unknown=True),dict(budget_pending=True),dict(policy_intact=False),dict(source_intact=False),
    dict(cancelled=True),dict(remaining_retries=0),dict(remaining_model_calls=3),dict(remaining_seconds=69),
    dict(workflow_completed=False),dict(verification_status='unavailable'),dict(failure_kind='tool_error')])
def test_correction_never_bypasses_safety_or_unknown_state(changes):
    observed=replace(features(),episodes_used=1,**changes)
    assert decide(observed,ExpertPolicy(mode='public-failure-v1'),repair_cycle_version=VERSION).action=='stop'


def test_allocations_leave_runway_and_clip_correction_without_resetting_limits():
    cap=RuntimeLimits(max_model_calls=12,max_tool_calls=24,max_rounds=10,max_runtime_seconds=300,
                      max_total_tokens=200000,max_context_tokens=32000,max_output_tokens=2048)
    first=task_limits(cap,0)
    assert first.max_model_calls==8 and first.max_tool_calls==21 and first.max_total_tokens==180000
    next_cap=replace(cap,max_model_calls=7,max_total_tokens=90000,max_runtime_seconds=100)
    correction=task_limits(next_cap,1)
    assert correction.max_model_calls==6 and correction.max_total_tokens==90000 and correction.max_runtime_seconds==100
    with pytest.raises(BudgetExceeded):task_limits(replace(cap,max_model_calls=3),1)
    with pytest.raises(BudgetExceeded):task_limits(cap,2)


@pytest.mark.parametrize('requests_count,unknown',[(None,None),(None,0),(0,None),(-1,0),(True,0),(2,1)])
def test_missing_or_invalid_usage_does_not_crash_or_become_zero(requests_count,unknown):
    result=dict(model_requests=requests_count,unknown_requests=unknown,usage_complete=True)
    assert provider_accounting_unknown(result,[dict(status='known')])


def test_provider_accounting_preserves_known_calls_and_rejects_missing_transport():
    result=dict(model_requests=2,unknown_requests=0,usage_complete=True)
    assert not provider_accounting_unknown(result,[dict(status='known'),dict(status='known')])
    assert provider_accounting_unknown(result,[])
    assert provider_accounting_unknown(result,[dict(status='pending')])


def execute(tmp_path,*,first,feedback=None):
    source,task,checks,_=fixture(tmp_path)
    policy=ProjectPolicy(completion_version='explicit-v1',progress_version='progress-v1',repair_cycle_version=VERSION,
        task_limits=RuntimeLimits(max_model_calls=12,max_tool_calls=24,max_rounds=10,max_runtime_seconds=180,
            max_context_tokens=16000,max_output_tokens=512,max_total_tokens=200000),
        job_limits=JobLimits(max_model_calls=16,max_tool_calls=60,max_total_tokens=400000,max_runtime_seconds=600))
    target=tmp_path/'trial'
    result=ProjectEvaluationRunner().run(target,task,source,checks['public'],checks['f2p'],checks['p2p'],first,
        policy=policy,feedback_models=(feedback,) if feedback else (),
        expert_policy=ExpertPolicy(mode='public-failure-v1',initial_mode='code-first-v1',writer_min_model_calls=12),
        expert_models={'code_analysis':expert_model()},permission_recovery_mode='bounded-public-read-v1')
    io=ArtifactIO(target/'artifacts');raw=io.read_json(ArtifactRef.from_json((target/'raw-reference.json').read_text()))
    return result,raw,io,target


def extra_reads(model):
    return replace(model,turns=(model.turns[0],call('read_file',{'file_path':'.projectrc'},'config'),
        call('read_file',{'file_path':'LICENSE'},'license'),*model.turns[1:]))


def test_public_failure_corrected_by_second_worker_with_same_budget_and_no_oracle_leak(tmp_path):
    result,raw,io,target=execute(tmp_path,first=extra_reads(scripted(PARTIAL)),feedback=scripted(GOOD,PARTIAL))
    assert result['end_to_end_success'] and result['fail_to_pass'] and result['pass_to_pass']
    assert len(raw['consultations'])==1 and len(raw['rounds'])==2
    first,second=raw['rounds']
    assert first['result']['failure_kind']=='verification_failed' and first['public_receipt']['passed'] is False
    assert second['before']==first['after'] and first['result']['attempt_id']!=second['result']['attempt_id']
    assert all(r['result']['job_id']==raw['job_id'] for r in raw['rounds'])
    assert len({r['worker_pid'] for r in raw['records']})==3
    assert raw['routing_decisions'][-1]['decision']['reason_codes']==['bounded_public_correction']
    assert raw['budget']['charged']['model_calls']<=16 and not raw['budget']['pending']
    assert len(raw['budget']['settlements'])==3 and not any(s['usage_unknown'] for s in raw['budget']['settlements'].values())
    visible=json.dumps(requests(raw,io));assert SECRET not in visible and 'private-output' not in visible
    assert 'repair-cycle-v1' in visible and 'Previous public' in visible
    assert (target/'candidate/module.py').read_text()==GOOD and recompute(target)==result


def test_action_phase_keeps_tools_available_for_evidenced_edit(tmp_path):
    model=extra_reads(scripted())
    model=replace(model,turns=(*model.turns[:3],call('read_file',{'file_path':'module.py','offset':1,'limit':2},'decisive'),*model.turns[3:]))
    result,raw,io,_=execute(tmp_path,first=model)
    assert result['end_to_end_success']
    packets=requests(raw,io)
    phased=[p for p in packets if 'Controller execution phase: repair-cycle-v1' in json.dumps(p)]
    assert phased and any('edit_file' in json.dumps(p['request']['tools']) for p in phased)
    trace=io.read_json(ArtifactRef.from_dict(raw['records'][-1]['trace']))['events']
    assert any(e['event_type']=='coding.action_phase' for e in trace)


def test_denied_write_remains_stopped_and_cannot_use_correction(tmp_path):
    bad=replace(scripted(),turns=(call('edit_file',{'file_path':'LICENSE','old_string':'Fixture','new_string':'Changed'},'forbidden'),*scripted().turns))
    result,raw,_,target=execute(tmp_path,first=bad,feedback=scripted())
    assert not result['end_to_end_success'] and result['status']=='needs_review'
    assert len(raw['rounds'])==1 and not raw['final_checks']
    assert raw['permission_records'][0]['receipt']['tool_executed'] is False
    assert (target/'candidate/LICENSE').read_text()=='Fixture license\n'
