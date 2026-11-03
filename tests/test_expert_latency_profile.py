"""Versioned latency admission, real Worker deadlines, and strict stop controls."""
from dataclasses import replace
import hashlib,json,sys
import pytest

from researchcoder.evaluation.projects.routing import (
    ExpertPolicy, READINESS_EXPERT_LIMITS, STREAMING_EXPERT_LIMITS,
    decide, extract_public_features,
)

def policy():
    return ExpertPolicy(mode='public-failure-v1',initial_mode='code-first-v1',
        report_version='readiness-v2',latency_profile='streaming-v1',
        expert_limits=STREAMING_EXPERT_LIMITS,writer_min_seconds=90,
        writer_min_model_calls=12,schedule_version='serial-parallel-v2',max_workers=1)

def features(**changes):
    data=dict(public_log='AssertionError: public component contract',source_files=('module.py',),
        verification_status='failed',workflow_completed=True,failure_kind='verification_failed',
        remaining_model_calls=24,remaining_tool_calls=60,remaining_total_tokens=400000,
        remaining_seconds=600,remaining_retries=1)
    data.update(changes)
    return extract_public_features(**data)

def test_latency_is_explicit_and_old_wire_identity_is_preserved():
    old=ExpertPolicy(mode='public-failure-v1')
    assert 'latency_profile' not in old.to_dict()
    original=dict(schema_version=1,mode='public-failure-v1',max_episodes=1,
        expert_limits=old.expert_limits.to_dict(),writer_min_model_calls=3,
        writer_min_tool_calls=3,writer_min_total_tokens=20000,writer_min_seconds=15.0)
    assert old.to_dict()==original
    assert old.sha256==hashlib.sha256(json.dumps(original,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    assert ExpertPolicy.from_json(old.to_json())==old
    new=policy()
    assert ExpertPolicy.from_json(new.to_json())==new
    assert new.sha256!=replace(new,latency_profile='legacy-v1',expert_limits=READINESS_EXPERT_LIMITS).sha256

@pytest.mark.parametrize('changes',[
    {'latency_profile':'unbounded'},
    {'latency_profile':'legacy-v1'},
    {'expert_limits':replace(STREAMING_EXPERT_LIMITS,max_runtime_seconds=181)},
    {'expert_limits':replace(STREAMING_EXPERT_LIMITS,max_model_calls=7)},
    {'expert_limits':replace(STREAMING_EXPERT_LIMITS,max_total_tokens=80001)},
    {'report_version':'legacy-v1'},
    {'mode':'off','initial_mode':'off'},
])
def test_time_profile_cannot_raise_other_caps_or_change_legacy_implicitly(changes):
    with pytest.raises((ValueError,TypeError)):
        replace(policy(),**changes)

def test_admission_requires_same_job_time_and_keeps_call_reserves():
    p=policy()
    admitted=decide(features(),p,initial=True)
    assert admitted.action=='consult_experts'
    assert admitted.roles==('code_analysis',)
    too_short=decide(features(remaining_seconds=180),p,initial=True)
    assert too_short.action=='stop' and 'expert_allocation_unavailable' in too_short.reason_codes
    too_few=decide(features(remaining_model_calls=17),p,initial=True)
    assert too_few.action=='stop' and 'expert_allocation_unavailable' in too_few.reason_codes

@pytest.mark.parametrize('flag,reason',[
    ('usage_unknown','accounting_unknown'),('budget_pending','accounting_pending'),
    ('cancelled','interrupted'),('source_intact','source_changed'),('policy_intact','policy_violation'),
])
def test_network_latency_profile_does_not_bypass_safety_stops(flag,reason):
    value=False if flag.endswith('_intact') else True
    result=decide(features(**{flag:value}),policy(),initial=True)
    assert result.action=='stop' and result.reason_codes==(reason,)

def test_real_worker_short_deadline_interrupts_and_longer_deadline_finishes(tmp_path):
    from tests.test_runtime_worker import request,run
    command=(sys.executable,'-c',
        'import time;time.sleep(.4);from researchcoder.runtime.worker import main;raise SystemExit(main())')
    short=tmp_path/'short';short.mkdir()
    interrupted,_=run(request(short,seconds=.15),command=command)
    assert interrupted.reason=='timeout'
    long=tmp_path/'long';long.mkdir()
    completed,_=run(request(long,seconds=8),command=command)
    assert completed.reason is None and completed.response.result.status=='succeeded'

def test_streaming_profile_actual_expert_handoff_and_independent_checks(tmp_path):
    from tests.test_project_runner import fixture,policy as project_policy,scripted
    from tests.test_project_consultation import expert_model
    from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner,recompute
    from researchcoder.adapters.artifacts import ArtifactIO
    from researchcoder.contracts import ArtifactRef
    from researchcoder.orchestration.budget import JobLimits
    source,task,checks,_=fixture(tmp_path)
    settings=project_policy(completion_version='explicit-v1',progress_version='progress-v1',
        max_feedback_rounds=1,job_limits=JobLimits(max_model_calls=24,max_tool_calls=60,
            max_total_tokens=400000,max_runtime_seconds=600))
    directory=tmp_path/'trial'
    result=ProjectEvaluationRunner().run(directory,task,source,
        checks['public'],checks['f2p'],checks['p2p'],scripted(),policy=settings,
        expert_policy=policy(),expert_models={'code_analysis':expert_model()})
    assert result['end_to_end_success'] and result['fail_to_pass'] and result['pass_to_pass']
    assert recompute(directory)==result
    io=ArtifactIO(directory/'artifacts')
    raw=io.read_json(ArtifactRef.from_json((directory/'raw-reference.json').read_text()))
    assert len(raw['consultations'])==1 and raw['consultations'][0]['safe_to_continue']
    assert len({r['worker_pid'] for r in raw['records'] if r.get('worker_pid')})==2
    assert raw['budget']['charged']['model_calls']==6 and not raw['budget']['pending']
    assert all(not s['usage_unknown'] for s in raw['budget']['settlements'].values())
