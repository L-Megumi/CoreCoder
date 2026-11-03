"""Real diagnostic tools and a bounded report correction, never synthetic success."""
from dataclasses import replace
import json
import pytest
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.evaluation.projects.routing import ExpertPolicy,STREAMING_EXPERT_LIMITS
from researchcoder.context.policy import ContextPolicy
from researchcoder.runtime.protocol import ScriptedTurn
from tests.test_m10_evidence_ready import run_diagnostic,search_turn,report
from tests.test_orchestration_static import call

def turns():
    return (call('log','extract_log_evidence'),search_turn(),
        call('read','read_project_file',{'path':'training.py'}),
        call('read2','read_project_file',{'path':'training.py'}))

def test_v3_uses_two_existing_slots_and_v2_remains_strict():
    with pytest.raises(ValueError):
        RuntimeOptimization(compact_reports=True,report_reserve_calls=2,version='evidence-ready-v2')
    opt=RuntimeOptimization(compact_reports=True,report_reserve_calls=2,version='evidence-ready-v3')
    assert RuntimeOptimization.from_json(opt.to_json())==opt
    with pytest.raises(ValueError):replace(opt,report_reserve_calls=1)
    p=ExpertPolicy(mode='public-failure-v1',report_version='readiness-v3',
        latency_profile='streaming-v1',expert_limits=STREAMING_EXPERT_LIMITS)
    assert ExpertPolicy.from_json(p.to_json())==p

def test_real_tools_invalid_first_report_corrected_within_six_requests(tmp_path):
    broken=json.loads(report().content)
    broken['current_state']['observation_ids']=['obs_not_in_this_task']
    result,ledger,model,trace=run_diagnostic(tmp_path,
        (*turns(),ScriptedTurn(content=json.dumps(broken)),report()),version='evidence-ready-v3')
    assert result['status']=='completed',result
    assert result['validation']['structure_and_references']
    assert ledger.model_calls==6 and ledger.tool_calls==4 and not ledger.pending
    assert ledger.unknown_requests==0 and len(ledger.settlements)==6
    assert model.requests[3]['tools'] and model.requests[4]['tools']==[]
    assert not model.requests[5]['tools']
    assert result['metrics']['report_repairs']==1
    attempts=result['report_attempts']
    assert len(attempts)==2 and 'obs_not_in_this_task' in attempts[0]['response']
    assert attempts[0]['validation_error'] and attempts[1]['validation_error'] is None
    assert result['output_policy']['version']=='evidence-ready-v3'

def test_two_invalid_reports_remain_incomplete_and_no_seventh_request(tmp_path):
    broken=ScriptedTurn(content='{truncated JSON')
    result,ledger,model,_=run_diagnostic(tmp_path,
        (*turns(),broken,broken,report()),version='evidence-ready-v3')
    assert result['status']=='incomplete' and result['diagnosis'] is None
    assert ledger.model_calls==len(model.requests)==6
    assert len(result['report_attempts'])==2

def test_unknown_usage_does_not_fund_report_correction(tmp_path):
    result,ledger,model,_=run_diagnostic(tmp_path,
        (*turns(),ScriptedTurn(content='{invalid'),report()),
        version='evidence-ready-v3',unknown_at=5)
    assert result['status']=='incomplete' and result['diagnosis'] is None
    assert ledger.model_calls==len(model.requests)==5 and ledger.unknown_requests==1

def test_focused_writer_guidance_is_opt_in_and_requires_real_repair_context():
    old=ContextPolicy()
    assert 'writer_guidance_version' not in old.to_dict()
    with pytest.raises(ValueError):replace(old,writer_guidance_version='focused-repair-v1')
    new=ContextPolicy(role='repair',allow_recovery=True,coding_source_mode='observed-source-v2',
        repair_state_version='state-v1',writer_guidance_version='focused-repair-v1')
    assert ContextPolicy.from_json(new.to_json())==new
    from researchcoder.context.repair_state import GUIDANCE_SHA256,FOCUSED_GUIDANCE_SHA256
    assert GUIDANCE_SHA256!=FOCUSED_GUIDANCE_SHA256
