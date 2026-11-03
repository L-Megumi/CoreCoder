from dataclasses import replace
import copy,json
import pytest
from researchcoder.context.engine import EvidenceContextEngine
from researchcoder.runtime.worker import execute_request
from tests.test_coding_handoff_context import fixture

def projected(tmp_path):
    manager,_,runtime,io,envelope,context,_=fixture(tmp_path)
    context['excerpt_policy']=dict(version='bounded-excerpts-v1',prefix_chars=7)
    for row in context['observations']:row['excerpt']=row['excerpt'][:7]
    envelope['consultations'][0]['context_ref']=io.put_json('job',context).to_dict()
    policy=replace(manager.request.context_policy,coding_source_mode='observed-source-v2',
        repair_state_version='state-v1',handoff_view_version='bounded-excerpts-v1')
    request=replace(manager.request,context_policy=policy,evidence_context=json.dumps(envelope))
    return request,io,runtime,envelope,context

def test_bounded_projection_is_accepted_and_complete_originals_remain_restorable(tmp_path):
    request,io,runtime,_,context=projected(tmp_path)
    engine=EvidenceContextEngine(request,io,runtime.ledger,runtime.trace)
    assert len(engine.coding_handoff.observations)==len(context['observations'])
    assert all(len(row['record'])>0 for row in engine.coding_handoff.observations)
    assert any('return value + 0' in json.dumps(row['record']) for row in engine.coding_handoff.observations)

@pytest.mark.parametrize('change',['unapproved','version','length','prefix','producer','digest'])
def test_projection_never_authorizes_tampered_content_or_provenance(tmp_path,change):
    request,io,runtime,envelope,context=projected(tmp_path)
    if change=='unapproved':request=replace(request,context_policy=replace(request.context_policy,handoff_view_version='legacy-v1'))
    elif change=='version':context['excerpt_policy']['version']='accept-anything'
    elif change=='length':context['excerpt_policy']['prefix_chars']=1201
    elif change=='prefix':context['observations'][0]['excerpt']='forged '
    elif change=='producer':envelope['consultations'][0]['selected_producers']=['forged']
    elif change=='digest':context['observations'][0]['reference']['source_sha256']='e'*64
    envelope['consultations'][0]['context_ref']=io.put_json('job',context).to_dict()
    request=replace(request,evidence_context=json.dumps(envelope))
    with pytest.raises((ValueError,TypeError)):EvidenceContextEngine(request,io,runtime.ledger,runtime.trace)

def test_context_setup_failure_returns_complete_zero_request_accounting(tmp_path):
    request,io,_,envelope,context=projected(tmp_path)
    context['observations'][0]['excerpt']='forged '
    envelope['consultations'][0]['context_ref']=io.put_json('job',context).to_dict()
    request=replace(request,evidence_context=json.dumps(envelope))
    response=execute_request(request)
    assert response.result.status=='failed' and response.result.failure_kind=='input_invalid'
    budget=json.loads(response.budget_json)
    assert budget['model_calls']==budget['tool_calls']==budget['unknown_requests']==0
    assert budget['pending']=={} and budget['settlements']==[]
