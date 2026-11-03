"""Typed failure metadata and actionable citations without relaxed authority."""
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from researchcoder.context.expert_followup import TOOL
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.oracle_receipt import adapt_original_result, PUBLIC_EXECUTION_CONTRACT
from tests.test_coding_context import run_tool
from tests.test_m11_repair_continuity import enriched_handoff,bind,state_policy,state_record,project
from tests.test_m12_expert_followup import decision


def setup(tmp_path):
    old,_,_,io,_=enriched_handoff(tmp_path)
    manager,agent,hooks,io=bind(replace(old.request,context_policy=replace(state_policy(),expert_followup_version='followup-v3')),io)
    hooks.allowed_tools=frozenset((*hooks.allowed_tools,TOOL))
    return manager,agent,hooks,io


@pytest.mark.parametrize('invalid',['rendered','upstream','unknown','stale','no_source'])
def test_invalid_citation_is_not_imported_and_feedback_contains_only_current_local_ids(tmp_path,invalid):
    manager,agent,hooks,io=setup(tmp_path)
    _,read=run_tool(manager,agent,'read_file',dict(file_path='module.py'))
    binding=manager.coding_sources.entries[-1]
    ref=binding['rendered_document_id'] if invalid=='rendered' else manager.coding_handoff.expert_reports[0]['binding']['document_id'] if invalid=='upstream' else 'unknown-document'
    if invalid=='stale':
        run_tool(manager,agent,'edit_file',dict(file_path='module.py',old_string='return value + 0',new_string='return value + 1'),'edit')
        ref=binding['document_id']
    row=decision(manager,refs=[] if invalid=='no_source' else [ref])
    before=hooks.ledger.tool_calls
    output,_=run_tool(manager,agent,TOOL,dict(decisions=[row]),'bad')
    value=json.loads(output)
    assert value['failure_code']=='invalid_source_references' and value['batch_committed'] is False
    assert not manager.expert_followup.decisions and not manager.expert_followup.history
    assert hooks.ledger.tool_calls==before+1
    valid={b['document_id'] for b in manager.coding_sources.entries if manager.coding_sources.current_eligible(b)}
    assert {x['document_id'] for x in value['current_source_documents']}<=valid
    assert ref not in {x['document_id'] for x in value['current_source_documents']}
    if invalid=='stale':assert not value['current_source_documents']


def test_explicit_correction_uses_same_budget_and_persists_only_valid_batch(tmp_path):
    manager,agent,hooks,io=setup(tmp_path)
    _,read=run_tool(manager,agent,'read_file',dict(file_path='module.py'))
    output,bad=run_tool(manager,agent,TOOL,dict(decisions=[decision(manager,refs=['not-local'])]),'bad')
    feedback=json.loads(output)
    row=decision(manager,refs=[feedback['current_source_documents'][0]['document_id']])
    output,good=run_tool(manager,agent,TOOL,dict(decisions=[row]),'corrected')
    assert json.loads(output)['recorded']==[row['action_id']]
    assert hooks.ledger.tool_calls==3 and len(manager.expert_followup.history)==1
    assert manager.expert_followup.rejection_count==1
    actual=project(manager,agent,hooks,io,read+bad+good)
    item=next(i for i in state_record(actual)['expert_followup']['action_items'] if i['action_id']==row['action_id'])
    assert item['decision']['status']=='adopted' and item['decision']['verified'] is False


def test_rich_feedback_is_bounded_and_does_not_echo_arbitrary_untrusted_text(tmp_path):
    manager,agent,hooks,io=setup(tmp_path)
    run_tool(manager,agent,'read_file',dict(file_path='module.py'))
    hostile='arbitrary-private-path-or-instruction'*1000
    for index in range(3):
        output,_=run_tool(manager,agent,TOOL,dict(decisions=[decision(manager,refs=[hostile])]),'bad'+str(index))
        value=json.loads(output);assert hostile not in output and len(output)<6000
        if index<2:assert value['current_source_documents']
        else:assert value['current_source_documents']==[] and value['registered_action_ids']==[]
    assert hooks.ledger.tool_calls==4 and not manager.expert_followup.history


def test_original_failure_type_preserved_without_exception_bodies_or_private_rows(tmp_path):
    (tmp_path/'training.py').write_text('value=0\n')
    raw=json.dumps(dict(passed=False,failure_kind='ValueError'))
    value=adapt_original_result(raw,1,tmp_path,kind='basic',profile='discovery')
    assert value['failure_kind']=='ValueError' and value['passed'] is False
    assert value['checks']==[dict(check='original_oracle_aggregate',passed=False)]
    good=json.dumps(dict(passed=True,profile='holdout',rows=[dict(seed=12345,passed=True),dict(seed=45678,passed=True)]))
    value=adapt_original_result(good,0,tmp_path,kind='basic',profile='holdout')
    assert value['passed'] and value['failure_kind'] is None
    assert '12345' not in json.dumps(value) and 'rows' not in value
    assert 'getattr' in PUBLIC_EXECUTION_CONTRACT
    assert not any(word in PUBLIC_EXECUTION_CONTRACT for word in ('prepare_features','logsigmoid','cpx_','holdout'))


@pytest.mark.parametrize('raw,code',[
 ('{"passed":false,"passed":true}',1),
 ('{"passed":false,"failure_kind":{"body":"private"}}',1),
 ('{"passed":true,"failure_kind":"ValueError"}',0),
 ('{"passed":false,"failure_kind":"Error: private body"}',1),
 ('{"passed":false,"failure_kind":"ValueError"}',0)])
def test_receipt_does_not_convert_invalid_metadata_or_failure_into_pass(tmp_path,raw,code):
    (tmp_path/'training.py').write_text('value=0\n')
    with pytest.raises(ValueError):adapt_original_result(raw,code,tmp_path,kind='basic',profile='discovery')


def test_real_worker_rejects_then_accepts_explicit_correction(tmp_path):
    from tests.test_project_runner import fixture,policy,scripted,call,requests,SECRET
    from tests.test_project_consultation import expert_model
    from tests.test_m10_evidence_ready import LIMITS
    from tests.test_m9_code_first import specialist_requests
    from researchcoder.evaluation.projects.routing import ExpertPolicy
    from researchcoder.evaluation.projects.handoff import HandoffPolicy
    from researchcoder.evaluation.projects.runner import ProjectEvaluationRunner,recompute
    from researchcoder.orchestration.budget import JobLimits
    source,task,checks,_=fixture(tmp_path)
    settings=policy(completion_version='explicit-v1',progress_version='progress-v1',max_feedback_rounds=1,
        task_limits=replace(policy().task_limits,max_model_calls=12,max_rounds=10,max_tool_calls=24,
            max_total_tokens=200000,max_output_tokens=2048),
        job_limits=JobLimits(max_model_calls=24,max_tool_calls=60,max_total_tokens=400000,max_runtime_seconds=180))
    writer=scripted()
    # Test-only provider resolves dynamically generated IDs from its actual input.
    writer=replace(writer,turns=(writer.turns[0],call(TOOL,{'test_invalid':True},'bad-decision'),call(TOOL,{'test_correction':True},'correct-decision'),*writer.turns[1:]))
    directory=tmp_path/'trial'
    result=ProjectEvaluationRunner().run(directory,task,source,checks['public'],checks['f2p'],checks['p2p'],writer,
        policy=settings,expert_policy=ExpertPolicy(mode='public-failure-v1',initial_mode='code-first-v1',
            report_version='readiness-v2',assignment_version='focused-v1',expert_limits=LIMITS,writer_min_model_calls=12,
            schedule_version='serial-parallel-v2',max_workers=1),expert_models={'code_analysis':expert_model()},
        writer_context_policy=replace(state_policy(),expert_followup_version='followup-v3'),
        worker_command=(sys.executable,'-B',str(Path(__file__).resolve()),'--worker'),
        handoff_policy=HandoffPolicy(mode='local-stop-v3',closure_version='reserve-incomplete-v1',request_version='explicit-request-v1'))
    io=__import__('researchcoder.adapters.artifacts',fromlist=['ArtifactIO']).ArtifactIO(directory/'artifacts')
    raw=io.read_json(ArtifactRef.from_json((directory/'raw-reference.json').read_text()))
    assert result['end_to_end_success'] and result['fail_to_pass'] and result['pass_to_pass'],raw
    actual=requests(raw,io)
    states=[state_record(q['request']['messages']) for q in actual]
    assert len(states)==5
    assert any(x['decision'] and x['decision']['status']=='adopted' for x in states[-1]['expert_followup']['action_items'])
    assert 'focused-v1' in json.dumps(specialist_requests(raw,io))
    assert SECRET not in json.dumps(actual+specialist_requests(raw,io))
    assert raw['budget']['charged']['model_calls']==8 and not raw['budget']['pending']
    assert recompute(directory)==result


if __name__=='__main__' and sys.argv[1:2]==['--worker']:
    from researchcoder.runtime.worker import ProtocolScript,main
    original=ProtocolScript.chat
    def scripted_chat(self,messages,tools=None,on_token=None):
        response=original(self,messages,tools,on_token)
        for call in response.tool_calls:
            if call.name==TOOL and call.arguments in ({'test_invalid':True},{'test_correction':True}):
                state=state_record(messages)
                if call.arguments=={'test_invalid':True}:
                    refs=['not-a-source-document']
                else:
                    feedback=[]
                    for message in messages:
                        if message.get('role')!='tool':continue
                        try:value=json.loads(message.get('content',''))
                        except (ValueError,TypeError):continue
                        if isinstance(value,dict) and value.get('error')=='expert_decision_rejected':feedback.append(value)
                    assert feedback and feedback[-1]['batch_committed'] is False
                    refs=[feedback[-1]['current_source_documents'][0]['document_id']]
                call.arguments.clear()
                call.arguments['decisions']=[dict(action_id=state['expert_followup']['action_items'][0]['action_id'],
                    status='adopted',reason='Explicit correction using returned canonical source IDs.',source_document_ids=refs)]
        return response
    ProtocolScript.chat=scripted_chat
    raise SystemExit(main(sys.argv[2:]))
