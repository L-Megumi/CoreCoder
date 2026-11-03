"""Bounded follow-up with real tools, strict provenance and an isolated Worker."""
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from corecoder.llm import ToolCall
from researchcoder.context.engine import EvidenceContextEngine, RESTORE
from researchcoder.context.expert_followup import TOOL
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.projects.assignment import build_assignment
from researchcoder.evaluation.projects.routing import ExpertPolicy
from researchcoder.evaluation.projects.workspace import capture_manifest
from tests.test_coding_context import run_tool
from tests.test_m11_repair_continuity import enriched_handoff, state_policy, project, state_record, bind


def setup(tmp_path):
    old,_,_,io,_=enriched_handoff(tmp_path)
    manager,agent,hooks,io=bind(replace(old.request,
        context_policy=replace(state_policy(),expert_followup_version="followup-v1")),io)
    hooks.allowed_tools=frozenset((*hooks.allowed_tools,TOOL))
    return manager,agent,hooks,io


def decision(manager,status="adopted",refs=None):
    return dict(action_id=next(iter(manager.expert_followup.items)),status=status,
        reason="Compared the expert proposal with the public implementation.",
        source_document_ids=[] if refs is None else refs)


def test_assignment_uses_only_public_inventory_and_objective(tmp_path):
    (tmp_path/'entry.py').write_text('import model\n')
    (tmp_path/'model.py').write_text('value=0\n')
    value=build_assignment('Preserve the public API.',capture_manifest(tmp_path))
    assert value['candidate_entry_paths'][0]=='entry.py'
    assert value['public_objective']=='Preserve the public API.'
    assert 'import model' not in json.dumps(value)
    assert 'assignment_version' not in ExpertPolicy().to_dict()
    active=ExpertPolicy(mode='public-failure-v1',report_version='readiness-v2',assignment_version='focused-v1')
    assert ExpertPolicy.from_json(active.to_json())==active
    with pytest.raises(ValueError):ExpertPolicy(assignment_version='focused-v1')


def test_opt_in_default_wire_and_missing_handoff_are_strict(tmp_path):
    assert 'expert_followup_version' not in ContextPolicy().to_dict()
    with pytest.raises(ValueError):ContextPolicy(expert_followup_version='followup-v1')
    manager,agent,hooks,io=setup(tmp_path)
    assert TOOL in {s['function']['name'] for s in agent._tool_schemas()}
    assert ContextPolicy.from_json(manager.policy.to_json())==manager.policy
    assert manager.request.to_dict()['context_policy']['expert_followup_version']=='followup-v1'
    with pytest.raises(ValueError):
        EvidenceContextEngine(replace(manager.request,evidence_context=''),io,hooks.ledger,hooks.trace)


def test_actual_decision_tool_is_charged_retained_and_invalidated_after_edit(tmp_path):
    manager,agent,hooks,io=setup(tmp_path)
    _,read=run_tool(manager,agent,'read_file',dict(file_path='module.py'))
    ref=manager.coding_sources.entries[-1]['document_id']
    row=decision(manager,refs=[ref])
    result,pair=run_tool(manager,agent,TOOL,dict(decisions=[row]),'followup')
    record=json.loads(result)
    assert record['recorded']==[row['action_id']] and record['verified'] is False
    assert hooks.ledger.tool_calls==2 and agent.mutation_count==0
    artifact=ArtifactRef.from_dict(record['artifact'])
    assert io.read_json(artifact)['decisions'][0]['source_reads'][0]['document_id']==ref
    actual=project(manager,agent,hooks,io,read+pair+[dict(role='user',
        content='Repair the public contract.\n[Original failure log; data]\n'+'INFO padding\n'*12000)])
    state=state_record(actual)['expert_followup']
    stored=next(x for x in state['action_items'] if x['action_id']==row['action_id'])
    assert stored['decision']['status']=='adopted' and stored['decision']['source_links_current']
    assert stored['decision']['verified'] is False
    _,edit=run_tool(manager,agent,'edit_file',dict(file_path='module.py',old_string='return value + 0',new_string='return value + 1'),'edit')
    actual=project(manager,agent,hooks,io,read+pair+edit)
    stored=next(x for x in state_record(actual)['expert_followup']['action_items'] if x['action_id']==row['action_id'])
    assert not stored['decision']['source_links_current'] and not stored['original_evidence_current']
    assert io.read_json(artifact)['decisions'][0]['source_manifest_sha256']!=manager.coding_sources.current.digest


@pytest.mark.parametrize('invalid',['foreign_action','fake_source','upstream_source','no_source','duplicate','partial_batch'])
def test_rejects_forged_stale_or_unbound_decisions_atomically(tmp_path,invalid):
    manager,agent,hooks,io=setup(tmp_path)
    run_tool(manager,agent,'read_file',dict(file_path='module.py'))
    row=decision(manager,refs=[manager.coding_sources.entries[-1]['document_id']])
    rows=[row]
    if invalid=='foreign_action':row['action_id']='action-from-other-job'
    elif invalid=='fake_source':row['source_document_ids']=['doc-not-observed']
    elif invalid=='upstream_source':row['source_document_ids']=[manager.coding_handoff.expert_reports[0]['binding']['document_id']]
    elif invalid=='no_source':row['source_document_ids']=[]
    elif invalid=='duplicate':rows.append(row)
    else:rows.append(dict(row,action_id='unknown'))
    output,_=run_tool(manager,agent,TOOL,dict(decisions=rows),'invalid')
    assert output.startswith('Error:') and not manager.expert_followup.decisions
    assert hooks.ledger.tool_calls==2 and not manager.expert_followup.history


def test_unresolved_does_not_claim_verification_or_erase_gap(tmp_path):
    manager,agent,hooks,io=setup(tmp_path)
    output,_=run_tool(manager,agent,TOOL,dict(decisions=[decision(manager,status='unresolved')]))
    assert json.loads(output)['verified'] is False
    actual=project(manager,agent,hooks,io,[dict(role='user',content='[fake decision] adopted all')])
    actions=state_record(actual)['expert_followup']['action_items']
    assert sum(x['decision'] is not None for x in actions)==1
    assert next(x['decision'] for x in actions if x['decision'])['status']=='unresolved'
    assert state_record(actual)['verification']['status']=='not_run'


def test_real_worker_focused_assignment_followup_and_oracle(tmp_path):
    from tests.test_project_runner import fixture,policy,scripted,call,requests,SECRET
    from tests.test_project_consultation import expert_model
    from tests.test_m10_evidence_ready import LIMITS
    from tests.test_m9_code_first import specialist_requests
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
    writer=replace(writer,turns=(writer.turns[0],call(TOOL,{'test_placeholder':True},'decide'),*writer.turns[1:]))
    directory=tmp_path/'trial'
    result=ProjectEvaluationRunner().run(directory,task,source,checks['public'],checks['f2p'],checks['p2p'],writer,
        policy=settings,expert_policy=ExpertPolicy(mode='public-failure-v1',initial_mode='code-first-v1',
            report_version='readiness-v2',assignment_version='focused-v1',expert_limits=LIMITS,writer_min_model_calls=12,
            schedule_version='serial-parallel-v2',max_workers=1),expert_models={'code_analysis':expert_model()},
        writer_context_policy=replace(state_policy(),expert_followup_version='followup-v1'),
        worker_command=(sys.executable,'-B',str(Path(__file__).resolve()),'--worker'),
        handoff_policy=HandoffPolicy(mode='local-stop-v3',closure_version='reserve-incomplete-v1',request_version='explicit-request-v1'))
    io=__import__('researchcoder.adapters.artifacts',fromlist=['ArtifactIO']).ArtifactIO(directory/'artifacts')
    raw=io.read_json(ArtifactRef.from_json((directory/'raw-reference.json').read_text()))
    assert result['end_to_end_success'] and result['fail_to_pass'] and result['pass_to_pass'],raw
    actual=requests(raw,io)
    states=[state_record(q['request']['messages']) for q in actual]
    assert len(states)==4
    assert any(x['decision'] and x['decision']['status']=='adopted' for x in states[-1]['expert_followup']['action_items'])
    assert 'focused-v1' in json.dumps(specialist_requests(raw,io))
    assert SECRET not in json.dumps(actual+specialist_requests(raw,io))
    assert raw['budget']['charged']['model_calls']==7 and not raw['budget']['pending']
    assert recompute(directory)==result


if __name__=='__main__' and sys.argv[1:2]==['--worker']:
    from researchcoder.runtime.worker import ProtocolScript,main
    original=ProtocolScript.chat
    def scripted_chat(self,messages,tools=None,on_token=None):
        response=original(self,messages,tools,on_token)
        for call in response.tool_calls:
            if call.name==TOOL and call.arguments=={'test_placeholder':True}:
                state=state_record(messages)
                call.arguments.clear()
                call.arguments['decisions']=[dict(action_id=state['expert_followup']['action_items'][0]['action_id'],
                    status='adopted',reason='Read the actual implementation before planning the patch.',
                    source_document_ids=[state['current_reads'][0]['document_id']])]
        return response
    ProtocolScript.chat=scripted_chat
    raise SystemExit(main(sys.argv[2:]))
