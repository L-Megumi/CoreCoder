from dataclasses import replace
import json
import pytest
from corecoder.diagnose import run_diagnosis
from corecoder.diagnostic_output import COMPACT_PROMPT_SHA256
from corecoder.llm import LLMResponse, LLM
from researchcoder.runtime.worker import ProtocolScript
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, RuntimeLimits
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.budget import BudgetLedger, BudgetedModel, RuntimeHooks
from researchcoder.runtime.tracing import TraceSink
from tests.test_orchestration_static import call, report_turn


def diagnose(tmp_path, *, reports=None, reasons=None, compact=True, limit=6):
    project=tmp_path/'project';project.mkdir();(project/'training.py').write_text('loss = 1\n')
    log=project/'run.log';log.write_text('WARNING: missing backbone gradients\n')
    turns=(call('extract','extract_log_evidence'),call('read','read_project_file',{'path':'training.py'}))
    turns+=tuple(reports or (report_turn(),))
    turns=tuple(replace(t,prompt_tokens=100,completion_tokens=10) for t in turns)
    class Script(ProtocolScript):
        def __init__(self):super().__init__(ModelSpec(turns=turns));self.schemas=[];self.reasons=list(reasons or [None]*len(turns))
        def chat(self,messages,tools=None,on_token=None):
            self.schemas.append(tools)
            response=super().chat(messages,tools,on_token)
            return replace(response,finish_reason=self.reasons.pop(0))
    script=Script();trace=TraceSink('job','diagnose','attempt')
    ledger=BudgetLedger(RuntimeLimits(max_model_calls=limit,max_rounds=8,max_tool_calls=8),trace)
    hooks=RuntimeHooks(ledger,trace)
    if compact:hooks.optimization=RuntimeOptimization(compact_reports=True);ledger.optimization=hooks.optimization
    model=BudgetedModel(script,ledger,trace)
    result=run_diagnosis(log,project,model,max_rounds=8,max_tool_calls=8,execution_hooks=hooks,compact_report=compact)
    return result,ledger,script,trace


def test_compact_report_retains_original_strict_local_citations_and_sha(tmp_path):
    result,b,script,trace=diagnose(tmp_path)
    assert result['status']=='completed' and result['validation']['structure_and_references']
    assert result['output_policy']['prompt_sha256']==COMPACT_PROMPT_SHA256
    assert set(result['diagnosis']['current_state']['observation_ids'])<=set(result['current_observation_ids'])
    assert b.model_calls==3 and b.tool_calls==2 and not b.pending


def test_soft_finalization_withholds_tools_inside_same_model_allowance(tmp_path):
    result,b,script,trace=diagnose(tmp_path,limit=4)
    assert result['status']=='completed' and not script.schemas[-1]
    assert any(e['event_type']=='diagnosis.report_reserved' for e in trace.events)
    assert b.model_calls==3 and len(b.settlements)==3


@pytest.mark.parametrize('reason,expected',[('length','response_length_exceeded'),(None,'invalid_json')])
def test_truncated_json_is_never_auto_completed_and_repair_is_charged(tmp_path,reason,expected):
    invalid=ScriptedTurn(content='{"summary": "unfinished')
    result,b,_,_=diagnose(tmp_path,reports=(invalid,invalid),reasons=(None,None,reason,reason))
    assert result['status']=='incomplete' and result['diagnosis'] is None
    assert result['output_failure_kind']==expected
    assert len(result['report_attempts'])==2 and result['metrics']['report_repairs']==1
    assert all(r['response']==invalid.content for r in result['report_attempts'])
    assert b.model_calls==4 and b.prompt_tokens==400 and not b.pending


def test_model_confirmed_bounded_format_repair_preserves_first_failure(tmp_path):
    invalid=ScriptedTurn(content='{"summary": "unfinished')
    result,b,_,_=diagnose(tmp_path,reports=(invalid,report_turn()),reasons=(None,None,'length','stop'))
    assert result['status']=='completed' and result['output_failure_kind'] is None
    assert result['report_attempts'][0]['failure_kind']=='response_length_exceeded'
    assert result['report_attempts'][1]['validation_error'] is None and b.model_calls==4


def test_upstream_or_forged_id_still_rejected_in_compact_mode(tmp_path):
    value=json.loads(report_turn().content);value['current_state']['observation_ids']=['forged-upstream-id']
    invalid=ScriptedTurn(content=json.dumps(value))
    result,_,_,_=diagnose(tmp_path,reports=(invalid,invalid))
    assert result['diagnosis'] is None and result['output_failure_kind']=='invalid_report'
    assert 'unknown or invalid reference' in result['failure_reason']


def test_legacy_output_shape_and_failure_semantics_are_not_changed(tmp_path):
    result,_,_,_=diagnose(tmp_path,compact=False)
    assert result['status']=='completed'
    assert 'output_policy' not in result and 'output_failure_kind' not in result


def test_model_cannot_force_tools_in_reserved_finalization(tmp_path):
    result,b,_,_=diagnose(tmp_path,reports=(call('forbidden','read_project_file',{'path':'training.py'}),),limit=4)
    assert result['status']=='incomplete' and result['diagnosis'] is None
    assert result['output_failure_kind']=='invalid_report'
    assert b.model_calls==3 and b.tool_calls==2 and len(result['tool_events'])==2


def test_format_repair_cannot_invoke_a_tool(tmp_path):
    invalid=ScriptedTurn(content='{"summary": "unfinished')
    result,b,_,_=diagnose(tmp_path,reports=(invalid,call('forbidden','read_project_file',{'path':'training.py'})))
    assert result['status']=='incomplete' and result['diagnosis'] is None
    assert result['output_failure_kind']=='invalid_report'
    assert b.model_calls==4 and b.tool_calls==2 and result['report_attempts'][-1]['failure_kind']=='invalid_report'


def test_real_sdk_finish_reason_and_usage_survive_streaming(monkeypatch):
    import httpx2
    monkeypatch.setattr('socket.socket.connect',lambda *a,**k:pytest.fail('Offline only'))
    def handle(request):
        chunk=dict(id='fixture',object='chat.completion.chunk',created=1,model='fixture',choices=[dict(index=0,delta=dict(content='{"summary":"cut'),finish_reason='length')],usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30))
        return httpx2.Response(200,headers={'content-type':'text/event-stream'},content=('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode())
    with httpx2.Client(transport=httpx2.MockTransport(handle)) as client:
        llm=LLM('fixture','offline-placeholder',base_url='https://api.deepseek.com')
        llm.client=llm.client.with_options(http_client=client,max_retries=0)
        result=llm.chat([{'role':'user','content':'fixture'}])
    assert result.finish_reason=='length' and result.usage_complete and result.completion_tokens==20
