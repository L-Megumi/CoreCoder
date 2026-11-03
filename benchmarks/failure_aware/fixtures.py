"""Same fixed response policy across A0..A4; known-action engineering only."""

from dataclasses import replace
from pathlib import Path
import json,sys
from researchcoder.runtime.protocol import ModelSpec,ScriptedTurn,ScriptedCall

SCENARIOS={'role_quota':'cpx_03','report_length':'cpx_03','ownership':'cpx_04',
           'unknown_usage':'cpx_03','unknown_after_edit':'cpx_03','invalid_citation':'cpx_03','permission_denied':'cpx_03'}


def decorate(built,scenario):
    if scenario not in SCENARIOS:raise ValueError('Unknown controlled fixture')
    command=(sys.executable,'-B',str(Path(__file__).with_name('worker_fixture.py')),scenario)
    def known(model):return replace(model,turns=tuple(replace(t,prompt_tokens=100,completion_tokens=10) for t in model.turns))
    if 'spec' in built:
        built['config']=replace(built['config'],model=known(built['config'].model),worker_command=command,record_requests=True)
        return built
    templates={}
    for key,t in built['config'].templates.items():
        model=t.config.model
        if t.role in ('log_analysis','code_analysis','diagnosis') and scenario=='ownership':
            # Identical fixed action policy for every ablation: batch independent
            # source reads, preserving every tool action and its local ID order.
            # This isolates the public-check mechanism within the unchanged
            # 48-request Job allowance; it does not grant retry budget.
            middle=model.turns[1:-1]
            assert all(r.calls and not r.content for r in middle)
            model=replace(model,turns=(model.turns[0],ScriptedTurn(calls=tuple(c for r in middle for c in r.calls)),model.turns[-1]))
        if t.role=='log_analysis' and scenario=='role_quota':
            turns=list(model.turns)
            burst=ScriptedTurn(calls=tuple(ScriptedCall(call_id='burst-'+str(i),name='read_project_file',
                arguments_json=json.dumps({'path':'pipeline.py' if i==0 else 'entry.py'})) for i in range(7)))
            last=ScriptedTurn(calls=(ScriptedCall(call_id='last',name='read_project_file',arguments_json='{"path":"entry.py"}'),))
            model=replace(model,turns=(turns[0],burst,last,turns[-1]))
        if t.role in ('log_analysis','code_analysis','diagnosis') and scenario=='report_length':
            marker=ScriptedTurn(content='__FAKE_VERBOSE_TRUNCATION__')
            model=replace(model,turns=model.turns[:-1]+(marker,marker,model.turns[-1]))
        if t.role=='log_analysis' and scenario=='permission_denied':
            model=replace(model,turns=(ScriptedTurn(calls=(ScriptedCall(call_id='denied',name='search_project_source',arguments_json='{"query":"loss"}'),)),))
        templates[key]=replace(t,config=replace(t.config,model=known(model),worker_command=command,record_requests=True))
    built['config']=replace(built['config'],templates=templates)
    return built
