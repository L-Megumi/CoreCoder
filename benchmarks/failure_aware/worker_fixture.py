"""Tool-aware ScriptedLLM fixture, isolated in the actual Worker process.

Its conditional responses deliberately model finalization, short-report and
public-feedback behaviors. These are test assumptions, NOT measured LLM gains.
Reference actions remain fixture-owned and this launcher rejects live providers.
"""

from dataclasses import replace
from pathlib import Path
import json,re,sys
from corecoder.llm import LLMResponse
from researchcoder.runtime.worker import ProtocolScript


class FixtureModel(ProtocolScript):
    def __init__(self,model,request,scenario):
        super().__init__(model);self.request=request;self.scenario=scenario
        self.report=next((r for r in reversed(self._turns) if r.content.lstrip().startswith('{') and 'historical_failure' in r.content),None)
        self.seen_edit=False

    def current_reads(self,messages):
        calls={};sources={}
        for m in messages:
            for c in m.get('tool_calls',[]):
                if c['function']['name']=='read_file':calls[c['id']]=json.loads(c['function']['arguments'])['file_path']
            if m.get('role')=='tool' and m.get('tool_call_id') in calls:
                text=m.get('content','')
                if text.startswith('Error'):continue
                text=re.sub(r'^\[Unchanged read reused[^\n]*\n','',text)
                sources[Path(calls[m['tool_call_id']]).name]=re.sub(r'(?m)^\d+\t','',text)
        return sources

    def chat(self,messages,tools=None,on_token=None):
        if self.scenario=='unknown_usage':
            from openai import APIConnectionError
            import httpx2
            raise APIConnectionError(request=httpx2.Request('POST','https://offline.invalid'))
        compact=any('Compact diagnosis output policy v1' in str(m.get('content','')) for m in messages if m.get('role')=='system')
        if (self.request.spec.kind=='diagnose' and not tools and self.report is not None
                and self._turns and self._turns[0].tool_calls):
            self._turns=[self.report]
        if not self._turns and self.report is not None:self._turns=[self.report]
        if self._turns and self._turns[0].content=='__FAKE_VERBOSE_TRUNCATION__':
            if compact:
                while self._turns and self._turns[0].content=='__FAKE_VERBOSE_TRUNCATION__':self._turns.pop(0)
            else:
                self._turns.pop(0)
                return LLMResponse(content='{"summary": "unfinished fixture',prompt_tokens=100,completion_tokens=2048,usage_complete=True,finish_reason='length')
        if self.request.spec.kind in ('repair','coding') and self._turns:
            if self.scenario=='unknown_after_edit' and self.seen_edit and not self._turns[0].tool_calls:
                from openai import APIConnectionError
                import httpx2
                raise APIConnectionError(request=httpx2.Request('POST','https://offline.invalid'))
            if self.scenario=='ownership' and self.request.spec.kind=='repair':
                sources=self.current_reads(messages)
                while self._turns and self._turns[0].tool_calls:
                    turn=self._turns[0];updated=[]
                    for c in turn.tool_calls:
                        args=dict(c.arguments)
                        if c.name=='edit_file':
                            name=Path(args['file_path']).name
                            if self.request.spec.task_id.startswith('r1.') and name in sources:
                                args['old_string']=sources[name]+('\n' if args['old_string'].endswith('\n') and not sources[name].endswith('\n') else '')
                            if name=='settings.py' and not self.request.spec.task_id.startswith('r1.'):
                                args['new_string']='import torch\n\ndef optimizer(model, lr):\n    return torch.optim.SGD(model.parameters(), lr=lr)\n'
                            if args['old_string']==args['new_string']:continue
                        updated.append(replace(c,arguments=args))
                    if updated:
                        self._turns[0]=replace(turn,tool_calls=updated);break
                    self._turns.pop(0)
        if self.request.spec.kind=='diagnose' and self._turns and not self._turns[0].tool_calls:
            try:report=json.loads(self._turns[0].content)
            except ValueError:report=None
            if report and self.scenario=='ownership' and self.request.spec.task_id.startswith('r1.'):
                summary='The authorized public component check failed exact optimizer ownership. Current candidate source was read; preserve intentional freezing and all public trainability flags.'
                report['summary']=summary;report['current_state']['assessment']=summary
                report['hypotheses'][0]['explanation']=summary
                self._turns[0]=replace(self._turns[0],content=json.dumps(report))
            if report and self.scenario=='invalid_citation':
                report['current_state']['observation_ids']=['forged-upstream-observation']
                self._turns[0]=replace(self._turns[0],content=json.dumps(report))
        response=super().chat(messages,tools,on_token)
        if any(c.name=='edit_file' for c in response.tool_calls):self.seen_edit=True
        return response


def main(argv=None):
    from researchcoder.runtime.protocol import WorkerRequest
    import researchcoder.runtime.worker as worker
    scenario,request_path,response_path=sys.argv[1:] if argv is None else argv
    request=WorkerRequest.from_json(Path(request_path).read_text())
    if request.model.provider!='scripted':raise ValueError('Fixture launcher cannot invoke a live provider')
    worker.ProtocolScript=lambda config:FixtureModel(config,request,scenario)
    return worker.main([request_path,response_path])


if __name__=='__main__':raise SystemExit(main())
