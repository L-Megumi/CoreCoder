from dataclasses import replace
import json
from corecoder.llm import ScriptedLLM, LLMResponse
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import ArtifactRef, TaskInput, ExecutionPolicy, DiagnosticOptions
from researchcoder.runtime import ResearchHarness
from researchcoder.runtime.budget import BudgetLedger, BudgetedModel
from researchcoder.runtime.protocol import RuntimeLimits, ModelSpec, ScriptedTurn, WorkerRequest
from researchcoder.runtime.request_records import RequestRecorder
from researchcoder.runtime.tracing import TraceSink
from tests.test_runtime_harness import case


def test_request_capture_preserves_input_and_budget_but_redacts_local_artifact(tmp_path):
    io = ArtifactIO(tmp_path)
    trace = TraceSink("job", "task", "attempt")
    ledger = BudgetLedger(RuntimeLimits(), trace)
    recorder = RequestRecorder(io, trace, dict(model="fake", max_output_tokens=2048), secrets=("private-fixture-value",))

    class Capture(ScriptedLLM):
        def chat(self, messages, **kwargs):
            self.seen = messages
            return super().chat(messages, **kwargs)

    delegate = Capture([LLMResponse(content="ok", prompt_tokens=12, completion_tokens=2)])
    messages = [dict(role="user", content="private-fixture-value sk-fakeCredential12345")]
    BudgetedModel(delegate, ledger, trace, request_recorder=recorder).chat(messages)
    assert delegate.seen == messages and "private-fixture-value" in messages[0]["content"]
    raw = io.read_json(recorder.records[0])
    assert raw["redacted"] and "private-fixture-value" not in json.dumps(raw) and "sk-fakeCredential" not in json.dumps(raw)
    assert raw["request"]["messages"][0]["content"].count("REDACTED") == 2
    assert ledger.model_calls == 1 and ledger.prompt_tokens == 12
    assert all("private-fixture" not in json.dumps(e) for e in trace.events)


def test_opt_in_real_worker_records_and_legacy_wire_identity(tmp_path):
    spec, config, io, log = case(tmp_path)
    spec = replace(spec, kind="diagnose", task_id="capture", objective="Investigate", verification=None,
                   inputs=(TaskInput(role="raw_log", reference=log),), policy=ExecutionPolicy(),
                   diagnostic_options=DiagnosticOptions(completion_mode="off"))
    request = WorkerRequest(spec=spec, attempt_id="a", artifact_root=str(io.root), trace_path="/unused", parent_event_id="p")
    assert "record_requests" not in request.to_dict()
    assert WorkerRequest.from_json(request.to_json()) == request
    config = replace(config, record_requests=True, model=ModelSpec(turns=(ScriptedTurn(content="Unable to diagnose"),)))
    result = ResearchHarness().execute(spec, config)
    manifests = [io.read_json(ref) for ref in result.outputs]
    manifest = next(m for m in manifests if "request_records" in m)
    assert len(manifest["request_records"]) == result.usage.llm_calls == 1
    record = io.read_json(ArtifactRef.from_dict(manifest["request_records"][0]))
    assert (record["job_id"], record["task_id"], record["attempt_id"]) == (result.job_id, result.task_id, result.attempt_id)
    assert record["request"]["messages"][0]["role"] == "system"
    assert result.status == "incomplete"  # Captured requests never fabricate completion.


def test_fixed_script_citations_use_live_scope_not_compressor_or_upstream_ids():
    from researchcoder.runtime.worker import ProtocolScript
    scope = dict(scope="current_investigation_only", evidence_ids=["local-e"],
                 current_observation_ids=["local-o1", "local-o2"])
    messages = [dict(role="system", content="[Trusted live citation scope v1]\nRules\n" +
                     json.dumps(scope) + "\n[End citation scope]"),
                dict(role="tool", content=json.dumps(dict(
                    evidence=[dict(id="compressed-e")], observations=[dict(id="upstream-o"), dict(id="local-o2")])))]
    script = ProtocolScript(ModelSpec(turns=(ScriptedTurn(content="$E0 $O0"), ScriptedTurn(content="$E0 $O1"))))
    assert script.chat(messages).content == "local-e local-o2"
    messages[-1]["content"] = json.dumps(dict(observations=[dict(id="local-o1")]))
    assert script.chat(messages).content == "local-e local-o1"


def test_context_projection_keeps_live_citation_system_before_untrusted_hints(tmp_path):
    import copy
    from tests.test_context_runtime import engine
    from corecoder.diagnostic_tools import Investigation, investigation_tools
    from corecoder.diagnostic_policy import DiagnosticAgent
    from corecoder.citation_scope import citation_message
    manager, _ = engine(tmp_path)
    spec = manager.request.spec
    log = manager.store.io.put_bytes(spec.job_id, b"WARNING: missing gradients\n", media_type="text/plain")
    state = Investigation(manager.store.io.resolve(log), spec.workspace.root)
    state.execute("extract_log_evidence", {})
    agent = DiagnosticAgent(ScriptedLLM([]), state, tools=investigation_tools(state))
    agent._system += '\n用户提供的源码入口提示（路径数据）：["untrusted-hint.py"]'
    original = agent._full_messages()
    before = copy.deepcopy(original)
    view = manager.prepare(original, [])
    assert view[0]["role"] == "system" and citation_message(state) in view[0]["content"]
    assert "untrusted-hint.py" not in view[0]["content"]
    assert any("untrusted-hint.py" in str(m.get("content")) and m["role"] == "user" for m in view[1:])
    assert original == before
