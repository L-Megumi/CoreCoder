"""One isolated process executes one Adapter attempt. No repository access."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from openai import APIError

from corecoder.llm import ScriptedLLM, LLMResponse, ToolCall, LLM
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.adapters.corecoder import CoreCoderAdapter, AdapterContext
from researchcoder.contracts import TaskResult
from .protocol import WorkerRequest, WorkerResponse, PlanningTaskResult, CodingTaskResult
from .tracing import TraceSink
from .budget import BudgetLedger, BudgetedModel, RuntimeHooks


class ProtocolScript(ScriptedLLM):
    """Fixed ScriptedLLM actions with references filled only from actual tool replies."""

    def __init__(self, config):
        self._observed_ids = {"E": [], "O": []}
        super().__init__(
            [
                LLMResponse(
                    content=t.content,
                    tool_calls=[ToolCall(c.call_id, c.name, json.loads(c.arguments_json)) for c in t.calls],
                    prompt_tokens=t.prompt_tokens,
                    completion_tokens=t.completion_tokens,
                )
                for t in config.turns
            ],
            model=config.model,
        )

    def chat(self, messages, tools=None, on_token=None):
        if self._turns:
            evidence = []
            observations = []
            documents = []
            for message in messages:
                is_memory = message.get("role") == "user" and str(message.get("content", "")).startswith("[Evidence context:")
                if message.get("role") != "tool" and not is_memory:
                    continue
                try:
                    payload = message.get("content", "")
                    data = json.loads(payload.split("\n", 1)[1] if is_memory else payload)
                except (TypeError, ValueError):
                    continue
                if not isinstance(data, dict):
                    continue
                evidence.extend(r["id"] for r in data.get("evidence", []) if "id" in r)
                observations.extend(r["id"] for r in data.get("observations", []) if "id" in r)
                for record in data.get("records", []):
                    if record.get('document_id'):
                        documents.append(record['document_id'])
                    if record.get("id") and record.get("classification") == "observed_record":
                        (observations if record.get("kind") == "source" else evidence).append(record["id"])
            content = self._turns[0].content
            calls = self._turns[0].tool_calls
            serialized = [json.dumps(c.arguments) for c in calls]
            # The compressor can parse raw logs into its own cards. Those IDs,
            # just like upstream IDs, are not local Investigation citations.
            # This only binds fixed test placeholders; real provider output and
            # the production validator are never rewritten here.
            if messages and messages[0].get("role") == "system":
                system = str(messages[0].get("content", ""))
                marker = "[Trusted live citation scope v1]\n"
                if marker in system:
                    block = system.split(marker, 1)[1].split("\n[End citation scope]", 1)[0]
                    scope = json.loads(block.splitlines()[-1])
                    if scope.get("scope") != "current_investigation_only":
                        raise ValueError("Invalid fixed-script citation scope")
                    for prefix, values, key in (("E", evidence, "evidence_ids"),
                                                ("O", observations, "current_observation_ids")):
                        allowed = scope[key]
                        seen = self._observed_ids[prefix]
                        seen[:] = list(dict.fromkeys([i for i in seen + values if i in allowed]))
                        values[:] = seen + [i for i in allowed if i not in seen]
            for prefix, ids in [("E", list(dict.fromkeys(evidence))), ("O", list(dict.fromkeys(observations))), ("D",list(dict.fromkeys(documents)))]:
                for index, identity in reversed(list(enumerate(ids))):
                    content = content.replace(f"${prefix}{index}", identity)
                    serialized = [s.replace(f"${prefix}{index}", identity) for s in serialized]
            self._turns[0] = replace(self._turns[0], content=content,
                                   tool_calls=[replace(c,arguments=json.loads(s)) for c,s in zip(calls,serialized)])
        return super().chat(messages, tools, on_token)


def execute_request(request):
    spec = request.spec
    trace = TraceSink(spec.job_id, spec.task_id, request.attempt_id, path=request.trace_path, parent_id=request.parent_event_id)
    trace.emit("worker.ready", pid=os.getpid())
    ledger = BudgetLedger(request.limits, trace)
    ledger.optimization = request.optimization
    ledger.coding_continuation = request.coding_continuation
    hooks = RuntimeHooks(ledger, trace, request.diagnostic_tools if spec.kind == "diagnose" else spec.policy.allowed_tool_names)
    hooks.optimization = request.optimization
    io = ArtifactIO(request.artifact_root, namespace=request.artifact_namespace)
    if request.permission_recovery is not None:
        from .permissions import PermissionRecoverySession
        hooks.permission_recovery_session = PermissionRecoverySession(request.permission_recovery)
        hooks.permission_artifacts = io
        hooks.permission_workspace_revision = spec.workspace.source_manifest_sha256
    recorder = None
    if request.record_requests:
        from .request_records import RequestRecorder
        recorder = RequestRecorder(io, trace,
            dict(model=request.model.model, provider=request.model.provider,
                 max_output_tokens=request.limits.max_output_tokens,
                 options=request.model_options.to_dict() if request.model_options else None),
            secrets=(os.environ.get(request.model.credential_env),))
    engine = None
    if request.context_policy is not None:
        from researchcoder.context.engine import EvidenceContextEngine, RESTORE
        try:
            engine = EvidenceContextEngine(request, io, ledger, trace)
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            # Context admission happens before a model factory or tool dispatch.
            # Return the actual zero-request ledger instead of losing accounting.
            trace.emit("worker.context_rejected", status="error", failure_kind="input_invalid", reason=type(error).__name__)
            result_type = {'plan':PlanningTaskResult,'coding':CodingTaskResult}.get(spec.kind,TaskResult)
            result = result_type(job_id=spec.job_id,task_id=spec.task_id,attempt_id=request.attempt_id,
                kind=spec.kind,status="failed",failure_kind="input_invalid",
                usage=ledger.usage("live" if request.model.provider == "openai" else "injected"))
            trace.emit("worker.result", status=result.status, failure_kind=result.failure_kind)
            return WorkerResponse(result=result,budget_json=json.dumps(ledger.snapshot(),allow_nan=False))
        hooks.context_engine = engine
        if request.context_policy.allow_recovery:
            hooks.allowed_tools = frozenset(hooks.allowed_tools or ()) | {RESTORE}
        if engine.expert_followup is not None:
            from researchcoder.context.expert_followup import TOOL
            hooks.allowed_tools = frozenset(hooks.allowed_tools or ()) | {TOOL}
    if request.journal_endpoint is not None:
        from researchcoder.durable.journal import JournalClient
        hooks.effect_gateway = JournalClient(request.journal_endpoint)
        hooks.effect_artifacts = io
    result_class = {'plan':PlanningTaskResult,'coding':CodingTaskResult}.get(spec.kind,TaskResult)

    def factory():
        config = request.model
        if config.provider == "scripted":
            model = ProtocolScript(config)
        else:
            key = os.environ.get(config.credential_env)
            if not key:
                raise ValueError("Controller credential not configured")
            from .protocol import model_request_kwargs
            model = LLM(model=config.model, api_key=key, base_url=config.base_url, max_tokens=request.limits.max_output_tokens,
                        **model_request_kwargs(config, request.model_options))
            # SDK automatic attempts would bypass per-request accounting.
            model.client = model.client.with_options(max_retries=0, timeout=min(30, request.limits.max_runtime_seconds))
        return BudgetedModel(model, ledger, trace, system_instructions=request.system_instructions,
                             evidence_context=request.evidence_context, context_engine=engine, request_recorder=recorder)

    def protected_guard(*args):
        for protected in request.protected_files:
            if hashlib.sha256(Path(protected.path).read_bytes()).hexdigest() != protected.sha256:
                raise ValueError("Protected verifier input changed")

    try:
        protected_guard()
        if spec.kind == 'plan':
            from corecoder.agent import Agent
            from corecoder.workflow import workspace_revision
            from researchcoder.contracts import _unique

            if workspace_revision(spec.workspace.root) != spec.workspace.source_manifest_sha256:
                raise ValueError("Planning source version changed")
            inputs = [io.read_json(i.reference) for i in spec.inputs]
            agent = Agent(factory(),tools=[],max_rounds=request.limits.max_rounds,
                          max_context_tokens=request.limits.max_context_tokens)
            agent.execution_hooks = hooks
            text = agent.chat(spec.objective+'\n[Controller planning input; data]\n'+json.dumps(inputs))
            raw = json.loads(text,object_pairs_hook=_unique)
            if not isinstance(raw,dict):
                raise ValueError("Planning must produce a JSON object")
            artifact = io.put_json(spec.job_id,raw)
            result = PlanningTaskResult(job_id=spec.job_id,task_id=spec.task_id,attempt_id=request.attempt_id,
                                        status='succeeded',workflow_completed=True,outputs=(artifact,))
        elif spec.kind == 'coding':
            from researchcoder.orchestration.baseline import run_single_agent

            result = run_single_agent(spec,request,io,factory,hooks,protected_guard)
        else:
            result = CoreCoderAdapter().execute(spec,AdapterContext(
                attempt_id=request.attempt_id,
                artifacts=io,
                model_factory=factory,
                mode="live" if request.model.provider == "openai" else "injected",
                permitted_verifications=request.permitted_verifications,
                verification_guard=protected_guard,
                execution_hooks=hooks,
            ))
        protected_guard()
    except KeyboardInterrupt:
        result = result_class(
            job_id=spec.job_id,
            task_id=spec.task_id,
            attempt_id=request.attempt_id,
            kind=spec.kind,
            status="cancelled",
            failure_kind="interrupted",
        )
    except APIError as error:
        trace.emit("worker.execution_error", status="error", failure_kind="model_error", reason=type(error).__name__)
        result = result_class(
            job_id=spec.job_id,
            task_id=spec.task_id,
            attempt_id=request.attempt_id,
            kind=spec.kind,
            status="incomplete",
            failure_kind="model_error",
        )
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        trace.emit("worker.execution_error", status="error", failure_kind="tool_error", reason=type(error).__name__)
        result = result_class(
            job_id=spec.job_id,
            task_id=spec.task_id,
            attempt_id=request.attempt_id,
            kind=spec.kind,
            status="incomplete",
            failure_kind="tool_error",
        )
    if ledger.stop_kind:
        result = replace(
            result,
            status="failed" if ledger.stop_kind == "timeout" else "incomplete",
            failure_kind=ledger.stop_kind,
            workflow_completed=False,
        )
        if request.optimization is not None and spec.kind == 'diagnose' and ledger.stop_kind == 'budget_exhausted':
            result = replace(result, failure_kind='local_budget_exhausted')
    elif result.status == "incomplete" and result.failure_kind == "unknown" and ledger.last_failure_kind:
        result = replace(result, failure_kind=ledger.last_failure_kind)
    result = replace(result, usage=ledger.usage("live" if request.model.provider == "openai" else "injected"))
    if hooks.permission_refs:
        result = replace(result, outputs=result.outputs + tuple(hooks.permission_refs))
    if getattr(hooks, 'permission_partial_refs', ()):
        result = replace(result, outputs=result.outputs + tuple(hooks.permission_partial_refs))
    if getattr(hooks, 'coding_partial_refs', ()):
        result = replace(result, outputs=result.outputs + tuple(hooks.coding_partial_refs))
    if engine is not None:
        result = engine.finalize(result)
    if recorder is not None:
        result = replace(result, outputs=result.outputs + (recorder.manifest(),))
    trace.emit("worker.result", status=result.status, failure_kind=result.failure_kind)
    return WorkerResponse(result=result, budget_json=json.dumps(ledger.snapshot(), allow_nan=False))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        return 2
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        request = WorkerRequest.from_json(Path(argv[0]).read_text(encoding="utf-8"))
        response = execute_request(request)
        with open(argv[1], "x", encoding="utf-8") as stream:
            stream.write(response.to_json())
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        # No provider exception bodies, stdout payloads or environment dumps.
        print(type(error).__name__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
