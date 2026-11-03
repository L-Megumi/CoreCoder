"""Producer-scoped clues never silently become current-task report citations."""

import copy
from dataclasses import replace
import json
import os

import pytest
from corecoder.citation_scope import citation_scope, UPSTREAM_SCOPE
from corecoder.diagnose import run_diagnosis, validate_diagnosis
from corecoder.diagnostic_policy import DiagnosticAgent
from corecoder.diagnostic_tools import Investigation, investigation_tools
from corecoder.llm import ScriptedLLM, LLMResponse, ToolCall
from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import TaskSpec, WorkspaceRef, ArtifactRef, TaskInput
from researchcoder.context.contracts import AccessScope
from researchcoder.context.engine import EvidenceContextEngine
from researchcoder.context.policy import ContextPolicy
from researchcoder.context.store import EvidenceStore
from researchcoder.orchestration.evidence import EvidenceExchange
from researchcoder.runtime.budget import BudgetLedger, BudgetedModel, RuntimeHooks
from researchcoder.runtime.protocol import WorkerRequest, RuntimeLimits
from researchcoder.runtime.tracing import TraceSink
from tests.test_orchestration_evidence import sample


def investigation(root):
    root.mkdir()
    (root / "training.py").write_text("x = features.detach()\n")
    (root / "run.log").write_text("ERROR: backbone gradient missing\n")
    state = Investigation(root / "run.log", root)
    state.execute("extract_log_evidence", {})
    state.execute("read_project_file", {"path": "training.py"})
    return state


def report(state):
    e = citation_scope(state)["evidence_ids"]
    o = citation_scope(state)["current_observation_ids"]
    return dict(
        summary="Detach interrupts backbone gradients in training.py",
        historical_failure=dict(description="Missing backbone gradient", evidence_ids=e),
        current_state=dict(assessment="Current source was independently read", observation_ids=o),
        hypotheses=[dict(explanation="Detached features interrupt gradients", evidence_ids=e,
                         observation_ids=o, verification="Independent gradient oracle")],
        recommended_actions=["Remove unintended detach"], limitations=["Runtime validation pending"],
    )


def test_snapshot_id_is_not_candidate_citation_even_with_identical_bytes(tmp_path):
    upstream, current = [investigation(tmp_path / name) for name in ("snapshot", "candidate")]
    a, b = [next(iter(s.observations.values())) for s in (upstream, current)]
    assert a["data"]["file_sha256"] == b["data"]["file_sha256"] and a["id"] != b["id"]
    text = report(current)
    text["current_state"]["observation_ids"].append(a["id"])
    before = copy.deepcopy(current.observations)
    with pytest.raises(ValueError, match="current_state.observation_ids contains an unknown"):
        validate_diagnosis(json.dumps(text), current)
    assert current.observations == before
    # Independently constructed valid local report; no mutation/filtering of model output.
    assert validate_diagnosis(json.dumps(report(current)), current) == report(current)


def test_live_scope_updates_after_source_change_without_mutating_history(tmp_path):
    state = investigation(tmp_path / "candidate")
    agent = DiagnosticAgent(ScriptedLLM([]), state, tools=investigation_tools(state))
    agent.citation_identity = dict(job_id="j", task_id="t", attempt_id="a")
    agent.messages.append(dict(role="user", content="Investigate"))
    history = copy.deepcopy(agent.messages)
    old = citation_scope(state)["current_observation_ids"]
    (state.root / "training.py").write_text("x = features\n")
    state.execute("read_project_file", {"path": "training.py"})
    live = citation_scope(state, agent.citation_identity)
    assert set(old) <= set(live["observation_ids"]) and not set(old) & set(live["current_observation_ids"])
    view = agent._full_messages()
    assert json.dumps(live, ensure_ascii=False, sort_keys=True) in view[0]["content"]
    assert agent.messages == history and "Trusted live citation" not in agent._system
    invalid = report(state)
    invalid["current_state"]["observation_ids"] = old
    with pytest.raises(ValueError, match="superseded"):
        validate_diagnosis(json.dumps(invalid), state)


@pytest.mark.parametrize("selection", ["all", "guided"])
def test_model_visible_tools_never_exceed_trusted_role_permissions(tmp_path, selection):
    state = investigation(tmp_path / "candidate")
    agent = DiagnosticAgent(ScriptedLLM([]), state, tools=investigation_tools(state), tool_selection=selection)
    trace = TraceSink("job", "log", "attempt")
    allowed = ("extract_log_evidence", "locate_source", "read_project_file", "inspect_paths")
    agent.execution_hooks = RuntimeHooks(BudgetLedger(RuntimeLimits(), trace), trace, allowed_tools=allowed)
    names = {s["function"]["name"] for s in agent._tool_schemas()}
    assert names <= set(allowed) and {"read_project_file", "inspect_paths"} <= names
    assert "resolve_project_symbol" not in names and "search_project_source" not in names


@pytest.mark.parametrize("field,value", [
    ("producer_task_id", "forged"), ("producer_attempt_id", "forged-attempt"),
    ("workspace_revision", "c" * 64), ("source_path", "/forged.py"),
])
def test_bundle_render_rejects_forged_identity_or_metadata(tmp_path, field, value):
    io = ArtifactIO(tmp_path)
    exchange = EvidenceExchange("job", io)
    result = sample(io, "producer")
    exchange.submit(result, "a" * 64)
    bundle = exchange.bundle(("producer",), "a" * 64)
    ref = replace(bundle.items[0].reference, **{field: value})
    with pytest.raises(ValueError):
        altered = replace(bundle, items=(replace(bundle.items[0], reference=ref),) + bundle.items[1:])
        exchange.context(altered)


def test_compound_ids_restoration_access_and_sha_are_preserved(tmp_path):
    io = ArtifactIO(tmp_path)
    exchange = EvidenceExchange("job", io)
    for name in ("one", "two"):
        exchange.submit(sample(io, name), "a" * 64)
    bundle = exchange.bundle(("one", "two"), "a" * 64)
    data = json.loads(exchange.context(bundle))
    assert data["citation_scope"] == UPSTREAM_SCOPE
    assert len({x["exchange_id"] for x in data["observations"]}) == 4
    store = EvidenceStore(io, "job")
    refs = [store.add_record(i.reference) for i in bundle.items]
    scope = AccessScope(job_id="job", task_id="diagnosis", attempt_id="d",
                        workspace_revision="a" * 64, producers=(("one", "attempt-one"),))
    assert store.restore_record(refs[0], scope)["record"]["id"] == refs[0].origin_id
    for bad in (replace(scope, job_id="foreign"), replace(scope, workspace_revision="c" * 64)):
        with pytest.raises(ValueError):
            store.restore_record(refs[0], bad)
    with pytest.raises(ValueError):
        store.restore_record(refs[2], scope)
    with pytest.raises(ValueError):
        store.add_record(replace(refs[0], artifact=replace(refs[0].artifact, sha256="d" * 64)))


def test_real_exchange_context_tools_and_validator_with_live_local_registry(tmp_path):
    upstream = investigation(tmp_path / "snapshot")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "training.py").write_text((upstream.root / "training.py").read_text())
    log = candidate / "run.log"
    log.write_text(upstream.log_text)
    io = ArtifactIO(tmp_path / "artifacts")
    revision = workspace_revision(candidate)
    from researchcoder.contracts import TaskResult
    original = dict(evidence=upstream.evidence, observations=list(upstream.observations.values()))
    artifact = io.put_json("job", original)
    result = TaskResult(job_id="job", task_id="log", attempt_id="a-log", kind="diagnose",
                        status="succeeded", workflow_completed=True, outputs=(artifact,),
                        evidence_refs=io.index_evidence(artifact, task_id="log", attempt_id="a-log", workspace_revision=revision))
    exchange = EvidenceExchange("job", io)
    exchange.submit(result, revision)
    bundle = exchange.bundle(("log",), revision)
    bundle_ref = io.put_json("job", bundle.to_dict())
    spec = TaskSpec(job_id="job", task_id="diagnosis", kind="diagnose", objective="Investigate gradients",
                    inputs=(TaskInput(role="raw_log", reference=io.put_bytes("job", log.read_bytes())),),
                    workspace=WorkspaceRef(workspace_id="w", root=str(candidate), source_manifest_sha256=revision))
    request = WorkerRequest(spec=spec, attempt_id="a-diagnosis", artifact_root=str(io.root),
                            trace_path="/unused", parent_event_id="root", evidence_bundle=bundle_ref,
                            context_policy=ContextPolicy(role="diagnosis", strategy="evidence"))
    trace = TraceSink("job", "diagnosis", "a-diagnosis")
    ledger = BudgetLedger(request.limits, trace)
    engine = EvidenceContextEngine(request, io, ledger, trace)
    scopes = []

    class RegistryModel(ScriptedLLM):
        def __init__(self):
            super().__init__([LLMResponse(tool_calls=[ToolCall(id="extract", name="extract_log_evidence", arguments={})]),
                              LLMResponse(tool_calls=[ToolCall(id="read", name="read_project_file", arguments={"path": "training.py"})])])

        def chat(self, messages, tools=None, on_token=None):
            prefix = messages[0]["content"].split("[Trusted live citation scope v1]\n", 1)[1]
            live = json.loads(prefix.split("\n[End citation scope]", 1)[0].splitlines()[-1])
            scopes.append(live)
            if self._turns:
                return super().chat(messages, tools, on_token)
            assert live["identity"] == dict(job_id="job", task_id="diagnosis", attempt_id="a-diagnosis")
            assert not set(live["observation_ids"]) & set(upstream.observations)
            return LLMResponse(content=json.dumps(report(engine.agent.state)), prompt_tokens=100, completion_tokens=100)

    hooks = RuntimeHooks(ledger, trace)
    hooks.context_engine = engine
    model = BudgetedModel(RegistryModel(), ledger, trace, context_engine=engine)
    output = run_diagnosis(log, candidate, model, execution_hooks=hooks)
    assert output["status"] == "completed", output["failure_reason"]
    assert len(scopes) == 3 and scopes[0]["current_observation_ids"] == []
    assert json.loads(engine.external[0]["content"])["citation_scope"] == UPSTREAM_SCOPE
    assert io.read_json(artifact) == original
    assert ledger.model_calls == 3 and engine.views
    assert set(output["diagnosis"]["current_state"]["observation_ids"]) <= set(output["current_observation_ids"])


def test_autograd_evidence_runner_independent_oracle_after_citation_change(tmp_path):
    from researchcoder.evaluation.contracts import EvaluationConfig
    from researchcoder.evaluation.runner import EvaluationRunner, recompute
    config = EvaluationConfig(case_ids=("autograd_detach",), modes=("evidence",), max_trials=1)
    result = EvaluationRunner().run(tmp_path / "run", config, torch_python=os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"])
    assert result == recompute(tmp_path / "run")
    ref = ArtifactRef.from_json((tmp_path / "run/records/trial-0000.json").read_text())
    saved = ArtifactIO(tmp_path / "run/trials/trial-0000/artifacts").read_json(ref)
    assert saved["status"] == "succeeded" and saved["metrics"]["fail_to_pass"] and saved["metrics"]["pass_to_pass"]
