from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import TaskResult, ArtifactRef
from researchcoder.orchestration.evidence import EvidenceExchange, EvidenceBundle
from researchcoder.orchestration.profiles import AgentRunner, PROFILES
from tests.test_orchestration_static import setup_graph


def sample(io, task, *, version="a" * 64, source_hash="b" * 64):
    raw = {
        "status": "completed",
        "evidence": [dict(id="same", source="log.txt", line_no=1, end_line_no=1, message="WARNING gradient missing")],
        "observations": [
            dict(
                id="obs_same",
                kind="file_read",
                data=dict(resolved_path="/project/training.py", file_sha256=source_hash, start_line=1, end_line=3),
            )
        ],
        "diagnosis": {"hypotheses": [{"explanation": "Hypothesis, not observed fact"}]},
    }
    artifact = io.put_json("job", raw)
    refs = io.index_evidence(artifact, task_id=task, attempt_id="attempt-" + task, workspace_revision=version)
    return TaskResult(
        job_id="job",
        task_id=task,
        attempt_id="attempt-" + task,
        kind="diagnose",
        status="succeeded",
        workflow_completed=True,
        outputs=(artifact,),
        evidence_refs=refs,
    )


def test_same_original_ids_preserve_compound_task_provenance(tmp_path):
    io = ArtifactIO(tmp_path)
    exchange = EvidenceExchange("job", io)
    a, b = [sample(io, t) for t in ("a", "b")]
    exchange.submit(a, "a" * 64)
    exchange.submit(b, "a" * 64)
    bundle = exchange.bundle(("a", "b"), "a" * 64)
    assert len(bundle.items) == 4 and len({i.exchange_id for i in bundle.items}) == 4
    assert [i.reference.origin_id for i in bundle.items].count("same") == 2
    assert EvidenceBundle.from_json(bundle.to_json()) == bundle
    data = json.loads(exchange.context(bundle))
    assert all(r["classification"] == "observed_record" for r in data["observations"])
    assert all(r["classification"] == "unverified_hypothesis" for r in data["hypotheses"])
    assert io.read_json(a.outputs[0]) == io.read_json(b.outputs[0])
    with pytest.raises(ValueError):
        exchange.submit(a, "a" * 64)
    with pytest.raises(ValueError):
        exchange.bundle(("unknown",), "a" * 64)
    with pytest.raises(ValueError):
        exchange.bundle(("a",), "c" * 64)


def test_forged_provenance_metadata_and_conflicting_observations_rejected(tmp_path):
    io = ArtifactIO(tmp_path)
    exchange = EvidenceExchange("job", io)
    a = sample(io, "a")
    ref = a.evidence_refs[0]
    with pytest.raises(ValueError):
        exchange.submit(replace(a, evidence_refs=(replace(ref, producer_task_id="wrong"),)), "a" * 64)
    with pytest.raises(ValueError):
        exchange.submit(replace(a, evidence_refs=(replace(ref, source_path="/forged"),)), "a" * 64)
    exchange.submit(a, "a" * 64)
    b = sample(io, "b", source_hash="c" * 64)
    exchange.submit(b, "a" * 64)
    with pytest.raises(ValueError):
        exchange.bundle(("a", "b"), "a" * 64)


def test_role_profiles_restrict_tools_permissions_and_budget(tmp_path):
    _, cfg, _ = setup_graph(tmp_path)
    template = cfg.templates["read"]
    a = AgentRunner.configure("log_analysis", template.spec, template.config)
    b = AgentRunner.configure("code_analysis", template.spec, template.config)
    assert a.system_instructions != b.system_instructions
    assert "resolve_project_symbol" not in a.diagnostic_tools and "resolve_project_symbol" in b.diagnostic_tools
    assert a.limits.max_total_tokens <= PROFILES["log_analysis"].max_tokens
    with pytest.raises(ValueError):
        AgentRunner.configure(
            "log_analysis",
            replace(template.spec, policy=replace(template.spec.policy, writable_paths=("training.py",))),
            template.config,
        )
    assert not template.config.system_instructions


def test_authorized_context_is_metered_without_mutating_message_history():
    from researchcoder.runtime.budget import BudgetLedger, BudgetedModel
    from researchcoder.runtime.protocol import RuntimeLimits
    from researchcoder.runtime.tracing import TraceSink
    from corecoder.llm import LLMResponse

    class Model:
        model = "fake"

        def chat(self, messages, **kwargs):
            self.seen = messages
            return LLMResponse(content="ok")

    trace = TraceSink("j", "t", "a")
    ledger = BudgetLedger(RuntimeLimits(), trace)
    model = Model()
    messages = [dict(role="system", content="original")]
    context = json.dumps({"fact": "bound-evidence-" * 300})
    BudgetedModel(model, ledger, trace, system_instructions="Specialist", evidence_context=context).chat(messages)
    assert messages == [dict(role="system", content="original")]
    assert context in model.seen[0]["content"] and "Specialist" in model.seen[0]["content"]
    assert ledger.estimated_tokens > 1000 and context not in json.dumps(trace.events)


def test_autograd_specialist_dag_real_tools_and_independent_oracle(tmp_path):
    from examples.multi_agent_autograd_demo import run_demo

    result = run_demo(tmp_path / "multi-demo")
    assert result["provider_requests"] == 0 and result["initial_status"] == "failed"
    assert result["result"]["status"] == "succeeded", result["execution"]
    records = result["execution"]["records"]
    assert len({r["worker_pid"] for r in records.values()}) == 5
    assert records["diagnose"]["started_ns"] >= max(records[t]["ended_ns"] for t in ("log", "code"))
    assert records["repair"]["ended_ns"] <= records["verify"]["started_ns"]
    io = ArtifactIO(tmp_path / "multi-demo/artifacts")
    names = set()
    for task in ("log", "code"):
        trace = io.read_json(ArtifactRef.from_dict(records[task]["trace"]))["events"]
        names |= {e["tool"] for e in trace if e["event_type"] == "tool.call.start"}
        assert any(e["event_type"] == "model.request.start" for e in trace)
    assert {"extract_log_evidence", "locate_source", "read_project_file", "resolve_project_symbol"} <= names
    scheduler_trace = io.read_json(ArtifactRef.from_dict(result["result"]["trace_ref"]))["events"]
    assert any(e["event_type"] == "evidence.aggregated" and e["logical_task_id"] == "diagnose" for e in scheduler_trace)
    verification = result["result"]["results"][-1]["verification"]
    assert verification["status"] == "passed" and verification["exit_code"] == 0
    oracle = json.loads(verification["stdout"]["text"])
    assert oracle["passed"] and {r["seed"] for r in oracle["rows"]} == {142, 314}
    assert (
        hashlib.sha256((Path(__file__).parents[1] / "benchmarks/pytorch/autograd_detach/oracle.py").read_bytes()).hexdigest()
        == "9b68411dcf34594c30c8e107e8b06c70fd37f4f5af39c645bb5e062b48fe772d"
    )
    assert all(p["grad_present"] and p["update_norm"] > 0 for r in oracle["rows"] for p in r["parameters"].values())
    assert all(
        s["workspace_revision"] == result["result"]["states"][0]["workspace_revision"] for s in result["result"]["states"][:4]
    )
