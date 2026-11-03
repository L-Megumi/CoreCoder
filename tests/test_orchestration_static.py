from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import threading
import pytest

from tests.test_runtime_harness import case
from researchcoder.contracts import TaskInput, ExecutionPolicy, ArtifactRef
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, ScriptedCall, RuntimeLimits
from researchcoder.runtime.isolation import WorkspaceLease
from researchcoder.orchestration import (
    PlanRevision,
    OrchestrationTask,
    TaskDependency,
    DAGScheduler,
    SchedulerConfig,
    TaskTemplate,
)
from researchcoder.orchestration.workspaces import SourceSnapshot
from researchcoder.orchestration.budget import JobLimits, JobBudget
from researchcoder.runtime.budget import BudgetExceeded


def call(identity, name, args=None):
    return ScriptedTurn(calls=(ScriptedCall(call_id=identity, name=name, arguments_json=json.dumps(args or {})),))


def report_turn():
    return ScriptedTurn(
        content=json.dumps(
            dict(
                summary="Investigated actual gradient evidence",
                historical_failure=dict(description="Warning of missing gradients", evidence_ids=["$E0"]),
                current_state=dict(assessment="Actual source inspected", observation_ids=["$O0"]),
                hypotheses=[
                    dict(
                        explanation="Check joint gradient flow",
                        evidence_ids=["$E0"],
                        observation_ids=["$O0"],
                        verification="Independent oracle",
                    )
                ],
                recommended_actions=["Verify the gradient contract"],
                limitations=["Scripted decisions only"],
            )
        )
    )


def setup_graph(tmp_path, worker_command=None):
    spec, cfg, io, log = case(tmp_path)
    snap = SourceSnapshot.create(spec.workspace.root, tmp_path / "snapshot")
    model = ModelSpec(
        turns=(call("log", "extract_log_evidence"), call("read", "read_project_file", {"path": "training.py"}), report_turn())
    )
    diag = replace(
        spec, kind="diagnose", verification=None, policy=ExecutionPolicy(), inputs=(TaskInput(role="raw_log", reference=log),)
    )
    limits = RuntimeLimits(max_model_calls=6, max_tool_calls=6, max_total_tokens=50000, max_runtime_seconds=8)
    read = TaskTemplate("log_analysis", diag, replace(cfg, model=model, limits=limits, worker_command=worker_command))
    verify = TaskTemplate("verification", spec, replace(cfg, limits=limits), "candidate")
    config = SchedulerConfig(
        templates={"read": read, "verify": verify},
        artifact_root=str(io.root),
        snapshot=snap,
        job_limits=JobLimits(max_model_calls=40, max_tool_calls=40, max_total_tokens=500000),
    )
    tasks = (
        OrchestrationTask(task_id="a", template_id="read", role="log_analysis"),
        OrchestrationTask(task_id="b", template_id="read", role="log_analysis"),
        OrchestrationTask(
            task_id="v",
            template_id="verify",
            role="verification",
            dependencies=(TaskDependency(task_id="a"), TaskDependency(task_id="b")),
        ),
    )
    return PlanRevision(job_id="job", revision_id="r0", tasks=tasks), config, io


def test_contract_roundtrip_and_unknown_permissions_rejected(tmp_path):
    plan, cfg, _ = setup_graph(tmp_path)
    assert PlanRevision.from_json(plan.to_json()) == plan and plan.depth() == 2
    data = plan.to_dict()
    data["tasks"][0]["tool_policy"] = {"bash": True}
    with pytest.raises(ValueError):
        PlanRevision.from_dict(data)
    with pytest.raises(ValueError):
        DAGScheduler().validate(plan, replace(cfg, max_depth=1))
    with pytest.raises(ValueError):
        DAGScheduler().validate(plan, replace(cfg, max_tasks=1))


@pytest.mark.parametrize("mode", ["cycle", "missing", "duplicate", "role", "identity"])
def test_bad_dags_rejected(mode):
    a = OrchestrationTask(task_id="a", template_id="read", role="log_analysis")
    b = OrchestrationTask(task_id="b", template_id="read", role="code_analysis", dependencies=(TaskDependency(task_id="a"),))
    with pytest.raises((ValueError, TypeError)):
        if mode == "cycle":
            PlanRevision(job_id="j", revision_id="r", tasks=(replace(a, dependencies=(TaskDependency(task_id="b"),)), b))
        elif mode == "missing":
            PlanRevision(job_id="j", revision_id="r", tasks=(b,))
        elif mode == "duplicate":
            PlanRevision(job_id="j", revision_id="r", tasks=(a, a))
        elif mode == "role":
            replace(a, role="root_shell")
        else:
            replace(a, task_id="../bad")


def test_snapshot_is_independent_and_detects_mutation(tmp_path):
    plan, cfg, _ = setup_graph(tmp_path)
    original = Path(cfg.templates["verify"].spec.workspace.root) / "training.py"
    original.write_text("changed = 2\n")
    assert (Path(cfg.snapshot.workspace.root) / "training.py").read_text() == "loss = 1\n"
    cfg.snapshot.verify()
    target = Path(cfg.snapshot.workspace.root) / "training.py"
    target.chmod(0o644)
    target.write_text("changed = 3\n")
    with pytest.raises(ValueError):
        cfg.snapshot.verify()


def test_shared_read_and_exclusive_write_leases(tmp_path):
    with WorkspaceLease(tmp_path, shared=True), WorkspaceLease(tmp_path, shared=True):
        with pytest.raises(RuntimeError):
            with WorkspaceLease(tmp_path):
                pass
    with WorkspaceLease(tmp_path):
        with pytest.raises(RuntimeError):
            with WorkspaceLease(tmp_path, shared=True):
                pass
        with pytest.raises(RuntimeError):
            with WorkspaceLease(tmp_path):
                pass


def test_atomic_job_budget_and_unknown_debt():
    budget = JobBudget(JobLimits(max_model_calls=6, max_tool_calls=6, max_total_tokens=50000))
    limits = RuntimeLimits(max_model_calls=6, max_tool_calls=6, max_total_tokens=50000)
    barrier = threading.Barrier(3)
    outcomes = []

    def reserve(name):
        barrier.wait()
        try:
            budget.reserve(name, limits)
            outcomes.append("reserved")
        except BudgetExceeded:
            outcomes.append("denied")

    threads = [threading.Thread(target=reserve, args=(str(i),)) for i in range(2)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["denied", "reserved"]
    key = next(iter(budget.pending))
    row = budget.settle(key, None)
    assert row["usage_unknown"] and row["charged"]["total_tokens"] == 50000
    with pytest.raises(ValueError):
        budget.settle(key, None)
    with pytest.raises(BudgetExceeded):
        budget.reserve("third", limits)


def test_real_process_barrier_overlap_static_dag_and_artifacts(tmp_path):
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    program = r"""
import os,time,json,pathlib,sys
import researchcoder.runtime.worker as worker
directory=pathlib.Path(sys.argv[1]);del sys.argv[1]
original=worker.ProtocolScript.chat
first=True
def chat(self,*args,**kwargs):
 global first
 if first:
  first=False
  marker=directory/(str(os.getpid())+'.json')
  marker.write_text(json.dumps({'pid':os.getpid(),'entered_ns':time.monotonic_ns()}))
  deadline=time.monotonic()+5
  while len(list(directory.glob('*.json')))<2:
   if time.monotonic()>deadline:raise RuntimeError('Barrier peer missing')
   time.sleep(.005)
  row=json.loads(marker.read_text());row['released_ns']=time.monotonic_ns();marker.write_text(json.dumps(row))
 return original(self,*args,**kwargs)
worker.ProtocolScript.chat=chat
raise SystemExit(worker.main())
"""
    plan, cfg, io = setup_graph(tmp_path, (sys.executable, "-B", "-c", program, str(barrier)))
    scheduler = DAGScheduler()
    result = scheduler.execute(plan, cfg)
    assert result.status == "succeeded", io.read_json(result.summary_ref)
    assert type(result).from_json(result.to_json()) == result
    with pytest.raises(ValueError):
        replace(result, states=(replace(result.states[0], status="blocked"),) + result.states[1:])
    with pytest.raises(ValueError):
        replace(plan, revision_id="r0", generation=1, parent_revision_id="r0")
    a, b, v = [scheduler.records[n] for n in ("a", "b", "v")]
    assert len({a["worker_pid"], b["worker_pid"], v["worker_pid"]}) == 3
    rows = [json.loads(p.read_text()) for p in barrier.glob("*.json")]
    assert len(rows) == 2
    assert max(r["entered_ns"] for r in rows) <= min(r["released_ns"] for r in rows)
    assert max(a["started_ns"], b["started_ns"]) < min(a["ended_ns"], b["ended_ns"])
    assert v["started_ns"] >= max(a["ended_ns"], b["ended_ns"])
    for logical, execution in zip(("a", "b", "v"), result.results):
        assert execution.task_id == "r0." + logical
        assert all("/tasks/r0." + logical + "/" in ref.storage_key for ref in execution.outputs)
        for ref in execution.outputs:
            io.read_bytes(ref)
    summary = io.read_json(result.summary_ref)
    assert summary["budget"]["charged"]["model_calls"] == 6
    assert summary["budget"]["charged"]["tool_calls"] == 5
    assert not summary["budget"]["pending"]
    assert all(r["usage_unknown"] for r in summary["budget"]["settlements"].values() if r["charged"]["model_calls"])
    events = io.read_json(result.trace_ref)["events"]
    assert len([e for e in events if e["event_type"] == "graph.dependency"]) == 2
    assert {e["artifact_id"] for e in events if e["event_type"] == "task.collected"} == {
        r["trace"]["artifact_id"] for r in scheduler.records.values()
    }
    with pytest.raises(ValueError):
        scheduler.execute(plan, cfg)


@pytest.mark.parametrize("mode", ["crash", "timeout", "cancel", "budget"])
def test_failures_propagate_and_do_not_run_downstream(tmp_path, mode):
    plan, cfg, io = setup_graph(tmp_path)
    template = cfg.templates["read"]
    if mode == "cancel":
        cancel = threading.Event()
        cancel.set()
        cfg = replace(cfg, cancel_event=cancel)
    elif mode == "budget":
        cfg = replace(cfg, job_limits=JobLimits(max_model_calls=1, max_tool_calls=1, max_total_tokens=1))
    else:
        command = (sys.executable, "-c", "import os;os._exit(9)" if mode == "crash" else "import time;time.sleep(30)")
        template = replace(
            template,
            config=replace(
                template.config, worker_command=command, limits=replace(template.config.limits, max_runtime_seconds=0.2)
            ),
        )
        cfg = replace(cfg, templates={**cfg.templates, "read": template})
    scheduler = DAGScheduler()
    result = scheduler.execute(plan, cfg)
    states = {s.task_id: s for s in result.states}
    assert result.status != "succeeded"
    assert states["v"].status in ("blocked", "cancelled") and states["v"].attempt_id is None
    assert not io.read_json(result.summary_ref)["budget"]["pending"]


def test_candidate_operations_serialize_and_verification_binds_version(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    tasks = tuple(OrchestrationTask(task_id=str(i), template_id="verify", role="verification") for i in range(2))
    scheduler = DAGScheduler()
    result = scheduler.execute(replace(plan, tasks=tasks), cfg)
    assert result.status == "succeeded"
    a, b = scheduler.records.values()
    assert b["started_ns"] >= a["ended_ns"]
    for r in result.results:
        assert r.verification.checked_workspace_revision == cfg.templates["verify"].spec.workspace.source_manifest_sha256


def test_namespaced_artifacts_collision_and_cross_task_reads(tmp_path):
    from researchcoder.adapters.artifacts import ArtifactIO

    a, b = [ArtifactIO(tmp_path, namespace=n) for n in ("a", "b")]
    ra = a.put_bytes("j", b"one", artifact_id="same")
    rb = b.put_bytes("j", b"two", artifact_id="same")
    assert ra.storage_key != rb.storage_key and a.read_bytes(rb) == b"two" and b.read_bytes(ra) == b"one"
    with pytest.raises(FileExistsError):
        a.put_bytes("j", b"bad", artifact_id="same")
    assert hashlib.sha256(a.read_bytes(ra)).hexdigest() == ra.sha256
    assert ArtifactRef.from_json(rb.to_json()) == rb


def test_job_deadline_keeps_timeout_distinct_from_user_cancellation(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    template = cfg.templates["read"]
    template = replace(
        template, config=replace(template.config, worker_command=(sys.executable, "-c", "import time;time.sleep(30)"))
    )
    cfg = replace(cfg, templates={**cfg.templates, "read": template}, job_limits=replace(cfg.job_limits, max_runtime_seconds=0.2))
    result = DAGScheduler().execute(plan, cfg)
    assert result.status == "failed"
    assert result.results and all(r.status == "failed" and r.failure_kind == "timeout" for r in result.results)
    assert next(s for s in result.states if s.task_id == "v").reason == "timeout"
    assert not io.read_json(result.summary_ref)["budget"]["pending"]
