from dataclasses import replace
import json
import pytest
from tests.test_orchestration_static import setup_graph
from researchcoder.contracts import TaskSpec, TaskResult
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, PlanningTaskSpec, PlanningTaskResult
from researchcoder.orchestration import DAGScheduler, PlanRevision, OrchestrationTask, TaskDependency
from researchcoder.orchestration.budget import JobBudget
from researchcoder.orchestration.planning import (
    Planner,
    PlanningJob,
    PlanningConstraints,
    PlanningRejected,
    ReplanningController,
    ReplanningLimits,
)


def model(plan):
    data = plan.to_dict() if hasattr(plan, "to_dict") else plan
    return ModelSpec(turns=(ScriptedTurn(content=json.dumps(data)),))


def test_lead_plans_in_real_worker_and_phase1_wire_kinds_unchanged(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    planner = Planner()
    budget = JobBudget(cfg.job_limits)
    actual = planner.plan(
        PlanningJob("job", "Plan the authorized workflow"),
        [],
        PlanningConstraints(config=cfg, model=model(plan), budget=budget, required_roles=("log_analysis", "verification")),
    )
    assert actual == plan
    row = planner.records[0]
    assert row["accepted"] and row["record"]["worker_pid"]
    assert row["result"]["kind"] == "plan"
    assert PlanningTaskResult.from_dict(row["result"]).kind == "plan"
    with pytest.raises(ValueError):
        TaskResult.from_dict(row["result"])
    data = cfg.templates["read"].spec.to_dict()
    data["kind"] = "plan"
    data["policy"]["allowed_tool_names"] = []
    assert PlanningTaskSpec.from_dict(data).kind == "plan"
    with pytest.raises(ValueError):
        TaskSpec.from_dict(data)
    assert budget.charged["model_calls"] == 1 and budget.snapshot()["settlements"]["planning.r0"]["usage_unknown"]
    assert not budget.pending


@pytest.mark.parametrize("bad", ["cycle", "permissions", "role", "missing_goal", "depth", "ancestry", "retry"])
def test_invalid_generated_plans_rejected_and_metered(tmp_path, bad):
    plan, cfg, io = setup_graph(tmp_path)
    raw = plan.to_dict()
    if bad == "cycle":
        raw["tasks"][0]["dependencies"] = [TaskDependency(task_id="v").to_dict()]
    elif bad == "permissions":
        raw["tasks"][0]["tool_policy"] = {"bash": True}
    elif bad == "role":
        raw["tasks"][0]["role"] = "lead"
    elif bad == "missing_goal":
        raw["tasks"] = raw["tasks"][:2]
    elif bad == "depth":
        cfg = replace(cfg, max_depth=1)
    elif bad == "ancestry":
        raw["parent_revision_id"] = "fake"
    elif bad == "retry":
        raw["tasks"][0]["retry_of"] = "unexecuted"
    planner = Planner()
    budget = JobBudget(cfg.job_limits)
    with pytest.raises(PlanningRejected):
        planner.plan(
            PlanningJob("job", "Plan"),
            [],
            PlanningConstraints(config=cfg, model=model(raw), budget=budget, required_roles=("verification",)),
        )
    assert len(planner.records) == 1 and not planner.records[0]["accepted"]
    assert budget.charged["model_calls"] == 1 and not budget.pending


def test_version_and_input_bound_result_reuse_and_invalidation(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    scheduler = DAGScheduler()
    first = scheduler.execute(plan, cfg)
    reuse = dict(scheduler.completed)
    assert first.status == "succeeded"
    second_plan = replace(plan, revision_id="r1", generation=1, parent_revision_id="r0")
    budget = JobBudget(cfg.job_limits)
    second = scheduler.execute(second_plan, cfg, reuse=reuse, budget=budget)
    assert second.status == "succeeded" and all(s.status == "reused" and s.attempts == 0 for s in second.states)
    assert [r.attempt_id for r in first.results] == [r.attempt_id for r in second.results]
    assert budget.charged == dict(model_calls=0, tool_calls=0, total_tokens=0)
    third_plan = replace(
        second_plan,
        revision_id="r2",
        generation=2,
        parent_revision_id="r1",
        tasks=tuple(replace(t, objective="New input changes investigation") if t.task_id == "a" else t for t in plan.tasks),
    )
    third = scheduler.execute(third_plan, cfg, reuse=dict(scheduler.completed))
    states = {s.task_id: s for s in third.states}
    assert states["a"].status == "succeeded" and states["b"].status == "reused" and states["v"].status == "succeeded"
    assert third.results[0].attempt_id != first.results[0].attempt_id
    from pathlib import Path
    from researchcoder.orchestration.workspaces import SourceSnapshot

    candidate = Path(cfg.templates["verify"].spec.workspace.root)
    (candidate / "training.py").write_text("loss = 2\n")
    updated = replace(cfg, snapshot=SourceSnapshot.create(candidate, tmp_path / "snapshot-new-version"))
    fourth = scheduler.execute(
        replace(plan, revision_id="r3", generation=3, parent_revision_id="r2"), updated, reuse=dict(scheduler.completed)
    )
    assert fourth.status == "succeeded" and all(s.status == "succeeded" for s in fourth.states)


def replan_case(tmp_path, *, noop=False):
    from examples.multi_agent_autograd_demo import prepare_demo, call
    from researchcoder.runtime.protocol import ScriptedTurn

    plan, cfg, io, _ = prepare_demo(tmp_path / "replan-demo", bad_patch=True)
    templates = dict(cfg.templates)
    repair = templates["repair"]
    if noop:
        noop_model = ModelSpec(
            turns=(
                call("r", "read_file", {"file_path": "training.py"}),
                ScriptedTurn(content="No candidate change; request independent verification."),
            )
        )
        templates["repair"] = replace(repair, config=replace(repair.config, model=noop_model))
    else:
        fixed_model = ModelSpec(
            turns=(
                call("r", "read_file", {"file_path": "training.py"}),
                call(
                    "w",
                    "edit_file",
                    {
                        "file_path": "training.py",
                        "old_string": "features = model.backbone(x).clone().detach()",
                        "new_string": "features = model.backbone(x)",
                    },
                ),
                ScriptedTurn(content="Correct gradient-flow patch complete."),
            )
        )
        templates["repair_final"] = replace(repair, config=replace(repair.config, model=fixed_model))
    cfg = replace(cfg, templates=templates)
    next_plan = replace(
        plan,
        revision_id="r1",
        generation=1,
        parent_revision_id="r0",
        tasks=tuple(
            replace(t, template_id="repair" if noop else "repair_final", retry_of="r0.repair") if t.role == "repair" else t
            for t in plan.tasks
        ),
    )
    return plan, next_plan, cfg, io


def test_failed_verification_triggers_bounded_real_replanning_and_new_version(tmp_path):
    plan, next_plan, cfg, io = replan_case(tmp_path)
    result = ReplanningController().execute(
        PlanningJob("autograd", "Repair the joint training failure"), cfg, (model(plan), model(next_plan))
    )
    (tmp_path / "replanning-summary.json").write_text(json.dumps(result, indent=2))
    assert result["status"] == "succeeded" and result["replans"] == 1, result
    assert result["runs"][0]["status"] == "failed" and result["runs"][1]["status"] == "succeeded"
    assert result["runs"][0]["results"][-1]["verification"]["status"] == "failed"
    assert result["runs"][1]["results"][-1]["verification"]["status"] == "passed"
    a, b = result["runs"][0]["states"][0], result["runs"][1]["states"][0]
    assert a["workspace_revision"] != b["workspace_revision"] and b["status"] == "succeeded"
    assert all(r["accepted"] for r in result["planning_records"])
    assert len({r["record"]["worker_pid"] for r in result["planning_records"]}) == 2
    assert not result["budget"]["pending"]
    assert result["budget"]["charged"]["model_calls"] == sum(
        v["charged"]["model_calls"] for v in result["budget"]["settlements"].values()
    )


def test_repeated_failure_without_novel_evidence_stops(tmp_path):
    plan, next_plan, cfg, io = replan_case(tmp_path, noop=True)
    third = replace(next_plan, revision_id="r2", generation=2, parent_revision_id="r1")
    result = ReplanningController().execute(
        PlanningJob("autograd", "Repair gradients"),
        cfg,
        (model(plan), model(next_plan), model(third)),
        limits=ReplanningLimits(max_replans=2),
    )
    assert result["status"] == "incomplete" and result["reason"] == "no_new_evidence" and len(result["runs"]) == 2, result
    assert len(result["planning_records"]) == 2


def test_replan_and_cumulative_task_limits_are_enforced(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    controller = ReplanningController()
    result = controller.execute(
        PlanningJob("job", "Plan authorized tasks"),
        cfg,
        (model(plan),),
        limits=ReplanningLimits(max_replans=0, max_total_tasks=1),
        required_roles=("log_analysis", "verification"),
    )
    assert result["reason"] == "task_limit" and not result["runs"]
    assert result["budget"]["charged"]["model_calls"] == 1


def test_all_three_engineering_comparison_arms_have_raw_traces_and_real_checks(tmp_path):
    from examples.orchestration_comparison import run_comparison
    from researchcoder.adapters.artifacts import ArtifactIO
    from researchcoder.contracts import ArtifactRef
    from pathlib import Path

    result = run_comparison(tmp_path / "comparison", repetitions=1)
    assert len(result["rows"]) == 3 and result["provider_requests"] == 0
    assert {r["mode"] for r in result["rows"]} == {
        "single_agent_sequential",
        "multi_agent_sequential",
        "multi_agent_dag_parallel",
    }
    assert all(r["status"] == "succeeded" and r["verification_result"] == "passed" for r in result["rows"]), result
    assert all(r["actual_prompt_tokens"] is None and r["token_usage_unknown"] and r["model_requests"] > 0 for r in result["rows"])
    assert len({json.dumps(r["job_limits"], sort_keys=True) for r in result["rows"]}) == 1
    assert [r["worker_count"] for r in result["rows"]] == [1, 6, 6]
    for path in result["raw_records"]:
        raw = json.loads(Path(path).read_text())
        io = ArtifactIO(Path(path).parent / "artifacts")
        if "record" in raw:
            io.read_json(ArtifactRef.from_dict(raw["record"]["trace"]))
        else:
            io.read_json(ArtifactRef.from_dict(raw["result"]["trace_ref"]))
            assert raw["planning"][0]["accepted"]


def test_cancellation_after_real_edit_preserves_partial_modification(tmp_path):
    from examples.multi_agent_autograd_demo import prepare_demo
    from pathlib import Path
    import sys
    import threading
    import time
    from researchcoder.contracts import TaskInput

    plan, cfg, io, _ = prepare_demo(tmp_path / "partial")
    initial = replace(plan, tasks=plan.tasks[:3])
    scheduler = DAGScheduler()
    first = scheduler.execute(initial, cfg)
    assert first.status == "succeeded"
    diagnosis = next(r for r in first.results if r.task_id.endswith(".diagnose"))
    marker = tmp_path / "edited.marker"
    cancel = threading.Event()
    program = r"""
import pathlib,sys,time
import researchcoder.runtime.worker as worker
from researchcoder.runtime.budget import RuntimeHooks
marker=pathlib.Path(sys.argv[1]);del sys.argv[1]
original=RuntimeHooks.after_tool
def after(self,token,status):
 original(self,token,status)
 start=next(e for e in self.trace.events if e['event_id']==token[0])
 if start['tool']=='edit_file' and status=='ok':
  marker.write_text('real edit completed')
  time.sleep(30)
RuntimeHooks.after_tool=after
raise SystemExit(worker.main())
"""
    template = cfg.templates["repair"]
    template = replace(
        template,
        spec=replace(template.spec, inputs=(TaskInput(role="diagnosis_run", reference=diagnosis.outputs[0]),)),
        config=replace(template.config, worker_command=(sys.executable, "-B", "-c", program, str(marker))),
    )
    cfg = replace(cfg, templates={"repair": template}, cancel_event=cancel)
    next_plan = PlanRevision(
        job_id="autograd",
        revision_id="r1",
        generation=1,
        parent_revision_id="r0",
        tasks=(OrchestrationTask(task_id="repair", template_id="repair", role="repair"),),
    )

    def observe():
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        cancel.set()

    observer = threading.Thread(target=observe)
    observer.start()
    try:
        result = scheduler.execute(next_plan, cfg)
    finally:
        observer.join()
    assert marker.exists() and result.status == "cancelled"
    current = (Path(template.spec.workspace.root) / "training.py").read_text()
    assert "features = model.backbone(x)\n" in current
    assert "features = model.backbone(x).detach()" not in current
    state = result.states[0]
    assert state.status == "cancelled" and state.resulting_revision != state.workspace_revision
    assert result.results[0].status == "cancelled"


def test_two_writers_serialize_and_stale_handoff_cannot_repeat_edit(tmp_path):
    from examples.multi_agent_autograd_demo import prepare_demo
    from researchcoder.contracts import TaskInput

    plan, cfg, io, _ = prepare_demo(tmp_path / "writers")
    scheduler = DAGScheduler()
    initial = scheduler.execute(replace(plan, tasks=plan.tasks[:3]), cfg)
    assert initial.status == "succeeded"
    diagnosis = next(r for r in initial.results if r.task_id.endswith(".diagnose"))
    template = cfg.templates["repair"]
    template = replace(
        template, spec=replace(template.spec, inputs=(TaskInput(role="diagnosis_run", reference=diagnosis.outputs[0]),))
    )
    cfg = replace(cfg, templates={"repair": template})
    writers = PlanRevision(
        job_id="autograd",
        revision_id="r1",
        generation=1,
        parent_revision_id="r0",
        tasks=(
            OrchestrationTask(task_id="write_a", template_id="repair", role="repair"),
            OrchestrationTask(task_id="write_b", template_id="repair", role="repair"),
        ),
    )
    result = scheduler.execute(writers, cfg)
    a, b = [scheduler.records[k] for k in ("write_a", "write_b")]
    assert a["ended_ns"] <= b["started_ns"]
    assert result.results[0].status == "succeeded"
    assert result.results[1].status == "incomplete" and result.results[1].failure_kind == "reference_stale"
    assert a["resulting_revision"] == b["workspace_revision"] == b["resulting_revision"]
