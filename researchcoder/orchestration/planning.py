"""A tool-free Lead uses the existing Agent/Harness; the Controller validates authority."""

from dataclasses import dataclass, replace
import json
from researchcoder.contracts import TaskInput, ExecutionPolicy, identifier, text
from researchcoder.runtime import ResearchHarness, ExecutionConfig
from researchcoder.runtime.protocol import PlanningTaskSpec, RuntimeLimits
from researchcoder.runtime.budget import BudgetExceeded
from .contracts import PlanRevision
from .scheduler import DAGScheduler
from .profiles import PROFILES


class PlanningRejected(ValueError):
    pass


@dataclass(frozen=True)
class PlanningJob:
    job_id: str
    objective: str

    def __post_init__(self):
        identifier(self.job_id)
        text(self.objective)


@dataclass(frozen=True)
class PlanningConstraints:
    config: object
    model: object
    budget: object
    revision_id: str = "r0"
    generation: int = 0
    parent_revision_id: str | None = None
    prior_execution_ids: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ("log_analysis", "code_analysis", "diagnosis", "repair", "verification")
    max_model_calls: int = 2
    max_total_tokens: int = 60000


class Planner:
    def __init__(self):
        self.records = []

    def plan(self, job, available_evidence, constraints) -> PlanRevision:
        cfg = constraints.config
        from researchcoder.adapters.artifacts import ArtifactIO

        io = ArtifactIO(cfg.artifact_root, namespace="planning." + constraints.revision_id)
        allowed = []
        for name, template in cfg.templates.items():
            allowed.append(dict(template_id=name, role=template.role, workspace_mode=template.workspace_mode))
        payload = dict(
            job_id=job.job_id,
            objective=job.objective,
            revision_id=constraints.revision_id,
            generation=constraints.generation,
            parent_revision_id=constraints.parent_revision_id,
            allowed_templates=allowed,
            required_roles=list(constraints.required_roles),
            max_tasks=cfg.max_tasks,
            max_depth=cfg.max_depth,
            available_evidence=available_evidence,
            output_contract="PlanRevision schema_version=1: job_id, revision_id, tasks[{task_id,template_id,role,dependencies:[{task_id,input_role,schema_version}],objective,workspace_revision,retry_of,schema_version}], generation,parent_revision_id,schema_version",
        )
        raw = json.dumps(payload, ensure_ascii=False)
        if len(raw) > 80000:
            raise PlanningRejected("Planning input too large; use evidence references")
        artifact = io.put_json(job.job_id, payload)
        limits = RuntimeLimits(
            max_model_calls=constraints.max_model_calls,
            max_tool_calls=1,
            max_rounds=2,
            max_total_tokens=constraints.max_total_tokens,
            max_output_tokens=6000,
            max_runtime_seconds=min(30, cfg.job_limits.max_runtime_seconds),
        )
        reservation = "planning." + constraints.revision_id
        constraints.budget.reserve(reservation, limits)
        record = None
        try:
            workspace = cfg.snapshot.verify()
            spec = PlanningTaskSpec(
                job_id=job.job_id,
                task_id=reservation,
                objective=job.objective,
                workspace=workspace,
                inputs=(TaskInput(role="evidence", reference=artifact),),
                policy=ExecutionPolicy(allowed_tool_names=()),
            )
            harness = ResearchHarness()
            import time

            worker_started = time.monotonic_ns()
            result = harness.execute(
                spec,
                ExecutionConfig(
                    artifact_root=cfg.artifact_root,
                    allowed_workspaces=(workspace.root,),
                    max_policy=spec.policy,
                    model=constraints.model,
                    limits=limits,
                    artifact_namespace=reservation,
                    system_instructions=PROFILES["lead"].instructions,
                    cancel_event=cfg.cancel_event,
                ),
            )
            record = dict(harness.records[result.attempt_id], started_ns=worker_started, ended_ns=time.monotonic_ns())
            row = dict(result=result.to_dict(), record=record, accepted=False, revision_id=constraints.revision_id)
            self.records.append(row)
            if result.status != "succeeded":
                raise PlanningRejected("Lead execution did not complete")
            try:
                plan = PlanRevision.from_dict(io.read_json(result.outputs[0]))
                if (plan.job_id, plan.revision_id, plan.generation, plan.parent_revision_id) != (
                    job.job_id,
                    constraints.revision_id,
                    constraints.generation,
                    constraints.parent_revision_id,
                ):
                    raise ValueError("Wrong plan identity/ancestry")
                DAGScheduler().validate(plan, cfg)
                if not set(constraints.required_roles) <= {t.role for t in plan.tasks}:
                    raise ValueError("Plan omits required workflow roles")
                by_id = {t.task_id: t for t in plan.tasks}

                def ancestors(task):
                    result = set()
                    for d in task.dependencies:
                        result.add(d.task_id)
                        result.update(ancestors(by_id[d.task_id]))
                    return result

                for task in plan.tasks:
                    if task.retry_of and task.retry_of not in constraints.prior_execution_ids:
                        raise ValueError("Unknown retry predecessor")
                    roles = {by_id[k].role for k in ancestors(task)}
                    if task.role == "verification" and "repair" in constraints.required_roles and "repair" not in roles:
                        raise ValueError("Verification must follow repair")
                    if task.role == "repair" and not any(
                        d.input_role == "diagnosis_run" and by_id[d.task_id].role == "diagnosis" for d in task.dependencies
                    ):
                        raise ValueError("Repair must use an explicit diagnosis handoff")
                    if (
                        task.role == "diagnosis"
                        and {"log_analysis", "code_analysis"} <= set(constraints.required_roles)
                        and not {"log_analysis", "code_analysis"} <= roles
                    ):
                        raise ValueError("Diagnosis must wait for both specialists")
            except (ValueError, TypeError, KeyError) as error:
                row["rejection_type"] = type(error).__name__
                raise PlanningRejected("Lead plan rejected by controller") from None
            row["accepted"] = True
            row["plan"] = plan.to_dict()
            return plan
        finally:
            constraints.budget.settle(reservation, record)


@dataclass(frozen=True)
class ReplanningLimits:
    max_replans: int = 2
    max_total_tasks: int = 32

    def __post_init__(self):
        if not 0 <= self.max_replans <= 5 or not 1 <= self.max_total_tasks <= 128:
            raise ValueError("Invalid bounded replanning limits")


class ReplanningController:
    def __init__(self, planner=None):
        self.planner = planner or Planner()

    def execute_authorized(self, plan, config, *, policy, budget=None):
        """Bounded controller-authored revisions over the original Scheduler."""
        from .remediation import execute_authorized
        return execute_authorized(plan, config, policy=policy, budget=budget)

    def execute(
        self,
        job,
        config,
        models,
        *,
        limits=ReplanningLimits(),
        required_roles=("log_analysis", "code_analysis", "diagnosis", "repair", "verification"),
    ):
        import hashlib
        import time
        import uuid
        from pathlib import Path
        from researchcoder.adapters.artifacts import ArtifactIO
        from researchcoder.contracts import TaskInput
        from .budget import JobBudget
        from .workspaces import SourceSnapshot
        from researchcoder.runtime.tracing import TraceSink

        started = time.monotonic()
        budget = JobBudget(config.job_limits)
        io = ArtifactIO(config.artifact_root)
        trace = TraceSink(job.job_id, "replanning", "controller")
        trace.parent_id = trace.emit("replanning.started")
        scheduler = DAGScheduler()
        runs = []
        reuse = {}
        available = []
        signatures = set()
        total_tasks = 0
        reason = "max_replans"
        status = "incomplete"
        parent = None
        cfg = config
        for generation in range(limits.max_replans + 1):
            remaining = config.job_limits.max_runtime_seconds - (time.monotonic() - started)
            if remaining < 0.1:
                reason = "timeout"
                status = "failed"
                break
            if config.cancel_event is not None and config.cancel_event.is_set():
                reason = "interrupted"
                status = "cancelled"
                break
            if generation >= len(models):
                reason = "planner_model_unavailable"
                break
            revision = "r" + str(generation)
            cfg = replace(cfg, job_limits=replace(config.job_limits, max_runtime_seconds=remaining))
            trace.emit("planning.requested", revision_id=revision, count=generation)
            constraints = PlanningConstraints(
                config=cfg,
                model=models[generation],
                budget=budget,
                revision_id=revision,
                generation=generation,
                parent_revision_id=parent,
                required_roles=required_roles,
                prior_execution_ids=tuple(r.task_id for run in runs for r in run.results),
            )
            try:
                plan = self.planner.plan(job, available, constraints)
            except (PlanningRejected, BudgetExceeded) as error:
                reason = "budget_exhausted" if isinstance(error, BudgetExceeded) else "invalid_plan"
                trace.emit("planning.rejected", status="incomplete", revision_id=revision, reason=reason)
                break
            total_tasks += len(plan.tasks)
            if total_tasks > limits.max_total_tasks:
                reason = "task_limit"
                break
            # Planning time is part of the same Job deadline.
            remaining = config.job_limits.max_runtime_seconds - (time.monotonic() - started)
            if remaining < 0.1:
                reason = "timeout"
                status = "failed"
                break
            cfg = replace(cfg, job_limits=replace(config.job_limits, max_runtime_seconds=remaining))
            trace.emit("revision.started", revision_id=revision)
            run = scheduler.execute(plan, cfg, budget=budget, reuse=reuse)
            runs.append(run)
            parent = revision
            reuse = dict(scheduler.completed)
            trace.emit("revision.finished", revision_id=revision, status=run.status, artifact_id=run.trace_ref.artifact_id)
            if run.status == "succeeded":
                status = "succeeded"
                reason = "verified_workflow"
                break
            if run.status == "cancelled":
                status = "cancelled"
                reason = "interrupted"
                break
            if any(s.reason in ("budget_exhausted", "timeout") for s in run.states):
                reason = "budget_exhausted" if any(s.reason == "budget_exhausted" for s in run.states) else "timeout"
                break
            feedback = []
            for result in run.results:
                if result.verification:
                    feedback.append(
                        dict(
                            task_id=result.task_id,
                            status=result.verification.status,
                            revision=result.verification.checked_workspace_revision,
                            exit_code=result.verification.exit_code,
                            stdout=result.verification.stdout.text,
                            stderr=result.verification.stderr.text,
                            original_result_artifact=result.outputs[0].to_dict() if result.outputs else None,
                        )
                    )
            if not feedback:
                # Insufficient evidence can request a bounded investigation revision.
                feedback = [
                    dict(status=s.status, reason=s.reason, revision=s.resulting_revision or s.workspace_revision)
                    for s in run.states
                    if s.status not in SUCCESS_STATUSES
                ]
            # Exclude generated IDs/receipts/timestamps from the novelty signal.
            semantic = [{k: v for k, v in row.items() if k not in ("task_id", "original_result_artifact")} for row in feedback]
            signature = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
            if signature in signatures:
                reason = "no_new_evidence"
                trace.emit("replanning.stopped", reason=reason)
                break
            signatures.add(signature)
            available = dict(
                previous_revision=revision,
                feedback=feedback,
                summary=run.summary_ref.to_dict(),
                states=[s.to_dict() for s in run.states],
            )
            if generation == limits.max_replans:
                reason = "max_replans"
                break
            # A new candidate version invalidates old read observations. Preserve the
            # previous immutable snapshot and create a new one without rewriting it.
            candidates = [t.spec.workspace.root for t in cfg.templates.values() if t.workspace_mode == "candidate"]
            if candidates:
                source = candidates[0]
                next_snapshot = SourceSnapshot.create(
                    source,
                    Path(config.snapshot.workspace.root).parent
                    / ("snapshot-" + str(generation + 1) + "-" + uuid.uuid4().hex[:8]),
                )
                templates = dict(cfg.templates)
                failed_verification = next((f for f in feedback if f.get("status") == "failed"), None)
                failed_output = next((f.get("stdout") for f in feedback if f.get("stdout")), None)
                if failed_verification is not None:
                    # JSON-only verifier output need not contain an ERROR word.
                    # Supply the observed exit status as a log envelope, without
                    # changing the original verification bytes or novelty signal.
                    failed_output = "ERROR Authorized verification failed; inspect current source.\n" + (failed_verification.get("stdout") or "")
                if failed_output:
                    log = io.put_bytes(job.job_id, failed_output.encode(), media_type="text/plain")
                    for name, t in list(templates.items()):
                        if t.spec.kind == "diagnose":
                            templates[name] = replace(
                                t,
                                spec=replace(
                                    t.spec,
                                    inputs=tuple(i for i in t.spec.inputs if i.role != "raw_log")
                                    + (TaskInput(role="raw_log", reference=log),),
                                ),
                            )
                cfg = replace(cfg, snapshot=next_snapshot, templates=templates)
            trace.emit(
                "replanning.requested",
                revision_id="r" + str(generation + 1),
                reason="verification_failed_or_insufficient_evidence",
            )
        trace.emit(
            "replanning.finished",
            status=status,
            reason=reason,
            count=max(0, len(runs) - 1),
            elapsed_seconds=time.monotonic() - started,
        )
        trace_ref = io.put_json(job.job_id, {"schema_version": 1, "events": trace.merged([])})
        summary = dict(
            schema_version=1,
            status=status,
            reason=reason,
            replans=max(0, len(runs) - 1),
            runs=[r.to_dict() for r in runs],
            planning_records=self.planner.records,
            budget=budget.snapshot(),
            total_declared_tasks=total_tasks,
            trace_ref=trace_ref.to_dict(),
            elapsed_seconds=time.monotonic() - started,
            scope="Bounded in-memory planning over isolated Workers; no persistent recovery.",
        )
        summary_ref = io.put_json(job.job_id, summary)
        return dict(summary, summary_ref=summary_ref.to_dict())


SUCCESS_STATUSES = {"succeeded", "reused"}
