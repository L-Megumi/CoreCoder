"""Controller-owned static DAG dispatch; Workers retain the original Agent Loop."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from concurrent.futures import wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import threading
import time

from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import TaskSpec, TaskInput
from researchcoder.runtime import ExecutionConfig
from researchcoder.runtime.budget import BudgetExceeded
from researchcoder.runtime.pool import HarnessWorkerPool
from researchcoder.runtime.tracing import TraceSink
from .contracts import PlanRevision, DAGExecutionState, SchedulerResult, Role
from .budget import JobBudget, JobLimits
from .workspaces import SourceSnapshot
from .profiles import AgentRunner, RoleBudgetOverride
from .evidence import EvidenceExchange

SUCCESS = {"succeeded", "reused"}
TERMINAL = SUCCESS | {"failed", "incomplete", "cancelled", "blocked"}
KIND = {
    "log_analysis": "diagnose",
    "code_analysis": "diagnose",
    "diagnosis": "diagnose",
    "repair": "repair",
    "verification": "verify",
}


@dataclass(frozen=True)
class TaskTemplate:
    role: Role
    spec: TaskSpec
    config: ExecutionConfig
    workspace_mode: str = "snapshot"

    def __post_init__(self):
        if self.role not in KIND or self.spec.kind != KIND[self.role]:
            raise ValueError("Role must adapt an existing authorized workflow")
        if self.workspace_mode not in ("snapshot", "candidate"):
            raise ValueError("Unknown workspace mode")
        if self.workspace_mode == "snapshot" and (self.spec.kind != "diagnose" or self.spec.policy.writable_paths):
            raise ValueError("Snapshot is for read-only diagnostic roles")
        if self.role in ("repair", "verification") and self.workspace_mode != "candidate":
            raise ValueError("Mutation/verification requires a candidate version")


@dataclass(frozen=True)
class SchedulerConfig:
    templates: dict[str, TaskTemplate]
    artifact_root: str
    snapshot: SourceSnapshot
    max_workers: int = 2
    max_tasks: int = 32
    max_depth: int = 12
    job_limits: JobLimits = field(default_factory=JobLimits)
    cancel_event: object | None = None
    role_profiles: bool = False
    role_overrides: tuple[RoleBudgetOverride, ...] = ()

    def __post_init__(self):
        if not 1 <= self.max_workers <= 16 or not 1 <= self.max_tasks <= 64 or not 1 <= self.max_depth <= 16:
            raise ValueError("Invalid Scheduler constraints")
        if not Path(self.artifact_root).is_absolute() or not self.templates:
            raise ValueError("Trusted template registry and artifact root required")
        if any(not isinstance(t, TaskTemplate) for t in self.templates.values()):
            raise TypeError("Typed templates required")
        if (any(not isinstance(o, RoleBudgetOverride) for o in self.role_overrides)
                or len({o.role for o in self.role_overrides}) != len(self.role_overrides)):
            raise ValueError('Unique typed read-only role allowances required')


class Cancellation:
    def __init__(self, *events):
        self.events = tuple(e for e in events if e is not None)

    def is_set(self):
        return any(e.is_set() for e in self.events)

    @property
    def failure_kind(self):
        return (
            "timeout" if any(e.is_set() and getattr(e, "failure_kind", None) == "timeout" for e in self.events) else "interrupted"
        )


class StopSignal(threading.Event):
    failure_kind = "interrupted"


class DAGScheduler:
    def __init__(self):
        self._owner = threading.get_ident()
        self._executed = set()
        self.records = {}
        self.completed = {}

    def validate(self, plan, config):
        plan = PlanRevision.from_json(plan.to_json())
        if len(plan.tasks) > config.max_tasks or plan.depth() > config.max_depth:
            raise ValueError("Plan exceeds controller graph limits")
        for task in plan.tasks:
            if task.template_id not in config.templates:
                raise ValueError("Unknown trusted task template")
            template = config.templates[task.template_id]
            if task.role != template.role or template.spec.job_id != plan.job_id:
                raise ValueError("Planner cannot alter a template's role or job")
            if Path(template.config.artifact_root).resolve() != Path(config.artifact_root).resolve():
                raise ValueError("Templates must share the authorized Job artifact root")
        return plan

    def _prepare(self, plan, task, config, results):
        template = config.templates[task.template_id]
        spec = template.spec
        if template.workspace_mode == "snapshot":
            workspace = config.snapshot.verify()
        else:
            root = Path(spec.workspace.root).resolve()
            if root not in {Path(p).resolve() for p in template.config.allowed_workspaces}:
                raise ValueError("Candidate is outside the trusted template workspace set")
            workspace = replace(spec.workspace, source_manifest_sha256=workspace_revision(root))
            writers = [
                self.completed[d.task_id]["state"].resulting_revision
                for d in task.dependencies
                if d.task_id in self.completed
                and config.templates[next(t.template_id for t in plan.tasks if t.task_id == d.task_id)].role == "repair"
            ]
            if writers and any(v != workspace.source_manifest_sha256 for v in writers):
                raise ValueError("Candidate changed since upstream repair")
        if task.workspace_revision and workspace.source_manifest_sha256 != task.workspace_revision:
            raise ValueError("Explicit task source revision is stale")
        inputs = {i.role: i for i in spec.inputs}
        for dependency in task.dependencies:
            if dependency.input_role == "diagnosis_run":
                upstream = results[dependency.task_id]
                if not upstream.outputs or upstream.kind != "diagnose":
                    raise ValueError("Repair needs an original diagnosis artifact")
                inputs["diagnosis_run"] = TaskInput(role="diagnosis_run", reference=upstream.outputs[0])
        execution_id = plan.revision_id + "." + task.task_id
        spec = replace(spec, task_id=execution_id, objective=task.objective, workspace=workspace, inputs=tuple(inputs.values()))
        cfg = replace(
            template.config,
            shared_read=template.workspace_mode == "snapshot",
            artifact_namespace=execution_id,
            allowed_workspaces=(workspace.root,),
        )
        if config.role_profiles:
            override = next((o for o in config.role_overrides if o.role == task.role), None)
            cfg = AgentRunner.configure(task.role, spec, cfg, override=override)
        if cfg.context_policy is not None:
            cfg = replace(cfg, context_policy=replace(cfg.context_policy, role=task.role))
        producers = [results[d.task_id].task_id for d in task.dependencies if d.input_role == "evidence"]
        if producers:
            bundle = self.exchange.bundle(producers, workspace.source_manifest_sha256)
            bundle_ref = self.exchange.io.put_json(plan.job_id, bundle.to_dict())
            spec = replace(
                spec,
                inputs=tuple(i for i in spec.inputs if i.role != "evidence")
                + (TaskInput(role="evidence", reference=bundle_ref),),
            )
            cfg = (replace(cfg, evidence_bundle=bundle_ref, evidence_context="") if cfg.context_policy is not None
                   else replace(cfg, evidence_context=self.exchange.context(bundle)))
        canonical_spec = replace(spec, task_id=task.task_id).to_dict()
        canonical_spec["inputs"] = [
            dict(
                role=i.role,
                sha256=getattr(i.reference, "sha256", None) or i.reference.artifact.sha256,
                record_key=getattr(i.reference, "record_key", None),
            )
            for i in spec.inputs
        ]
        fingerprint_data = dict(
            template=task.template_id,
            role=task.role,
            spec=canonical_spec,
            model=cfg.model.to_dict(),
            limits=cfg.limits.to_dict(),
            instructions=cfg.system_instructions,
            context_policy=cfg.context_policy.to_dict() if cfg.context_policy is not None else None,
            diagnostic_tools=cfg.diagnostic_tools,
            maximum_policy=cfg.max_policy.to_dict(),
            permitted_verifications=[v.to_dict() for v in cfg.permitted_verifications],
            protected_files=[v.to_dict() for v in cfg.protected_files],
            upstream=[
                dict(
                    task_id=results[d.task_id].task_id,
                    attempt_id=results[d.task_id].attempt_id,
                    outputs=[r.sha256 for r in results[d.task_id].outputs],
                )
                for d in task.dependencies
            ],
        )
        if cfg.model_options is not None:
            fingerprint_data["model_options"] = cfg.model_options.to_dict()
        if cfg.record_requests:
            fingerprint_data["record_requests"] = True
        if cfg.optimization is not None:
            fingerprint_data['optimization'] = cfg.optimization.to_dict()
        fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return spec, cfg, fingerprint

    def execute(self, plan: PlanRevision, config: SchedulerConfig, *, budget=None, reuse=None, durable=None, resume=None):
        if threading.get_ident() != self._owner:
            raise RuntimeError("DAG state belongs to its Controller thread")
        plan = self.validate(plan, config)
        identity = (plan.job_id, plan.revision_id)
        if identity in self._executed:
            raise ValueError("Revision already submitted; no implicit replay")
        self._executed.add(identity)
        started = time.monotonic()
        budget = budget or JobBudget(config.job_limits)
        io = ArtifactIO(config.artifact_root)
        trace = TraceSink(plan.job_id, "scheduler", plan.revision_id)
        root_event = trace.emit("graph.started", revision_id=plan.revision_id)
        trace.parent_id = root_event
        states = {
            t.task_id: DAGExecutionState(job_id=plan.job_id, revision_id=plan.revision_id, task_id=t.task_id) for t in plan.tasks
        }
        results, running = {}, {}
        self.records, self.completed = {}, {}
        self.exchange = EvidenceExchange(plan.job_id, io)
        for logical, entry in (resume or {}).items():
            if logical not in states or entry["state"].revision_id != plan.revision_id:
                raise ValueError("Foreign recovery seed")
            state, result = entry["state"], entry["result"]
            self.exchange.submit(result, state.workspace_revision)
            states[logical], results[logical] = state, result
            self.records[logical] = entry["record"]
            self.completed[logical] = entry
            trace.emit("task.restored",logical_task_id=logical,revision_id=plan.revision_id,status=state.status)
        stop = StopSignal()
        cancel = Cancellation(stop, config.cancel_event)
        pool = HarnessWorkerPool(config.max_workers)
        stop_reason = None
        try:
            while any(s.status not in TERMINAL for s in states.values()):
                if stop_reason is None:
                    if config.cancel_event is not None and config.cancel_event.is_set():
                        stop_reason = "interrupted"
                    elif time.monotonic() - started >= config.job_limits.max_runtime_seconds:
                        stop_reason = "timeout"
                if stop_reason:
                    stop.failure_kind = stop_reason
                    stop.set()
                for task in plan.tasks:
                    state = states[task.task_id]
                    if state.status not in ("pending", "ready"):
                        continue
                    deps = [states[d.task_id].status for d in task.dependencies]
                    reason = stop_reason or ("upstream_failed" if any(s in TERMINAL - SUCCESS for s in deps) else None)
                    if reason:
                        status = "cancelled" if reason == "interrupted" else "blocked"
                        states[task.task_id] = replace(state, status=status, reason=reason)
                        trace.emit(
                            "task.blocked",
                            status=status,
                            logical_task_id=task.task_id,
                            revision_id=plan.revision_id,
                            reason=reason,
                        )
                        continue
                    if not all(s in SUCCESS for s in deps):
                        continue
                    states[task.task_id] = replace(state, status="ready")
                    if len(running) >= config.max_workers:
                        continue
                    template = config.templates[task.template_id]
                    # Snapshot reads are independent. Candidate access is exclusive,
                    # including verification commands; at most one writer is active.
                    candidate = template.workspace_mode == "candidate"
                    if candidate and any(entry["candidate"] for entry in running.values()):
                        continue
                    try:
                        spec, cfg, fingerprint = self._prepare(plan, task, config, results)
                        prior = (reuse or {}).get(task.task_id)
                        if (
                            prior
                            and prior["result"].status == "succeeded"
                            and prior["state"].input_fingerprint == fingerprint
                            and prior["state"].resulting_revision == spec.workspace.source_manifest_sha256
                        ):
                            old = prior["result"]
                            if old.attempt_id in {r.attempt_id for r in results.values()}:
                                raise ValueError("Duplicate reused completion")
                            self.exchange.submit(old, spec.workspace.source_manifest_sha256)
                            state = replace(
                                states[task.task_id],
                                status="reused",
                                execution_task_id=old.task_id,
                                attempt_id=old.attempt_id,
                                input_fingerprint=fingerprint,
                                workspace_revision=spec.workspace.source_manifest_sha256,
                                resulting_revision=prior["state"].resulting_revision,
                            )
                            states[task.task_id], results[task.task_id] = state, old
                            self.records[task.task_id] = dict(
                                prior["record"], reused=True, reused_from_revision=prior["state"].revision_id
                            )
                            self.completed[task.task_id] = dict(result=old, record=self.records[task.task_id], state=state)
                            trace.emit(
                                "task.reused",
                                logical_task_id=task.task_id,
                                revision_id=plan.revision_id,
                                workspace_revision=state.workspace_revision,
                            )
                            continue
                        key = plan.revision_id + "." + task.task_id
                        budget.reserve(key, cfg.limits)
                    except BudgetExceeded:
                        if running:
                            continue  # Retry admission after another reservation settles.
                        states[task.task_id] = replace(states[task.task_id], status="incomplete", reason="budget_exhausted")
                        trace.emit(
                            "job.budget_denied",
                            logical_task_id=task.task_id,
                            status="incomplete",
                            failure_kind="budget_exhausted",
                        )
                        continue
                    except (ValueError, TypeError, OSError, KeyError) as error:
                        states[task.task_id] = replace(states[task.task_id], status="incomplete", reason=type(error).__name__)
                        trace.emit(
                            "task.preflight_error", logical_task_id=task.task_id, status="incomplete", reason=type(error).__name__
                        )
                        continue
                    task_event = trace.emit(
                        "task.submitted",
                        logical_task_id=task.task_id,
                        role=task.role,
                        revision_id=plan.revision_id,
                        workspace_revision=spec.workspace.source_manifest_sha256,
                    )
                    if cfg.evidence_context:
                        trace.emit(
                            "evidence.aggregated",
                            parent_id=task_event,
                            logical_task_id=task.task_id,
                            workspace_revision=spec.workspace.source_manifest_sha256,
                        )
                    for d in task.dependencies:
                        trace.emit(
                            "graph.dependency",
                            parent_id=task_event,
                            logical_task_id=task.task_id,
                            upstream_task_id=d.task_id,
                            revision_id=plan.revision_id,
                        )
                    trace.emit("job.budget_reserved", parent_id=task_event, reservation_id=key)
                    states[task.task_id] = replace(
                        states[task.task_id],
                        status="running",
                        execution_task_id=spec.task_id,
                        attempts=1,
                        input_fingerprint=fingerprint,
                        workspace_revision=spec.workspace.source_manifest_sha256,
                    )
                    cfg = replace(cfg, cancel_event=Cancellation(cancel, cfg.cancel_event))
                    if durable is not None:
                        cfg = durable.dispatch(task, spec, cfg, fingerprint, budget)
                    future = pool.submit(spec, cfg)
                    running[future] = dict(task=task, key=key, event=task_event, candidate=candidate, spec=spec)
                if not running:
                    if any(s.status not in TERMINAL for s in states.values()):
                        raise RuntimeError("Scheduler made no progress in a validated DAG")
                    break
                finished, _ = wait(tuple(running), timeout=0.03, return_when=FIRST_COMPLETED)
                for future in finished:
                    item = running.pop(future)
                    task, key = item["task"], item["key"]
                    result, record = None, None
                    try:
                        execution = future.result()
                        result, record = execution.result, execution.record
                        if (
                            result.task_id != key
                            or result.job_id != plan.job_id
                            or result.attempt_id in {r.attempt_id for r in results.values()}
                        ):
                            raise ValueError("Duplicate or foreign task completion")
                        after = workspace_revision(item["spec"].workspace.root)
                        if not item["candidate"]:
                            config.snapshot.verify()
                        self.exchange.submit(result, states[task.task_id].workspace_revision)
                        state = replace(
                            states[task.task_id],
                            status=result.status,
                            attempt_id=result.attempt_id,
                            resulting_revision=after,
                            reason=result.failure_kind,
                        )
                        states[task.task_id] = state
                        results[task.task_id] = result
                        record = dict(
                            record,
                            started_ns=execution.started_ns,
                            ended_ns=execution.ended_ns,
                            revision_id=plan.revision_id,
                            logical_task_id=task.task_id,
                            role=task.role,
                            input_fingerprint=state.input_fingerprint,
                            workspace_revision=state.workspace_revision,
                            resulting_revision=after,
                        )
                        self.records[task.task_id] = record
                        self.completed[task.task_id] = dict(result=result, record=record, state=state)
                    except (ValueError, TypeError, OSError, RuntimeError) as error:
                        states[task.task_id] = replace(states[task.task_id], status="failed", reason=type(error).__name__)
                        trace.emit("task.collection_error", parent_id=item["event"], status="failed", reason=type(error).__name__)
                    settlement = budget.settle(key, record)
                    if durable is not None:
                        durable.complete(task, states[task.task_id], result, record, budget)
                    trace.emit(
                        "job.budget_settled",
                        parent_id=item["event"],
                        reservation_id=key,
                        usage={"unknown": settlement["usage_unknown"], "charged_tokens": settlement["charged"]["total_tokens"]},
                    )
                    trace.emit(
                        "task.collected",
                        parent_id=item["event"],
                        logical_task_id=task.task_id,
                        revision_id=plan.revision_id,
                        status=states[task.task_id].status,
                        pid=(record or {}).get("worker_pid"),
                        artifact_id=((record or {}).get("trace") or {}).get("artifact_id"),
                    )
        finally:
            stop.set()
            pool.close()
        elapsed = time.monotonic() - started
        statuses = [s.status for s in states.values()]
        status = (
            "succeeded"
            if all(s in SUCCESS for s in statuses)
            else "cancelled"
            if stop_reason == "interrupted"
            else "failed"
            if "failed" in statuses or stop_reason == "timeout"
            else "incomplete"
        )
        trace.emit(
            "graph.finished", status=status, revision_id=plan.revision_id, elapsed_seconds=elapsed, failure_kind=stop_reason
        )
        trace_ref = io.put_json(plan.job_id, {"schema_version": 1, "events": trace.merged([])})
        summary = dict(
            schema_version=1,
            plan=plan.to_dict(),
            states=[s.to_dict() for s in states.values()],
            records=self.records,
            budget=budget.snapshot(),
            elapsed_seconds=elapsed,
            status=status,
        )
        summary_ref = io.put_json(plan.job_id, summary)
        return SchedulerResult(
            job_id=plan.job_id,
            revision_id=plan.revision_id,
            status=status,
            states=tuple(states.values()),
            results=tuple(results[t.task_id] for t in plan.tasks if t.task_id in results),
            trace_ref=trace_ref,
            summary_ref=summary_ref,
            elapsed_seconds=elapsed,
        )
