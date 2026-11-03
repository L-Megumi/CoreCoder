"""Checkpoint/reconcile outside the original DAG Scheduler. No replacement loop."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

from corecoder.workflow import workspace_revision
from researchcoder.contracts import Contract, ArtifactRef, TaskResult
from researchcoder.orchestration import DAGScheduler
from researchcoder.orchestration.contracts import (
    PlanRevision,
    DAGExecutionState,
    SchedulerResult,
)
from researchcoder.orchestration.budget import JobBudget, JobLimits
from researchcoder.runtime.tracing import TraceSink
from .journal import (
    OperationJournal,
    JournalServer,
    JournalEndpoint,
    process_identity,
    file_hash,
)
from .store import CheckpointError, digest, boot_id


@dataclass(frozen=True, kw_only=True)
class RecoveryLimits(Contract):
    max_recoveries: int = 3
    max_replans: int = 3
    max_total_tasks: int = 256
    schema_version: int = 1

    def validate(self):
        if (
            not 0 <= self.max_recoveries <= 10
            or not 0 <= self.max_replans <= 10
            or not 1 <= self.max_total_tasks <= 640
        ):
            raise ValueError("Invalid recovery bounds")


def authority(config):
    """Only caller-supplied templates reconstruct executable settings on restart."""
    templates = {}
    for key, t in config.templates.items():
        c = t.config
        templates[key] = dict(
            role=t.role,
            workspace_mode=t.workspace_mode,
            spec=t.spec.to_dict(),
            allowed_workspaces=list(c.allowed_workspaces),
            max_policy=c.max_policy.to_dict(),
            model=c.model.to_dict(),
            limits=c.limits.to_dict(),
            permitted_verifications=[v.to_dict() for v in c.permitted_verifications],
            protected_files=[p.to_dict() for p in c.protected_files],
            diagnostic_tools=list(c.diagnostic_tools),
            context_policy=c.context_policy.to_dict() if c.context_policy else None,
            system_instructions=c.system_instructions,
            worker_command=c.worker_command,
            worker_python=c.worker_python,
        )
        if c.model_options is not None:
            templates[key]["model_options"] = c.model_options.to_dict()
        if c.record_requests:
            templates[key]["record_requests"] = True
        if c.optimization is not None:
            templates[key]['optimization'] = c.optimization.to_dict()
    body = (
        dict(
            templates=templates,
            snapshot=config.snapshot.workspace.to_dict(),
            manifest=config.snapshot.manifest,
            artifact_root=config.artifact_root,
            job_limits=config.job_limits.to_dict(),
            max_workers=config.max_workers,
            max_tasks=config.max_tasks,
            max_depth=config.max_depth,
            role_profiles=config.role_profiles,
        )
    )
    if config.role_overrides:
        body['role_overrides'] = [o.to_dict() for o in config.role_overrides]
    return digest(body)


def state_body(job_id, status, **values):
    return dict(schema_version=1, job_id=job_id, status=status, **values)


class TaskObserver:
    def __init__(self, coordinator, key, spec, authority_ref):
        self.owner = coordinator
        self.key = key
        self.spec = spec
        self.authority_ref = authority_ref
        self.endpoint = None

    def attempt_started(self, spec, attempt):
        store = self.owner.store
        row = store.get("tasks", self.key)
        if row is None or row["status"] != "running" or row["value"].get("attempt_id"):
            raise CheckpointError("Duplicate durable attempt")
        value = dict(row["value"], attempt_id=attempt)
        attempt_value = state_body(
            spec.job_id,
            "running",
            task_id=spec.task_id,
            attempt_id=attempt,
            revision_id=self.owner.plan.revision_id,
            logical_task_id=value["logical_task_id"],
            authority_ref=self.authority_ref.to_dict(),
        )
        store.commit(
            [
                ("tasks", self.key, row["version"], "running", value),
                ("attempts", attempt, None, "running", attempt_value),
            ],
            event_type="attempt.started",
        )
        self.owner.journal.register(spec, attempt, self.authority_ref)
        self.endpoint = JournalEndpoint(
            socket_path=self.owner.server.path,
            job_id=spec.job_id,
            task_id=spec.task_id,
            attempt_id=attempt,
        )

    def worker_started(self, attempt, pid, directory):
        identity = process_identity(pid)
        if identity is None or identity["pgid"] != pid:
            raise CheckpointError("Supervised process identity unavailable")
        value = state_body(
            self.spec.job_id,
            "running",
            **identity,
            attempt_id=attempt,
            task_id=self.spec.task_id,
            revision_id=self.owner.plan.revision_id,
            request_path=str(directory / "request.json"),
            response_path=str(directory / "response.json"),
        )
        self.owner.store.commit(
            [("workers", attempt, None, "running", value)], event_type="worker.leased"
        )
        self.owner.fault("worker_started")

    def worker_finished(self, attempt, reason, exit_code):
        row = self.owner.store.get("workers", attempt)
        if row:
            value = dict(
                row["value"], status="exited", reason=reason, exit_code=exit_code
            )
            self.owner.store.commit(
                [("workers", attempt, row["version"], "exited", value)],
                event_type="worker.exited",
            )


class RecoveryCoordinator:
    def __init__(self, store, *, limits=RecoveryLimits(), fault=None):
        self.store = store
        self.limits = limits
        self.fault = fault or (lambda _: None)
        self.journal = OperationJournal(store, self.fault)
        self.server = None
        self.last_result = None
        self.last_report = None

    def _budget(self, job_id):
        row = self.store.get("budgets", job_id)
        saved = row["value"]["snapshot"]
        b = JobBudget(JobLimits.from_dict(saved["limits"]))
        b.pending = saved["pending"]
        b.settlements = saved["settlements"]
        b.charged = saved["charged"]
        if any(type(v) is not int or v < 0 for v in b.charged.values()):
            raise CheckpointError("Invalid budget accounting")
        computed = {
            k: sum(s["charged"][k] for s in b.settlements.values()) for k in b.charged
        }
        if computed != b.charged:
            raise CheckpointError("Budget settlement inconsistency")
        for allocation in list(b.pending.values()) + [
            s["charged"] for s in b.settlements.values()
        ]:
            if set(allocation) != set(b.charged) or any(
                type(v) is not int or v < 0 for v in allocation.values()
            ):
                raise CheckpointError("Invalid saved allocation")
        return b

    def _budget_update(self, job, budget):
        row = self.store.get("budgets", job)
        return (
            "budgets",
            job,
            row["version"] if row else None,
            "active",
            state_body(job, "active", snapshot=budget.snapshot()),
        )

    def _open_server(self):
        if self.server is None:
            directory = Path(tempfile.mkdtemp(prefix="researchcoder-journal-"))
            directory.chmod(0o700)
            self.server = JournalServer(self.journal, directory / "rpc.sock")

    def _append(self, plan, config, budget, *, recovery=False, seed=None):
        store = self.store
        job = store.get("jobs", plan.job_id)
        fingerprint = authority(config)
        if plan.generation > self.limits.max_replans:
            raise CheckpointError("Persisted revision limit exhausted")
        if job:
            old = job["value"]
            if (
                old["authority"] != fingerprint
                or old["recovery_limits"] != self.limits.to_dict()
            ):
                raise CheckpointError("Trusted execution authority changed")
            if (
                plan.parent_revision_id != old["latest_revision"]
                or plan.generation != old["generation"] + 1
            ):
                raise CheckpointError("Stale PlanRevision ancestry")
            if old["status"] == "cancelled" or old.get("failure_kind") in (
                "budget_exhausted",
                "timeout",
            ):
                raise CheckpointError("Terminal stop cannot be bypassed")
            parent = store.get("plans", plan.job_id + "/" + old["latest_revision"])
            updated = dict(
                old,
                status="running",
                latest_revision=plan.revision_id,
                generation=plan.generation,
                total_declared_tasks=old["total_declared_tasks"] + len(plan.tasks),
            )
            updates = [
                (
                    "plans",
                    parent["id"],
                    parent["version"],
                    "superseded",
                    dict(parent["value"], status="superseded"),
                )
            ]
        else:
            if plan.generation != 0:
                raise CheckpointError("Initial revision must have no parent")
            roots = {
                str(Path(t.spec.workspace.root).resolve())
                for t in config.templates.values()
            } | {config.snapshot.workspace.root}
            updated = state_body(
                plan.job_id,
                "open",
                authority=fingerprint,
                latest_revision=plan.revision_id,
                generation=0,
                recovery_limits=self.limits.to_dict(),
                recoveries=0,
                total_declared_tasks=len(plan.tasks),
                boot_id=boot_id(),
                deadline=time.monotonic() + config.job_limits.max_runtime_seconds,
                workspace_versions={r: workspace_revision(r) for r in roots},
                result_ref=None,
                failure_kind=None,
            )
            updates = []
        if updated["total_declared_tasks"] > self.limits.max_total_tasks:
            raise CheckpointError("Persistent task count limit exceeded")
        plan_ref = store.io.put_json(plan.job_id, plan.to_dict())
        updates.extend(
            [
                (
                    "jobs",
                    plan.job_id,
                    job["version"] if job else None,
                    updated["status"],
                    updated,
                ),
                (
                    "plans",
                    plan.job_id + "/" + plan.revision_id,
                    None,
                    "committed",
                    state_body(
                        plan.job_id,
                        "committed",
                        plan=plan.to_dict(),
                        plan_ref=plan_ref.to_dict(),
                        recovery_revision=recovery,
                    ),
                ),
                self._budget_update(plan.job_id, budget),
            ]
        )
        for t in plan.tasks:
            key = plan.job_id + "/" + plan.revision_id + "/" + t.task_id
            initial = state_body(
                plan.job_id,
                "pending",
                logical_task_id=t.task_id,
                revision_id=plan.revision_id,
                attempt_id=None,
                role=t.role,
            )
            updates.append(("tasks", key, None, "pending", initial))
            if t.task_id in (seed or {}):
                entry = seed[t.task_id]
                state = replace(
                    entry["state"],
                    revision_id=plan.revision_id,
                    status="reused"
                    if entry["state"].status in ("succeeded", "reused")
                    else entry["state"].status,
                )
                entry = dict(entry, state=state)
                seed[t.task_id] = entry
                packet = self.store.io.put_json(
                    plan.job_id,
                    dict(
                        state=state.to_dict(),
                        result=entry["result"].to_dict(),
                        record=entry["record"],
                    ),
                )
                updates.append(
                    (
                        "tasks",
                        key,
                        0,
                        state.status,
                        dict(
                            initial,
                            status=state.status,
                            packet=packet.to_dict(),
                            attempt_id=state.attempt_id,
                        ),
                    )
                )
        self.fault("before_plan_commit")
        store.commit(updates, event_type="plan.committed")
        self.fault("after_plan_commit")
        if not job:
            self.fault("job_created")

    def execute(self, plan, config, *, budget=None):
        plan = DAGScheduler().validate(plan, config)
        if Path(self.store.io.root) != Path(config.artifact_root).resolve():
            raise CheckpointError("Artifact store changed")
        if any(
            self.store.path.is_relative_to(Path(t.spec.workspace.root).resolve())
            for t in config.templates.values()
        ):
            raise CheckpointError("Checkpoint must be outside editable workspaces")
        budget = budget or (
            self._budget(plan.job_id)
            if self.store.get("jobs", plan.job_id)
            else JobBudget(config.job_limits)
        )
        self._append(plan, config, budget)
        return self._run(plan, config, budget, {})

    def _run(self, plan, config, budget, seed):
        self.plan = plan
        self.config = config
        self._open_server()
        row = self.store.get("jobs", plan.job_id)
        if row["value"]["boot_id"] != boot_id():
            raise CheckpointError("Host boot changed; explicit review required")
        remaining = row["value"]["deadline"] - time.monotonic()
        if remaining < 0.1:
            raise CheckpointError("Original Job deadline exhausted")
        if row["status"] != "running":
            self.store.commit(
                [
                    (
                        "jobs",
                        plan.job_id,
                        row["version"],
                        "running",
                        dict(row["value"], status="running"),
                    )
                ],
                event_type="job.running",
            )
        cfg = replace(
            config,
            job_limits=replace(
                config.job_limits,
                max_runtime_seconds=min(
                    config.job_limits.max_runtime_seconds, remaining
                ),
            ),
        )
        scheduler = DAGScheduler()
        self.scheduler = scheduler
        outcome = scheduler.execute(plan, cfg, budget=budget, durable=self, resume=seed)
        result_ref = self.store.io.put_json(plan.job_id, outcome.to_dict())
        row = self.store.get("jobs", plan.job_id)
        failure = next(
            (
                s.reason
                for s in outcome.states
                if s.reason in ("budget_exhausted", "timeout", "interrupted")
            ),
            None,
        )
        if failure is None:
            failure = next(
                (r.failure_kind for r in outcome.results if r.failure_kind), None
            )
        updated = dict(
            row["value"],
            status=outcome.status,
            result_ref=result_ref.to_dict(),
            failure_kind=failure,
        )
        updates = [
            ("jobs", plan.job_id, row["version"], outcome.status, updated),
            self._budget_update(plan.job_id, budget),
        ]
        for state in outcome.states:
            key = plan.job_id + "/" + plan.revision_id + "/" + state.task_id
            task = self.store.get("tasks", key)
            if task["status"] == "pending":
                updates.append(
                    (
                        "tasks",
                        key,
                        task["version"],
                        state.status,
                        dict(task["value"], status=state.status, state=state.to_dict()),
                    )
                )
        self.store.commit(updates, event_type="job.finished")
        self.last_result = outcome
        return outcome

    def dispatch(self, task, spec, cfg, fingerprint, budget):
        key = self.plan.job_id + "/" + self.plan.revision_id + "/" + task.task_id
        row = self.store.get("tasks", key)
        if row["status"] != "pending":
            raise CheckpointError("Task is not pending")
        authority_ref = self.store.io.put_json(
            spec.job_id,
            dict(
                schema_version=1,
                spec=spec.to_dict(),
                authority=authority(self.config),
                fingerprint=fingerprint,
            ),
        )
        value = dict(
            row["value"],
            status="running",
            fingerprint=fingerprint,
            workspace_revision=spec.workspace.source_manifest_sha256,
            workspace_root=spec.workspace.root,
            authority_ref=authority_ref.to_dict(),
        )
        self.store.commit(
            [
                ("tasks", key, row["version"], "running", value),
                self._budget_update(spec.job_id, budget),
            ],
            event_type="task.dispatched",
        )
        self.fault("task_dispatched")
        return replace(
            cfg, durable_observer=TaskObserver(self, key, spec, authority_ref)
        )

    def complete(self, task, state, result, record, budget):
        if result is None or record is None:
            raise CheckpointError("Cannot checkpoint absent result")
        if (
            result.verification is not None
            and result.verification.status == "passed"
            and result.verification.checked_workspace_revision
            != state.resulting_revision
        ):
            raise CheckpointError(
                "Verification changed source version; result requires reconciliation"
            )
        key = self.plan.job_id + "/" + self.plan.revision_id + "/" + task.task_id
        row = self.store.get("tasks", key)
        if (
            row["status"] != "running"
            or row["value"].get("attempt_id") != result.attempt_id
        ):
            raise CheckpointError("Late or foreign completion")
        packet = self.store.io.put_json(
            result.job_id,
            dict(state=state.to_dict(), result=result.to_dict(), record=record),
        )
        self.fault("task_artifact_published")
        value = dict(
            row["value"],
            status=state.status,
            packet=packet.to_dict(),
            resulting_revision=state.resulting_revision,
        )
        attempt = self.store.get("attempts", result.attempt_id)
        job = self.store.get("jobs", result.job_id)
        roots = dict(job["value"]["workspace_versions"])
        roots[row["value"]["workspace_root"]] = state.resulting_revision
        updates = [
            ("tasks", key, row["version"], state.status, value),
            (
                "attempts",
                result.attempt_id,
                attempt["version"],
                result.status,
                dict(
                    attempt["value"], status=result.status, result_ref=packet.to_dict()
                ),
            ),
            (
                "jobs",
                result.job_id,
                job["version"],
                "running",
                dict(job["value"], workspace_versions=roots),
            ),
            self._budget_update(result.job_id, budget),
        ]
        if result.verification and result.verification.status == "passed":
            for operation in self.store.rows("operations", result.job_id):
                op = operation["value"]
                if (
                    operation["status"] == "applied"
                    and op["path"]
                    and file_hash(Path(op["path"])) == op["after_sha256"]
                ):
                    updates.append(
                        (
                            "operations",
                            operation["id"],
                            operation["version"],
                            "verified",
                            dict(
                                op, status="verified", verification_ref=packet.to_dict()
                            ),
                        )
                    )
        self.store.commit(updates, event_type="task.completed")
        if result.kind == "repair" and result.status == "succeeded":
            self.fault("after_patch_before_verification")

    def _completed(self, plan):
        result = {}
        for row in self.store.rows("tasks", plan.job_id):
            v = row["value"]
            if v["revision_id"] != plan.revision_id or not v.get("packet"):
                continue
            packet = self.store.io.read_json(ArtifactRef.from_dict(v["packet"]))
            state = DAGExecutionState.from_dict(packet["state"])
            task_result = TaskResult.from_dict(packet["result"])
            state = replace(state, revision_id=plan.revision_id, status=row["status"])
            if (
                state.attempt_id != task_result.attempt_id
                or task_result.task_id != state.execution_task_id
            ):
                raise CheckpointError("Result/attempt identity mismatch")
            result[state.task_id] = dict(
                state=state, result=task_result, record=packet["record"]
            )
        return result

    def _cleanup_workers(self, job_id):
        decisions = []
        for row in self.store.rows("workers", job_id):
            if row["status"] != "running":
                continue
            v = row["value"]
            identity = process_identity(v["pid"])
            if identity and identity["state"] != "Z":
                if (
                    identity["boot_id"] != v["boot_id"]
                    or identity["start_ticks"] != v["start_ticks"]
                    or identity["pgid"] != v["pid"]
                ):
                    raise CheckpointError(
                        "Worker PID reused; refuse unrelated process termination"
                    )
                argv = Path(f"/proc/{v['pid']}/cmdline").read_bytes().split(b"\x00")
                if (
                    b"researchcoder.durable.guard" not in argv
                    or v["request_path"].encode() not in argv
                ):
                    raise CheckpointError(
                        "Process does not match supervised guard command"
                    )
                request = Path(v["request_path"])
                if (
                    request.is_symlink()
                    or request.name != "request.json"
                    or not request.parent.name.startswith("researchcoder-worker-")
                    or request.parent.parent != Path("/tmp")
                ):
                    raise CheckpointError("Untrusted Worker lease path")
                data = json.loads(request.read_text())
                if (
                    data["spec"]["job_id"],
                    data["spec"]["task_id"],
                    data["attempt_id"],
                ) != (job_id, v["task_id"], v["attempt_id"]):
                    raise CheckpointError("Worker lease/request identity mismatch")
                os.killpg(v["pid"], signal.SIGTERM)
                time.sleep(0.1)
                try:
                    os.killpg(v["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.store.commit(
                [
                    (
                        "workers",
                        row["id"],
                        row["version"],
                        "lost",
                        dict(v, status="lost"),
                    )
                ],
                event_type="worker.reconciled",
            )
            decisions.append((v["task_id"], "controller_crash_worker_fenced"))
        return decisions

    def _authorize_operation(self, operation, config):
        op = operation["value"]
        attempt = self.store.get("attempts", op["attempt_id"])
        if not attempt or attempt["value"]["task_id"] != op["task_id"]:
            raise CheckpointError("Unregistered operation owner")
        revision = attempt["value"]["revision_id"]
        logical = attempt["value"]["logical_task_id"]
        plan = PlanRevision.from_dict(
            self.store.get("plans", op["job_id"] + "/" + revision)["value"]["plan"]
        )
        task = next(t for t in plan.tasks if t.task_id == logical)
        template = config.templates[task.template_id]
        if op["tool"] in ("edit_file", "write_file"):
            root = Path(template.spec.workspace.root).resolve()
            given = Path(op["path"])
            if not given.is_absolute() or not given.resolve().is_relative_to(root):
                raise CheckpointError("Stored operation escaped trusted workspace")
            if (
                given.relative_to(root).as_posix()
                not in template.spec.policy.writable_paths
                or op["tool"] not in template.spec.policy.allowed_tool_names
            ):
                raise CheckpointError("Stored operation exceeds trusted tool policy")
            if any(
                p.is_symlink()
                for p in (given, *given.parents)
                if p.is_relative_to(root)
            ):
                raise CheckpointError("Stored operation symlink")
            patch = self.store.io.read_json(ArtifactRef.from_dict(op["patch"]))
            if (
                patch["path"] != str(given)
                or patch["after"]["sha256"] != op["expected_after_sha256"]
            ):
                raise CheckpointError("Intent/patch mismatch")
        elif (
            op["tool"] != "verify_command"
            or not template.spec.policy.allow_verification
        ):
            raise CheckpointError("Stored command not authorized")

    def recover(self, job_id, config):
        started = time.monotonic()
        self.store.validate()
        job = self.store.get("jobs", job_id)
        if not job:
            raise CheckpointError("No Job checkpoint")
        value = job["value"]
        if self.store.io.root != Path(config.artifact_root).resolve():
            raise CheckpointError("Recovery ArtifactIO root changed")
        if (
            value["authority"] != authority(config)
            or value["recovery_limits"] != self.limits.to_dict()
        ):
            raise CheckpointError("Trusted recovery configuration mismatch")
        plan = PlanRevision.from_dict(
            self.store.get("plans", job_id + "/" + value["latest_revision"])["value"][
                "plan"
            ]
        )
        DAGScheduler().validate(plan, config)
        config.snapshot.verify()
        budget = self._budget(job_id)
        before = budget.snapshot()
        if budget.limits != config.job_limits:
            raise CheckpointError("Persisted budget limits differ from trusted limits")
        trace = TraceSink(
            job_id,
            "recovery",
            "recovery-" + str(value["recoveries"]) + "-" + uuid.uuid4().hex,
        )
        trace.emit("recovery.started", revision_id=plan.revision_id)
        if (
            job["status"] in ("open", "running")
            and value["recoveries"] < self.limits.max_recoveries
        ):
            self.store.commit(
                [
                    (
                        "jobs",
                        job_id,
                        job["version"],
                        job["status"],
                        dict(value, recoveries=value["recoveries"] + 1),
                    )
                ],
                event_type="recovery.entered",
            )
        decisions = self._cleanup_workers(job_id)
        unknown = []
        lost = []
        if value["boot_id"] != boot_id():
            unknown.append("host_boot_changed")
        if value["recoveries"] >= self.limits.max_recoveries:
            unknown.append("recovery_limit")
        for op in self.store.rows("operations", job_id):
            self._authorize_operation(op, config)
            if op["status"] in ("prepared", "started", "unknown"):
                reconciled = self.journal.reconcile(op)
                decisions.append(
                    (
                        reconciled["task_id"],
                        reconciled["reconcile_reason"] or reconciled["status"],
                    )
                )
                if reconciled["status"] in ("applied", "unknown"):
                    unknown.append(reconciled["task_id"])
        trusted_roots = {
            str(Path(t.spec.workspace.root).resolve())
            for t in config.templates.values()
        } | {config.snapshot.workspace.root}
        if set(value["workspace_versions"]) != trusted_roots:
            raise CheckpointError("Stored workspace roots exceed trusted configuration")
        for root, expected in value["workspace_versions"].items():
            if workspace_revision(root) != expected:
                unknown.append("workspace_revision_mismatch")
        for task in self.store.rows("tasks", job_id):
            v = task["value"]
            if v["revision_id"] != plan.revision_id or task["status"] not in (
                "running",
                "lost",
                "unknown",
            ):
                continue
            lost.append(task)
            key = plan.revision_id + "." + v["logical_task_id"]
            if key in budget.pending:
                budget.settle(key, None)
            # Preserve the reservation identity but free the new revision's key.
            decisions.append((v["logical_task_id"], "lost_attempt_charged_unknown"))
            if config.templates[
                next(
                    t.template_id
                    for t in plan.tasks
                    if t.task_id == v["logical_task_id"]
                )
            ].config.worker_command:
                unknown.append(v["logical_task_id"])
            if any(
                op["value"]["attempt_id"] == v.get("attempt_id")
                and op["status"] in ("applied", "verified", "unknown")
                for op in self.store.rows("operations", job_id)
            ):
                unknown.append(v["logical_task_id"])
                decisions.append(
                    (
                        v["logical_task_id"],
                        "effect_completed_without_task_result_no_replay",
                    )
                )
        seed = self._completed(plan)
        for entry in seed.values():
            checked = entry["result"].verification
            if (
                checked is not None
                and checked.status == "passed"
                and checked.checked_workspace_revision
                != entry["state"].resulting_revision
            ):
                unknown.append(entry["state"].task_id)
        updates = [self._budget_update(job_id, budget)]
        for task in lost:
            if task["status"] != "running":
                continue
            v = task["value"]
            updates.append(
                (
                    "tasks",
                    task["id"],
                    task["version"],
                    "unknown" if unknown else "lost",
                    dict(v, status="unknown" if unknown else "lost"),
                )
            )
            if v.get("attempt_id"):
                attempt = self.store.get("attempts", v["attempt_id"])
                updates.append(
                    (
                        "attempts",
                        attempt["id"],
                        attempt["version"],
                        "unknown" if unknown else "lost",
                        dict(attempt["value"], status="unknown" if unknown else "lost"),
                    )
                )
        self.store.commit(updates, event_type="attempts_and_budget.reconciled")
        job = self.store.get("jobs", job_id)
        self.fault("recovery_reconciled")
        disposition = (
            "needs_review"
            if unknown
            else "completed"
            if job["status"] == "succeeded"
            else "stopped"
            if job["status"] in ("cancelled", "failed", "incomplete")
            else "resumed"
        )
        if not unknown and job["status"] == "succeeded":
            self.last_result = SchedulerResult.from_dict(
                self.store.io.read_json(
                    ArtifactRef.from_dict(job["value"]["result_ref"])
                )
            )
        elif disposition == "resumed":
            if job["value"]["deadline"] <= time.monotonic():
                disposition = "stopped"
                decisions.append(("job", "timeout_no_budget_reset"))
                self.store.commit(
                    [
                        (
                            "jobs",
                            job_id,
                            job["version"],
                            "failed",
                            dict(job["value"], status="failed", failure_kind="timeout"),
                        )
                    ],
                    event_type="recovery.deadline_exhausted",
                )
            else:
                if lost:
                    plan = replace(
                        plan,
                        revision_id="r" + str(plan.generation + 1),
                        generation=plan.generation + 1,
                        parent_revision_id=plan.revision_id,
                    )
                    self._append(plan, config, budget, recovery=True, seed=seed)
                job = self.store.get("jobs", job_id)
                self.store.commit(
                    [
                        (
                            "jobs",
                            job_id,
                            job["version"],
                            "running",
                            dict(job["value"], status="running"),
                        )
                    ],
                    event_type="recovery.admitted",
                )
                self._run(plan, config, budget, seed)
        if unknown:
            job = self.store.get("jobs", job_id)
            if job["status"] != "needs_review":
                self.store.commit(
                    [
                        (
                            "jobs",
                            job_id,
                            job["version"],
                            "needs_review",
                            dict(job["value"], status="needs_review"),
                        )
                    ],
                    event_type="recovery.quarantined",
                )
        for task, reason in decisions:
            trace.emit(
                "recovery.decision",
                logical_task_id=task,
                reason=reason,
                revision_id=plan.revision_id,
            )
        unknown_tasks = sorted(
            t.task_id
            for t in plan.tasks
            if t.task_id in unknown or plan.revision_id + "." + t.task_id in unknown
        )
        trace.emit(
            "recovery.finished",
            status=disposition,
            count=len(unknown_tasks),
            elapsed_seconds=time.monotonic() - started,
        )
        ref = self.store.io.put_json(
            job_id, dict(schema_version=1, events=trace.merged([]))
        )
        report = dict(
            schema_version=1,
            job_id=job_id,
            revision_id=plan.revision_id,
            status=disposition,
            attempts=[
                {
                    k: r["value"][k]
                    for k in (
                        "job_id",
                        "task_id",
                        "attempt_id",
                        "revision_id",
                        "status",
                    )
                }
                for r in self.store.rows("attempts", job_id)
            ],
            original_job_status=value["status"],
            original_failure_kind=value.get("failure_kind"),
            unknown_tasks=unknown_tasks,
            review_reasons=sorted(set(unknown)),
            decisions=decisions,
            elapsed_seconds=time.monotonic() - started,
            trace_ref=ref.to_dict(),
            budget_before=before,
            budget_after=budget.snapshot(),
            orphan_artifacts=list(self.store.orphans()),
            result_status=self.last_result.status if self.last_result else None,
        )
        report_ref = self.store.io.put_json(job_id, report)
        self.store.commit(
            [],
            job_id=job_id,
            event_type="recovery.report",
            details=dict(report=report_ref.to_dict()),
        )
        self.last_report = report
        return report

    def close(self):
        if self.server is not None:
            self.server.close()
            self.server = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
