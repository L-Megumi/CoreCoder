"""Trusted controller entry point; state and workers are deliberately separate."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import threading
import time
import uuid

from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import ArtifactRef
from researchcoder.contracts import TaskSpec, TaskResult, ExecutionPolicy, VerificationSpec
from .protocol import RuntimeLimits, ModelSpec, WorkerRequest, ProtectedFile, PlanningTaskResult, CodingTaskResult, ModelRequestOptions, model_request_kwargs
from .state import TaskRepository
from .tracing import TraceSink
from .supervisor import WorkerSupervisor
from .isolation import WorkspaceLease
from .optimization import RuntimeOptimization
from .permissions import PermissionRecoverySpec
from .coding import CodingContinuationPolicy


def _result_class(spec):
    return {'plan':PlanningTaskResult,'coding':CodingTaskResult}.get(spec.kind,TaskResult)


@dataclass(frozen=True, kw_only=True)
class ExecutionConfig:
    """Trusted process-local configuration; never accepted from a model tool call."""

    artifact_root: str
    allowed_workspaces: tuple[str, ...]
    max_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    permitted_verifications: tuple[VerificationSpec, ...] = ()
    protected_files: tuple[ProtectedFile, ...] = ()
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    model: ModelSpec = field(default_factory=ModelSpec)
    model_options: ModelRequestOptions | None = None
    cancel_event: threading.Event | None = None
    worker_command: tuple[str, ...] | None = None
    worker_python: str | None = None
    diagnostic_tools: tuple[str, ...] = (
        "extract_log_evidence",
        "locate_source",
        "read_project_file",
        "inspect_paths",
        "search_project_source",
        "resolve_project_symbol",
    )
    shared_read: bool = False
    artifact_namespace: str | None = None
    system_instructions: str = ""
    evidence_context: str = ""
    context_policy: ContextPolicy | None = None
    evidence_bundle: ArtifactRef | None = None
    record_requests: bool = False
    optimization: RuntimeOptimization | None = None
    permission_recovery: PermissionRecoverySpec | None = None
    coding_continuation: CodingContinuationPolicy | None = None
    durable_observer: object | None = None

    def __post_init__(self):
        if self.coding_continuation is not None and type(self.coding_continuation) is not CodingContinuationPolicy:
            raise TypeError("Trusted typed Coding continuation configuration required")
        if self.permission_recovery is not None and not isinstance(self.permission_recovery, PermissionRecoverySpec):
            raise TypeError("Trusted typed permission recovery configuration required")
        if type(self.record_requests) is not bool:
            raise TypeError("Request capture is trusted boolean configuration")
        if self.optimization is not None and not isinstance(self.optimization, RuntimeOptimization):
            raise TypeError("Typed runtime optimization required")
        model_request_kwargs(self.model, self.model_options)
        if self.context_policy is not None and not isinstance(self.context_policy, ContextPolicy):
            raise TypeError("Trusted typed context policy required")
        if self.evidence_bundle is not None and not isinstance(self.evidence_bundle, ArtifactRef):
            raise TypeError("Typed evidence bundle reference required")
        if (
            not isinstance(self.model, ModelSpec)
            or not isinstance(self.limits, RuntimeLimits)
            or not isinstance(self.max_policy, ExecutionPolicy)
        ):
            raise TypeError("Typed controller model, limits and policy required")
        if not isinstance(self.artifact_root, str) or not Path(self.artifact_root).is_absolute():
            raise ValueError("Absolute artifact directory required")
        if any(not isinstance(p, str) or not Path(p).is_absolute() for p in self.allowed_workspaces):
            raise ValueError("Absolute controller workspace paths required")
        if any(not isinstance(v, VerificationSpec) for v in self.permitted_verifications):
            raise TypeError("Typed verification permissions required")
        if any(not isinstance(v, ProtectedFile) for v in self.protected_files):
            raise TypeError("Typed protected files required")
        if self.worker_command is not None and (
            not self.worker_command or any(not isinstance(p, str) or not p for p in self.worker_command)
        ):
            raise ValueError("Invalid trusted worker command")


class ResearchHarness:
    def __init__(self):
        self.repository = TaskRepository()
        self.records = {}
        self._partial = {}
        self._job_stores = {}
        self._active = threading.Lock()

    def execute(self, task_spec, execution_config):
        if not isinstance(task_spec, TaskSpec) or not isinstance(execution_config, ExecutionConfig):
            raise TypeError("TaskSpec and trusted ExecutionConfig required")
        if not self._active.acquire(blocking=False):
            raise RuntimeError("Phase 2 runs one worker per Harness")
        try:
            return self._execute(task_spec, execution_config)
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            try:
                state = self.repository.task(task_spec.job_id, task_spec.task_id)
            except KeyError:
                raise error
            if state.status != "running":
                raise
            result = self._partial.pop(state.attempt_id, None)
            if result is None:
                result = _result_class(task_spec)(
                    job_id=task_spec.job_id,
                    task_id=task_spec.task_id,
                    attempt_id=state.attempt_id,
                    kind=task_spec.kind,
                    status="incomplete",
                    failure_kind="tool_error",
                )
            elif result.status == "succeeded":
                result = replace(result, status="incomplete", failure_kind="tool_error", workflow_completed=False)
            result = replace(
                result, limitations=result.limitations + ("Controller could not archive complete execution metadata.",)
            )
            self.repository.finish(result)
            self.records[state.attempt_id] = {
                "status": result.status,
                "failure_kind": result.failure_kind,
                "archive_error_type": type(error).__name__,
                "trace": None,
            }
            return result
        finally:
            self._active.release()

    def _execute(self, spec, config):
        # Roundtrip validates the complete persistent input at the controller boundary.
        spec = type(spec).from_json(spec.to_json())
        attempt = "attempt-" + uuid.uuid4().hex
        io = ArtifactIO(config.artifact_root, namespace=config.artifact_namespace)
        root = Path(spec.workspace.root).resolve()
        if io.root.is_relative_to(root):
            raise ValueError("Runtime artifacts must be outside the editable workspace")
        trace = TraceSink(spec.job_id, spec.task_id, attempt)
        try:
            self.repository.job(spec.job_id)
        except KeyError:
            self.repository.create_job(spec.job_id)
            self._job_stores[spec.job_id] = io
            job_event = trace.emit("job.created")
        else:
            if self._job_stores[spec.job_id].root != io.root:
                raise ValueError("Job artifact store cannot change")
            job_event = trace.emit("job.attached")
        self.repository.add_task(spec)
        self.repository.start(spec.job_id, spec.task_id, attempt)
        task_span = trace.start("task.execution", parent_id=job_event)
        trace.parent_id = task_span[0]
        supervised = None
        if config.durable_observer is not None:
            config.durable_observer.attempt_started(spec, attempt)
        try:
            maximum = config.max_policy
            if (
                root not in [Path(p).resolve() for p in config.allowed_workspaces]
                or not set(spec.policy.allowed_tool_names) <= set(maximum.allowed_tool_names)
                or not set(spec.policy.writable_paths) <= set(maximum.writable_paths)
                or not set(spec.policy.allowed_read_roots) <= set(maximum.allowed_read_roots)
                or spec.policy.allow_verification
                and not maximum.allow_verification
            ):
                raise ValueError("Task exceeds controller execution policy")
            if spec.verification and spec.verification not in config.permitted_verifications:
                raise ValueError("Verification not authorized")
            limits = replace(
                config.limits,
                max_rounds=min(config.limits.max_rounds, spec.budget.max_rounds),
                max_tool_calls=min(config.limits.max_tool_calls, spec.budget.max_tool_calls),
                max_context_tokens=min(config.limits.max_context_tokens, spec.budget.max_context_tokens),
            )
            request = WorkerRequest(
                spec=spec,
                attempt_id=attempt,
                artifact_root=str(io.root),
                trace_path="/unused",
                parent_event_id=task_span[0],
                limits=limits,
                model=config.model,
                model_options=config.model_options,
                permitted_verifications=config.permitted_verifications,
                protected_files=config.protected_files,
                diagnostic_tools=config.diagnostic_tools,
                artifact_namespace=config.artifact_namespace,
                system_instructions=config.system_instructions,
                evidence_context=config.evidence_context,
                context_policy=config.context_policy,
                evidence_bundle=config.evidence_bundle,
                record_requests=config.record_requests,
                optimization=config.optimization,
                permission_recovery=config.permission_recovery,
                coding_continuation=config.coding_continuation,
                journal_endpoint=config.durable_observer.endpoint if config.durable_observer else None,
            )
            if config.shared_read and (spec.kind != "diagnose" or spec.policy.writable_paths):
                raise ValueError("Shared lease requires a read-only diagnostic task")
            with WorkspaceLease(root, shared=config.shared_read):
                supervised = WorkerSupervisor().run(
                    request,
                    trace,
                    cancel_event=config.cancel_event,
                    command=config.worker_command,
                    python=config.worker_python,
                    started_at=task_span[1],
                    observer=config.durable_observer,
                )
                if config.shared_read:
                    from corecoder.workflow import workspace_revision

                    if workspace_revision(root) != spec.workspace.source_manifest_sha256:
                        raise ValueError("Read-only workspace changed during execution")
            if supervised.response is not None:
                result = supervised.response.result
                # A worker cannot satisfy completion with invalid or foreign artifact references.
                for ref in (*result.outputs, *((result.file_changes_ref,) if result.file_changes_ref else ())):
                    if ref.job_id != spec.job_id:
                        raise ValueError("Foreign output artifact")
                    io.read_bytes(ref)
                for ref in result.evidence_refs:
                    io.read_record(ref)
                self._partial[attempt] = result
                worker_response_ref = io.put_json(spec.job_id, supervised.response.to_dict())
                result = replace(result, outputs=result.outputs + (worker_response_ref,))
            else:
                result = _result_class(spec)(
                    job_id=spec.job_id,
                    task_id=spec.task_id,
                    attempt_id=attempt,
                    kind=spec.kind,
                    status="failed",
                    failure_kind="tool_error",
                )
            if supervised.reason:
                result = replace(
                    result,
                    status="cancelled" if supervised.reason == "interrupted" else "failed",
                    failure_kind=supervised.reason,
                    workflow_completed=False,
                )
        except KeyboardInterrupt:
            result = _result_class(spec)(
                job_id=spec.job_id,
                task_id=spec.task_id,
                attempt_id=attempt,
                kind=spec.kind,
                status="cancelled",
                failure_kind="interrupted",
            )
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            trace.emit("controller.error", status="error", failure_kind="input_invalid", reason=type(error).__name__)
            result = _result_class(spec)(
                job_id=spec.job_id,
                task_id=spec.task_id,
                attempt_id=attempt,
                kind=spec.kind,
                status="incomplete",
                failure_kind="input_invalid",
            )
        worker_events = list(supervised.worker_events) if supervised else []
        try:
            trace.merged(worker_events)
        except ValueError:
            result = replace(result, status="failed", failure_kind="tool_error", workflow_completed=False)
            trace.emit("worker.trace_rejected", status="error", failure_kind="tool_error")
            worker_events = []
        trace.finish(task_span, "task.execution", status=result.status, failure_kind=result.failure_kind)
        events = trace.merged(worker_events)
        if supervised is None or supervised.response is None:
            result = replace(
                result,
                usage=replace(
                    result.usage,
                    llm_calls=sum(e["event_type"] == "budget.reserved" for e in worker_events),
                    tool_calls=sum(e["event_type"] in ("tool.call.start", "verification.command.start") for e in worker_events),
                ),
            )
        result = replace(
            result,
            usage=replace(
                result.usage,
                elapsed_seconds=max(0, time.monotonic() - task_span[1]),
                mode="live" if config.model.provider == "openai" else "injected",
            ),
        )
        trace_ref = io.put_json(spec.job_id, {"schema_version": 1, "events": events})
        budget = (
            json.loads(supervised.response.budget_json)
            if supervised and supervised.response
            else {"usage": "unknown", "reason": "No complete worker accounting response; do not infer zero usage."}
        )
        record = {
            "counter_semantics": "Complete worker accounting"
            if supervised and supervised.response
            else "Observed trace lower bounds; usage remains unknown",
            "schema_version": 1,
            "job_id": spec.job_id,
            "task_id": spec.task_id,
            "attempt_id": attempt,
            "trace": trace_ref.to_dict(),
            "budget": budget,
            "worker_pid": supervised.pid if supervised else None,
            "worker_exit_code": supervised.exit_code if supervised else None,
            "worker_stdout": supervised.stdout if supervised else None,
            "worker_stderr": supervised.stderr if supervised else None,
            "status": result.status,
            "failure_kind": result.failure_kind,
            "limitations": [
                "Same-session process-group cleanup; escaping daemon sessions are not an OS sandbox.",
                "No DAG scheduling, persistent task repository or recovery in Phase 2.",
            ],
        }
        receipt = io.put_json(spec.job_id, record)
        result = replace(
            result, outputs=result.outputs + (receipt,), limitations=result.limitations + tuple(record["limitations"])
        )
        final_ref = io.put_json(spec.job_id, result.to_dict())
        result = replace(result, outputs=result.outputs + (final_ref,))
        self.repository.finish(result)
        self._partial.pop(attempt, None)
        self.records[attempt] = record
        return result

    def close_job(self, job_id):
        self.repository.close_job(job_id)
        trace = TraceSink(job_id, "job-lifecycle", "job-close-" + uuid.uuid4().hex)
        trace.emit("job.closed", status="closed")
        return self._job_stores[job_id].put_json(job_id, {"schema_version": 1, "events": trace.events})
