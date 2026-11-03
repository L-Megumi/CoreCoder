"""Planning values contain no tool policy, filesystem roots or model credentials."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from researchcoder.contracts import Contract, TaskResult, ArtifactRef, identifier, digest, text

Role = Literal["lead", "log_analysis", "code_analysis", "diagnosis", "repair", "verification", "aggregation"]
NodeStatus = Literal["pending", "ready", "running", "succeeded", "failed", "incomplete", "cancelled", "blocked", "reused"]


@dataclass(frozen=True, kw_only=True)
class TaskDependency(Contract):
    task_id: str
    input_role: Literal["none", "diagnosis_run", "evidence"] = "none"
    schema_version: int = 1

    def validate(self):
        identifier(self.task_id)


@dataclass(frozen=True, kw_only=True)
class OrchestrationTask(Contract):
    task_id: str
    template_id: str
    role: Role
    dependencies: tuple[TaskDependency, ...] = ()
    objective: str = "Execute the authorized task"
    workspace_revision: str | None = None
    retry_of: str | None = None
    schema_version: int = 1

    def validate(self):
        identifier(self.task_id)
        identifier(self.template_id)
        text(self.objective)
        if len(self.task_id) > 45:
            raise ValueError("Logical task ID too long for revision-qualified execution")
        if self.workspace_revision is not None:
            digest(self.workspace_revision)
        if self.retry_of is not None:
            identifier(self.retry_of)
        ids = [d.task_id for d in self.dependencies]
        if len(set(ids)) != len(ids) or self.task_id in ids:
            raise ValueError("Duplicate or self dependency")
        if sum(d.input_role == "diagnosis_run" for d in self.dependencies) > 1:
            raise ValueError("Ambiguous repair handoff")


@dataclass(frozen=True, kw_only=True)
class PlanRevision(Contract):
    job_id: str
    revision_id: str
    tasks: tuple[OrchestrationTask, ...]
    generation: int = 0
    parent_revision_id: str | None = None
    schema_version: int = 1

    def validate(self):
        identifier(self.job_id)
        identifier(self.revision_id)
        if len(self.revision_id) > 45 or not 1 <= len(self.tasks) <= 64 or not 0 <= self.generation <= 10:
            raise ValueError("Plan bounds exceeded")
        if self.parent_revision_id is not None:
            identifier(self.parent_revision_id)
            if self.parent_revision_id == self.revision_id:
                raise ValueError("A revision cannot be its own parent")
        if (self.generation == 0) != (self.parent_revision_id is None):
            raise ValueError("Revision ancestry required")
        by_id = {t.task_id: t for t in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("Duplicate task ID")
        visited, visiting = {}, set()

        def depth(identity):
            if identity not in by_id:
                raise ValueError("Missing dependency")
            if identity in visiting:
                raise ValueError("Cyclic DAG")
            if identity not in visited:
                visiting.add(identity)
                visited[identity] = 1 + max((depth(d.task_id) for d in by_id[identity].dependencies), default=0)
                visiting.remove(identity)
            return visited[identity]

        for task in self.tasks:
            depth(task.task_id)
        if max(visited.values()) > 16:
            raise ValueError("DAG too deep")

    def depth(self):
        depths = {}
        pending = list(self.tasks)
        while pending:
            for task in list(pending):
                if all(d.task_id in depths for d in task.dependencies):
                    depths[task.task_id] = 1 + max((depths[d.task_id] for d in task.dependencies), default=0)
                    pending.remove(task)
        return max(depths.values())


@dataclass(frozen=True, kw_only=True)
class DAGExecutionState(Contract):
    job_id: str
    revision_id: str
    task_id: str
    status: NodeStatus = "pending"
    execution_task_id: str | None = None
    attempt_id: str | None = None
    attempts: int = 0
    input_fingerprint: str | None = None
    workspace_revision: str | None = None
    resulting_revision: str | None = None
    reason: str | None = None
    schema_version: int = 1

    def validate(self):
        for v in (self.job_id, self.revision_id, self.task_id):
            identifier(v)
        for v in (self.execution_task_id, self.attempt_id):
            if v is not None:
                identifier(v)
        for v in (self.input_fingerprint, self.workspace_revision, self.resulting_revision):
            if v is not None:
                digest(v)
        if self.attempts not in (0, 1):
            raise ValueError("One attempt per task per revision; retries require a new revision")
        if self.reason is not None:
            text(self.reason, 300)


@dataclass(frozen=True, kw_only=True)
class SchedulerResult(Contract):
    job_id: str
    revision_id: str
    status: Literal["succeeded", "failed", "incomplete", "cancelled"]
    states: tuple[DAGExecutionState, ...]
    results: tuple[TaskResult, ...]
    trace_ref: ArtifactRef
    summary_ref: ArtifactRef
    elapsed_seconds: float
    schema_version: int = 1

    def validate(self):
        identifier(self.job_id)
        identifier(self.revision_id)
        if self.elapsed_seconds < 0 or any(s.job_id != self.job_id or s.revision_id != self.revision_id for s in self.states):
            raise ValueError("Foreign execution state")
        if len({s.task_id for s in self.states}) != len(self.states):
            raise ValueError("Duplicate execution state")
        if any(ref.job_id != self.job_id for ref in (self.trace_ref, self.summary_ref)):
            raise ValueError("Foreign orchestration artifacts")
        if any(r.job_id != self.job_id for r in self.results):
            raise ValueError("Foreign task result")
        if len({r.attempt_id for r in self.results}) != len(self.results):
            raise ValueError("Duplicate task attempt result")
        identities = {(s.execution_task_id, s.attempt_id) for s in self.states}
        if any((r.task_id, r.attempt_id) not in identities for r in self.results):
            raise ValueError("Result is not bound to an execution state")
        if any(s.status in ("pending", "ready", "running") for s in self.states):
            raise ValueError("SchedulerResult requires terminal graph states")
        if self.status == "succeeded" and any(s.status not in ("succeeded", "reused") for s in self.states):
            raise ValueError("Incomplete graph cannot claim success")
