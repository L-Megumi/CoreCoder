"""Controller-owned in-memory execution states; intentionally not Checkpoints."""

from dataclasses import dataclass, replace
import threading
from researchcoder.contracts import TaskSpec, TaskResult, identifier

TERMINAL = frozenset({"succeeded", "failed", "incomplete", "cancelled"})


@dataclass(frozen=True)
class JobState:
    job_id: str
    status: str = "open"


@dataclass(frozen=True)
class TaskState:
    job_id: str
    task_id: str
    status: str = "pending"
    attempt_id: str | None = None
    version: int = 0


@dataclass(frozen=True)
class AttemptState:
    job_id: str
    task_id: str
    attempt_id: str
    status: str = "running"


class TaskRepository:
    def __init__(self):
        self._owner = threading.get_ident()
        self._jobs = {}
        self._tasks = {}
        self._attempts = {}
        self._results = {}
        self._specs = {}

    def _controller(self):
        if threading.get_ident() != self._owner:
            raise RuntimeError("Only the controller thread may update task state")

    def create_job(self, job_id):
        self._controller()
        identifier(job_id)
        if job_id in self._jobs:
            raise ValueError("Job already exists")
        self._jobs[job_id] = JobState(job_id)
        return self._jobs[job_id]

    def add_task(self, spec):
        self._controller()
        if not isinstance(spec, TaskSpec) or self._jobs.get(spec.job_id, JobState("", "closed")).status != "open":
            raise ValueError("Task requires an open job")
        key = (spec.job_id, spec.task_id)
        if key in self._tasks:
            raise ValueError("Duplicate task identity; no implicit re-execution")
        self._tasks[key] = TaskState(*key)
        self._specs[key] = spec
        return self._tasks[key]

    def start(self, job_id, task_id, attempt_id):
        self._controller()
        identifier(attempt_id)
        key = (job_id, task_id)
        state = self._tasks[key]
        if state.status != "pending" or attempt_id in self._attempts:
            raise ValueError("Illegal attempt start")
        self._tasks[key] = replace(state, status="running", attempt_id=attempt_id, version=state.version + 1)
        self._attempts[attempt_id] = AttemptState(job_id, task_id, attempt_id)
        return self._tasks[key]

    def finish(self, result):
        self._controller()
        if not isinstance(result, TaskResult):
            raise TypeError("TaskResult required")
        key = (result.job_id, result.task_id)
        state = self._tasks[key]
        if (
            state.status != "running"
            or state.attempt_id != result.attempt_id
            or result.status not in TERMINAL
            or self._specs[key].kind != result.kind
        ):
            raise ValueError("Illegal or duplicate completion")
        self._tasks[key] = replace(state, status=result.status, version=state.version + 1)
        self._attempts[result.attempt_id] = replace(self._attempts[result.attempt_id], status=result.status)
        self._results[result.attempt_id] = result
        return self._tasks[key]

    def close_job(self, job_id):
        self._controller()
        if self._jobs[job_id].status != "open":
            raise ValueError("Job already closed")
        if any(s.job_id == job_id and s.status not in TERMINAL for s in self._tasks.values()):
            raise ValueError("Job still has unfinished tasks")
        self._jobs[job_id] = replace(self._jobs[job_id], status="closed")

    def job(self, job_id):
        return self._jobs[job_id]

    def task(self, job_id, task_id):
        return self._tasks[(job_id, task_id)]

    def attempt(self, attempt_id):
        return self._attempts[attempt_id]

    def result(self, attempt_id):
        return self._results[attempt_id]
