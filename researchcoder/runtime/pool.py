"""Threads supervise independent processes; no legacy workflow runs in these threads."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from .harness import ResearchHarness


@dataclass(frozen=True)
class PoolResult:
    result: object
    record: dict
    started_ns: int
    ended_ns: int


class HarnessWorkerPool:
    def __init__(self, max_workers):
        if type(max_workers) is not int or not 1 <= max_workers <= 16:
            raise ValueError("Worker concurrency out of bounds")
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="worker-supervision")

    def submit(self, spec, config):
        def supervise():
            # Each task controller owns its own Repository on its creating thread.
            # The DAG Controller alone commits cross-task state and Job budget.
            harness = ResearchHarness()
            started = time.monotonic_ns()
            result = harness.execute(spec, config)
            return PoolResult(result, harness.records[result.attempt_id], started, time.monotonic_ns())

        return self.executor.submit(supervise)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
