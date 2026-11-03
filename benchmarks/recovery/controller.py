"""Actual disposable Controller process, killed at named durable boundaries."""

from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time

from researchcoder.durable import DurableStore, CheckpointError
from researchcoder.durable.recovery import RecoveryCoordinator
from .case import trusted_case
from researchcoder.orchestration import OrchestrationTask, TaskDependency


class Crash:
    def __init__(self, directory, mode):
        self.directory = directory
        self.mode = mode
        self.counts = {}
        self.lock = threading.Lock()
        self.store = None

    def kill(self, point):
        with (self.directory / "faults.jsonl").open("a") as stream:
            stream.write(
                json.dumps(dict(point=point, pid=os.getpid(), signal="SIGKILL")) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.kill(os.getpid(), signal.SIGKILL)

    def __call__(self, point):
        with self.lock:
            self.counts[point] = self.counts.get(point, 0) + 1
            count = self.counts[point]
        mapping = {
            "job_created": "job_created",
            "artifact_uncommitted": "task_artifact_published",
            "transaction_interrupted": "before_db_commit",
            "patch_prepared": "operation_prepared",
            "patch_started": "operation_started",
            "patch_written": "after_effect_before_journal",
            "patch_committed": "after_patch_before_verification",
            "revision_before": "before_plan_commit",
            "revision_after": "after_plan_commit",
            "recovery_again": "recovery_reconciled",
        }
        if mapping.get(self.mode) == point:
            target = (
                2
                if self.mode
                in ("transaction_interrupted", "revision_before", "revision_after")
                else 1
            )
            if count == target:
                self.kill(point)
        if self.mode == "parallel_reads" and point == "tool_ready":
            if count == 2:
                self.kill(point)
            threading.Event().wait(10)
        if (
            self.mode == "verification_running"
            and point == "operation_started"
            and count == 2
        ):

            def watch():
                until = time.monotonic() + 10
                while time.monotonic() < until:
                    if (self.directory / "verify-count.txt").exists():
                        self.kill("verification_subprocess_observed")
                    time.sleep(0.005)

            threading.Thread(target=watch, daemon=True).start()
        if self.mode == "worker_crash" and point == "tool_ready" and count == 1:
            rows = self.store.rows("workers", "recovery")
            for row in rows:
                children = Path(
                    f"/proc/{row['value']['pid']}/task/{row['value']['pid']}/children"
                )
                if children.exists() and children.read_text().strip():
                    child = int(children.read_text().split()[0])
                    os.kill(child, signal.SIGKILL)
                    (self.directory / "worker-killed.json").write_text(
                        json.dumps({"pid": child, "signal": "SIGKILL"})
                    )
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("mode")
    parser.add_argument("action")
    parser.add_argument("--fault", default="none")
    args = parser.parse_args()
    directory = args.directory
    plan, cfg, io = trusted_case(directory, args.mode)
    crash = Crash(directory, args.fault)
    output = None
    try:
        with (
            DurableStore(directory / "state.sqlite", io, fault=crash) as store,
            RecoveryCoordinator(store, fault=crash) as runner,
        ):
            crash.store = store
            if args.action == "start":
                result = runner.execute(plan, cfg)
                if args.mode in ("revision_before", "revision_after"):
                    revised = (
                        OrchestrationTask(
                            task_id="diagnose_again",
                            role="diagnosis",
                            template_id="diagnose",
                        ),
                        OrchestrationTask(
                            task_id="repair_again",
                            role="repair",
                            template_id="repair_second",
                            dependencies=(
                                TaskDependency(
                                    task_id="diagnose_again", input_role="diagnosis_run"
                                ),
                            ),
                        ),
                        OrchestrationTask(
                            task_id="verify_again",
                            role="verification",
                            template_id="verify",
                            dependencies=(TaskDependency(task_id="repair_again"),),
                        ),
                    )
                    result = runner.execute(
                        replace(
                            plan,
                            revision_id="r1",
                            generation=1,
                            parent_revision_id="r0",
                            tasks=revised,
                        ),
                        cfg,
                    )
                output = dict(
                    status="executed",
                    result_status=result.status,
                    budget_after=store.get("budgets", "recovery")["value"]["snapshot"],
                )
            else:
                if store.get("jobs", "recovery") is None:
                    result = runner.execute(plan, cfg)
                    output = dict(
                        status="bootstrap_after_rollback",
                        result_status=result.status,
                        budget_before=None,
                        budget_after=store.get("budgets", "recovery")["value"][
                            "snapshot"
                        ],
                    )
                else:
                    output = runner.recover("recovery", cfg)
            output["state_consistent"] = store.validate()
            output["operations"] = [r["value"] for r in store.rows("operations")]
            output["attempts"] = [r["value"] for r in store.rows("attempts")]
            output["job"] = store.get("jobs", "recovery")["value"]
    except (
        CheckpointError,
        sqlite3.DatabaseError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        output = dict(
            status="rejected",
            error_type=type(error).__name__,
            reason=str(error)[:300],
            state_consistent=False,
        )
    (directory / (args.action + "-outcome.json")).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
