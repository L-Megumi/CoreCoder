"""Real SIGKILL matrix, independent file-write/command counters, raw recovery records."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import sqlite3
import struct
import subprocess
import sys
import time

from .case import trusted_case

CASES = (
    "job_created",
    "parallel_reads",
    "artifact_uncommitted",
    "transaction_interrupted",
    "patch_prepared",
    "patch_written",
    "patch_committed",
    "verification_running",
    "revision_before",
    "revision_after",
    "checkpoint_corrupt",
    "workspace_changed",
    "recovery_again",
    "unknown_usage",
    "worker_crash",
    "budget_exhausted",
    "verification_failed",
    "patch_started",
    "recovery_again_with_pending",
)
AUTO = {
    "job_created",
    "parallel_reads",
    "artifact_uncommitted",
    "transaction_interrupted",
    "patch_prepared",
    "patch_committed",
    "revision_after",
    "recovery_again",
    "patch_started",
    "recovery_again_with_pending",
}
REVIEW = {"patch_written", "verification_running", "workspace_changed"}


def group_members(groups):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(stat[2]) in groups:
                members.append((int(entry.name), stat[0]))
        except (OSError, ValueError):
            pass
    return members


class WriteCounter:
    def __init__(self, path):
        library = ctypes.CDLL(None, use_errno=True)
        self.fd = library.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if (
            self.fd < 0
            or library.inotify_add_watch(self.fd, os.fsencode(path), 0x8) < 0
        ):
            raise OSError("inotify unavailable")
        self.writes = 0

    def drain(self):
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return
        offset = 0
        while offset + 16 <= len(data):
            _, mask, _, size = struct.unpack_from("iIII", data, offset)
            offset += 16 + size
            if mask & 0x8:
                self.writes += 1

    def close(self):
        self.drain()
        os.close(self.fd)


def child(directory, mode, action, watch, *, fault="none"):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for key in list(env):
        if any(k in key.upper() for k in ("API_KEY", "AUTH_TOKEN")):
            env.pop(key, None)
    command = [
        sys.executable,
        "-B",
        "-m",
        "benchmarks.recovery.controller",
        str(directory),
        mode,
        action,
        "--fault",
        fault,
    ]
    name = action + "-" + fault
    with (directory / (name + ".log")).open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env
        )
        until = time.monotonic() + 40
        while process.poll() is None:
            watch.drain()
            if time.monotonic() > until:
                process.kill()
                process.wait()
                raise RuntimeError("Controller scenario timeout: " + mode)
            time.sleep(0.01)
    watch.drain()
    return process.returncode


def run_case(directory, mode):
    directory = Path(directory)
    directory.mkdir(parents=True)
    trusted_case(directory, mode, initialize=True)
    # Reap only descendants in the recorded supervised groups after Controller death.
    if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError("subreaper unavailable")
    watch = WriteCounter(directory / "candidate/training.py")
    codes = []
    fault = (
        "job_created"
        if mode in ("checkpoint_corrupt", "workspace_changed")
        else mode
        if mode in AUTO | REVIEW | {"revision_before", "worker_crash"}
        else "none"
    )
    if mode == "recovery_again":
        fault = "job_created"
    if mode == "recovery_again_with_pending":
        fault = "parallel_reads"
    try:
        codes.append(child(directory, mode, "start", watch, fault=fault))
        if fault != "none" and mode != "worker_crash":
            assert codes[0] == -signal.SIGKILL, (mode, codes)
        groups = set()
        dbpath = directory / "state.sqlite"
        with sqlite3.connect(dbpath) as db:
            for row in db.execute("SELECT body FROM workers"):
                groups.add(json.loads(row[0])["pid"])
            snapshot = db.execute(
                "SELECT body FROM budgets WHERE id='recovery'"
            ).fetchone()
            original_budget = json.loads(snapshot[0])["snapshot"] if snapshot else None
            stages = [
                json.loads(row[0])
                for row in db.execute("SELECT body FROM events")
                if json.loads(row[0])["event_type"] == "worker.stage"
            ]
        if mode == "parallel_reads":
            assert len({e["details"]["pid"] for e in stages}) == 2
        if mode == "checkpoint_corrupt":
            with sqlite3.connect(dbpath) as db:
                db.execute("UPDATE jobs SET body='{}'")
        if mode == "workspace_changed":
            # Deliberate external mutation is outside duplicate-effect accounting.
            watch.drain()
            (directory / "candidate/training.py").write_text("value = 999\n")
            watch.drain()
            watch.writes -= 1
        if mode in ("recovery_again", "recovery_again_with_pending"):
            codes.append(
                child(directory, mode, "recover", watch, fault="recovery_again")
            )
            assert codes[-1] == -signal.SIGKILL
        started = time.perf_counter()
        codes.append(child(directory, mode, "recover", watch))
        elapsed = time.perf_counter() - started
        assert codes[-1] == 0
        output = json.loads((directory / "recover-outcome.json").read_text())
        if mode in ("recovery_again", "recovery_again_with_pending"):
            assert output["job"]["recoveries"] == 2
        if mode in AUTO:
            matched = output.get("result_status") == "succeeded" and output[
                "status"
            ] in ("resumed", "bootstrap_after_rollback")
        elif mode in REVIEW:
            matched = output["status"] == "needs_review"
        elif mode == "checkpoint_corrupt":
            matched = output["status"] == "rejected"
        elif mode in (
            "revision_before",
            "worker_crash",
            "budget_exhausted",
            "verification_failed",
        ):
            matched = output["status"] == "stopped"
        else:
            matched = (
                output["status"] == "completed"
                and output["result_status"] == "succeeded"
            )
        command_count = (
            len((directory / "verify-count.txt").read_text().splitlines())
            if (directory / "verify-count.txt").exists()
            else 0
        )
        expected_commands = (
            2
            if mode == "revision_after"
            else 1
            if mode
            in AUTO
            | {
                "revision_before",
                "unknown_usage",
                "verification_running",
                "verification_failed",
            }
            else 0
        )
        patch_expected = (
            2
            if mode == "revision_after"
            else 0
            if mode
            in (
                "checkpoint_corrupt",
                "workspace_changed",
                "worker_crash",
                "budget_exhausted",
            )
            else 1
        )
        watch.drain()
        duplicate_effects = max(0, watch.writes - patch_expected) + max(
            0, command_count - expected_commands
        )
        if mode == "verification_running":
            assert command_count == 1
        if mode == "patch_written":
            assert command_count == 0 and watch.writes == 1
        until = time.monotonic() + 3
        while time.monotonic() < until:
            members = group_members(groups)
            for pid, status in members:
                if status == "Z":
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            if not group_members(groups):
                break
            time.sleep(0.02)
        remaining = group_members(groups)
        after = output.get("budget_after")
        if original_budget and after:
            assert all(
                after["charged"][key] >= original_budget["charged"][key]
                for key in after["charged"]
            )
            assert set(original_budget["settlements"]) <= set(after["settlements"])
        row = dict(
            case=mode,
            expected_disposition_matched=matched,
            status=output["status"],
            result_status=output.get("result_status"),
            recovery_seconds=elapsed,
            duplicate_effects=duplicate_effects,
            observed_patch_writes=watch.writes,
            verification_invocations=command_count,
            state_consistency_passed=output["state_consistent"],
            unknown_tasks=output.get("unknown_tasks", []),
            review_reasons=output.get("review_reasons", []),
            budget_before=original_budget,
            budget_after=after,
            supervised_processes_remaining=remaining,
            process_exit_codes=codes,
        )
        (directory / "case-result.json").write_text(json.dumps(row, indent=2))
        assert matched, (mode, output)
        assert duplicate_effects == 0 and not remaining, (mode, row)
        return row
    finally:
        watch.close()


def run(directory, cases=CASES):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [run_case(directory / mode, mode) for mode in cases]
    eligible = [r for r in rows if r["case"] in AUTO]
    result = dict(
        schema_version=1,
        provider_requests=0,
        rows=rows,
        automatic_recovery_eligible=len(eligible),
        automatic_recovery_succeeded=sum(
            r["result_status"] == "succeeded" for r in eligible
        ),
        expected_safe_dispositions=sum(r["expected_disposition_matched"] for r in rows),
        cases=len(rows),
        mean_recovery_seconds=sum(r["recovery_seconds"] for r in rows) / len(rows),
        duplicate_effects=sum(r["duplicate_effects"] for r in rows),
        state_consistency_passed=sum(r["state_consistency_passed"] for r in rows),
        manual_review_cases=sum(r["status"] == "needs_review" for r in rows),
        unknown_task_count=sum(len(r["unknown_tasks"]) for r in rows),
        metric_scope="Real independent Controller/Worker processes and SIGKILL. Functional counter oracle + inotify; not model accuracy or universal exactly-once.",
    )
    (directory / "results.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = run(args.output_dir)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))
