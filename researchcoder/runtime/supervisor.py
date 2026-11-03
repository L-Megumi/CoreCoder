"""Linux process supervision, bounded stream metadata and process-group cleanup."""

from dataclasses import dataclass, replace
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .protocol import WorkerResponse


@dataclass(frozen=True)
class SupervisedResult:
    response: WorkerResponse | None
    reason: str | None
    exit_code: int | None
    worker_events: tuple
    stdout: dict
    stderr: dict
    pid: int | None


class StreamDigest:
    def __init__(self):
        self.count = 0
        self.hash = hashlib.sha256()

    def drain(self, stream):
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                self.count += len(chunk)
                self.hash.update(chunk)
        finally:
            stream.close()

    def metadata(self):
        return {"bytes": self.count, "sha256": self.hash.hexdigest(), "retained_bytes": 0, "content_omitted": True}


def _subreaper():
    # Reap only this supervised process group, never unrelated child processes.
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError("Linux subreaper unavailable")


def _cleanup(process, grace=0.25):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=3)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            child, _ = os.waitpid(-process.pid, os.WNOHANG)
            if child == 0:
                time.sleep(0.01)
        except ChildProcessError:
            break


class WorkerSupervisor:
    def run(self, request, trace, *, cancel_event=None, command=None, python=None, started_at=None, observer=None):
        started = time.monotonic() if started_at is None else started_at
        directory = Path(tempfile.mkdtemp(prefix="researchcoder-worker-"))
        request_file = directory / "request.json"
        response_file = directory / "response.json"
        events_file = directory / "events.jsonl"
        span = trace.start("worker.lifecycle")
        request = replace(request, trace_path=str(events_file), parent_event_id=span[0])
        request_file.write_text(request.to_json(), encoding="utf-8")
        request_file.chmod(0o600)
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR") if key in os.environ}
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        if request.model.provider == "openai" and request.model.credential_env in os.environ:
            env[request.model.credential_env] = os.environ[request.model.credential_env]
        import sys

        argv = list(command) if command else [python or sys.executable, "-B", "-m", "researchcoder.runtime.worker"]
        argv += [str(request_file), str(response_file)]
        if observer is not None:
            argv = [python or sys.executable, "-B", "-m", "researchcoder.durable.guard", str(os.getpid())] + argv
        out, err = StreamDigest(), StreamDigest()
        process = None
        reason = None
        worker_events = []
        response = None
        try:
            _subreaper()
            if cancel_event is not None and cancel_event.is_set():
                reason = "timeout" if getattr(cancel_event,"failure_kind",None) == "timeout" else "interrupted"
            elif time.monotonic() - started >= request.limits.max_runtime_seconds:
                reason = "timeout"
            else:
                process = subprocess.Popen(
                    argv,
                    cwd=directory,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    close_fds=True,
                )
                trace.emit("worker.spawned", parent_id=span[0], pid=process.pid)
                if observer is not None:
                    observer.worker_started(request.attempt_id, process.pid, directory)
                    with (directory/"lease.ready").open("x") as fence:
                        fence.write("ready")
                        fence.flush()
                        os.fsync(fence.fileno())
                readers = [
                    threading.Thread(target=sink.drain, args=(stream,), daemon=True)
                    for sink, stream in [(out, process.stdout), (err, process.stderr)]
                ]
                for reader in readers:
                    reader.start()
                try:
                    while process.poll() is None:
                        if cancel_event is not None and cancel_event.is_set():
                            reason = "timeout" if getattr(cancel_event,"failure_kind",None) == "timeout" else "interrupted"
                            break
                        if time.monotonic() - started >= request.limits.max_runtime_seconds:
                            reason = "timeout"
                            break
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    reason = "interrupted"
                finally:
                    _cleanup(process)
                    for reader in readers:
                        reader.join(timeout=2)
                if response_file.is_file():
                    if response_file.stat().st_size > 10_000_000:
                        raise ValueError("Oversized worker response")
                    response = WorkerResponse.from_json(response_file.read_text(encoding="utf-8"))
                    result = response.result
                    expected = request.spec
                    if (result.job_id, result.task_id, result.attempt_id, result.kind) != (
                        expected.job_id,
                        expected.task_id,
                        request.attempt_id,
                        expected.kind,
                    ):
                        raise ValueError("Worker result identity mismatch")
                if process.returncode != 0 or response is None:
                    reason = reason or "tool_error"
                if events_file.is_file():
                    if events_file.stat().st_size > 3_000_000:
                        raise ValueError("Oversized worker trace")
                    for line in events_file.read_text().splitlines():
                        try:
                            worker_events.append(json.loads(line))
                        except ValueError:
                            break  # A killed writer may leave only a partial final line.
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            response = None
            reason = reason or "tool_error"
            trace.emit("worker.error", parent_id=span[0], status="error", failure_kind=reason, reason=type(error).__name__)
        finally:
            if process is not None and process.poll() is None:
                _cleanup(process)
            trace.finish(
                span,
                "worker.lifecycle",
                status="error" if reason else "ok",
                failure_kind=reason,
                exit_code=process.returncode if process else None,
                stdout_bytes=out.count,
                stderr_bytes=err.count,
            )
        if observer is not None and process is not None:
            observer.worker_finished(request.attempt_id, reason, process.returncode)
        return SupervisedResult(
            response,
            reason,
            process.returncode if process else None,
            tuple(worker_events),
            out.metadata(),
            err.metadata(),
            process.pid if process else None,
        )
