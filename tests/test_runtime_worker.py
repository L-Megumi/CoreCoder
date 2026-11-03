from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import threading
import pytest

from researchcoder.contracts import TaskSpec, WorkspaceRef, ExecutionPolicy, VerificationSpec
from corecoder.workflow import workspace_revision
from researchcoder.runtime.protocol import WorkerRequest, WorkerResponse, RuntimeLimits, ModelSpec
from researchcoder.runtime.tracing import TraceSink
from researchcoder.runtime.supervisor import WorkerSupervisor


def request(tmp_path, seconds=5):
    root = tmp_path / "project"
    root.mkdir()
    command = VerificationSpec(argv=(sys.executable, "-c", "print('verified')"))
    spec = TaskSpec(
        job_id="j",
        task_id="t",
        kind="verify",
        objective="Check",
        workspace=WorkspaceRef(workspace_id="p", root=str(root), source_manifest_sha256=workspace_revision(root)),
        policy=ExecutionPolicy(allow_verification=True),
        verification=command,
    )
    return WorkerRequest(
        spec=spec,
        attempt_id="a",
        artifact_root=str(tmp_path / "artifacts"),
        trace_path="/unused",
        parent_event_id="parent",
        permitted_verifications=(command,),
        limits=RuntimeLimits(max_runtime_seconds=seconds),
    )


def run(req, **kwargs):
    trace = TraceSink("j", "t", "a")
    trace.emit("job.created")
    result = WorkerSupervisor().run(req, trace, **kwargs)
    return result, trace


def test_real_worker_protocol_normal_completion_and_parent_cwd_unchanged(tmp_path):
    req = request(tmp_path)
    before = Path.cwd()
    assert WorkerRequest.from_json(req.to_json()) == req
    result, trace = run(req)
    assert result.reason is None and result.exit_code == 0 and result.response.result.status == "succeeded"
    assert WorkerResponse.from_json(result.response.to_json()) == result.response
    assert Path.cwd() == before and result.pid != os.getpid()
    events = trace.merged(list(result.worker_events))
    assert any(e["event_type"] == "worker.ready" for e in events)
    assert all(e["parent_id"] is None or any(p["event_id"] == e["parent_id"] for p in events) for e in events)


@pytest.mark.parametrize(
    "command,code", [(("/missing-runtime-worker-executable",), None), ((sys.executable, "-c", "import os;os._exit(9)"), 9)]
)
def test_launch_failure_and_abnormal_exit(tmp_path, command, code):
    result, _ = run(request(tmp_path), command=command)
    assert result.reason == "tool_error" and result.response is None and result.exit_code == code


def test_timeout_and_same_group_grandchild_cleanup(tmp_path):
    req = request(tmp_path, 0.5)
    pid_file = tmp_path / "child.pid"
    program = (
        "import subprocess,sys,pathlib,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid));time.sleep(30)"
    )
    result, _ = run(req, command=(sys.executable, "-c", program))
    assert result.reason == "timeout"
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancellation_and_pre_cancel_do_not_hang(tmp_path):
    req = request(tmp_path)
    cancel = threading.Event()
    timer = threading.Timer(0.15, cancel.set)
    timer.start()
    try:
        result, _ = run(req, cancel_event=cancel, command=(sys.executable, "-c", "import time;time.sleep(30)"))
    finally:
        timer.join()
    assert result.reason == "interrupted"
    result, _ = run(req, cancel_event=cancel)
    assert result.reason == "interrupted" and result.pid is None


def test_normal_exit_still_cleans_unwaited_child(tmp_path):
    req = request(tmp_path)
    pid_file = tmp_path / "orphan.pid"
    program = (
        "import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))"
    )
    result, _ = run(req, command=(sys.executable, "-c", program))
    assert result.exit_code == 0 and result.reason == "tool_error"  # no legitimate result protocol
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_stdout_stderr_are_bounded_metadata_not_trace_content(tmp_path, monkeypatch):
    req = request(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE_ENV_CANARY")
    program = "import sys;print('PRIVATE_FILE_CANARY'*10000);print('PRIVATE_STDERR_CANARY',file=sys.stderr)"
    result, trace = run(req, command=(sys.executable, "-c", program))
    assert result.stdout["bytes"] > 100000 and result.stdout["retained_bytes"] == 0
    assert result.stderr["bytes"] > 0 and result.stderr["content_omitted"]
    assert "CANARY" not in json.dumps(trace.events)


@pytest.mark.parametrize("payload", ["{}", '{"schema_version":1,"invalid":"PRIVATE_RESPONSE_CANARY"}'])
def test_invalid_response_is_rejected_without_echoing_payload(tmp_path, payload):
    req = request(tmp_path)
    program = f"import sys,pathlib;pathlib.Path(sys.argv[2]).write_text({payload!r})"
    result, trace = run(req, command=(sys.executable, "-c", program))
    assert result.reason == "tool_error" and result.response is None
    assert "PRIVATE_RESPONSE_CANARY" not in json.dumps(trace.events)


def test_protocol_rejects_clients_secrets_and_unknown_fields(tmp_path):
    req = request(tmp_path)
    data = req.to_dict()
    data["api_key"] = "DO_NOT_SEND"
    with pytest.raises(ValueError):
        WorkerRequest.from_dict(data)
    with pytest.raises((ValueError, TypeError)):
        replace(req, model=object())
    with pytest.raises(ValueError):
        ModelSpec(provider="openai", base_url="https://user:secret@example.com/v1")
    with pytest.raises(ValueError):
        ModelSpec(provider="openai", base_url="https://example.com?api_key=secret")
    with pytest.raises(ValueError):
        replace(req, spec=replace(req.spec, objective="Do not send sk-EXAMPLESECRET000000000"))
