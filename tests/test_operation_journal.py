from dataclasses import replace
import json
import pytest
from corecoder.tools import EditFileTool
from researchcoder.contracts import ExecutionPolicy
from researchcoder.durable.store import DurableStore, CheckpointError
from researchcoder.durable.journal import OperationJournal, JournalEndpoint
from tests.test_runtime_harness import case


def setup(tmp_path):
    spec, cfg, io, _ = case(tmp_path)
    spec = replace(
        spec,
        policy=ExecutionPolicy(
            writable_paths=("training.py",), allow_verification=True
        ),
    )
    store = DurableStore(tmp_path / "state.sqlite", io)
    store.commit(
        [
            (
                "attempts",
                "a",
                None,
                "running",
                dict(
                    schema_version=1,
                    job_id=spec.job_id,
                    status="running",
                    task_id=spec.task_id,
                ),
            )
        ]
    )
    journal = OperationJournal(store)
    journal.register(spec, "a", "f" * 64)
    path = tmp_path / "project/training.py"
    args = dict(file_path=str(path), old_string="loss = 1", new_string="loss = 2")
    return store, journal, spec, path, args


def test_apply_original_tool_and_reuse_idempotent_outcome(tmp_path):
    store, j, spec, path, args = setup(tmp_path)
    with store:
        prepared = j.prepare("a", "c", "edit_file", args)
        token = prepared["operation_id"]
        assert not j.start("a", token)["reuse"]
        actual = EditFileTool().execute(**args)
        j.finish("a", token, {"result": actual})
        assert (
            path.read_text() == "loss = 2\n"
            and store.get("operations", token)["status"] == "applied"
        )
        again = j.prepare("a", "c", "edit_file", args)
        assert again["operation_id"] == token and j.start("a", token)["reuse"]
        assert len(store.rows("operations")) == 1
        with pytest.raises(CheckpointError):
            j.prepare("a", "c", "edit_file", dict(args, new_string="other"))


@pytest.mark.parametrize(
    "phase", ["prepared", "started", "applied_uncommitted", "partial"]
)
def test_reconcile_real_file_bytes_without_replaying_patch(tmp_path, phase):
    store, j, spec, path, args = setup(tmp_path)
    with store:
        op = j.prepare("a", "c", "edit_file", args)
        if phase != "prepared":
            j.start("a", op["operation_id"])
        if phase == "applied_uncommitted":
            EditFileTool().execute(**args)
        if phase == "partial":
            path.write_text("other bytes\n")
        before = path.read_bytes()
        result = j.reconcile(store.get("operations", op["operation_id"]))
        assert path.read_bytes() == before
        assert result["status"] == (
            "applied"
            if phase == "applied_uncommitted"
            else "unknown"
            if phase == "partial"
            else "failed"
        )


def test_unknown_command_not_replayed_and_policy_not_expanded(tmp_path):
    store, j, spec, path, args = setup(tmp_path)
    with store:
        with pytest.raises(CheckpointError):
            j.prepare("a", "bad", "bash", {"command": "anything"})
        with pytest.raises(CheckpointError):
            j.prepare("a", "bad", "edit_file", dict(args, file_path="/tmp/outside.py"))
        command = {
            "command": list(spec.verification.argv),
            "project": spec.workspace.root,
            "timeout": spec.verification.timeout_seconds,
        }
        op = j.prepare("a", "verify", "verify_command", command)
        j.start("a", op["operation_id"])
        result = j.reconcile(store.get("operations", op["operation_id"]))
        assert (
            result["status"] == "unknown" and "no_replay" in result["reconcile_reason"]
        )
        with pytest.raises(CheckpointError):
            j.start("a", op["operation_id"])


def test_endpoint_contains_only_ids_and_local_socket():
    endpoint = JournalEndpoint(
        socket_path="/tmp/context/j.sock", job_id="j", task_id="t", attempt_id="a"
    )
    assert JournalEndpoint.from_json(endpoint.to_json()) == endpoint
    assert set(json.loads(endpoint.to_json())) == {
        "schema_version",
        "socket_path",
        "job_id",
        "task_id",
        "attempt_id",
    }


def test_completed_operation_cannot_be_reused_after_source_changes(tmp_path):
    store, j, spec, path, args = setup(tmp_path)
    with store:
        op = j.prepare("a", "c", "edit_file", args)
        j.start("a", op["operation_id"])
        j.finish("a", op["operation_id"], {"result": EditFileTool().execute(**args)})
        path.write_text("loss = 999\n")
        with pytest.raises(CheckpointError):
            j.start("a", op["operation_id"])
