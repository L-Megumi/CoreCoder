"""Controller-authorized effect intents; original tools perform the actual effects."""

import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import threading

from corecoder.workflow import workspace_revision
from researchcoder.contracts import Contract, identifier
from dataclasses import dataclass
from .store import CheckpointError, digest, boot_id


@dataclass(frozen=True, kw_only=True)
class JournalEndpoint(Contract):
    socket_path: str
    job_id: str
    task_id: str
    attempt_id: str
    schema_version: int = 1

    def validate(self):
        for v in (self.job_id, self.task_id, self.attempt_id):
            identifier(v)
        if not self.socket_path.startswith("/tmp/") or len(self.socket_path) > 100:
            raise ValueError("Invalid local journal endpoint")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(
            pid=pid,
            start_ticks=int(stat[19]),
            boot_id=boot_id(),
            pgid=os.getpgid(pid),
            state=stat[0],
        )
    except (OSError, ValueError):
        return None


class OperationJournal:
    def __init__(self, store, fault=None):
        self.store = store
        self.bindings = {}
        self.fault = fault or (lambda _: None)

    def register(self, spec, attempt_id, authority_digest):
        row = self.store.get("attempts", attempt_id)
        if row is None or row["status"] != "running" or row["job_id"] != spec.job_id:
            raise CheckpointError("Unregistered running attempt")
        self.bindings[attempt_id] = (spec, authority_digest)

    def prepare(self, attempt_id, call_id, name, args):
        if attempt_id not in self.bindings:
            raise CheckpointError("Attempt not authorized")
        identifier(call_id)
        spec, authority = self.bindings[attempt_id]
        root = Path(spec.workspace.root).resolve()
        request = digest(dict(call_id=call_id, tool=name, arguments=args))
        op_id = "operation-" + digest([spec.job_id, spec.task_id, attempt_id, call_id])
        previous = self.store.get("operations", op_id)
        if previous:
            if previous["value"]["request_digest"] != request:
                raise CheckpointError("Idempotency key reused for different input")
            return previous["value"]
        before = after = None
        path = None
        patch = None
        if name in ("edit_file", "write_file"):
            if name not in spec.policy.allowed_tool_names:
                raise CheckpointError("Effect tool not authorized")
            given = Path(args["file_path"]).expanduser()
            given = given if given.is_absolute() else root / given
            path = given.resolve()
            if (
                not path.is_relative_to(root)
                or path.relative_to(root).as_posix() not in spec.policy.writable_paths
            ):
                raise CheckpointError("Effect path not authorized")
            if any(
                p.is_symlink()
                for p in (given, *given.parents)
                if p.is_relative_to(root)
            ):
                raise CheckpointError("Symlink effect target")
            raw = path.read_bytes() if path.is_file() else b""
            if len(raw) > 1_000_000:
                raise CheckpointError("Effect input too large")
            before = file_hash(path)
            if name == "edit_file":
                # Predict bytes only; the original guarded EditFileTool remains the writer.
                content = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                if not path.is_file() or content.count(args["old_string"]) != 1:
                    raise CheckpointError("Patch precondition failed")
                planned = content.replace(
                    args["old_string"], args["new_string"], 1
                ).encode("utf-8")
            else:
                planned = args["content"].encode("utf-8")
            if len(planned) > 1_000_000:
                raise CheckpointError("Effect output too large")
            after = hashlib.sha256(planned).hexdigest()
            before_ref = self.store.io.put_bytes(
                spec.job_id, raw, media_type="text/plain"
            )
            after_ref = self.store.io.put_bytes(
                spec.job_id, planned, media_type="text/plain"
            )
            patch = self.store.io.put_json(
                spec.job_id,
                dict(
                    before=before_ref.to_dict(),
                    after=after_ref.to_dict(),
                    path=str(path),
                ),
            )
        elif name == "verify_command":
            if (
                not spec.policy.allow_verification
                or spec.verification is None
                or tuple(args["command"]) != spec.verification.argv
            ):
                raise CheckpointError("Verification not authorized")
            if (
                Path(args["project"]).resolve() != root
                or args["timeout"] != spec.verification.timeout_seconds
            ):
                raise CheckpointError("Verification scope changed")
        else:
            raise CheckpointError("No safe journal contract for this effect")
        value = dict(
            schema_version=1,
            job_id=spec.job_id,
            task_id=spec.task_id,
            attempt_id=attempt_id,
            operation_id=op_id,
            idempotency_key=op_id,
            request_digest=request,
            authority_digest=authority.sha256
            if hasattr(authority, "sha256")
            else authority,
            authority_ref=authority.to_dict()
            if hasattr(authority, "to_dict")
            else None,
            tool=name,
            path=str(path) if path else None,
            before_sha256=before,
            expected_after_sha256=after,
            after_sha256=None,
            status="prepared",
            workspace_before=workspace_revision(root),
            workspace_after=None,
            patch=patch.to_dict() if patch else None,
            outcome=None,
            reconcile_reason=None,
        )
        self.store.commit(
            [("operations", op_id, None, "prepared", value)],
            event_type="operation.prepared",
        )
        self.fault("operation_prepared")
        return value

    def start(self, attempt_id, operation_id):
        row = self.store.get("operations", operation_id)
        if row is None or row["value"]["attempt_id"] != attempt_id:
            raise CheckpointError("Foreign operation")
        if row["status"] in ("applied", "verified"):
            value = row["value"]
            spec, _ = self.bindings[attempt_id]
            expected = (
                value["workspace_before"]
                if value["tool"] == "verify_command"
                else value["workspace_after"]
            )
            if (
                value["outcome"] is None
                or workspace_revision(spec.workspace.root) != expected
            ):
                raise CheckpointError(
                    "Completed operation no longer matches current source/result"
                )
            if (
                value["path"]
                and file_hash(Path(value["path"])) != value["after_sha256"]
            ):
                raise CheckpointError("Idempotent patch result became stale")
            return dict(reuse=True, outcome=value["outcome"])
        if row["status"] != "prepared":
            raise CheckpointError("Unknown/started effect cannot be replayed")
        value = dict(row["value"], status="started")
        self.store.commit(
            [("operations", operation_id, row["version"], "started", value)],
            event_type="operation.started",
        )
        self.fault("operation_started")
        return dict(reuse=False)

    def finish(self, attempt_id, operation_id, outcome):
        self.fault("after_effect_before_journal")
        row = self.store.get("operations", operation_id)
        if (
            row is None
            or row["value"]["attempt_id"] != attempt_id
            or row["status"] != "started"
        ):
            raise CheckpointError("Stale or duplicate effect finish")
        value = row["value"]
        spec, _ = self.bindings[attempt_id]
        artifact = self.store.io.put_json(spec.job_id, outcome)
        if value["tool"] == "verify_command":
            status = "verified" if outcome.get("status") == "passed" else "failed"
            if outcome.get("status") in ("timeout", "interrupted", "launch_error"):
                status = (
                    "unknown" if outcome.get("status") != "launch_error" else "failed"
                )
            if outcome.get("checked_workspace_revision") != value["workspace_before"]:
                raise CheckpointError("Verification revision mismatch")
            after = None
        else:
            path = Path(value["path"])
            after = file_hash(path)
            if after == value["expected_after_sha256"]:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
                status = "applied"
            elif after == value["before_sha256"]:
                status = "failed"
            else:
                status = "unknown"
        updated = dict(
            value,
            status=status,
            after_sha256=after,
            workspace_after=workspace_revision(spec.workspace.root),
            outcome=artifact.to_dict(),
        )
        self.store.commit(
            [("operations", operation_id, row["version"], status, updated)],
            event_type="operation." + status,
        )
        self.fault("operation_finished")
        return updated

    def reconcile(self, row):
        value = row["value"]
        if value["tool"] == "verify_command":
            status = "failed" if row["status"] == "prepared" else "unknown"
            reason = (
                "command_not_started"
                if status == "failed"
                else "command_outcome_unknown_no_replay"
            )
        else:
            current = file_hash(Path(value["path"]))
            if current == value["expected_after_sha256"]:
                status = "applied"
                reason = "expected_bytes_present_validation_pending"
            elif current == value["before_sha256"]:
                status = "failed"
                reason = "original_bytes_present_no_replay_performed"
            else:
                status = "unknown"
                reason = "workspace_bytes_do_not_match_intent"
        updated = dict(
            value,
            status=status,
            reconcile_reason=reason,
            after_sha256=file_hash(Path(value["path"])) if value["path"] else None,
        )
        # Prepared writes confirmed absent must pass through started only when an
        # actual start was recorded; no fictional execution event is generated.
        if row["status"] == "prepared" and status == "applied":
            status = "unknown"
            updated["status"] = status
        if row["status"] not in ("prepared", "started", "unknown"):
            return value
        if row["status"] == status:
            return value
        self.store.commit(
            [("operations", row["id"], row["version"], status, updated)],
            event_type="operation.reconciled",
        )
        return updated


class JournalClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.sequence = 0

    def call(self, action, **payload):
        e = self.endpoint
        request = dict(
            action=action,
            job_id=e.job_id,
            task_id=e.task_id,
            attempt_id=e.attempt_id,
            **payload,
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(15)
            sock.connect(e.socket_path)
            sock.sendall((json.dumps(request) + "\n").encode())
            stream = sock.makefile("rb")
            raw = stream.readline(2_000_001)
        if len(raw) > 2_000_000:
            raise CheckpointError("Oversized journal response")
        response = json.loads(raw)
        if not response.get("ok"):
            raise CheckpointError("Controller refused effect; no automatic retry")
        return response["result"]

    def before(self, name, args):
        self.sequence += 1
        prepared = self.call(
            "prepare", call_id="effect-" + str(self.sequence), name=name, args=args
        )
        start = self.call("start", operation_id=prepared["operation_id"])
        return dict(operation_id=prepared["operation_id"], **start)

    def after(self, token, outcome):
        return self.call("finish", operation_id=token["operation_id"], outcome=outcome)


class JournalServer:
    """A Controller-owned service, never a second Runtime or Worker repository."""

    def __init__(self, journal, socket_path):
        self.journal = journal
        self.path = str(socket_path)

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    raw = self.rfile.readline(2_000_001)
                    if len(raw) > 2_000_000:
                        raise CheckpointError("Oversized effect request")
                    request = json.loads(raw)
                    peer, uid, _ = struct.unpack(
                        "3i",
                        self.request.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, 12
                        ),
                    )
                    attempt = request["attempt_id"]
                    lease = journal.store.get("workers", attempt)
                    if uid != os.getuid() or not lease or lease["status"] != "running":
                        raise CheckpointError("No active Worker lease")
                    identity = process_identity(lease["value"]["pid"])
                    if (
                        not identity
                        or identity["start_ticks"] != lease["value"]["start_ticks"]
                        or identity["boot_id"] != lease["value"]["boot_id"]
                    ):
                        raise CheckpointError("Stale Worker process identity")
                    if os.getpgid(peer) != lease["value"]["pid"]:
                        raise CheckpointError(
                            "Caller is not in the supervised Worker group"
                        )
                    spec, _ = journal.bindings[attempt]
                    if (request["job_id"], request["task_id"]) != (
                        spec.job_id,
                        spec.task_id,
                    ):
                        raise CheckpointError("Foreign effect identity")
                    action = request["action"]
                    if action == "prepare":
                        result = journal.prepare(
                            attempt,
                            request["call_id"],
                            request["name"],
                            request["args"],
                        )
                    elif action == "start":
                        result = journal.start(attempt, request["operation_id"])
                    elif action == "finish":
                        result = journal.finish(
                            attempt, request["operation_id"], request["outcome"]
                        )
                    elif action == "stage":
                        journal.store.commit(
                            [],
                            job_id=spec.job_id,
                            event_type="worker.stage",
                            details=dict(
                                task_id=spec.task_id,
                                attempt_id=attempt,
                                pid=peer,
                                stage=request["name"],
                                revision_id=lease["value"]["revision_id"],
                            ),
                        )
                        journal.fault(request["name"])
                        result = {}
                    else:
                        raise CheckpointError("Unsupported journal action")
                    response = dict(ok=True, result=result)
                except (ValueError, TypeError, KeyError, OSError, RuntimeError):
                    response = dict(ok=False)
                try:
                    self.wfile.write((json.dumps(response) + "\n").encode())
                except OSError:
                    pass

        self.server = socketserver.ThreadingUnixStreamServer(self.path, Handler)
        self.server.daemon_threads = True
        os.chmod(self.path, 0o600)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True, name="controller-journal"
        )
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
