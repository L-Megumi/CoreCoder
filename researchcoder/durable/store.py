"""SQLite metadata transactions, immutable file artifacts and one Controller lease."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import uuid

from researchcoder.contracts import ArtifactRef, EvidenceRef

TABLES = ("jobs", "plans", "tasks", "attempts", "budgets", "operations", "workers")
TRANSITIONS = {
    "plans": {"committed": {"superseded"}},
    "budgets": {"active": {"active"}},
    "tasks": {
        "pending": {
            "running",
            "reused",
            "blocked",
            "cancelled",
            "incomplete",
            "failed",
        },
        "running": {
            "running",
            "succeeded",
            "failed",
            "incomplete",
            "cancelled",
            "lost",
            "unknown",
        },
    },
    "attempts": {
        "running": {"succeeded", "failed", "incomplete", "cancelled", "lost", "unknown"}
    },
    "operations": {
        "prepared": {"started", "failed", "unknown"},
        "started": {"applied", "verified", "failed", "unknown"},
        "applied": {"verified", "unknown"},
        "unknown": {"applied", "failed"},
    },
    "workers": {"running": {"exited", "lost"}},
    "jobs": {
        "open": {"open", "running", "needs_review", "failed"},
        "running": {
            "running",
            "succeeded",
            "failed",
            "incomplete",
            "cancelled",
            "needs_review",
        },
        "incomplete": {"running", "needs_review"},
        "needs_review": set(),
        "succeeded": {"needs_review"},
        "cancelled": {"needs_review"},
        "failed": {"running", "needs_review"},
    },
}


class CheckpointError(ValueError):
    pass


def encode(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


class DurableStore:
    """Only the owning Controller process writes; its service thread shares the lock.

    JSON hashes detect corruption, not a malicious same-UID rewrite of all state.
    No stored model/command configuration is an execution authority on reload.
    """

    def __init__(self, path, artifacts, *, journal_mode="auto", fault=None):
        self.path = Path(path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (self.path, *self.path.parents)):
            raise CheckpointError("Symlink checkpoint path")
        self.io = artifacts
        self.pid = os.getpid()
        self.lock = threading.RLock()
        self.fault = fault or (lambda _: None)
        lease_path = str(self.path) + ".controller"
        self.lease = os.fdopen(
            os.open(
                lease_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
            ),
            "a+b",
        )
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close()
            raise CheckpointError("Another Controller owns this checkpoint") from None
        try:
            mount = subprocess.run(
                ["findmnt", "-T", str(self.path.parent), "-n", "-o", "FSTYPE"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            fs = mount.stdout.strip() if mount.returncode == 0 else "unknown"
            local = fs in {"ext4", "xfs", "btrfs", "tmpfs", "overlay"}
            if journal_mode not in ("auto", "wal", "delete"):
                raise CheckpointError("Invalid journal mode")
            if not local and journal_mode == "auto":
                raise CheckpointError(
                    "Unverified filesystem; explicit tested rollback configuration required"
                )
            if journal_mode == "wal" and not local:
                raise CheckpointError("WAL requires a verified local filesystem")
            mode = "wal" if local and journal_mode != "delete" else "delete"
            self.db = sqlite3.connect(
                self.path, timeout=5, isolation_level=None, check_same_thread=False
            )
            os.chmod(self.path, 0o600)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA foreign_keys=ON")
            self.db.execute("PRAGMA synchronous=FULL")
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise CheckpointError("SQLite integrity failure")
            version = self.db.execute("PRAGMA user_version").fetchone()[0]
            existing = self.db.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            if version not in (0, 1) or (version == 0 and existing):
                raise CheckpointError("Incompatible checkpoint schema")
            selected = self.db.execute("PRAGMA journal_mode=" + mode).fetchone()[0]
            if selected.lower() != mode:
                raise CheckpointError("Journal mode unavailable")
            if not existing:
                with self.transaction():
                    for table in TABLES:
                        self.db.execute(
                            f"CREATE TABLE {table}(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,version INTEGER NOT NULL,status TEXT NOT NULL,body TEXT NOT NULL,sha256 TEXT NOT NULL)"
                        )
                    self.db.execute(
                        "CREATE TABLE events(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,body TEXT NOT NULL,sha256 TEXT NOT NULL)"
                    )
                    self.db.execute(
                        "CREATE TABLE artifacts(id TEXT PRIMARY KEY,job_id TEXT NOT NULL,body TEXT NOT NULL,sha256 TEXT NOT NULL)"
                    )
                    self.db.execute("PRAGMA user_version=1")
            self.filesystem = {
                "type": fs,
                "journal_mode": selected,
                "synchronous": "FULL",
                "schema_version": 1,
            }
            self.validate()
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self.lease.close()
            raise

    @contextmanager
    def transaction(self):
        if os.getpid() != self.pid:
            raise CheckpointError("Worker cannot write Controller repository")
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.fault("before_db_commit")
                self.db.execute("COMMIT")
                self.fault("after_db_commit")
            except BaseException:
                if self.db.in_transaction:
                    self.db.execute("ROLLBACK")
                raise

    def get(self, table, key):
        if table not in TABLES:
            raise CheckpointError("Invalid table")
        with self.lock:
            row = self.db.execute(
                f"SELECT * FROM {table} WHERE id=?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            body = json.loads(row["body"])
        except ValueError as exc:
            raise CheckpointError("Invalid checkpoint JSON") from exc
        if digest(body) != row["sha256"] or body.get("schema_version") != 1:
            raise CheckpointError("Checkpoint body/version mismatch")
        if (
            body.get("job_id") != row["job_id"]
            or body.get("status", row["status"]) != row["status"]
        ):
            raise CheckpointError("Checkpoint identity/status mismatch")
        return dict(row) | {"value": body}

    def rows(self, table, job_id=None):
        if table not in TABLES:
            raise CheckpointError("Invalid table")
        with self.lock:
            ids = self.db.execute(
                f"SELECT id FROM {table}" + (" WHERE job_id=?" if job_id else ""),
                ((job_id,) if job_id else ()),
            ).fetchall()
            return [self.get(table, r[0]) for r in ids]

    def _refs(self, value, job_id, found=None):
        found = {} if found is None else found
        if len(found) > 10000:
            raise CheckpointError("Reference graph too large")
        if isinstance(value, dict):
            if {"artifact_id", "storage_key", "sha256", "size_bytes"} <= value.keys():
                ref = ArtifactRef.from_dict(value)
                if ref.job_id != job_id:
                    raise CheckpointError("Cross-Job artifact")
                if ref.storage_key in found:
                    if found[ref.storage_key] != ref:
                        raise CheckpointError("Conflicting artifact identity")
                    return found
                raw = self.io.read_bytes(ref)
                found[ref.storage_key] = ref
                if ref.media_type == "application/json":
                    self._refs(json.loads(raw), job_id, found)
                return found
            if {"record_key", "producer_task_id", "artifact"} <= value.keys():
                ref = EvidenceRef.from_dict(value)
                if ref not in self.io.index_evidence(
                    ref.artifact,
                    task_id=ref.producer_task_id,
                    attempt_id=ref.producer_attempt_id,
                    workspace_revision=ref.workspace_revision,
                ):
                    raise CheckpointError("Forged evidence metadata")
            for key, item in value.items():
                if key.lower() in (
                    "api_key",
                    "password",
                    "environ",
                    "environment",
                    "private_key",
                    "client",
                    "lock",
                ):
                    raise CheckpointError("Runtime or secret material forbidden")
                self._refs(item, job_id, found)
        elif isinstance(value, list):
            for item in value:
                self._refs(item, job_id, found)
        return found

    def commit(
        self,
        updates,
        *,
        event_id=None,
        event_type="checkpoint",
        job_id=None,
        details=None,
    ):
        """Validate and publish artifacts first, CAS all rows and event in one transaction."""
        refs = {}
        for table, key, expected, status, body in updates:
            if table not in TABLES or body.get("schema_version") != 1:
                raise CheckpointError("Invalid checkpoint update")
            self._refs(body, body["job_id"], refs)
            encode(body)
        if details is not None:
            self._refs(details, job_id, refs)
        self.fault("after_artifact_publish")
        event_id = event_id or "event-" + uuid.uuid4().hex
        with self.transaction():
            if self.db.execute(
                "SELECT 1 FROM events WHERE id=?", (event_id,)
            ).fetchone():
                raise CheckpointError("Duplicate event")
            for table, key, expected, status, body in updates:
                old = self.get(table, key)
                if (old["version"] if old else None) != expected:
                    raise CheckpointError("Stale or duplicate update")
                if (
                    old
                    and table in TRANSITIONS
                    and status not in TRANSITIONS[table].get(old["status"], set())
                ):
                    raise CheckpointError("Illegal durable state transition")
                if body.get("status", status) != status:
                    raise CheckpointError("Body status mismatch")
                if (
                    old is None
                    and table in ("tasks", "operations", "attempts")
                    and status not in {"pending", "prepared", "running"}
                ):
                    raise CheckpointError("Invalid initial state")
                self.db.execute(
                    f"INSERT OR REPLACE INTO {table} VALUES(?,?,?,?,?,?)",
                    (
                        key,
                        body["job_id"],
                        0 if expected is None else expected + 1,
                        status,
                        encode(body),
                        digest(body),
                    ),
                )
            for ref in refs.values():
                body = ref.to_dict()
                prior = self.db.execute(
                    "SELECT sha256 FROM artifacts WHERE id=?", (ref.storage_key,)
                ).fetchone()
                if prior and prior[0] != digest(body):
                    raise CheckpointError("Artifact identity changed")
                self.db.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                    (ref.storage_key, ref.job_id, encode(body), digest(body)),
                )
            event = {
                "schema_version": 1,
                "event_id": event_id,
                "job_id": job_id or updates[0][4]["job_id"],
                "event_type": event_type,
                "timestamp": time.time(),
                "rows": [[u[0], u[1], u[3]] for u in updates],
                "details": details,
            }
            self.db.execute(
                "INSERT INTO events VALUES(?,?,?,?)",
                (event_id, event["job_id"], encode(event), digest(event)),
            )

    def validate(self):
        with self.lock:
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise CheckpointError("SQLite integrity failure")
            if self.db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise CheckpointError("Incompatible checkpoint")
            for table in TABLES:
                for row in self.rows(table):
                    self._refs(row["value"], row["job_id"])
            for table in ("events", "artifacts"):
                for row in self.db.execute(f"SELECT * FROM {table}"):
                    body = json.loads(row["body"])
                    if digest(body) != row["sha256"]:
                        raise CheckpointError("Corrupt event/artifact index")
                    if table == "artifacts":
                        self.io.read_bytes(ArtifactRef.from_dict(body))
        return True

    def orphans(self):
        with self.lock:
            known = {r[0] for r in self.db.execute("SELECT id FROM artifacts")}
        return tuple(
            sorted(
                p.relative_to(self.io.root).as_posix()
                for p in self.io.root.rglob("*.blob")
                if p.relative_to(self.io.root).as_posix() not in known
            )
        )

    def close(self):
        with self.lock:
            self.db.close()
            self.lease.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
