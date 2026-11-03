"""Metadata-only ordered events. Never log messages, tool arguments or environment."""

import datetime
import json
import re
import threading
import time
import uuid

FIELDS = {
    "model",
    "tool",
    "usage",
    "elapsed_seconds",
    "failure_kind",
    "reservation_id",
    "amount",
    "exit_code",
    "pid",
    "limit",
    "count",
    "stdout_bytes",
    "stderr_bytes",
    "reason",
    "source_sequence",
    "revision_id",
    "logical_task_id",
    "role",
    "upstream_task_id",
    "workspace_revision",
    "artifact_id",
}


def safe_label(value):
    value = str(value)[:160]
    value = re.sub(r"(?:sk-[\w-]{6,}|Bearer\s+\S+)", "[redacted]", value, flags=re.IGNORECASE)
    return value


class TraceSink:
    def __init__(self, job_id, task_id, attempt_id, path=None, parent_id=None):
        self.job_id, self.task_id, self.attempt_id = job_id, task_id, attempt_id
        self.path = path
        self.parent_id = parent_id
        self.events = []
        self._lock = threading.RLock()

    def emit(self, event_type, *, parent_id=None, status="ok", **metadata):
        if set(metadata) - FIELDS:
            raise ValueError("Trace cannot contain arbitrary payload fields")
        # Usage is numeric/unknown metadata only, not a back door for content.
        if "usage" in metadata:
            usage = metadata["usage"]
            if not isinstance(usage, dict) or any(type(v) not in (int, float, bool, type(None)) for v in usage.values()):
                raise ValueError("Trace usage must contain scalar counters")
            if not set(usage) <= {"prompt_tokens", "completion_tokens", "estimated_tokens", "unknown", "charged_tokens"}:
                raise ValueError("Unexpected usage key")
        with self._lock:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            event = {
                "job_id": self.job_id,
                "task_id": self.task_id,
                "attempt_id": self.attempt_id,
                "event_id": "event-" + uuid.uuid4().hex,
                "parent_id": parent_id or self.parent_id,
                "event_type": event_type,
                "sequence": len(self.events) + 1,
                "monotonic_ns": time.monotonic_ns(),
                "start_timestamp": now,
                "end_timestamp": None if status == "running" else now,
                "status": status,
                "model": None,
                "tool": None,
                "usage": None,
                "elapsed_seconds": None,
                "failure_kind": None,
            }
            event.update({k: safe_label(v) if isinstance(v, str) else v for k, v in metadata.items()})
            self.events.append(event)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
            return event["event_id"]

    def start(self, event_type, **metadata):
        token = self.emit(event_type + ".start", status="running", **metadata)
        return token, time.monotonic()

    def finish(self, span, event_type, status="ok", **metadata):
        return self.emit(
            event_type + ".end", parent_id=span[0], status=status, elapsed_seconds=max(0, time.monotonic() - span[1]), **metadata
        )

    def merged(self, worker_events):
        events = self.events + worker_events
        seen = set()
        for event in events:
            required = {
                "job_id",
                "task_id",
                "attempt_id",
                "event_id",
                "parent_id",
                "event_type",
                "sequence",
                "monotonic_ns",
                "start_timestamp",
                "end_timestamp",
                "status",
                "model",
                "tool",
                "usage",
                "elapsed_seconds",
                "failure_kind",
            }
            if not isinstance(event, dict) or not required <= set(event) or set(event) - required - FIELDS:
                raise ValueError("Malformed trace event")
            if type(event["monotonic_ns"]) is not int:
                raise ValueError("Invalid trace clock")
            if (event.get("job_id"), event.get("task_id"), event.get("attempt_id")) != (
                self.job_id,
                self.task_id,
                self.attempt_id,
            ):
                raise ValueError("Trace identity mismatch")
            if event["event_id"] in seen:
                raise ValueError("Duplicate trace event")
            seen.add(event["event_id"])
        ordered = sorted(events, key=lambda e: e["monotonic_ns"])
        for index, event in enumerate(ordered, 1):
            event["sequence"] = index
        if any(e["parent_id"] and e["parent_id"] not in seen for e in ordered):
            raise ValueError("Trace contains dangling parent")
        return ordered
