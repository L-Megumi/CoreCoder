"""Optional, allowlisted retrospective Langfuse export; no agent instrumentation swap.

Current API: Langfuse Python v5 start_observation/create_score. Official references:
https://langfuse.com/docs/compatibility
https://python.reference.langfuse.com/langfuse
No legacy trace()/generation()/api.trace.list, no prompt/body/Artifact upload.
"""

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import re
from researchcoder.contracts import ArtifactRef
from researchcoder.adapters.artifacts import ArtifactIO

ROLES = {
    "single",
    "log_analysis",
    "code_analysis",
    "diagnosis",
    "repair",
    "verification",
    "lead",
    "aggregation",
}
STATES = {
    "ok",
    "running",
    "error",
    "failed",
    "incomplete",
    "cancelled",
    "needs_review",
    "succeeded",
    "passed",
    "timeout",
    "unknown",
    "stream_opened",
    "closed",
}
CATEGORIES = {
    "job",
    "task",
    "worker",
    "model",
    "tool",
    "budget",
    "verification",
    "context",
    "graph",
    "recovery",
}


def opaque(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def number(value):
    return (
        float(value)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0
        else None
    )


def projection(result, raw, events):
    """A fresh allowlist projection; never recursively forward arbitrary metadata."""
    root_id = opaque(result.run_id + result.trial.trial_id)
    root = dict(
        id=root_id,
        parent_id=None,
        name="evaluation.trial",
        type="span",
        metadata=dict(
            run_id=opaque(result.run_id),
            job_id=opaque(result.job_id),
            case_id=opaque(result.trial.case_id),
            mode=result.trial.mode,
            repeat=result.trial.repeat,
            status=result.status,
            elapsed_seconds=result.metrics.wall_seconds,
            historical_replay=True,
        ),
    )
    observations = {root_id: root}
    aliases = {}
    for event in events:
        if event.get("event_type", "").endswith(".end") and event.get("parent_id"):
            aliases[event["event_id"]] = event["parent_id"]

    def resolve(value):
        visited = set()
        while value in aliases:
            if value in visited:
                raise ValueError("Trace parent cycle")
            visited.add(value)
            value = aliases[value]
        return value

    endings = {
        resolve(e["parent_id"]): e
        for e in events
        if e.get("event_type", "").endswith(".end") and e.get("parent_id")
    }
    ids = {e["event_id"] for e in events}
    for event in events:
        if event["event_id"] in aliases:
            continue
        end = endings.get(event["event_id"], event)
        category = event.get("event_type", "").split(".")[0]
        category = category if category in CATEGORIES else "other"
        state = end.get("status") if end.get("status") in STATES else "unknown"
        parent = resolve(event["parent_id"]) if event.get("parent_id") in ids else None
        metadata = dict(
            job_id=opaque(event.get("job_id")),
            task_id=opaque(event.get("task_id")),
            attempt_id=opaque(event.get("attempt_id")),
            event_id=opaque(event["event_id"]),
            parent_id=opaque(event["parent_id"]) if event.get("parent_id") else None,
            revision_id=opaque(event.get("revision_id")),
            event_type=opaque(event.get("event_type")),
            status=state,
            role=event.get("role") if event.get("role") in ROLES else "single",
            elapsed_seconds=number(end.get("elapsed_seconds")),
        )
        for field in ("start_timestamp", "end_timestamp"):
            value = (event if field == "start_timestamp" else end).get(field)
            if isinstance(value, str) and re.fullmatch(
                r"\d{4}-\d\d-\d\dT[\d:.+Z-]{8,40}", value
            ):
                metadata["historical_" + field] = value
        usage = end.get("usage") or {}
        known = usage.get("unknown") is False and all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        observation = dict(
            id=opaque(event["event_id"]),
            parent_id=opaque(parent) if parent else root_id,
            name=category + ".observation",
            type="generation"
            if category == "model"
            and event.get("event_type", "").startswith("model.request")
            else "tool"
            if category == "tool"
            else "span",
            metadata=metadata,
        )
        if event.get("model"):
            observation["model"] = "model-" + opaque(event["model"])[:16]
        if known:
            observation["usage_details"] = dict(
                input=usage["prompt_tokens"], output=usage["completion_tokens"]
            )
        metadata["usage_unknown"] = not known
        if number(usage.get("estimated_tokens")) is not None:
            metadata["estimated_input_tokens"] = number(usage["estimated_tokens"])
        observations[observation["id"]] = observation
    # Parent-before-child SDK construction; reject malformed cycles rather than fake a tree.
    ordered = []
    pending = dict(observations)
    while pending:
        ready = [
            k
            for k, v in pending.items()
            if v["parent_id"] is None or any(p["id"] == v["parent_id"] for p in ordered)
        ]
        if not ready:
            raise ValueError("Dangling/cyclic trace export")
        for key in ready:
            ordered.append(pending.pop(key))
    scores = {
        name: float(getattr(result.metrics, name))
        for name in (
            "end_to_end_success",
            "diagnosis_correct",
            "localization_correct",
            "needs_review",
        )
    }
    for key in (
        "fail_to_pass",
        "pass_to_pass",
        "evidence_retention",
        "recovery_success",
        "duplicate_side_effects",
    ):
        value = getattr(result.metrics, key)
        if value is not None:
            scores[key] = float(value)
    return dict(
        schema_version=1,
        trace_id=root_id[:32],
        observations=ordered,
        scores=scores,
        scope="Allowlisted metrics and pseudonymous IDs only. Historical timestamps are metadata; exported SDK span timing is export time.",
    )


def payload_for_result(result, directory):
    directory = Path(directory).resolve()
    io = ArtifactIO(directory / "trials" / result.trial.trial_id / "artifacts")
    raw = io.read_json(result.raw)
    events = []
    records = raw.get("records", [])
    for record in records:
        ref = record.get("trace")
        if ref:
            reference = ArtifactRef.from_dict(ref)
            if reference.job_id != result.job_id:
                raise ValueError("Foreign trace")
            for event in io.read_json(reference)["events"]:
                if event["job_id"] != result.job_id:
                    raise ValueError("Foreign trace event")
                events.append(
                    dict(
                        event,
                        role=record.get("role", "single"),
                        revision_id=record.get("revision_id"),
                    )
                )
    graph = raw.get("scheduler_result")
    if graph:
        reference = ArtifactRef.from_dict(graph["trace_ref"])
        if reference.job_id != result.job_id:
            raise ValueError("Foreign graph")
        events.extend(io.read_json(reference)["events"])
    return projection(result, raw, events)


def send_payload(payload, client):
    """Current SDK surface, separately dependency-injected in offline contract tests."""
    mapped = {}
    for item in payload["observations"]:
        context = dict(trace_id=payload["trace_id"])
        if item["parent_id"] is not None:
            context["parent_span_id"] = mapped[item["parent_id"]].id
        kwargs = dict(
            trace_context=context,
            name=item["name"],
            as_type=item["type"],
            metadata=item["metadata"],
        )
        if item["type"] == "generation":
            if item.get("model"):
                kwargs["model"] = item["model"]
            if item.get("usage_details"):
                kwargs["usage_details"] = item["usage_details"]
        span = client.start_observation(**kwargs)
        mapped[item["id"]] = span
        span.end()
    for name, value in payload["scores"].items():
        client.create_score(
            name=name,
            value=value,
            trace_id=payload["trace_id"],
            data_type="NUMERIC",
            score_id=opaque(payload["trace_id"] + name),
            metadata=dict(scorer="researchcoder-independent-oracle"),
        )
    client.flush()


class LangfuseExporter:
    def __init__(self, *, mode="disabled", allow_network=False):
        if mode not in ("disabled", "offline", "langfuse"):
            raise ValueError("Unknown exporter mode")
        self.mode = mode
        self.allow_network = allow_network

    def export_result(self, result, directory):
        if self.mode == "disabled":
            return dict(status="disabled", sent=False)
        try:
            payload = payload_for_result(result, directory)
            if self.mode == "offline":
                return dict(status="offline", sent=False, payload=payload)
            if not self.allow_network:
                return dict(status="network_not_authorized", sent=False)
            if not all(
                os.environ.get(k)
                for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
            ):
                return dict(status="not_configured", sent=False)
            version = importlib.metadata.version("langfuse")
            if version.split(".")[0] != "5":
                return dict(status="unsupported_sdk", sent=False)
            from langfuse import Langfuse

            client = Langfuse(
                timeout=3, debug=False, environment="researchcoder-evaluation"
            )
            send_payload(payload, client)
            return dict(
                status="submitted",
                sent=True,
                sdk_version=version,
                scope="SDK submission/flush only; server ingestion and visibility are not verified",
            )
        except Exception as exc:
            return dict(
                status="export_error", sent=False, error_type=type(exc).__name__
            )
