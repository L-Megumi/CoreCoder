"""Bounded recent source windows inside the existing context pipeline.

The registry accepts only results of executed, authorized tools. It neither
reads new paths nor grants citation authority. Retention means visible input,
not a claim that a model understood or used the source.
"""
import copy
import hashlib
import json
from pathlib import Path

from corecoder.context import estimate_tokens
from corecoder.evidence_context import project_tool_result
from corecoder.observation_state import current_observation_ids
from .coding_source import message_key, complete_manifest


class ActiveWorkingSet:
    MAX_CONTENT_CHARS = 12000
    MAX_GROUPS = 4
    MAX_TOKENS = 4000

    def __init__(self, engine):
        self.engine = engine
        self.entries = {}
        self.window_counts = {}
        self.repeated_recoveries = 0
        self.last = {}
        self.current_full = None

    def capture(self, call, result, original, *, restored=None):
        """Called after the original guarded read or EvidenceStore recovery."""
        if not isinstance(result, str) or len(result) > self.MAX_CONTENT_CHARS:
            return
        received = result
        source_artifact = None
        sources = self.engine.coding_sources
        message = dict(role="tool", tool_call_id=call.id, content=result)
        binding = sources.messages.get(message_key(message)) if sources else None
        observation_ids = []
        windows = []
        full_manifest = None
        if binding is not None:
            if not sources.current_eligible(binding):
                return
            start = binding["source_start_line"]
            end = binding["source_end_line"]
            if restored is not None:
                if not restored.get("current_eligible") or not isinstance(restored.get("content"), str):
                    return
                start = restored.get("source_start_line")
                if start is None:
                    return  # Envelope pages do not have source-relative line semantics.
                end = start + len(restored["content"].splitlines()) - 1
            else:
                rendered = next(d for d in self.engine.store.catalog.documents
                                if d.document_id == binding["rendered_document_id"])
                source_artifact = rendered.artifact
                result = self.engine.store.io.read_bytes(source_artifact).decode("utf-8")
                if len(result) > self.MAX_CONTENT_CHARS:
                    return
            windows.append((binding["source_path"], binding["file_sha256"], start, end))
        else:
            state = getattr(self.engine.agent, "state", None)
            if state is None or restored is not None:
                return
            try:
                payload = json.loads(result)
            except (TypeError, ValueError):
                return
            if not isinstance(payload, dict):
                return
            current = current_observation_ids(state)
            records = payload.get("observations", [])
            if not records:
                return
            root = Path(self.engine.request.spec.workspace.root).resolve()
            snapshot = complete_manifest(root)
            snapshot_files = {entry.path: entry.sha256 for entry in snapshot.entries}
            for record in records:
                if not isinstance(record, dict) or record.get("id") not in current:
                    return
                # Exact registered observations only, never a model-made source object.
                if state.observations.get(record["id"]) != record:
                    return
                data = record.get("data", {})
                snippet = data.get("snippet") or data
                if not isinstance(snippet.get("content"), str) or not data.get("file_sha256"):
                    return
                path = Path(data.get("resolved_path", ""))
                if not path.is_absolute() or not path.is_relative_to(root):
                    return  # Keep external-root observations on their original scoped path.
                if snapshot_files.get(path.relative_to(root).as_posix()) != data["file_sha256"]:
                    return
                observation_ids.append(record["id"])
                windows.append((data.get("resolved_path"), data["file_sha256"],
                                snippet.get("start_line", 1), snippet.get("end_line")))
            full_manifest = snapshot.digest
        key = hashlib.sha256(json.dumps([original.workspace_revision, windows], sort_keys=True).encode()).hexdigest()
        if restored is not None and key in self.window_counts:
            self.repeated_recoveries += 1
        self.window_counts[key] = self.window_counts.get(key, 0) + 1
        self.entries[call.id] = dict(
            tool_call_id=call.id, tool_name=call.name, arguments=copy.deepcopy(call.arguments),
            content=result, received_content=received, source_artifact=source_artifact,
            original=original, binding=copy.deepcopy(binding),
            full_manifest=full_manifest,
            observation_ids=tuple(observation_ids), key=key,
            sequence=len(self.entries) + 1,
        )

    def _current(self, entry):
        engine = self.engine
        if entry["original"].workspace_revision != engine.scope.workspace_revision:
            return False
        if entry["binding"] is not None:
            return engine.coding_sources.current_eligible(entry["binding"])
        state = getattr(engine.agent, "state", None)
        return (state is not None and entry["full_manifest"] == self.current_full
                and set(entry["observation_ids"]) <= current_observation_ids(state))

    def protect(self, messages, max_tokens):
        """Pin complete recent tool groups; older duplicate windows remain offloadable."""
        from .compression import groups, validate_pairs

        validate_pairs(messages)
        if any(entry["binding"] is None for entry in self.entries.values()):
            self.current_full = complete_manifest(self.engine.request.spec.workspace.root).digest
        budget = min(self.MAX_TOKENS, max(0, max_tokens // 3))
        used, selected, seen, duplicates, stale, oversized = 0, [], set(), 0, 0, 0
        for group in reversed(groups(messages)):
            if any(m.get("_context_recovery") for m in group):
                continue  # Mandatory first delivery has its own strict admission.
            calls = {c["id"]: c for c in group[0].get("tool_calls", [])}
            eligible = []
            for message in group:
                identity = message.get("tool_call_id")
                entry = self.entries.get(identity) if message.get("role") == "tool" else None
                if entry is None:
                    continue
                call = calls.get(identity)
                if (call is None or call["function"]["name"] != entry["tool_name"]
                        or json.loads(call["function"]["arguments"]) != entry["arguments"]):
                    raise ValueError("Active evidence tool pairing changed")
                if message.get("content") not in (entry["received_content"], project_tool_result(entry["received_content"])):
                    raise ValueError("Active evidence content changed")
                if not self._current(entry):
                    stale += 1
                    continue
                if entry["key"] in seen:
                    duplicates += 1
                    continue
                self.engine.store._authorize(self.engine.scope, entry["original"].task_id,
                    entry["original"].attempt_id, entry["original"].workspace_revision)
                # Re-validate the immutable artifact before reusing its source view.
                self.engine.store.io.read_bytes(entry["original"].artifact)
                if entry["source_artifact"] is not None:
                    self.engine.store.io.read_bytes(entry["source_artifact"])
                eligible.append((message, entry))
            if not eligible:
                continue
            candidate = copy.deepcopy(group)
            replacements = {e["tool_call_id"]: e["content"] for _, e in eligible}
            for m in candidate:
                if m.get("role") == "tool" and m.get("tool_call_id") in replacements:
                    m["content"] = replacements[m["tool_call_id"]]
            cost = estimate_tokens(candidate)
            if len(selected) >= self.MAX_GROUPS or used + cost > budget:
                oversized += len(eligible)
                continue
            for message, entry in eligible:
                message["content"] = entry["content"]
                seen.add(entry["key"])
            group[0]["_context_working_set"] = True
            for message, _ in eligible:
                message["_context_working_set"] = True
            used += cost
            selected.append(tuple(entry["tool_call_id"] for _, entry in eligible))
        self.last = dict(version="active-v1", selected_groups=len(selected),
            selected_tool_calls=[key for group in selected for key in group],
            retained_tokens_estimate=used, working_set_budget=budget,
            duplicate_windows_not_pinned=duplicates, stale_windows_excluded=stale,
            windows_outside_budget=oversized, repeated_recoveries=self.repeated_recoveries)
        return messages

    def observed(self, view):
        visible = {m.get("tool_call_id") for m in view.messages if m.get("role") == "tool"
                   and m.get("tool_call_id") in self.entries
                   and m.get("content") == self.entries[m["tool_call_id"]]["content"]}
        self.engine.trace.emit("context.working_set.built", count=len(visible),
            usage={"estimated_tokens": self.last.get("retained_tokens_estimate", 0)},
            reason="active-v1")

    def receipt(self):
        return dict(schema_version=1, version="active-v1", registered_windows=len(self.window_counts),
            repeated_recoveries=self.repeated_recoveries, last_view=copy.deepcopy(self.last),
            semantics="bounded recent source visibility; not semantic use or zero-cost tool calls")
