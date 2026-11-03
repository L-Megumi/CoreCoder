"""Views of successful guarded reads, never a new filesystem read capability.

The complete rendered tool reply stays immutable. Derived source pages preserve
the observed line numbering, not original file bytes/newline encoding. The file
hash and all-files-v1 hash bind those pages to their actual production state.
"""

import copy
import hashlib
import json
from pathlib import Path
import re

from corecoder.tools.read import ReadFileTool


def message_key(message):
    # Match CompressionPipeline._original without accepting metadata from models.
    return hashlib.sha256(json.dumps(
        {k: v for k, v in message.items() if not k.startswith("_context_")},
        sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def complete_manifest(root):
    # projects.__init__/workspace have only contracts/stdlib imports at module
    # scope. Lazy import keeps the general Context engine independent at import.
    from researchcoder.evaluation.projects.workspace import capture_manifest
    return capture_manifest(root)


def observed_lines(output, offset):
    """Decode the existing ReadFileTool rendering, not arbitrary model JSON."""
    if type(offset) is not int or offset < 1 or not isinstance(output, str):
        return None
    rows = output.splitlines()
    if rows and re.fullmatch(r"\.\.\. \(\d+ lines total, showing \d+-\d+\)", rows[-1]):
        rows.pop()
    if not rows:
        return None
    lines = []
    for index, row in enumerate(rows, offset):
        number, separator, text = row.partition("\t")
        if not separator or number != str(index):
            return None
        lines.append(text)
    return lines


class CodingSourceContext:
    """Task-local trusted read registry; no model-proposed producer registration."""

    def __init__(self, root, store, scope, objective, max_records, mode="observed-source-v1"):
        self.root = Path(root).resolve()
        self.store, self.scope = store, scope
        self.objective, self.max_records = objective, max_records
        self.mode = mode
        self.current = complete_manifest(self.root)
        self.bindings = {}
        self.ambiguous = set()
        self.messages = {}
        self.entries = []
        self.windows = {}

    def refresh(self, scope):
        self.scope = scope
        self.current = complete_manifest(self.root)

    def capture_read(self, agent, call, result, original, before, revision):
        """Register only a successful existing RepairAgent read/cache observation."""
        if call.name != "read_file" or not isinstance(agent._tool_by_name.get(call.name), ReadFileTool):
            return
        event = agent.events[-1] if getattr(agent, "events", ()) else None
        if not event or event.get("status") != "ok" or event.get("tool") != "read_file":
            return
        after = complete_manifest(self.root)
        self.current = after
        if before != after:
            raise ValueError("Complete source changed while capturing read evidence")
        arguments = call.arguments
        target = Path(arguments["file_path"]).expanduser().resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Observed source escaped candidate")
        logical = target.relative_to(self.root).as_posix()
        entry = next((e for e in after.entries if e.path == logical), None)
        if entry is None:
            raise ValueError("Observed source absent from full manifest")
        offset, limit = arguments.get("offset", 1), arguments.get("limit", 2000)
        if type(offset) is not int or type(limit) is not int or offset < 1 or limit < 1:
            return
        cached = getattr(agent, "read_cache", {}).get((str(target), offset, limit, entry.sha256))
        if cached is None:
            # Empty/error/non-stable reads never create evidence or edit authority.
            return
        rendered = cached[1]
        lines = observed_lines(rendered, offset)
        if not lines or len(lines) > limit:
            return
        origin = "source-read-" + hashlib.sha256(call.id.encode()).hexdigest()
        raw = self.store.add_document(rendered.encode(), task_id=self.scope.task_id,
            attempt_id=self.scope.attempt_id, workspace_revision=revision, kind="tool_output", origin_id=origin)
        page = self.store.add_document("\n".join(lines).encode(), task_id=self.scope.task_id,
            attempt_id=self.scope.attempt_id, workspace_revision=revision, kind="source",
            origin_id=origin,
            source_path=logical, source_start_line=offset)
        binding = dict(schema_version=1, document_id=page.document_id, source_path=logical,
            read_sequence=len(self.entries) + 1,
            source_start_line=offset, source_end_line=offset + len(lines) - 1,
            file_sha256=entry.sha256, full_manifest_sha256=after.digest,
            workspace_revision=revision, rendered_document_id=raw.document_id,
            rendered_sha256=raw.artifact.sha256, source_artifact_sha256=page.artifact.sha256,
            semantics="observed_rendered_source; not original file bytes")
        for identity in (page.document_id, raw.document_id, original.document_id):
            prior = self.bindings.get(identity)
            if prior is not None and prior != binding:
                if identity == original.document_id:
                    if any(prior[key] != binding[key] for key in ("source_path", "source_start_line",
                            "source_end_line", "file_sha256", "full_manifest_sha256", "workspace_revision")):
                        self.ambiguous.add(identity)
                    continue
                # A document can share bytes/provenance but must not silently
                # acquire a newer complete-source identity.
                raise ValueError("Conflicting complete source observation")
            self.bindings[identity] = binding
        self.entries.append(binding)
        self.messages[message_key(dict(role="tool", tool_call_id=call.id, content=result))] = binding
        self.windows[page.document_id] = (lines, offset, False)

    def current_eligible(self, binding):
        return (binding["workspace_revision"] == self.scope.workspace_revision
                and binding["full_manifest_sha256"] == self.current.digest)

    def check_restore(self, document_id, historical):
        if document_id in self.ambiguous:
            raise ValueError("Ambiguous source output; restore its producer-specific source page")
        binding = self.bindings.get(document_id)
        if binding is None:
            return None
        if not self.current_eligible(binding) and not historical:
            raise ValueError("Complete source version changed; explicit historical recovery required")
        return binding

    def capture_restore(self, call, result, encoded, original):
        resolved = result.get("reference", {}).get("document_id")
        binding = self.bindings.get(resolved)
        if binding is None:
            return
        prior = self.bindings.get(original.document_id)
        if prior is not None and prior != binding:
            self.ambiguous.add(original.document_id)
        else:
            self.bindings[original.document_id] = binding
        self.messages[message_key(dict(role="tool", tool_call_id=call.id, content=encoded))] = binding
        # Only the source-page reference has source-relative line semantics.
        # Restoring an original tool envelope does not fabricate a source page.
        if resolved != binding["document_id"]:
            return
        self.windows[binding["document_id"]] = (
            result["content"].splitlines(), result["source_start_line"], True)

    def knows(self, message):
        return message_key(message) in self.messages

    def cards(self, message):
        binding = self.messages.get(message_key(message))
        return [self.card(binding)] if binding else None

    def card(self, binding, *, reference_only=False):
        valid = self.current_eligible(binding)
        card = dict(kind="source", coding_source=True, **copy.deepcopy(binding),
            current_eligible=valid,
            classification="observed_record" if valid else "historical_observation")
        lines, start, restored = self.windows[binding["document_id"]]
        card["restored_window"] = restored
        if not reference_only:
            # Deterministic local selection; no hidden Oracle, LLM or case ID.
            terms = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", self.objective.lower()))
            ranked = sorted(range(len(lines)), key=lambda n: (
                -sum(term in lines[n].lower() for term in terms),
                0 if re.search(r"\b(?:def|class|return|assert|raise)\b", lines[n]) else 1, n))
            selected = set(range(min(8, len(lines)))) if restored else set()
            for index in ranked[:4]:
                selected.update(range(max(0, index - 1), min(len(lines), index + 3)))
            chosen, chars = [], 0
            for index in sorted(selected)[:20]:
                text = lines[index][:240]
                if chars + len(text) > 2200:
                    break
                chosen.append(dict(line_no=start + index, text=text,
                                   line_truncated=len(lines[index]) > len(text)))
                chars += len(text)
            card["excerpt"] = chosen
        return card

    def reference_cards(self):
        # Most recent unique observed pages stay discoverable after old complete
        # tool groups are dropped. The immutable catalog retains every page.
        found, cards = set(), []
        for binding in reversed(self.entries):
            if binding["document_id"] in found:
                continue
            found.add(binding["document_id"])
            cards.append(self.card(binding, reference_only=True))
            if len(cards) >= min(4, self.max_records):
                break
        return cards

    def reference_index(self):
        return [{key: card[key] for key in ("document_id", "source_path", "source_start_line",
                    "source_end_line", "classification", "current_eligible")}
                for card in self.reference_cards()]

    def receipt(self):
        return dict(schema_version=1, mode=self.mode, algorithm="all-files-v1",
                    current_full_manifest_sha256=self.current.digest,
                    reads=copy.deepcopy(self.entries))


class RecoveryDelivery:
    """V2 only: exact successful recovery replies await a confirmed model call.

    Each call owns an immutable returned window, even when several calls recover
    the same document. This is delivery bookkeeping, not a new tool or retry.
    """

    def __init__(self, sources, store, trace):
        self.sources, self.store, self.trace = sources, store, trace
        self.entries, self.attempts = {}, []
        self.prepared = {}

    def capture(self, call, result, encoded, original):
        if call.id in self.entries:
            raise ValueError("Recovery tool call identity already registered")
        reference = result.get("reference", {})
        document_id = reference.get("document_id")
        if document_id is None:
            # Record recovery remains authorized by the existing EvidenceStore.
            document_id = reference.get("ref_id")
        if not document_id or not isinstance(result, dict):
            raise ValueError("Authorized recovery identity missing")
        binding = result.get("coding_source") or result.get("coding_handoff") or {}
        full = binding.get("full_manifest_sha256", self.sources.current.digest)
        revision = binding.get("workspace_revision", reference.get("workspace_revision"))
        content = result.get("content")
        if content is None:
            content = json.dumps(result.get("record"), ensure_ascii=False, sort_keys=True)
        if not isinstance(content, str):
            raise ValueError("Invalid recovery content")
        self.entries[call.id] = dict(tool_call_id=call.id, document_id=document_id,
            arguments=copy.deepcopy(call.arguments), result=copy.deepcopy(result), encoded=encoded,
            original=original, status="pending", full_manifest_sha256=full, workspace_revision=revision,
            captured_current=bool(result.get("current_eligible")),
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            result_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
            source_start_line=result.get("source_start_line"), returned_line_count=len(content.splitlines()),
            acknowledged_request=None)
        self.trace.emit("context.recovery.pending", artifact_id=original.artifact.artifact_id,
                        revision_id=hashlib.sha256(call.id.encode()).hexdigest(), count=1)

    def _project(self, entry):
        # The original recovered record must still exist and retain its digest.
        self.store.io.read_bytes(entry["original"].artifact)
        result = copy.deepcopy(entry["result"])
        reference = result.get("reference", {})
        if "artifact" in reference:
            from researchcoder.contracts import ArtifactRef
            self.store.io.read_bytes(ArtifactRef.from_dict(reference["artifact"]))
        current = (entry["captured_current"]
                   and entry["full_manifest_sha256"] == self.sources.current.digest
                   and entry["workspace_revision"] == self.sources.scope.workspace_revision)
        result["current_eligible"] = current
        if result.get("classification") != "unverified_hypothesis":
            result["classification"] = "observed_record" if current else "historical_observation"
        if isinstance(result.get("coding_handoff"), dict):
            result["coding_handoff"]["current_eligible"] = current
            result["coding_handoff"]["classification"] = result["classification"]
        # The recovered content remains byte-for-byte the same string. Only
        # outer freshness metadata follows a later candidate edit.
        return json.dumps(result, ensure_ascii=False)

    def protect(self, messages):
        from .compression import groups, validate_pairs
        validate_pairs(messages)
        pending = {key: value for key, value in self.entries.items() if value["status"] == "pending"}
        self.prepared = {}
        if not pending:
            return messages
        observed = set()
        for group in groups(messages):
            replies = [m for m in group if m.get("role") == "tool" and m.get("tool_call_id") in pending]
            if not replies:
                continue
            calls = {c["id"]: c for c in group[0].get("tool_calls", [])}
            for reply in replies:
                identity = reply["tool_call_id"]
                entry, call = pending[identity], calls.get(identity)
                if (identity in observed or call is None or call["function"]["name"] != "restore_context_evidence"
                        or json.loads(call["function"]["arguments"]) != entry["arguments"]
                        or reply.get("content") != entry["encoded"]):
                    raise ValueError("Recovered window or tool pairing was altered")
                projected = self._project(entry)
                reply["content"] = projected
                self.prepared[identity] = projected
                observed.add(identity)
            for message in group:
                message["_context_recovery"] = True
        if observed != set(pending):
            raise ValueError("Pending recovered window is missing from model history")
        return messages

    def begin(self, messages, *, parent_id):
        if not self.prepared:
            return None
        from .compression import validate_pairs
        validate_pairs(messages)
        actual = {}
        for message in messages:
            identity = message.get("tool_call_id")
            if message.get("role") == "tool" and identity in self.prepared:
                if identity in actual:
                    raise ValueError("Duplicate pending recovery reply")
                actual[identity] = message.get("content")
        if actual != self.prepared:
            raise ValueError("Compression dropped or changed a pending recovered window")
        attempt = dict(request_event_id=parent_id, tool_call_ids=list(self.prepared), status="pending",
            projected_sha256={key: hashlib.sha256(value.encode()).hexdigest() for key, value in self.prepared.items()})
        self.attempts.append(attempt)
        self.trace.emit("context.recovery.delivery_started", parent_id=parent_id, count=len(self.prepared))
        return len(self.attempts) - 1

    def finish(self, token, *, delivered, reason):
        if token is None:
            return
        attempt = self.attempts[token]
        if attempt["status"] != "pending":
            raise ValueError("Recovery delivery already settled")
        attempt.update(status="delivered" if delivered else "unconfirmed", reason=reason)
        if delivered:
            for identity in attempt["tool_call_ids"]:
                self.entries[identity].update(status="delivered", acknowledged_request=attempt["request_event_id"])
        self.trace.emit("context.recovery.delivered" if delivered else "context.recovery.unconfirmed",
            parent_id=attempt["request_event_id"], status="ok" if delivered else "unknown",
            count=len(attempt["tool_call_ids"]), reason=reason)

    def receipt(self):
        fields = ("tool_call_id", "document_id", "status", "full_manifest_sha256", "workspace_revision",
                  "content_sha256", "result_sha256", "source_start_line", "returned_line_count", "acknowledged_request")
        return dict(schema_version=1, mode="observed-source-v2",
            entries=[{key: value[key] for key in fields} for value in self.entries.values()],
            attempts=copy.deepcopy(self.attempts),
            semantics="delivered only after a successful model response with known per-request Usage; not semantic understanding")
