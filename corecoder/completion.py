"""Read-free completion requirements derived from observed evidence and configurations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from .locator import _CONFIG_REFERENCE
from .observation_state import current_observation_ids
from .source_search import needs_source_search, suggested_query
from .source_visibility import visible_end_line

_MISSING = re.compile(r'''(?:missing path\s*|No such file or directory:\s*)["']([^"'\r\n]+)["']''')


def scalar(value):
    """A deliberately restricted YAML scalar subset; never evaluate tags, variables or aliases."""
    value = value.strip()
    if value.startswith("'"):
        match = re.fullmatch(r"'((?:[^']|'')*)'(?:\s+#.*)?\s*", value)
        return match.group(1).replace("''", "'") if match else None
    if value.startswith('"'):
        try:
            result, end = json.JSONDecoder().raw_decode(value)
        except ValueError:
            return None
        return result if isinstance(result, str) and (not value[end:].strip() or value[end:].lstrip().startswith("#")) else None
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if not value or re.search(r"[\[\]{}&*!|><$~\\]", value) or value.lower() in ("null", "true", "false", "none"):
        return None
    return value


def dataset_validation_path(text):
    """Resolve only an absolute root + scalar val in the recognized Ultralytics-style profile."""
    values = {}
    for line in text.splitlines():
        match = re.match(r"^(path|val)\s*:\s*(.*)$", line)
        if match:
            key, value = match.groups()
            if key in values:
                return None, "Duplicate dataset path/val key"
            values[key] = scalar(value)
    val = values.get("val")
    if "val" not in values:
        return None, "Scalar val absent from observed lines"
    if not val:
        return None, "Unsupported scalar val"
    if Path(val).is_absolute():
        return os.path.normpath(val), None
    root = values.get("path")
    if "path" not in values:
        return None, "Scalar dataset path absent from observed lines"
    if not root or not Path(root).is_absolute():
        return None, "Relative/unsupported dataset root requires framework-specific resolution"
    return os.path.normpath(str(Path(root) / val)), None


class CompletionChecklist:
    def __init__(self, state, source_hints=()):
        self.state = state
        self.source_hints = tuple(source_hints)

    def canonical(self, reference):
        path = Path(reference)
        return os.path.normcase(os.path.normpath(str(path if path.is_absolute() else self.state.root / path)))

    def _matching(self, reference, kinds):
        target = self.canonical(reference)
        current = current_observation_ids(self.state)
        return [item for item in self.state.observations.values() if item["id"] in current and item["kind"] in kinds and any(
            self.canonical(item["data"][key]) == target
            for key in ("requested_path", "resolved_path") if item["data"].get(key)
        )]

    def status(self):
        checks = []

        def add(kind, subject, observations=(), action=None, blocked_reason=None):
            identifier = "check_" + hashlib.sha256(f"{kind}:{subject}".encode()).hexdigest()[:12]
            if any(check["id"] == identifier for check in checks):
                return
            status = "observed" if observations else "blocked" if blocked_reason else "pending"
            checks.append({"id": identifier, "kind": kind, "subject": subject, "status": status,
                           "observation_ids": [item["id"] for item in observations],
                           "blocked_reason": blocked_reason, "next_action": action if status == "pending" else None})

        def read(reference, kind="config_read"):
            kinds = ("source_location", "file_excerpt") if kind == "source_hint_read" else ("config_file", "file_excerpt")
            matches = self._matching(reference, kinds)
            readable = [item for item in matches if item["data"].get("file_sha256") and (
                (item["data"].get("snippet") or {}).get("content") is not None or "content" in item["data"])]
            add(kind, reference, readable, {"tool": "read_project_file", "arguments": {"path": reference}},
                None if readable or not matches else matches[-1]["data"]["status"])
            return readable

        def path(reference, kind):
            matches = self._matching(reference, ("path_status",))
            observed = [item for item in matches if item["data"]["status"] in ("exists", "missing")]
            add(kind, reference, observed, {"tool": "inspect_paths", "arguments": {"paths": [reference]}},
                None if observed or not matches else matches[-1]["data"]["status"])

        state = self.state
        if state.evidence is None:
            add("extract", "bound log", action={"tool": "extract_log_evidence", "arguments": {}})
        else:
            critical = [item for item in state.evidence if item["type"] in ("ERROR", "NUMERIC_ANOMALY")]
            unexposed = [item for item in critical if item["id"] not in state.exposed_evidence_ids]
            if unexposed:
                offset = next(index for index, item in enumerate(state.evidence) if item["id"] == unexposed[0]["id"])
                add("evidence_page", "unexposed critical evidence", action={
                    "tool": "extract_log_evidence", "arguments": {"offset": offset, "limit": 20},
                })
            for evidence in critical[:64]:
                if evidence["id"] not in state.exposed_evidence_ids:
                    continue
                text = evidence.get("message") or evidence["content"]
                if needs_source_search(evidence) and not self.source_hints:
                    self._search_checks(evidence, add)
                if evidence.get("frames"):
                    locations = [item for item in state.observations.values()
                                 if item["kind"] == "source_location" and evidence["id"] in item["evidence_ids"]
                                 and item["id"] in current_observation_ids(state)]
                    usable = [item for item in locations if item["data"].get("file_sha256")]
                    add("source_lookup", evidence["id"], usable, {
                        "tool": "locate_source", "arguments": {"evidence_ids": [evidence["id"]]},
                    }, None if usable or not locations else "Source lookup attempted without a readable current source")
                for reference in dict.fromkeys(_MISSING.findall(text)):
                    path(reference, "historical_missing_path")
                for reference in dict.fromkeys(_CONFIG_REFERENCE.findall(text)):
                    readable = read(reference)
                    # Only this explicit profile has the absolute path + relative val convention.
                    if "images not found" not in text or not reference.lower().endswith((".yaml", ".yml")):
                        continue
                    if readable:
                        latest_hash = readable[-1]["data"]["file_sha256"]
                        fragments = [item["data"] for item in readable if item["data"]["file_sha256"] == latest_hash]
                        observed_lines = {}
                        stalled_lines = set()
                        last_end = 0
                        more = False
                        for data in fragments:
                            snippet = data.get("snippet") or data
                            start = snippet.get("start_line", 1)
                            end = visible_end_line(snippet)
                            text_lines = snippet.get("content", "").split("\n")[:max(0, end - start + 1)]
                            if not text_lines and (snippet.get("char_truncated") or snippet.get("truncated")):
                                stalled_lines.add(start)
                            observed_lines.update({start + offset: line for offset, line in enumerate(text_lines)})
                            if end >= last_end:
                                last_end = end
                                more = bool(snippet.get("truncated") or snippet.get("char_truncated"))
                        content = "\n".join(observed_lines[number] for number in sorted(observed_lines))
                        resolved, reason = dataset_validation_path(content)
                        if resolved:
                            path(resolved, "current_validation_path")
                        elif reason in ("Scalar val absent from observed lines", "Scalar dataset path absent from observed lines") and more:
                            if last_end + 1 in stalled_lines:
                                add("config_resolution", reference, blocked_reason="Configuration line exceeds the excerpt character limit")
                            else:
                                add("config_resolution", reference, action={"tool": "read_project_file", "arguments": {
                                    "path": reference, "start_line": last_end + 1, "max_lines": 100,
                                }})
                        else:
                            add("config_resolution", reference, blocked_reason=reason)
            if len(critical) > 64:
                add("input_bound", "More than 64 critical records; provide a smaller complete log excerpt")
        for reference in self.source_hints:
            read(reference, "source_hint_read")
        self._initialization_checks(add)
        pending = [item for item in checks if item["status"] == "pending"]
        blocked = [item for item in checks if item["status"] == "blocked"]
        return {"ready_for_report": not pending, "checks": checks, "pending_count": len(pending),
                "blocked_count": len(blocked), "scope": "Observed checks, not proof of root cause or successful training"}

    def _initialization_checks(self, add):
        current = current_observation_ids(self.state)
        records = [item for item in self.state.observations.values() if item["id"] in current]
        for source in records:
            if source["kind"] not in ("source_location", "file_excerpt"):
                continue
            data = source["data"]
            if source["kind"] == "source_location" and data.get("function_verified") is not True:
                continue
            lead = (data.get("function") or {}).get("initialization_check")
            if not lead or not data.get("file_sha256") or not data.get("resolved_path"):
                continue
            subject = data["resolved_path"] + ":" + data["file_sha256"] + ":" + lead["method"]
            if lead["status"] != "candidate":
                add("initialization_read", subject, blocked_reason=lead["reason"])
                continue
            matching = [item for item in records if item["kind"] in ("source_location", "file_excerpt")
                        and item["data"].get("file_sha256") == data["file_sha256"]
                        and item["data"].get("resolved_path") == data["resolved_path"]]
            covered, used = set(), []
            for item in matching:
                span = item["data"].get("snippet") or item["data"]
                visible = set(range(span.get("start_line", 1), visible_end_line(span) + 1))
                if visible.intersection(lead["required_lines"]):
                    covered.update(visible)
                    used.append(item)
            missing = sorted(set(lead["required_lines"]) - covered)
            if missing:
                start = missing[0]
                stalled = any((item["data"].get("snippet") or item["data"]).get("start_line") == start
                              and (item["data"].get("snippet") or item["data"]).get("char_truncated")
                              and visible_end_line(item["data"].get("snippet") or item["data"]) < start for item in matching)
                add("initialization_read", subject, action={"tool": "read_project_file", "arguments": {
                    "path": data["resolved_path"], "start_line": start, "max_lines": min(60, missing[-1] - start + 1)}},
                    blocked_reason="Initializer line exceeds reader character bound" if stalled else None)
            elif lead["header_truncated"] or lead["required_lines_omitted"]:
                add("initialization_read", subject, blocked_reason="Initializer inspection metadata truncated; remaining source unknown")
            else:
                add("initialization_read", subject, used)
            if lead["unresolved_attributes"] or lead["attributes_omitted"]:
                add("initialization_unknown", subject, blocked_reason="Some instance attributes require inherited/dynamic or omitted initialization")

    def _search_checks(self, evidence, add):
        state = self.state
        identifier = evidence["id"]
        searches = [item for item in state.observations.values()
                    if item["kind"] == "source_search" and identifier in item["evidence_ids"]]
        action = {"tool": "search_project_source", "arguments": {
            "query": suggested_query(evidence), "evidence_ids": [identifier],
        }}
        if not searches:
            add("source_search", identifier, action=action)
            return
        latest = searches[-1]
        ids = latest["data"]["candidate_ids"]
        if not ids:
            add("source_search", identifier, blocked_reason=(
                "Partial search without matches; unscanned source remains unknown" if latest["data"]["partial"]
                else "No matches in the searched Python scope; source cause remains unknown"))
            return
        current = current_observation_ids(state)
        candidates = [state.observations[i] for i in ids if i in current]
        if not candidates:
            add("source_search", identifier, action=action)
            return
        add("source_search", identifier, [latest])
        stalled = []
        for candidate in candidates:
            data = candidate["data"]
            matching = self._matching(data["resolved_path"], ("file_excerpt",))
            reads = [item for item in matching
                     if item["data"].get("file_sha256") == data["file_sha256"]
                     and item["data"].get("start_line", 0) <= data["hit_line"] <= visible_end_line(item["data"])]
            if reads:
                add("candidate_read", identifier, reads)
                self._symbol_check(reads[0], identifier, add)
                return
            if any(item["data"].get("file_sha256") == data["file_sha256"]
                   and item["data"].get("start_line") == data["hit_line"]
                   and item["data"].get("char_truncated") and visible_end_line(item["data"]) < data["hit_line"]
                   for item in matching):
                stalled.append(candidate["id"])
        candidates = [item for item in candidates if item["id"] not in stalled]
        if not candidates:
            add("candidate_read", identifier, blocked_reason="Candidate source line exceeds the reader character limit")
            return
        candidate = candidates[0]["data"]
        # Reading from the hit line permits progress even when a preceding line is oversized.
        add("candidate_read", identifier, action={"tool": "read_project_file", "arguments": {
            "path": candidate["relative_path"], "start_line": candidate["hit_line"], "max_lines": 20,
        }})

    def _symbol_check(self, source, evidence_id, add):
        """A lexical hint schedules AST verification; it is never a binding assertion."""
        span = source["data"].get("snippet") or source["data"]
        text = "\n".join(span.get("content", "").split("\n")[:visible_end_line(span) - span.get("start_line", 1) + 1])
        evidence = next(item for item in self.state.evidence if item["id"] == evidence_id)
        query = suggested_query(evidence).lower()
        hints = []
        pattern = r"(?m)^\s*from\s+([\w.]+)\s+import\s+([A-Za-z_]\w*)(?:\s+as\s+([A-Za-z_]\w*))?"
        for match in re.finditer(pattern, text):
            module, imported, alias = match.groups()
            if module.split(".")[0] in sys.stdlib_module_names:
                continue
            name = alias or imported
            call = re.search(r"\b" + re.escape(name) + r"((?:\.[A-Za-z_]\w*)*)\s*\(", text[match.end():])
            if call is None:
                continue
            symbol = name + call.group(1)
            line = span["start_line"] + text[:match.end() + call.start()].count("\n")
            symbol_match = query in (imported + " " + symbol).lower()
            line_text = text.split("\n")[line - span["start_line"]]
            hints.append((not symbol_match, query not in line_text.lower(), line, symbol))
        if not hints:
            return
        _, _, line, symbol = min(hints)
        subject = source["id"] + ":" + symbol
        current = current_observation_ids(self.state)
        matches = [item for item in self.state.observations.values() if item["id"] in current
                   and item["kind"] == "symbol_resolution" and evidence_id in item["evidence_ids"]
                   and item["data"].get("source_observation_id") == source["id"]
                   and item["data"].get("symbol") == symbol]
        if matches:
            latest = matches[-1]
            if latest["data"]["candidates"]:
                add("symbol_lookup", subject, [latest])
            else:
                add("symbol_lookup", subject, blocked_reason="Static lookup unresolved: " + ", ".join(latest["data"]["reasons"]))
        else:
            add("symbol_lookup", subject, action={"tool": "resolve_project_symbol", "arguments": {
                "symbol": symbol, "source_observation_id": source["id"], "evidence_ids": [evidence_id], "at_line": line,
            }})

    def guidance(self):
        status = self.status()
        pending = [item for item in status["checks"] if item["status"] == "pending"]
        blocked = [item for item in status["checks"] if item["status"] == "blocked"]
        completed = [item for item in status["checks"] if item["status"] == "observed"]
        if self.state.evidence is None:
            pending = [item for item in pending if item["kind"] == "extract"]
        return {"ready_for_report": status["ready_for_report"], "pending_count": len(pending),
                "next_checks": pending[:6], "blocked_checks": blocked[:4],
                "completed_checks": [{"kind": item["kind"], "subject": item["subject"],
                                      "observation_ids": item["observation_ids"][:2]} for item in completed[:6]],
                "completed_checks_omitted": max(0, len(completed) - 6),
                "note": "Investigation data. Batch independent pending checks; only inspect literal targets, not every ancestor. "
                        "Reuse completed checks and their observation IDs; do not re-read them without a new question. "
                        "Blocked checks are unknown, not missing; do not repeat the same out-of-scope request. "
                        "Initializer metadata is only a reading lead; verify actual source lines before asserting defaults. "
                        "Report historical/current differences explicitly."}
