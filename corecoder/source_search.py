"""Bounded literal search: evidence-linked candidates, never root-cause assertions."""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

from .locator import _CONFIG_REFERENCE, _Locator, _scope_at

SEARCH_LIMITS = {"entries": 2000, "files": 80, "file_bytes": 200_000, "bytes": 2_000_000, "depth": 24}
EXCLUDED = {"venv", "env", "node_modules", "__pycache__", "build", "dist", "site-packages"}


def needs_source_search(item):
    text = item.get("message") or item.get("content", "")
    return (item["type"] in ("ERROR", "NUMERIC_ANOMALY") and not item.get("frames")
            and not _CONFIG_REFERENCE.search(text))


def suggested_query(item):
    text = item.get("message") or item.get("content", "")
    match = re.search(r"\b([A-Za-z_][A-Za-z_0-9]{1,60})\s*[:=]\s*[-+]?(?:nan|inf(?:inity)?)\b", text, re.IGNORECASE)
    return match.group(1) if match else item.get("exception_type") or "nan"


def _excluded(relative):
    return any(part.startswith(".") or part.lower() in EXCLUDED for part in relative.parts)


def search_source(state, query, evidence_ids, paths=None, max_results=8):
    if not isinstance(query, str) or not 2 <= len(query.strip()) <= 120 or any(c in query for c in "\x00\r\n"):
        raise ValueError("query must be a single-line literal of 2 to 120 characters")
    if (not isinstance(evidence_ids, list) or not 1 <= len(evidence_ids) <= 10
            or any(not isinstance(i, str) or i not in state.exposed_evidence_ids for i in evidence_ids)):
        raise ValueError("evidence_ids must contain 1 to 10 already exposed evidence IDs")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 20:
        raise ValueError("max_results must be an integer between 1 and 20")
    if paths is None:
        paths = ["."]
    if (not isinstance(paths, list) or not 1 <= len(paths) <= 5
            or any(not isinstance(p, str) or not p or "\x00" in p for p in paths)):
        raise ValueError("paths must contain 1 to 5 literal project files or directories")
    roots = []
    for reference in paths:
        lexical = Path(os.path.abspath(state.root / reference))
        target = lexical.resolve()
        if not lexical.is_relative_to(state.root) or not target.is_relative_to(state.root):
            raise ValueError("Search paths must stay inside the project")
        relative = lexical.relative_to(state.root)
        if _excluded(relative) or any(p.is_symlink() for p in [lexical, *lexical.parents] if p != state.root and p.is_relative_to(state.root)):
            raise ValueError("Hidden, dependency and symlink search scopes are excluded")
        roots.append(target)
    limits = dict(SEARCH_LIMITS)
    reader = _Locator(state.root, {}, {"max_file_bytes": limits["file_bytes"], "max_snippet_lines": 12,
                                     "max_snippet_chars": 800, "context_lines": 2})
    stats = {"entries": 0, "files": 0, "bytes_read": 0, "filtered": 0}
    reasons = Counter()
    candidates = []
    seen = set()
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    def walk(target, counted=False, depth=0):
        if target in seen:
            return
        seen.add(target)
        if not counted and stats["entries"] >= limits["entries"]:
            reasons["entry_limit"] += 1
            return
        if not counted:
            stats["entries"] += 1
        try:
            if target.is_symlink() or _excluded(target.relative_to(state.root)):
                stats["filtered"] += 1
                return
            if not target.resolve().is_relative_to(state.root):
                reasons["scope_changed"] += 1
                return
            if target.is_dir():
                if depth >= limits["depth"]:
                    reasons["depth_limit"] += 1
                    return
                # Bound enumeration itself, before sorting or materializing a large directory.
                entries = []
                with os.scandir(target) as stream:
                    for entry in stream:
                        if stats["entries"] >= limits["entries"]:
                            reasons["entry_limit"] += 1
                            break
                        entries.append(Path(entry.path))
                        stats["entries"] += 1
                for child in sorted(entries):
                    yield from walk(child, counted=True, depth=depth + 1)
            elif target.suffix.lower() == ".py" and target.is_file():
                yield target
            elif not target.exists():
                reasons["missing_scope"] += 1
            else:
                stats["filtered"] += 1
        except OSError:
            reasons["traversal_error"] += 1

    stop = False
    for root in roots:
        for target in walk(root):
            if stats["files"] >= limits["files"]:
                reasons["file_limit"] += 1
                stop = True
                break
            remaining = limits["bytes"] - stats["bytes_read"]
            if remaining <= 1:
                reasons["byte_limit"] += 1
                stop = True
                break
            # A concurrent growth can consume at most the remaining budget, including the sentinel byte.
            reader.limits["max_file_bytes"] = min(limits["file_bytes"], remaining - 1)
            before = reader.stats["bytes_read"]
            data = reader.read(str(target))
            stats["files"] += 1
            stats["bytes_read"] += reader.stats["bytes_read"] - before
            if data["status"] != "read":
                reason = "byte_limit" if data["status"] == "too_large" and remaining - 1 < limits["file_bytes"] else data["status"]
                reasons[reason] += 1
                continue
            for number, line in enumerate(data["lines"], 1):
                match = pattern.search(line)
                if not match:
                    continue
                scope = _scope_at(data["functions"], number)
                payload = {"status": "candidate", "requested_path": str(target.relative_to(state.root)),
                           "resolved_path": str(target), "relative_path": str(target.relative_to(state.root)),
                           "query": query, "file_sha256": data["file_sha256"], "hit_line": number,
                           "match_start_column": match.start() + 1,
                           "match_excerpt": line[max(0, match.start() - 120):match.end() + 120],
                           "match_excerpt_truncated": len(line) > len(line[max(0, match.start() - 120):match.end() + 120]),
                           "function": scope, "parse_status": data["parse_status"],
                           "snippet": reader.snippet(data["lines"], number, scope)}
                candidates.append(state.observe("source_candidate", payload, evidence_ids))
                if len(candidates) >= max_results:
                    reasons["result_limit"] += 1
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    summary = state.observe("source_search", {
        "status": "partial" if reasons else "searched", "query": query, "paths": paths,
        "limits": {**limits, "max_results": max_results}, "stats": stats, "partial": bool(reasons),
        "reasons": dict(reasons), "candidate_ids": [item["id"] for item in candidates],
        "matches_returned": len(candidates),
        "scope": "Literal case-insensitive Python search; hidden/dependency paths and symlinks excluded. "
                 "Candidates are clues, not causes. No matches only describes the scanned scope at search time.",
    }, evidence_ids)
    return {"observations": [*candidates, summary]}
