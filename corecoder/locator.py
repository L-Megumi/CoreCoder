"""Locate log evidence in current project files without importing project code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import stat
import sys
import tokenize
from pathlib import Path, PurePosixPath, PureWindowsPath

from .evidence import extract_evidence
from .initialization import initialization_lead
from .source_visibility import visible_end_line

_CONFIG_REFERENCE = re.compile(r"""["']([^"'\r\n]+\.(?:yaml|yml|json|toml|ini|cfg))["']""", re.IGNORECASE)


def _logical_path(value: str) -> PurePosixPath | PureWindowsPath:
    return PureWindowsPath(value) if "\\" in value or PureWindowsPath(value).drive else PurePosixPath(value)


def _function_scopes(tree: ast.AST) -> list[dict]:
    scopes = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.parents = []
            self.owners = []

        def visit_ClassDef(self, node):
            self.parents.append(node.name)
            self.owners.append(node)
            self.generic_visit(node)
            self.owners.pop()
            self.parents.pop()

        def visit_FunctionDef(self, node):
            scope = {
                "name": node.name,
                "qualname": ".".join([*self.parents, node.name]),
                "start_line": min([node.lineno, *(d.lineno for d in node.decorator_list)]),
                "body_start_line": node.body[0].lineno,
                "end_line": node.end_lineno,
            }
            if self.owners and isinstance(self.owners[-1], ast.ClassDef):
                lead = initialization_lead(self.owners[-1], node, scope["qualname"])
                if lead:
                    scope["initialization_check"] = lead
            scopes.append(scope)
            self.parents.append(node.name)
            self.owners.append(node)
            self.generic_visit(node)
            self.owners.pop()
            self.parents.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return scopes


def _scope_at(scopes: list[dict], line: int) -> dict | None:
    matches = [scope for scope in scopes if scope["body_start_line"] <= line <= scope["end_line"]]
    return min(matches, key=lambda s: s["end_line"] - s["start_line"]) if matches else None


def _function_agrees(expected: str | None, scope: dict | None, parsed: bool) -> bool | None:
    if not expected or not parsed:
        return None
    if expected == "<module>":
        return scope is None
    if expected.startswith("<"):  # Comprehensions and lambdas have synthetic runtime names.
        return None
    return scope is not None and expected in (scope["name"], scope["qualname"])


def _file_version(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


class _Locator:
    def __init__(self, root: Path, mappings: dict[str, str], limits: dict, *, verify_cached=False):
        self.root = root
        self.mappings = sorted(mappings.items(), key=lambda pair: len(_logical_path(pair[0]).parts), reverse=True)
        self.limits = limits
        self.cache = {}
        self._cache_versions = {}
        self.verify_cached = verify_cached
        self.stats = {"files_read": 0, "cache_hits": 0, "bytes_read": 0, "cache_invalidations": 0}

    def resolve(self, reference: str) -> dict:
        result = {
            "requested_path": reference, "resolved_path": None,
            "relative_path": None, "path_resolution": None,
        }
        if not isinstance(reference, str) or not reference or "\x00" in reference:
            return {**result, "status": "invalid_path"}
        if reference.startswith("<") and reference.endswith(">"):
            return {**result, "status": "virtual_path"}
        logical = _logical_path(reference)
        candidate = None
        for old, new in self.mappings:
            try:
                suffix = logical.relative_to(_logical_path(old))
            except ValueError:
                continue
            target = Path(new)
            if not target.is_absolute():
                target = self.root / target
            candidate = target.joinpath(*suffix.parts)
            result["path_resolution"] = "mapped"
            break
        if candidate is None:
            candidate = Path(reference)
            if logical.is_absolute() and not candidate.is_absolute():
                return {**result, "status": "unmapped_path"}
            if candidate.is_absolute():
                result["path_resolution"] = "absolute"
            else:
                result["path_resolution"] = "relative"
                candidate = self.root / candidate
        try:
            candidate = candidate.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            return {**result, "status": "invalid_path", "detail": str(exc)}
        result["resolved_path"] = str(candidate)
        if not candidate.is_relative_to(self.root):
            return {**result, "status": "outside_project"}
        result["relative_path"] = candidate.relative_to(self.root).as_posix()
        return {**result, "status": "resolved"}

    def read(self, path: str) -> dict:
        if path in self.cache:
            try:
                unchanged = _file_version(Path(path).stat()) == self._cache_versions[path]
            except OSError:
                unchanged = False
            if unchanged and not self.verify_cached:
                self.stats["cache_hits"] += 1
                return self.cache[path]
            if not unchanged:
                self.stats["cache_invalidations"] += 1
                self.cache.pop(path, None)
                self._cache_versions.pop(path, None)
        result = self._read(Path(path))
        if path in self.cache and result is not self.cache[path]:
            self.stats["cache_invalidations"] += 1
            self.cache.pop(path, None)
        # Missing/failed files may recover during the same investigation.
        if result["status"] == "read":
            self.cache[path] = result
        else:
            self._cache_versions.pop(path, None)
        return result

    def _read(self, path: Path) -> dict:
        try:
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode):
                return {"status": "not_file"}
            if metadata.st_size > self.limits["max_file_bytes"]:
                return {"status": "too_large"}
            with path.open("rb") as stream:
                opened = os.fstat(stream.fileno())
                if _file_version(opened) != _file_version(metadata):
                    return {"status": "file_changed_during_read"}
                raw = stream.read(self.limits["max_file_bytes"] + 1)
                finished = os.fstat(stream.fileno())
            self.stats["files_read"] += 1
            self.stats["bytes_read"] += len(raw)
            final = path.stat()
            if _file_version(opened) != _file_version(finished) or _file_version(finished) != _file_version(final):
                return {"status": "file_changed_during_read"}
            if len(raw) > self.limits["max_file_bytes"]:
                return {"status": "too_large"}
        except FileNotFoundError:
            return {"status": "missing"}
        except OSError as exc:
            return {"status": "read_error", "detail": str(exc)}
        digest = hashlib.sha256(raw).hexdigest()
        cached = self.cache.get(str(path))
        if cached and cached["file_sha256"] == digest:
            self._cache_versions[str(path)] = _file_version(final)
            self.stats["cache_hits"] += 1
            return cached
        try:
            encoding = tokenize.detect_encoding(io.BytesIO(raw).readline)[0] if path.suffix == ".py" else "utf-8-sig"
            text = raw.decode(encoding)
        except (UnicodeError, LookupError, SyntaxError) as exc:
            return {"status": "decode_error", "detail": str(exc)}
        result = {
            "status": "read", "lines": text.splitlines(), "functions": [],
            "parse_status": "not_python", "parse_error": None,
            "file_sha256": digest,
        }
        if path.suffix == ".py":
            try:
                result["functions"] = _function_scopes(ast.parse(text, filename=str(path)))
                result["parse_status"] = "parsed"
            except (SyntaxError, RecursionError) as exc:
                result["parse_status"] = "parse_error"
                result["parse_error"] = str(exc)
        self._cache_versions[str(path)] = _file_version(final)
        return result

    def snippet(self, lines: list[str], line: int, scope: dict | None = None) -> dict:
        maximum = self.limits["max_snippet_lines"]
        if scope and scope["end_line"] - scope["start_line"] + 1 <= maximum:
            start, end = scope["start_line"], scope["end_line"]
            kind = "function"
        else:
            radius = min(self.limits["context_lines"], (maximum - 1) // 2)
            start, end = max(1, line - radius), min(len(lines), line + radius)
            kind = "context"
        content = "\n".join(lines[start - 1:end])
        char_truncated = len(content) > self.limits["max_snippet_chars"]
        snippet = {
            "kind": kind, "start_line": start, "end_line": end,
            "content": content[:self.limits["max_snippet_chars"]],
            "char_truncated": char_truncated,
            "function_truncated": bool(scope and (start > scope["start_line"] or end < scope["end_line"])),
        }
        snippet["visible_end_line"] = visible_end_line(snippet)
        return snippet

    def frame(self, frame: dict, evidence_id: str, index: int) -> dict:
        result = {
            "evidence_id": evidence_id, "frame_index": index, "logged_frame": dict(frame),
            "current_line_no": None, "current_line_at_logged_line": None,
            "logged_line_matches": None, "function": None, "function_verified": None,
            "candidate_lines": [], "candidate_count": 0, "function_candidates": [],
            "snippet": None, "snippet_basis": None,
        }
        resolution = self.resolve(frame.get("file_path", ""))
        result.update(resolution)
        if resolution["status"] != "resolved":
            return result
        line_no = frame.get("line_no")
        if isinstance(line_no, bool) or not isinstance(line_no, int) or line_no < 1:
            return {**result, "status": "invalid_line"}
        data = self.read(resolution["resolved_path"])
        if data["status"] != "read":
            return {**result, **data}
        lines, scopes = data["lines"], data["functions"]
        parsed = data["parse_status"] == "parsed"
        expected = frame.get("function")
        result.update({key: data[key] for key in ("file_sha256", "parse_status", "parse_error")})
        valid = line_no <= len(lines)
        scope = _scope_at(scopes, line_no) if valid else None
        agrees = _function_agrees(expected, scope, parsed)
        snapshot = frame.get("source_line")
        if valid:
            result["current_line_at_logged_line"] = lines[line_no - 1]
        selected = None
        if snapshot is not None:
            result["logged_line_matches"] = bool(valid and lines[line_no - 1].strip() == snapshot.strip())
            if result["logged_line_matches"] and agrees is not False:
                selected, result["status"] = line_no, "matched"
            else:
                candidates = [
                    number for number, line in enumerate(lines, start=1)
                    if line.strip() == snapshot.strip()
                    and _function_agrees(expected, _scope_at(scopes, number), parsed) is not False
                ]
                result["candidate_lines"] = candidates[:20]
                result["candidate_count"] = len(candidates)
                if len(candidates) == 1:
                    selected, result["status"] = candidates[0], "relocated"
                elif candidates:
                    result["status"] = "ambiguous"
                else:
                    result["status"] = "source_changed" if valid else "line_out_of_range"
        elif valid and agrees is not False:
            selected, result["status"] = line_no, "unverified"
        else:
            result["status"] = "function_mismatch" if valid else "line_out_of_range"
        if selected is not None:
            scope = _scope_at(scopes, selected)
            result["current_line_no"] = selected
            result["function"] = scope
            result["function_verified"] = _function_agrees(expected, scope, parsed)
            result["snippet"] = self.snippet(lines, selected, scope)
            result["snippet_basis"] = "current_location"
        else:
            result["function_candidates"] = [
                scope for scope in scopes if expected in (scope["name"], scope["qualname"])
            ][:5]
            if valid:
                result["snippet"] = self.snippet(lines, line_no)
                result["snippet_basis"] = "logged_line_unverified"
        return result

    def config(self, reference: str, evidence_ids: list[str]) -> dict:
        result = {**self.resolve(reference), "evidence_ids": evidence_ids, "snippet": None}
        if result["status"] != "resolved":
            return result
        data = self.read(result["resolved_path"])
        result["status"] = data["status"]
        if data["status"] == "read":
            lines = data["lines"]
            count = min(len(lines), self.limits["max_snippet_lines"])
            content = "\n".join(lines[:count])
            result["file_sha256"] = data["file_sha256"]
            result["snippet"] = {
                "start_line": 1 if lines else 0, "end_line": count,
                "content": content[:self.limits["max_snippet_chars"]],
                "truncated": len(lines) > count or len(content) > self.limits["max_snippet_chars"],
                "char_truncated": len(content) > self.limits["max_snippet_chars"],
            }
            result["snippet"]["visible_end_line"] = visible_end_line(result["snippet"])
        elif "detail" in data:
            result["detail"] = data["detail"]
        return result


def locate_evidence(
    evidence: list[dict], project_root: str | Path, *,
    path_mappings: dict[str, str] | None = None, context_lines: int = 3,
    max_snippet_lines: int = 40, max_snippet_chars: int = 8000,
    max_file_bytes: int = 1_000_000, max_frames: int = 50, max_config_files: int = 10,
) -> dict:
    """Resolve current code/config snapshots, preserving log references and uncertainty."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("project_root must be an existing directory")
    limits = {
        "context_lines": context_lines, "max_snippet_lines": max_snippet_lines,
        "max_snippet_chars": max_snippet_chars, "max_file_bytes": max_file_bytes,
        "max_frames": max_frames, "max_config_files": max_config_files,
    }
    for name, value in limits.items():
        minimum = 0 if name == "context_lines" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    locator = _Locator(root, path_mappings or {}, limits)
    locations = []
    requested_frames = 0
    references = {}
    for item in evidence:
        for index, frame in enumerate(item.get("frames", [])):
            requested_frames += 1
            if len(locations) < max_frames:
                locations.append(locator.frame(frame, item["id"], index))
        for reference in _CONFIG_REFERENCE.findall(item.get("message") or item.get("content", "")):
            ids = references.setdefault(reference, [])
            if item["id"] not in ids:
                ids.append(item["id"])
    configs = [locator.config(path, ids) for path, ids in list(references.items())[:max_config_files]]
    return {
        "schema_version": 1, "project_root": str(root), "limits": limits,
        "locations": locations, "config_files": configs,
        "stats": {
            **locator.stats, "frames_requested": requested_frames,
            "frames_omitted": requested_frames - len(locations),
            "config_references_omitted": max(0, len(references) - len(configs)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Offline entry point combining evidence extraction and source location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--map-root", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        mappings = {}
        for mapping in args.map_root:
            old, separator, new = mapping.partition("=")
            if not separator or not old or not new:
                parser.error("--map-root requires OLD=NEW")
            mappings[old] = new
        items = extract_evidence(args.log_file.read_text(encoding="utf-8-sig"), source=str(args.log_file))
        result = locate_evidence(items, args.project, path_mappings=mappings, max_frames=args.max_frames)
        if args.output:
            observed = {
                item["resolved_path"] for item in [*result["locations"], *result["config_files"]]
                if item.get("resolved_path")
            }
            target = args.output.resolve()
            if target == args.log_file.resolve() or str(target) in observed:
                parser.error("--output must differ from the input log and located project files")
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(payload, end="")
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
