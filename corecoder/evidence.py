"""Extract traceable log evidence without invoking an LLM or executing log text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

NUMERIC_ANOMALY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
EXCEPTION_LINE_PATTERN = re.compile(
    r"^(?P<exception_type>[\w.<>]*?(?:Error|Exception)|KeyboardInterrupt|SystemExit|"
    r"GeneratorExit|StopIteration|StopAsyncIteration|ExceptionGroup|BaseExceptionGroup)"
    r"(?::(?P<message>.*))?$"
)
_CUSTOM_EXCEPTION_PATTERN = re.compile(
    r"^(?P<exception_type>[A-Za-z_]\w*(?:\.(?:[A-Za-z_]\w*|<locals>))*)(?::(?P<message>.*))?$"
)
_FRAME_PATTERN = re.compile(
    r'^\s*File "(?P<file_path>.+)", line (?P<line_no>\d+)(?:, in (?P<function>.*))?$'
)
_LOG_BOUNDARY_PATTERN = re.compile(
    r"^(?:(?:\d{4}-\d{2}-\d{2}[T ][\d:.,+Z-]+)\s+)?"
    r"(?:\[[^\]\n]+\]\s*)?"
    r"(?:DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL|TRACE)\b(?:\s*[:\]\-]|$)",
    re.IGNORECASE,
)
_METRIC_BOUNDARY_PATTERN = re.compile(
    r"^(?:epoch|step|iter(?:ation)?|batch|(?:train_|val_)?loss|accuracy|acc|lr|learning_rate)"
    r"\b\s*(?:[:=]|\d)",
    re.IGNORECASE,
)
_WARNING_PATTERN = re.compile(r"^(?:[\w.]*Warning)(?::|$)")
_CHAIN_MARKERS = {
    "The above exception was the direct cause of the following exception:": "cause_id",
    "During handling of the above exception, another exception occurred:": "context_id",
}


def is_exception_line(line: str) -> bool:
    """Recognize standard exception names, including qualified Error/Exception names."""
    return EXCEPTION_LINE_PATTERN.fullmatch(line.strip()) is not None


def _is_traceback(line: str) -> bool:
    return line.startswith("Traceback (most recent call last):")


def _is_log_boundary(line: str) -> bool:
    return bool(
        _LOG_BOUNDARY_PATTERN.match(line)
        or _METRIC_BOUNDARY_PATTERN.match(line)
        or _WARNING_PATTERN.match(line)
    )


def _record(lines: list[str], start: int, end: int, kind: str, source: str | None) -> dict:
    raw = "\n".join(lines[start:end])
    identity = json.dumps([source, start + 1, end, kind, raw], ensure_ascii=False)
    return {
        "id": "ev_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "source": source,
        "line_no": start + 1,
        "end_line_no": end,
        "type": kind,
        "content": raw,
        "raw_content": raw,
    }


def _exception_fields(match: re.Match, frames: list[dict], truncated: bool) -> dict:
    return {
        "exception_type": match.group("exception_type"),
        "message": (match.group("message") or "").lstrip(" "),
        "frames": frames,
        "truncated": truncated,
        "cause_id": None,
        "context_id": None,
    }


def _traceback(lines: list[str], start: int, source: str | None) -> tuple[dict, int]:
    """Consume a flat Python traceback and multiline message up to a visible boundary."""
    frames = []
    exception = None
    exception_at = None
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        raw = lines[end]
        line = raw.strip()
        if _is_traceback(line) or line in _CHAIN_MARKERS:
            break
        if exception is not None:
            if not line:
                following = end
                while following < len(lines) and not lines[following].strip():
                    following += 1
                if following == len(lines):
                    break
                next_line = lines[following].strip()
                if _is_traceback(next_line) or next_line in _CHAIN_MARKERS or _is_log_boundary(next_line):
                    break
                # Keep blank lines inside an unstructured multiline message.
                end = following
                continue
            if _is_log_boundary(line):
                break
        known_exception = None
        if len(raw) - len(raw.lstrip()) <= indent:
            known_exception = EXCEPTION_LINE_PATTERN.fullmatch(line)
        if exception is None and known_exception is None and _is_log_boundary(line):
            break

        if exception is None:
            frame = _FRAME_PATTERN.fullmatch(raw)
            if frame:
                frames.append({
                    "file_path": frame.group("file_path"),
                    "line_no": int(frame.group("line_no")),
                    "function": frame.group("function"),
                    "log_line_no": end + 1,
                    "source_line": None,
                })
                if end + 1 < len(lines):
                    following = lines[end + 1]
                    frame_indent = len(raw) - len(raw.lstrip())
                    following_indent = len(following) - len(following.lstrip())
                    if following_indent > frame_indent and following.strip():
                        frames[-1]["source_line"] = following.strip()
            elif line and len(raw) - len(raw.lstrip()) <= indent:
                # Custom exception final lines have traceback indentation,
                # unlike indented source-code lines.
                exception = known_exception or (_CUSTOM_EXCEPTION_PATTERN.fullmatch(line) if frames else None)
                if exception:
                    exception_at = end
        end += 1

    result = _record(lines, start, end, "ERROR", source)
    result.update({
        "exception_type": None,
        "message": None,
        "frames": frames,
        "truncated": exception is None,
        "cause_id": None,
        "context_id": None,
    })
    if exception is not None:
        result.update(_exception_fields(exception, frames, False))
        if exception_at + 1 < end:
            result["message"] += "\n" + "\n".join(lines[exception_at + 1:end])
    return result, end


def extract_evidence(text: str, *, source: str | None = None) -> list[dict]:
    """Extract errors, warnings, numeric anomalies and linked Python tracebacks.

    Existing line_no, type and content fields remain available.
    cause_id represents explicit raise-from chaining; context_id represents
    implicit exception handling, not proven causality. A source and its
    unchanged line spans produce deterministic evidence IDs.
    """
    lines = text.splitlines()
    source = str(source) if source is not None else None
    evidence = []
    previous = None
    pending_relation = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line in _CHAIN_MARKERS:
            pending_relation = _CHAIN_MARKERS[line] if previous is not None else None
            index += 1
            continue
        if _is_traceback(line):
            item, index = _traceback(lines, index, source)
            if pending_relation is not None and previous is not None:
                item[pending_relation] = previous["id"]
            item["chain_id"] = previous["chain_id"] if item["cause_id"] or item["context_id"] else item["id"]
            evidence.append(item)
            previous = item
            pending_relation = None
            continue

        exception = EXCEPTION_LINE_PATTERN.fullmatch(line)
        kinds = []
        lower = line.lower()
        if exception or "error" in lower or "exception" in lower:
            kinds.append("ERROR")
        if "warning" in lower:
            kinds.append("WARNING")
        if NUMERIC_ANOMALY_PATTERN.search(line):
            kinds.append("NUMERIC_ANOMALY")
        last_exception = None
        for kind in kinds:
            item = _record(lines, index, index + 1, kind, source)
            item["content"] = line  # Keep the original standalone-line API.
            if kind == "ERROR" and exception is not None:
                item.update(_exception_fields(exception, [], False))
                if pending_relation is not None and previous is not None:
                    item[pending_relation] = previous["id"]
                item["chain_id"] = previous["chain_id"] if item["cause_id"] or item["context_id"] else item["id"]
                last_exception = item
            evidence.append(item)
        previous = last_exception
        pending_relation = None
        index += 1
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Offline JSON entry point: python -m corecoder.evidence LOG [--output JSON]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="UTF-8 experiment log")
    parser.add_argument("--output", type=Path, help="Write JSON to this file instead of stdout")
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() == args.log_file.resolve():
            parser.error("--output must differ from the input log file")
        text = args.log_file.read_text(encoding="utf-8-sig")
        result = extract_evidence(text, source=str(args.log_file))
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(payload, end="")
    except (OSError, UnicodeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
