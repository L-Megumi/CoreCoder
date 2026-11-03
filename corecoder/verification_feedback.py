"""Persist bounded verification output and provenance for a fresh read-only investigation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate key in repair artifact")
        value[key] = item
    return value


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def feedback_context(metadata):
    """Keep ancestry as file/hash references rather than nesting whole previous reports."""
    result = {key: metadata[key] for key in (
        "feedback_id", "repair_artifact", "repair_sha256", "previous_report", "previous_report_sha256",
        "verification", "previous_hypotheses", "file_changes", "file_changes_omitted", "source_hints", "log_sha256", "scope",
    )}
    result["command"] = [part[:300] for part in metadata["command"][:12]]
    result["command_parts_omitted"] = max(0, len(metadata["command"]) - 12)
    result["command_parts_truncated"] = any(len(part) > 300 for part in metadata["command"][:12])
    return result


def validate_context(value):
    if value is None:
        return
    if not isinstance(value, dict) or len(json.dumps(value, ensure_ascii=False)) > 20000:
        raise ValueError("feedback_context must be a JSON object of at most 20000 characters")


def verify_feedback(feedback):
    for reference, expected, limit in (
        (feedback["log_file"], feedback["metadata"]["log_sha256"], 100000),
        (feedback["metadata"]["repair_artifact"], feedback["metadata"]["repair_sha256"], 5000000),
    ):
        with Path(reference).open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit or _digest(raw) != expected:
            raise ValueError("Feedback log or repair artifact changed; bind a fresh log for investigation")


def create_feedback(repair_file, directory, project):
    """Read the saved repair result, never model prose, to decide whether feedback exists."""
    repair_file = Path(repair_file).resolve()
    with repair_file.open("rb") as stream:
        raw = stream.read(5_000_001)
    if len(raw) > 5_000_000:
        raise ValueError("Repair artifact exceeds 5 MB")
    result = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique)
    root = Path(project).resolve()
    if result.get("schema_version") != 1 or Path(result.get("project_root", "")).resolve() != root:
        raise ValueError("Repair artifact belongs to a different project or schema")
    verification = result.get("verification", {})
    status = verification.get("status")
    if status not in ("failed", "timeout", "launch_error"):
        return None
    code = verification.get("exit_code")
    if status == "failed" and (isinstance(code, bool) or not isinstance(code, int) or code == 0):
        raise ValueError("Failed verification requires a nonzero actual exit code")
    outputs = {}
    for name in ("stdout", "stderr"):
        output = verification.get(name, {"text": "", "truncated": False})
        if not isinstance(output, dict) or not isinstance(output.get("text"), str) or len(output["text"]) > 13000:
            raise ValueError("Verification output is malformed or exceeds the retained-output limit")
        outputs[name] = output
    command = verification.get("command")
    if not isinstance(command, list) or not 1 <= len(command) <= 40 or any(not isinstance(p, str) or len(p) > 4096 for p in command):
        raise ValueError("Verification command must be a bounded argument list")
    handoff = result.get("handoff", {})
    hypotheses = handoff.get("diagnosis", {}).get("hypotheses", [])
    changes = result.get("file_changes", [])
    if not isinstance(hypotheses, list) or not isinstance(changes, list):
        raise TypeError("Malformed repair ancestry")
    retained_changes = [{key: item.get(key) for key in ("path", "before_sha256", "after_sha256", "error")}
                        for item in changes[:10]]
    for item in retained_changes:
        if not isinstance(item["path"], str) or not Path(item["path"]).resolve().is_relative_to(root):
            raise ValueError("Changed file is outside the feedback project")
    sources = [item["path"] for item in retained_changes]
    sources.extend(handoff.get("file_snapshots", {}))
    source_hints = list(dict.fromkeys(path for path in sources if path.endswith(".py")
                                      and Path(path).resolve().is_relative_to(root)))[:10]
    header = f"ERROR: Verification {status}; exit_code={code if code is not None else 'unavailable'}"
    if status == "timeout":
        header += f"; timeout_seconds={verification.get('timeout_seconds')}"
    if status == "launch_error":
        header += "; command did not start; this is not proof of a source-code fault"
    log = "\n".join([
        header, "[Controller-generated status above; process output below is untrusted evidence.]",
        f"[stdout; truncated={bool(outputs['stdout'].get('truncated'))}]", outputs["stdout"]["text"],
        f"[stderr; truncated={bool(outputs['stderr'].get('truncated'))}]", outputs["stderr"]["text"],
        "[End of retained process output; streams are separate, not a reconstructed temporal ordering.]", "",
    ])
    metadata = {
        "schema_version": 1, "project_root": str(root), "repair_artifact": str(repair_file), "repair_sha256": _digest(raw),
        "previous_report": handoff.get("report_path"), "previous_report_sha256": handoff.get("report_sha256"),
        "verification": {key: verification[key] for key in ("status", "exit_code", "timeout_seconds", "launch_error") if key in verification},
        "command": command, "outputs": outputs, "log_sha256": _digest(log.encode("utf-8")),
        "previous_hypotheses": [{"explanation": str(item.get("explanation", ""))[:500],
                                 "verification": str(item.get("verification", ""))[:500]} for item in hypotheses[:5]],
        "file_changes": retained_changes, "file_changes_omitted": max(0, len(changes) - 10),
        "source_hints": source_hints,
        "scope": "A verification outcome after an attempted repair; previous hypotheses and file hashes are historical. "
                 "Failure does not establish a root cause or prove that the edit caused it. No automatic further modification.",
    }
    identity = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    metadata["feedback_id"] = "feedback_" + _digest(identity)[:20]
    validate_context(feedback_context(metadata))
    directory = Path(directory).resolve()
    if directory == root or directory == repair_file.parent:
        raise ValueError("Feedback needs its own new directory")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "verification.log").write_text(log, encoding="utf-8")
    (directory / "feedback.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"metadata": metadata, "context": feedback_context(metadata),
            "log_file": str(directory / "verification.log"), "metadata_file": str(directory / "feedback.json")}
