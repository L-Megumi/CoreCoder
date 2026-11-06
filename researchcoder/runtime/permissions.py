"""Bounded feedback for a denied public-check read; never a read capability.

The controller derives the exact public paths from its public verifier's command
and protected inputs. It also supplies the remaining *Job* correction allowance.
This module neither derives authority from logs nor retries or opens the denied
file. A session only consumes that allowance and returns safe audit metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import posixpath
import stat
import threading
from typing import Literal

from corecoder.permission import PermissionDenied, REASONS as _REASONS, TOOLS as _TOOLS
from researchcoder.contracts import Contract


_PRIVATE_DIRECTORIES = frozenset({
    ".git", ".hg", ".svn", ".ssh", ".aws", ".azure", ".gcloud", ".gnupg",
})
_PRIVATE_FILES = frozenset({
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials",
    "credentials.json", "secrets.json", "secrets.toml",
})


def _absolute(target):
    """Validate spelling without expanding variables, home paths or symlinks."""
    if (type(target) is not str or not 1 <= len(target) <= 4096
            or not target.startswith("/") or target.startswith("//")
            or "\\" in target or any(ord(c) < 32 or ord(c) == 127 for c in target)):
        return None
    try:
        target.encode("utf-8")
    except UnicodeEncodeError:
        return None
    return posixpath.normpath(target)


def normalized_target_sha256(target):
    """Hash a lexical absolute target; unavailable/invalid targets remain null."""
    normalized = _absolute(target)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized is not None else None


def _sensitive(target):
    parts = tuple(part.lower() for part in PurePosixPath(target).parts)
    return any(
        part in _PRIVATE_DIRECTORIES or part in _PRIVATE_FILES
        or part == ".env" or part.startswith(".env.")
        or PurePosixPath(part).suffix in {".key", ".pem", ".p12", ".pfx", ".jks"}
        for part in parts
    )


@dataclass(frozen=True, kw_only=True)
class PermissionRecoverySpec(Contract):
    """Controller-owned, versioned allowance; does not amend ExecutionPolicy.

    Zero allowance is carried to later Attempts/experts after a correction was
    consumed. A new session is not permission to replenish the Job allowance.
    Only the controller knows which protected checker is public, so deserializing
    this value alone is insufficient authorization for use by a Worker.
    """

    public_check_paths: tuple[str, ...] = ()
    remaining_corrections: int = 0
    mode: Literal["off", "bounded-public-read-v1"] = "off"
    schema_version: int = 1

    def validate(self):
        if self.remaining_corrections not in (0, 1):
            raise ValueError("Permission correction allowance must be zero or one")
        if len(self.public_check_paths) > 16 or len(set(self.public_check_paths)) != len(self.public_check_paths):
            raise ValueError("Public checker paths must be bounded and unique")
        for target in self.public_check_paths:
            if _absolute(target) != target or target == "/" or _sensitive(target):
                raise ValueError("Public checker path must be normalized and non-sensitive")

    @property
    def sha256(self):
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _file_state(target):
    """Metadata inspection only, and only after exact trusted-path matching."""
    path = Path(target)
    try:
        for component in (path, *path.parents):
            if stat.S_ISLNK(component.lstat().st_mode):
                return "symlink"
        if not stat.S_ISREG(path.lstat().st_mode):
            return "unavailable"
    except (OSError, ValueError, RuntimeError):
        return "unavailable"
    return "regular"


def classify_permission_denial(spec, tool, target, *, reason_code="outside_workspace"):
    """Classify an already denied call without executing or making it legal.

    Only a canonical absolute spelling of the exact public checker can receive
    corrective feedback. Source-root remapping, arbitrary traceback paths,
    relative aliases and imports are deliberately outside this policy version.
    """
    if type(spec) is not PermissionRecoverySpec:
        raise TypeError("Typed permission recovery specification required")
    safe_tool = tool if type(tool) is str and tool in _TOOLS else "unknown"
    original_reason = reason_code if type(reason_code) is str and reason_code in _REASONS else "guard_rejected"
    normalized = _absolute(target)
    target_hash = normalized_target_sha256(target)

    def receipt(reason, category, action="stop"):
        return PermissionDenied(
            reason_code=reason, target_category=category, action=action,
            tool=safe_tool, target_sha256=target_hash,
            correction_index=1 if action == "correct_parameters" else 0,
        ).receipt

    if normalized is not None and _sensitive(normalized):
        return receipt("sensitive_target", "sensitive")
    if safe_tool in ("write_file", "edit_file"):
        # Report the decision at the guard branch, without inspecting file
        # contents or guessing from exception text. All writes remain terminal.
        categories = {
            "current_read_required": "workspace_file",
            "source_changed_since_read": "workspace_file",
            "source_digest_unavailable": "workspace_file",
            "write_not_allowed": "protected_source",
            "outside_workspace": "outside_workspace",
            "tool_not_allowed": "tool",
            "path_unavailable": "unknown",
        }
        if original_reason in categories:
            return receipt(original_reason, categories[original_reason])
        return receipt("guard_rejected", "unknown")
    if safe_tool in ("bash", "agent"):
        return receipt("write_not_allowed", "protected_source" if target in spec.public_check_paths else "unknown")
    if safe_tool != "read_file":
        return receipt("tool_not_allowed", "tool")
    # Being in the public-check list cannot downgrade a different guard failure.
    if original_reason != "outside_workspace":
        return receipt(original_reason, "workspace_file" if original_reason == "current_read_required" else "unknown")
    if normalized is None or normalized != target:
        return receipt("unknown_target", "unknown")
    if spec.mode == "off":
        return receipt("policy_disabled", "outside_workspace")
    if target not in spec.public_check_paths:
        return receipt("unknown_target", "outside_workspace")
    state = _file_state(target)
    if state != "regular":
        return receipt("path_unavailable", "symlink" if state == "symlink" else "unknown")
    if not spec.remaining_corrections:
        return receipt("correction_exhausted", "public_check")
    return receipt("public_read_path_mismatch", "public_check", "correct_parameters")


class PermissionRecoverySession:
    """Per-Worker mutable counter; cumulative Job accounting belongs to Controller."""

    def __init__(self, spec):
        if type(spec) is not PermissionRecoverySpec:
            raise TypeError("Typed permission recovery specification required")
        self.spec = spec
        self._corrections_used = 0
        self._lock = threading.Lock()

    @property
    def corrections_used(self):
        with self._lock:
            return self._corrections_used

    @property
    def remaining_corrections(self):
        with self._lock:
            return self.spec.remaining_corrections - self._corrections_used

    def deny(self, tool, target, *, reason_code="outside_workspace"):
        from dataclasses import replace

        with self._lock:
            current = replace(self.spec, remaining_corrections=self.spec.remaining_corrections - self._corrections_used)
            result = classify_permission_denial(current, tool, target, reason_code=reason_code)
            if result["action"] == "correct_parameters":
                self._corrections_used += 1
            return result
