"""Versioned, immutable values. No model clients or runtime state belong here."""

from __future__ import annotations

import json
import math
import re
import types
from dataclasses import dataclass, field, fields
from pathlib import PurePosixPath
from typing import Literal, Union, get_args, get_origin, get_type_hints


def _typed(value, annotation, decode=False):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        for option in args:
            try:
                return _typed(value, option, decode)
            except (ValueError, TypeError):
                pass
        raise ValueError("Value does not match union contract")
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            raise ValueError("Unsupported literal value")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise TypeError("Expected a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_typed(v, args[0], decode) for v in value)
        if len(value) != len(args):
            raise ValueError("Invalid tuple length")
        return tuple(_typed(v, t, decode) for v, t in zip(value, args))
    if annotation is type(None):
        if value is not None:
            raise TypeError("Expected null")
        return value
    if isinstance(annotation, type) and issubclass(annotation, Contract):
        if decode and isinstance(value, dict):
            return annotation.from_dict(value)
        if type(value) is not annotation:
            raise TypeError("Expected contract value")
        return value
    if annotation is float:
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Expected finite number")
        return float(value)
    if type(value) is not annotation:
        raise TypeError("Wrong contract field type")
    return value


def _plain(value):
    if isinstance(value, Contract):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise TypeError("Runtime objects cannot be serialized")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


class Contract:
    def __post_init__(self):
        hints = get_type_hints(type(self))
        for f in fields(self):
            object.__setattr__(self, f.name, _typed(getattr(self, f.name), hints[f.name]))
        if self.schema_version != 1:
            raise ValueError("Unsupported schema_version")
        self.validate()

    def validate(self):
        pass

    def to_dict(self):
        return _plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or "schema_version" not in value:
            raise ValueError("Versioned JSON object required")
        hints = get_type_hints(cls)
        if set(value) - set(hints):
            raise ValueError("Unknown contract fields")
        return cls(**{key: _typed(item, hints[key], True) for key, item in value.items()})

    @classmethod
    def from_json(cls, value):
        if not isinstance(value, str) or len(value) > 10_000_000:
            raise ValueError("Expected bounded JSON text")
        return cls.from_dict(
            json.loads(
                value, object_pairs_hook=_unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON"))
            )
        )


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}", value):
        raise ValueError("Invalid identifier")


def digest(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected SHA256")


def text(value, maximum=6000):
    if not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError("Invalid bounded text")


def relative(value):
    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise ValueError("Invalid logical path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Logical path must be relative and normalized")


@dataclass(frozen=True, kw_only=True)
class ArtifactRef(Contract):
    job_id: str
    artifact_id: str
    storage_key: str
    sha256: str
    size_bytes: int
    media_type: str = "application/json"
    schema_version: int = 1

    def validate(self):
        identifier(self.job_id)
        identifier(self.artifact_id)
        relative(self.storage_key)
        if not self.storage_key.startswith(self.job_id + "/"):
            raise ValueError("Artifact key must belong to job")
        digest(self.sha256)
        text(self.media_type, 100)
        if not 0 <= self.size_bytes <= 10_000_000:
            raise ValueError("Artifact size out of bounds")


@dataclass(frozen=True, kw_only=True)
class WorkspaceRef(Contract):
    workspace_id: str
    root: str
    source_manifest_sha256: str
    target_platform: Literal["linux"] = "linux"
    git_head: str | None = None
    schema_version: int = 1

    def validate(self):
        identifier(self.workspace_id)
        if not PurePosixPath(self.root).is_absolute() or "\x00" in self.root:
            raise ValueError("Linux workspace root must be absolute")
        digest(self.source_manifest_sha256)
        if self.git_head is not None and not re.fullmatch(r"[0-9a-f]{40,64}", self.git_head):
            raise ValueError("Invalid git head")


@dataclass(frozen=True, kw_only=True)
class BudgetSpec(Contract):
    max_rounds: int = 8
    max_tool_calls: int = 16
    max_context_tokens: int = 128_000
    max_report_repairs: int = 1
    max_investigation_rounds: int = 8
    schema_version: int = 1

    def validate(self):
        for value, lo, hi in [
            (self.max_rounds, 1, 30),
            (self.max_tool_calls, 1, 100),
            (self.max_context_tokens, 1000, 1_000_000),
            (self.max_report_repairs, 0, 1),
            (self.max_investigation_rounds, 1, 30),
        ]:
            if not lo <= value <= hi:
                raise ValueError("Execution budget out of bounds")


@dataclass(frozen=True, kw_only=True)
class ExecutionPolicy(Contract):
    allowed_tool_names: tuple[str, ...] = ("read_file", "edit_file", "write_file")
    allowed_read_roots: tuple[str, ...] = ()
    writable_paths: tuple[str, ...] = ()
    allow_verification: bool = False
    profile: str = "default"
    schema_version: int = 1

    def validate(self):
        identifier(self.profile)
        if len(set(self.allowed_tool_names)) != len(self.allowed_tool_names):
            raise ValueError("Duplicate tools")
        if not set(self.allowed_tool_names) <= {"read_file", "edit_file", "write_file", "grep", "glob"}:
            raise ValueError("Phase 1 adapter does not expose shell or child agents")
        for path in self.writable_paths:
            relative(path)
        for path in self.allowed_read_roots:
            if not PurePosixPath(path).is_absolute():
                raise ValueError("Read roots must be absolute")


@dataclass(frozen=True, kw_only=True)
class VerificationSpec(Contract):
    argv: tuple[str, ...]
    timeout_seconds: int = 30
    cwd_scope: Literal["workspace"] = "workspace"
    oracle_profile: str = "explicit-command"
    schema_version: int = 1

    def validate(self):
        if not 1 <= len(self.argv) <= 40:
            raise ValueError("Expected nonempty argument vector")
        for part in self.argv:
            text(part, 4096)
        if not 1 <= self.timeout_seconds <= 300:
            raise ValueError("Verification timeout out of bounds")
        identifier(self.oracle_profile)


@dataclass(frozen=True, kw_only=True)
class UsageSummary(Contract):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    usage_complete: bool = False
    llm_calls: int = 0
    tool_calls: int = 0
    elapsed_seconds: float | None = None
    mode: Literal["live", "injected"] = "injected"
    schema_version: int = 1

    def validate(self):
        for value in (self.prompt_tokens, self.completion_tokens, self.elapsed_seconds, self.llm_calls, self.tool_calls):
            if value is not None and value < 0:
                raise ValueError("Negative usage")
        if self.usage_complete and (self.prompt_tokens is None or self.completion_tokens is None):
            raise ValueError("Complete usage requires provider token values")


@dataclass(frozen=True, kw_only=True)
class EvidenceRef(Contract):
    job_id: str
    ref_id: str
    origin_id: str
    record_type: Literal["log_evidence", "source_observation", "path_observation", "verification"]
    artifact: ArtifactRef
    record_key: str
    producer_task_id: str
    producer_attempt_id: str
    source_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    source_sha256: str | None = None
    workspace_revision: str | None = None
    relations: tuple[tuple[str, str], ...] = ()
    schema_version: int = 1

    def validate(self):
        for value in (self.job_id, self.ref_id, self.origin_id, self.producer_task_id, self.producer_attempt_id):
            identifier(value)
        if self.artifact.job_id != self.job_id:
            raise ValueError("Cross-job evidence")
        text(self.record_key, 200)
        if (self.start_line is None) != (self.end_line is None):
            raise ValueError("Both line bounds required")
        if self.start_line is not None and not 1 <= self.start_line <= self.end_line:
            raise ValueError("Invalid line range")
        for value in (self.source_sha256, self.workspace_revision):
            if value is not None:
                digest(value)
        if self.record_type == "source_observation" and self.workspace_revision is None:
            raise ValueError("Source observation requires revision")
        for kind, ref in self.relations:
            if kind not in ("cause", "context", "dependency"):
                raise ValueError("Unknown evidence relation")
            identifier(ref)


@dataclass(frozen=True, kw_only=True)
class TaskInput(Contract):
    role: Literal["raw_log", "diagnosis_run", "evidence"]
    reference: ArtifactRef | EvidenceRef
    schema_version: int = 1


@dataclass(frozen=True, kw_only=True)
class DiagnosticOptions(Contract):
    path_mappings: tuple[tuple[str, str], ...] = ()
    source_hints: tuple[str, ...] = ()
    tool_selection: Literal["all", "guided"] = "guided"
    context_mode: Literal["baseline", "evidence"] = "evidence"
    completion_mode: Literal["off", "checklist"] = "checklist"
    findings: bool = False
    schema_version: int = 1

    def validate(self):
        if len(self.source_hints) > 10 or len(self.path_mappings) > 20:
            raise ValueError("Excessive diagnostic options")
        for hint in self.source_hints:
            text(hint, 4096)


@dataclass(frozen=True, kw_only=True)
class TaskSpec(Contract):
    job_id: str
    task_id: str
    kind: Literal["diagnose", "repair", "verify"]
    objective: str
    workspace: WorkspaceRef
    inputs: tuple[TaskInput, ...] = ()
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    verification: VerificationSpec | None = None
    diagnostic_options: DiagnosticOptions = field(default_factory=DiagnosticOptions)
    schema_version: int = 1

    def validate(self):
        identifier(self.job_id)
        identifier(self.task_id)
        text(self.objective)
        roles = [i.role for i in self.inputs]
        if len(roles) != len(set(roles)):
            raise ValueError("Duplicate input roles")
        for item in self.inputs:
            if item.reference.job_id != self.job_id:
                raise ValueError("Cross-job task input")
            if item.role in ("raw_log", "diagnosis_run") and not isinstance(item.reference, ArtifactRef):
                raise ValueError("Full artifact required")
        required = {"diagnose": "raw_log", "repair": "diagnosis_run"}.get(self.kind)
        if required and required not in roles:
            raise ValueError("Missing required task input")
        if self.kind == "verify" and self.verification is None:
            raise ValueError("Verify requires command")
        if self.verification is not None and not self.policy.allow_verification:
            raise ValueError("Verification not permitted")
        if self.kind == "diagnose" and self.verification is not None:
            raise ValueError("Diagnosis is read-only")


@dataclass(frozen=True, kw_only=True)
class OutputSlice(Contract):
    text: str = ""
    truncated: bool = False
    original_chars: int = 0
    omitted_chars: int = 0
    artifact: ArtifactRef | None = None
    schema_version: int = 1

    def validate(self):
        if len(self.text) > 13000 or min(self.original_chars, self.omitted_chars) < 0:
            raise ValueError("Invalid output bounds")
        if not self.truncated and self.omitted_chars:
            raise ValueError("Omission requires truncation flag")


@dataclass(frozen=True, kw_only=True)
class VerificationResult(Contract):
    job_id: str
    task_id: str
    attempt_id: str
    verification_id: str
    status: Literal["not_run", "passed", "failed", "timeout", "launch_error", "interrupted"]
    spec: VerificationSpec | None = None
    exit_code: int | None = None
    stdout: OutputSlice = field(default_factory=OutputSlice)
    stderr: OutputSlice = field(default_factory=OutputSlice)
    elapsed_seconds: float | None = None
    timeout_seconds: int | None = None
    launch_error: str | None = None
    trigger: Literal["standalone", "completed_loop", "partial_file_changes"] = "standalone"
    checked_workspace_revision: str | None = None
    scope: str = "Only the explicit command and its checks are verified."
    schema_version: int = 1

    def validate(self):
        for value in (self.job_id, self.task_id, self.attempt_id, self.verification_id):
            identifier(value)
        if self.status != "not_run" and self.spec is None:
            raise ValueError("Executed verification requires specification")
        if self.status == "passed" and self.exit_code != 0:
            raise ValueError("Passed requires observed zero exit")
        if self.status == "failed" and (self.exit_code is None or self.exit_code == 0):
            raise ValueError("Failed requires observed nonzero exit")
        if self.status not in ("passed", "failed") and self.exit_code is not None:
            raise ValueError("No observed exit code for this state")
        if self.status != "not_run" and self.checked_workspace_revision is None:
            raise ValueError("Executed verification requires workspace version")
        if self.checked_workspace_revision is not None:
            digest(self.checked_workspace_revision)
        if self.elapsed_seconds is not None and self.elapsed_seconds < 0:
            raise ValueError("Negative verification time")
        if self.status == "timeout" and (self.timeout_seconds is None or self.timeout_seconds < 1):
            raise ValueError("Timeout requires limit")
        if self.status == "launch_error" and not self.launch_error:
            raise ValueError("Launch error requires category")


FailureKind = Literal[
    "input_invalid",
    "reference_stale",
    "budget_exhausted",
    "model_error",
    "tool_error",
    "verification_failed",
    "timeout",
    "launch_error",
    "interrupted",
    "unknown",
    "local_budget_exhausted",
    "response_length_exceeded",
    "invalid_json",
    "invalid_report",
]


@dataclass(frozen=True, kw_only=True)
class TaskResult(Contract):
    job_id: str
    task_id: str
    attempt_id: str
    kind: Literal["diagnose", "repair", "verify"]
    status: Literal["succeeded", "failed", "incomplete", "cancelled"]
    failure_kind: FailureKind | None = None
    summary: str = ""
    outputs: tuple[ArtifactRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    verification: VerificationResult | None = None
    file_changes_ref: ArtifactRef | None = None
    workflow_completed: bool = False
    usage: UsageSummary = field(default_factory=UsageSummary)
    limitations: tuple[str, ...] = ()
    schema_version: int = 1

    def validate(self):
        for value in (self.job_id, self.task_id, self.attempt_id):
            identifier(value)
        if len(self.summary) > 20000:
            raise ValueError("Summary too long")
        if self.status == "succeeded" and (self.failure_kind is not None or not self.workflow_completed):
            raise ValueError("Success requires completed workflow without failure")
        if self.status != "succeeded" and self.failure_kind is None:
            raise ValueError("Non-success requires failure category")
        if self.status == "cancelled" and self.failure_kind != "interrupted":
            raise ValueError("Cancellation requires interrupted category")
        for item in (*self.outputs, *self.evidence_refs, *((self.file_changes_ref,) if self.file_changes_ref else ())):
            if item.job_id != self.job_id:
                raise ValueError("Cross-job output")
        if self.verification:
            if (self.verification.job_id, self.verification.task_id, self.verification.attempt_id) != (
                self.job_id,
                self.task_id,
                self.attempt_id,
            ):
                raise ValueError("Verification identity mismatch")
            if self.status == "succeeded" and self.verification.status not in ("passed", "not_run"):
                raise ValueError("Failed verification cannot imply task success")
        if (
            self.kind == "verify"
            and self.status == "succeeded"
            and (self.verification is None or self.verification.status != "passed")
        ):
            raise ValueError("Standalone verification success requires a passed command")
