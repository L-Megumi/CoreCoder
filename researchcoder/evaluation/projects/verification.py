"""Trusted project checks over the existing Harness and public command verifier.

Receipt specifications are evaluator configuration, never Agent input. A valid
receipt proves that the protected checker produced the required observations;
it is not a substitute for candidate manifests or a complete repair workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Literal

from corecoder.workflow import workspace_revision
from researchcoder.contracts import Contract, ExecutionPolicy, TaskSpec, TaskResult, VerificationResult, VerificationSpec, WorkspaceRef, identifier
from researchcoder.runtime.harness import ExecutionConfig, ResearchHarness
from researchcoder.runtime.protocol import ModelSpec, ProtectedFile, RuntimeLimits


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(value):
    _require(isinstance(value, str) and bool(value) and "\\" not in value and "\x00" not in value,
             "Expected normalized relative source path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and all(p not in (".", "..") for p in path.parts)
             and path.as_posix() == value, "Source path escapes or is not normalized")


def _names(values, label):
    _require(len(values) == len(set(values)), "Duplicate " + label)
    for value in values:
        _require(isinstance(value, str) and 0 < len(value) <= 200 and "\x00" not in value, "Invalid " + label)


@dataclass(frozen=True, kw_only=True)
class ReceiptSpec(Contract):
    kind: Literal["checks_json", "public_json"]
    origin_field: str
    origin_relative_path: str
    required_fields: tuple[str, ...] = ()
    field_types: tuple[tuple[str, Literal["bool", "int", "float", "str", "list", "dict"]], ...] = ()
    nonempty_fields: tuple[str, ...] = ()
    exact_fields: tuple[tuple[str, str | int | bool | float | None], ...] = ()
    expected_checks: tuple[str, ...] = ()
    check_name_field: str = "check"
    check_exact_fields: tuple[tuple[str, str | int | bool | float | None], ...] = ()
    digest_field: str | None = None
    digest_relative_path: str | None = None
    schema_version: int = 1

    def validate(self):
        _relative(self.origin_relative_path)
        _names((self.origin_field, self.check_name_field), "receipt field")
        for names, label in ((self.required_fields, "required field"), (self.nonempty_fields, "nonempty field"),
                             (self.expected_checks, "check name"), (tuple(n for n, _ in self.field_types), "type field"),
                             (tuple(n for n, _ in self.exact_fields), "exact field"),
                             (tuple(n for n, _ in self.check_exact_fields), "exact check field")):
            _names(names, label)
        _require(bool(self.digest_field) == bool(self.digest_relative_path), "Digest field and source path must be paired")
        if self.digest_relative_path is not None:
            _relative(self.digest_relative_path)
            _names((self.digest_field,), "digest field")
        if self.kind == "checks_json":
            _require(bool(self.expected_checks), "Independent checks require a nonempty expected identity set")
        else:
            _require(not self.expected_checks and not self.check_exact_fields, "Public receipts do not specify hidden checks")
            _require(bool(self.required_fields), "Public receipts require explicit output fields")


@dataclass(frozen=True, kw_only=True)
class ProjectCheck(Contract):
    """Only trusted controller configuration may supply a command or receipt."""

    check_id: str
    argv_template: tuple[str, ...]
    protected_files: tuple[ProtectedFile, ...]
    receipt: ReceiptSpec
    timeout_seconds: int = 60
    schema_version: int = 1

    def validate(self):
        identifier(self.check_id)
        _require(1 <= len(self.argv_template) <= 40, "Invalid check argument count")
        _require(self.argv_template.count("{project}") == 1, "Exactly one standalone project placeholder is required")
        _require(all(isinstance(a, str) and a and len(a) <= 4096 and "\x00" not in a for a in self.argv_template),
                 "Invalid trusted check argument")
        _require(not any("{project}" in a and a != "{project}" for a in self.argv_template),
                 "Project placeholder must be a complete argument")
        _require(Path(self.argv_template[0]).is_absolute(), "Trusted interpreter must be absolute")
        _require(type(self.timeout_seconds) is int and 1 <= self.timeout_seconds <= 300, "Invalid check timeout")
        _require(bool(self.protected_files), "At least the checker must be protected")
        paths = tuple(p.path for p in self.protected_files)
        _require(len(paths) == len(set(paths)), "Duplicate protected checker path")
        _require(any(p in self.argv_template[1:] for p in paths), "Command must reference a protected checker")

    def command(self, project):
        root = _project(project)
        return VerificationSpec(argv=tuple(str(root) if arg == "{project}" else arg for arg in self.argv_template),
                                timeout_seconds=self.timeout_seconds, oracle_profile=self.check_id)

    def check_protected(self):
        for protected in self.protected_files:
            path = Path(protected.path)
            _require(not path.is_symlink() and path.is_file(), "Protected project checker is unavailable")
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == protected.sha256, "Protected project checker changed")


def _project(project):
    try:
        root = Path(project).resolve(strict=True)
        _require(root.is_dir(), "Candidate project must be a directory")
        return root
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError("Candidate project is unavailable") from exc


def _source(root, relative):
    _relative(relative)
    path = root / relative
    _require(not any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)),
             "Candidate source symlink is not permitted")
    try:
        path = path.resolve(strict=True)
        _require(path.is_file() and path.is_relative_to(root), "Candidate source escapes project")
        return path
    except (OSError, RuntimeError) as exc:
        raise ValueError("Candidate source is unavailable") from exc


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "Duplicate JSON receipt key")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("Non-JSON number in receipt")


def _exact(report, fields):
    for name, expected in fields:
        _require(name in report and type(report[name]) is type(expected) and report[name] == expected,
                 "Receipt field mismatch: " + name)


def validate_receipt(verification: VerificationResult, spec: ReceiptSpec, project):
    """Reject missing, truncated, foreign or contradictory checker output.

Public checks retain the checker's exit semantics; this function does not add
private assertions to public feedback. Independent checks also require every
expected unique check and a consistent strict-boolean aggregate.
"""
    if not isinstance(verification, VerificationResult) or not isinstance(spec, ReceiptSpec):
        raise TypeError("VerificationResult and trusted ReceiptSpec required")
    _require(not verification.stdout.truncated, "Checker output was truncated")
    _require(verification.status in ("passed", "failed") and type(verification.exit_code) is int
             and verification.exit_code in (0, 1), "Checker did not finish with a business exit code")
    passed = verification.exit_code == 0
    _require(verification.status == ("passed" if passed else "failed"), "Checker status disagrees with exit code")
    root = _project(project)
    _require(verification.checked_workspace_revision == workspace_revision(root), "Verification source version is stale")
    output = verification.stdout.text
    _require(bool(output.strip()), "Empty checker output")
    try:
        report = json.loads(output, object_pairs_hook=_unique, parse_constant=_invalid_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid checker JSON") from exc
    _require(isinstance(report, dict) and bool(report), "Checker output must be a nonempty object")
    _require(all(name in report for name in spec.required_fields), "Required checker output is missing")
    types = {"bool": bool, "int": int, "float": float, "str": str, "list": list, "dict": dict}
    for name, kind in spec.field_types:
        _require(type(report.get(name)) is types[kind], "Checker field has wrong type: " + name)
    for name in spec.nonempty_fields:
        value = report.get(name)
        _require(type(value) in (str, list, dict) and bool(value), "Empty checker field: " + name)
    _exact(report, spec.exact_fields)
    origin = _source(root, spec.origin_relative_path)
    value = report.get(spec.origin_field)
    _require(isinstance(value, str) and bool(value) and Path(value).is_absolute(), "Missing absolute candidate origin")
    try:
        _require(Path(value).resolve(strict=True) == origin, "Checker candidate origin mismatch")
    except (OSError, RuntimeError) as exc:
        raise ValueError("Checker candidate origin is unavailable") from exc
    source = _source(root, spec.digest_relative_path) if spec.digest_relative_path else origin
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if spec.digest_field is not None:
        _require(report.get(spec.digest_field) == source_sha, "Reported candidate source digest mismatch")
    checks = []
    if spec.kind == "checks_json":
        _require(type(report.get("passed")) is bool, "Checker aggregate must be boolean")
        rows = report.get("checks")
        _require(isinstance(rows, list) and len(rows) == len(spec.expected_checks), "Checker identity count mismatch")
        for row in rows:
            _require(isinstance(row, dict) and isinstance(row.get(spec.check_name_field), str), "Invalid checker identity")
            _require(type(row.get("passed")) is bool, "Each check result must be boolean")
            _exact(row, spec.check_exact_fields)
            checks.append({"name": row[spec.check_name_field], "passed": row["passed"]})
        names = [row["name"] for row in checks]
        _require(len(names) == len(set(names)) and set(names) == set(spec.expected_checks), "Checker identities mismatch")
        aggregate = all(row["passed"] for row in checks)
        _require(report["passed"] is aggregate and aggregate is passed, "Checker aggregate or exit status contradicts checks")
    return dict(schema_version=1, kind="project-check-receipt", receipt_kind=spec.kind, passed=passed,
                check_count=len(checks), checks=checks, exit_code=verification.exit_code, status=verification.status,
                candidate_origin=str(origin), source_path=str(source), source_sha256=source_sha,
                reported_source_sha256_verified=spec.digest_field is not None,
                checked_workspace_revision=verification.checked_workspace_revision,
                stdout_sha256=hashlib.sha256(output.encode("utf-8")).hexdigest(),
                receipt_spec_sha256=hashlib.sha256(spec.to_json().encode("utf-8")).hexdigest())


@dataclass(frozen=True, kw_only=True)
class CheckExecution:
    result: TaskResult
    record: dict
    receipt: dict | None
    receipt_error: str | None

    def to_dict(self):
        return dict(result=self.result.to_dict(), record=self.record, receipt=self.receipt, receipt_error=self.receipt_error)


def run_check(check, project, io, job_id, task_id, harness=None, *, cancel_event=None):
    """Execute one trusted check in a real Worker; never construct another verifier.

The caller owns full candidate-manifest checks, job-wide accounting and final
scoring. Receipts preserve a failing command as a failing command; invalid
receipt metadata does not rewrite the original TaskResult.
"""
    if not isinstance(check, ProjectCheck):
        raise TypeError("Trusted ProjectCheck required")
    root = _project(project)
    command = check.command(root)
    policy = ExecutionPolicy(allowed_tool_names=(), allowed_read_roots=(str(root),), writable_paths=(), allow_verification=True)
    task = TaskSpec(job_id=job_id, task_id=task_id, kind="verify", objective="Trusted project check " + check.check_id,
                    workspace=WorkspaceRef(workspace_id="candidate", root=str(root), source_manifest_sha256=workspace_revision(root)),
                    policy=policy, verification=command)
    config = ExecutionConfig(artifact_root=str(io.root), artifact_namespace=io.namespace,
                             allowed_workspaces=(str(root),), max_policy=policy, permitted_verifications=(command,),
                             protected_files=check.protected_files, model=ModelSpec(), cancel_event=cancel_event,
                             limits=RuntimeLimits(max_model_calls=1, max_tool_calls=1, max_rounds=1,
                                                  max_runtime_seconds=check.timeout_seconds + 5))
    harness = harness or ResearchHarness()
    result = harness.execute(task, config)
    record = harness.records[result.attempt_id]
    receipt, error = None, None
    try:
        check.check_protected()
        if result.verification is None:
            raise ValueError("Checker did not produce a verification record")
        receipt = validate_receipt(result.verification, check.receipt, root)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        error = str(exc)
    return CheckExecution(result=result, record=record, receipt=receipt, receipt_error=error)
