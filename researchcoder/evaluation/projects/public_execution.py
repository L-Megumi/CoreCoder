"""Opt-in public candidate exception provenance, separate from Oracle scoring.

The original public checker and its JSON validator remain unchanged. A failed
Python invocation may additionally supply a tightly bound exception receipt for
feedback. Missing JSON by itself never authorizes another attempt. This is an
application-level execution record, not protection from hostile Python code.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal
import uuid

from corecoder.workflow import workspace_revision
from researchcoder.contracts import Contract, VerificationResult, VerificationSpec, identifier
from researchcoder.runtime.protocol import ProtectedFile
from .verification import ProjectCheck
from .workspace import capture_manifest


ALLOWED_EXCEPTIONS = frozenset({"TypeError", "ValueError", "AssertionError", "RuntimeError",
    "AttributeError", "IndexError", "KeyError", "ZeroDivisionError", "OverflowError"})
MAX_RECEIPT_BYTES = 2_000_000


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "Duplicate public execution field")
        value[key] = item
    return value


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite public execution number")))


def _safe_file(path, limit=MAX_RECEIPT_BYTES):
    path = Path(path)
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "Public execution symlink refused")
    _require(path.is_file() and path.stat().st_size <= limit, "Public execution file missing or too large")
    raw = path.read_bytes()
    _require(len(raw) <= limit, "Public execution file exceeds bound")
    return raw


@dataclass(frozen=True, kw_only=True)
class PublicExecutionPolicy(Contract):
    mode: Literal["candidate-exception-v1"] = "candidate-exception-v1"
    schema_version: int = 1


@dataclass(frozen=True, kw_only=True)
class PreparedPublicExecution(Contract):
    job_id: str
    task_id: str
    invocation_id: str
    candidate_root: str
    command: VerificationSpec
    protected_files: tuple[ProtectedFile, ...]
    request_path: str
    receipt_path: str
    check_sha256: str
    schema_version: int = 1

    def validate(self):
        from researchcoder.contracts import digest
        for item in (self.job_id, self.task_id, self.invocation_id):
            identifier(item)
        digest(self.check_sha256)
        _require(all(Path(path).is_absolute() and len(path) <= 4096 and "\x00" not in path
                     for path in (self.candidate_root, self.request_path, self.receipt_path)),
                 "Public execution paths must be absolute")
        _require(Path(self.receipt_path).parent == Path(self.request_path).parent
                 and Path(self.receipt_path).name == "execution-receipt.json"
                 and Path(self.request_path).name == "execution-request.json", "Invalid public receipt location")
        directory = Path(self.request_path).parent
        _require(not directory.is_relative_to(Path(self.candidate_root))
                 and not Path(self.candidate_root).is_relative_to(directory), "Public execution directory overlaps candidate")
        argv = self.command.argv
        _require(len(argv) == 6 and Path(argv[0]).is_absolute() and argv[1:3] == ("-I", "-B")
                 and argv[3] == str(directory / "public_probe.py") and argv[4] == self.request_path,
                 "Invalid prepared public command")
        paths = tuple(item.path for item in self.protected_files)
        _require(len(paths) == len(set(paths)) and all(path in paths for path in (argv[3], argv[4], argv[5])),
                 "Public command inputs must be uniquely protected")

    def check_protected(self):
        for protected in self.protected_files:
            _require(_sha(_safe_file(protected.path)) == protected.sha256,
                     "Public execution configuration changed")

    def read_record(self):
        """Read the exclusive raw record for an Artifact; this is not validation."""
        self.check_protected()
        value = _json(_safe_file(self.receipt_path))
        _require(type(value) is dict, "Invalid public execution receipt")
        return value

    def read_files(self):
        """Bounded raw local records for private Artifacts, including invalid JSON.

        Missing receipts remain absent; the returned bytes are not an admission
        decision. A changed protected file may be retained for forensics without
        becoming trusted again. Symlinks/oversize files are still refused.
        """
        result = {}
        for name, value in (("probe", self.command.argv[3]), ("request", self.request_path),
                            ("receipt", self.receipt_path)):
            path = Path(value)
            if path.is_symlink() or path.exists():
                result[name] = _safe_file(path)
        return result


def prepare_public_execution(check, project, directory, *, job_id, task_id, policy=PublicExecutionPolicy()):
    """Prepare one exclusive public invocation; execution still uses the Verifier.

    Candidate files may legitimately change after preparation. The probe binds
    their complete identity immediately before executing the unchanged checker.
    """
    if type(check) is not ProjectCheck or type(policy) is not PublicExecutionPolicy:
        raise TypeError("Trusted public check and execution policy required")
    # `check` must be the controller-designated public check. ReceiptSpec.kind
    # describes output shape, not disclosure: some public fixtures use named
    # checks too. The runner never prepares an independent F2P/P2P invocation.
    identifier(job_id)
    identifier(task_id)
    root = Path(project).resolve(strict=True)
    capture_manifest(root)
    check.check_protected()
    original = check.command(root)
    argv = original.argv
    # Only the explicitly supported isolated Python script profile is wrapped.
    # Shells, -c/-m, interpreter environment changes and unknown flags fail closed.
    _require(len(argv) >= 5 and argv[1:3] == ("-I", "-B"), "Public probe requires Python -I -B script profile")
    checker = Path(argv[3])
    _require(checker.is_absolute() and checker.suffix == ".py"
             and any(p.path == str(checker) for p in check.protected_files), "Public script must be protected")
    _require(not checker.resolve().is_relative_to(root), "Public checker cannot be in candidate")
    directory = Path(directory).absolute()
    _require(not any(p.is_symlink() for p in (directory, *directory.parents)), "Public execution directory symlink refused")
    _require(not directory.resolve().is_relative_to(root) and not root.is_relative_to(directory.resolve()),
             "Public execution directory must be separate from candidate")
    directory.mkdir(parents=True, exist_ok=False)
    probe_path = directory / "public_probe.py"
    probe = _safe_file(Path(__file__).with_name("_public_probe.py"))
    with probe_path.open("xb") as stream:
        stream.write(probe)
    invocation = "public-" + uuid.uuid4().hex
    request = dict(schema_version=1, mode=policy.mode, invocation_id=invocation,
                   job_id=job_id, task_id=task_id, candidate_root=str(root), checker_path=str(checker),
                   checker_arguments=list(argv[4:]), original_argv=list(argv),
                   protected_files=[p.to_dict() for p in check.protected_files])
    data = json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    request_path = directory / "execution-request.json"
    with request_path.open("xb") as stream:
        stream.write(data)
    protected = (*check.protected_files,
                 ProtectedFile(path=str(probe_path), sha256=_sha(probe)),
                 ProtectedFile(path=str(request_path), sha256=_sha(data)))
    return PreparedPublicExecution(job_id=job_id, task_id=task_id, invocation_id=invocation,
        candidate_root=str(root), command=VerificationSpec(argv=(argv[0], "-I", "-B", str(probe_path), str(request_path), str(checker)),
            timeout_seconds=original.timeout_seconds, oracle_profile=original.oracle_profile),
        protected_files=protected, request_path=str(request_path), receipt_path=str(directory / "execution-receipt.json"),
        check_sha256=_sha(check.to_json().encode("utf-8")))


def validate_public_exception(verification, prepared, check, project):
    """Return a source-bound public failure receipt, or refuse ambiguous output.

    No successful task/Oracle receipt is created here. The caller must preserve
    the failed VerificationResult, original JSON error and complete raw receipt.
    """
    if type(verification) is not VerificationResult or type(prepared) is not PreparedPublicExecution or type(check) is not ProjectCheck:
        raise TypeError("Typed verification, prepared execution and public check required")
    root = Path(project).resolve(strict=True)
    _require(str(root) == prepared.candidate_root, "Foreign public candidate")
    _require((verification.job_id, verification.task_id) == (prepared.job_id, prepared.task_id), "Foreign public task")
    _require(verification.spec == prepared.command, "Unexpected public command")
    _require(verification.status == "failed" and verification.exit_code == 1
             and not verification.stdout.truncated and not verification.stderr.truncated,
             "Public execution was not a complete failed invocation")
    _require(verification.checked_workspace_revision == workspace_revision(root), "Public source revision is stale")
    _require(_sha(check.to_json().encode("utf-8")) == prepared.check_sha256, "Public check identity changed")
    check.check_protected()
    prepared.check_protected()
    request_raw = _safe_file(prepared.request_path)
    request = _json(request_raw)
    receipt_raw = _safe_file(prepared.receipt_path)
    receipt = _json(receipt_raw)
    _require(type(receipt) is dict, "Invalid public execution receipt")
    expected = dict(schema_version=1, kind="public-python-execution-v1", mode="candidate-exception-v1",
        invocation_id=prepared.invocation_id, job_id=prepared.job_id, task_id=prepared.task_id,
        request_sha256=_sha(request_raw), candidate_root=str(root),
        original_argv=list(check.command(root).argv), checker_path=request["checker_path"],
        checker_sha256=_sha(_safe_file(request["checker_path"])))
    _require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items()),
             "Public execution identity mismatch")
    _require(set(receipt) == set(expected) | {"candidate_before", "candidate_after", "outcome", "exception_type", "exception_module", "frames", "stdout_sha256", "stderr_sha256", "dependency_snapshot"},
             "Unexpected public execution receipt fields")
    _require(receipt["stdout_sha256"] == _sha(verification.stdout.text.encode("utf-8"))
             and receipt["stderr_sha256"] == _sha(verification.stderr.text.encode("utf-8")),
             "Public output differs from observed invocation")
    manifest = capture_manifest(root)
    entries = [entry.to_dict() for entry in manifest.entries]
    _require(receipt["candidate_before"] == entries == receipt["candidate_after"], "Public execution candidate changed")
    _require(receipt["outcome"] == "exception" and receipt["exception_module"] == "builtins"
             and receipt["exception_type"] in ALLOWED_EXCEPTIONS, "Public exception needs infrastructure or manual review")
    frames = receipt["frames"]
    _require(type(frames) is list and 1 <= len(frames) <= 128, "Invalid public traceback frames")
    source = {entry.path: entry for entry in manifest.entries}
    candidate_frames = []
    candidate_indices = []
    for index, frame in enumerate(frames):
        _require(type(frame) is dict and set(frame) == {"filename", "line", "function", "candidate_path",
            "source_sha256", "code_sha256", "source_code_matched", "dependency_path", "dependency_root",
            "dependency_before", "dependency_after"}, "Invalid public frame fields")
        _require(type(frame["line"]) is int and frame["line"] > 0 and type(frame["function"]) is str
                 and type(frame["filename"]) is str and type(frame["source_code_matched"]) is bool,
                 "Invalid public frame values")
        relative = frame["candidate_path"]
        if relative is not None:
            _require(type(relative) is str and relative in source
                     and str(root / relative) == frame["filename"]
                     and frame["source_sha256"] == source[relative].sha256
                     and frame["source_code_matched"] is True,
                     "Candidate traceback provenance mismatch")
            from researchcoder.contracts import digest
            digest(frame["code_sha256"])
            candidate_indices.append(index)
            candidate_frames.append(dict(path=relative, line=frame["line"], function=frame["function"],
                                         source_sha256=frame["source_sha256"], code_sha256=frame["code_sha256"]))
    _require(bool(candidate_frames), "Exception occurred outside candidate execution")
    attribution = "candidate_execution"
    dependency_frames = []
    if candidate_indices[-1] != len(frames) - 1:
        # A candidate ancestor alone is insufficient. One bounded exception is
        # a builtin TypeError traversing only preexisting, source-matched package
        # frames after the candidate call. This is an observed dependency-boundary
        # failure, not a claim that the candidate is the definitive root cause.
        snapshot = receipt["dependency_snapshot"]
        _require(receipt["exception_type"] == "TypeError" and type(snapshot) is dict
                 and snapshot.get("complete") is True
                 and snapshot.get("interpreter_executable") == str(Path(prepared.command.argv[0]).resolve()),
                 "Public exception terminal frame is outside candidate execution")
        roots = snapshot.get("roots")
        executable = Path(prepared.command.argv[0]).absolute()
        expected_prefix = executable.parent.parent if executable.parent.name == "bin" else executable.parent
        _require(type(roots) is list and roots and len(roots) <= 2
                 and snapshot.get("interpreter_prefix") == str(expected_prefix.resolve())
                 and all(type(name) is str and Path(name).is_absolute()
                         and Path(name).is_relative_to(expected_prefix.resolve())
                         and Path(name).name in ("site-packages", "dist-packages") for name in roots),
                 "Invalid dependency provenance roots")
        protected_paths = {p.path for p in prepared.protected_files}
        for frame in frames[candidate_indices[-1] + 1:]:
            path, dependency_root = frame["dependency_path"], frame["dependency_root"]
            _require(type(path) is str and path == frame["filename"] and path not in protected_paths
                     and dependency_root in roots and Path(path).is_relative_to(Path(dependency_root))
                     and not Path(path).is_relative_to(root) and frame["source_code_matched"] is True,
                     "Public exception terminal frame is outside candidate execution")
            before, after = frame["dependency_before"], frame["dependency_after"]
            _require(type(before) is list and len(before) == 4 and all(type(v) is int for v in before)
                     and before == after, "Dependency changed during public execution")
            info = Path(path).stat()
            _require(before == [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns]
                     and frame["source_sha256"] == _sha(_safe_file(path)), "Dependency version changed")
            digest(frame["code_sha256"])
            dependency_frames.append(dict(path=path, source_sha256=frame["source_sha256"],
                                          code_sha256=frame["code_sha256"], line=frame["line"]))
        attribution = "candidate_dependency_execution"
    return dict(schema_version=1, kind="public-candidate-exception-v1", passed=False,
                status=verification.status, exit_code=verification.exit_code,
                job_id=verification.job_id, task_id=verification.task_id, attempt_id=verification.attempt_id,
                verification_id=verification.verification_id, invocation_id=prepared.invocation_id,
                exception_type=receipt["exception_type"], candidate_frames=candidate_frames,
                attribution=attribution, dependency_frames=dependency_frames,
                source_manifest_sha256=manifest.digest, checked_workspace_revision=verification.checked_workspace_revision,
                public_check_sha256=prepared.check_sha256, execution_receipt_sha256=_sha(receipt_raw),
                request_sha256=_sha(request_raw), stdout_sha256=_sha(verification.stdout.text.encode("utf-8")),
                stderr_sha256=_sha(verification.stderr.text.encode("utf-8")))
