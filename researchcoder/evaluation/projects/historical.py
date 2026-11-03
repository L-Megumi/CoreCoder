"""Adapt trusted pinned upstream source slices to the existing project runner.

The candidate gets faulty source and the public objective only. Reference
versions, provenance metadata and the independent Oracle stay outside it.
"""
import hashlib
import json
from pathlib import Path

from researchcoder.contracts import _unique
from researchcoder.runtime.protocol import ProtectedFile
from .contracts import ProjectRunInput, source_path
from .verification import ProjectCheck, ReceiptSpec
from .workspace import capture_manifest


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_historical_case(case_root, python_executable):
    """Return existing contracts, never a second execution or scoring loop."""
    root = Path(case_root).absolute()
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Historical case symlink refused")
    root = root.resolve(strict=True)
    metadata_path = root / "case.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("Missing pinned case metadata")
    value = json.loads(metadata_path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported historical case version")
    oracle = root / "oracle.py"
    if oracle.is_symlink() or not oracle.is_file() or _sha(oracle) != value["oracle_sha256"]:
        raise ValueError("Historical Oracle changed")
    source = root / "faulty"
    manifest = capture_manifest(source)
    if {e.path: e.sha256 for e in manifest.entries} != value["manifests"]["faulty"]:
        raise ValueError("Historical faulty source changed")
    origin = value["origin_relative_path"]
    source_path(origin)
    if origin not in {e.path for e in manifest.entries}:
        raise ValueError("Historical origin is not candidate source")
    task = ProjectRunInput(project_id=value["case_id"] + "-development",
        source_revision=value["original_faulty_commit"], manifest=manifest,
        objective=value["objective"], public_log=value["public_log"],
        writable_paths=tuple(value["writable_paths"]))
    adapter = Path(__file__).with_name("historical_check.py").resolve(strict=True)
    protected = tuple(ProtectedFile(path=str(p), sha256=_sha(p)) for p in (oracle, metadata_path, adapter))
    interpreter = Path(python_executable)
    if not interpreter.is_absolute() or not interpreter.is_file():
        raise ValueError("Trusted absolute interpreter required")
    checks = {}
    for profile in ("public", "f2p", "p2p"):
        names = value["expected_checks"][profile]
        checks[profile] = ProjectCheck(check_id=value["case_id"] + "-" + profile,
            argv_template=(str(interpreter), "-I", "-B", str(adapter), str(metadata_path), "{project}", profile),
            protected_files=protected, timeout_seconds=20,
            receipt=ReceiptSpec(kind="checks_json", origin_field="candidate_origin", origin_relative_path=origin,
                required_fields=("case_id", "profile", "passed", "checks", "candidate_origin", "source_sha256", "adapter_version"),
                expected_checks=tuple(names), exact_fields=(("case_id", value["case_id"]), ("profile", profile),
                    ("adapter_version", "historical-original-v1")),
                digest_field="source_sha256", digest_relative_path=origin))
    return source, task, checks
