"""Stdlib-only protected adapter; delegate scoring to the unchanged Oracle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate checker field")
        value[key] = item
    return value


def main(argv=None):
    metadata_path, project, profile = (sys.argv[1:] if argv is None else argv)
    if profile not in ("public", "f2p", "p2p"):
        raise ValueError("Unknown profile")
    metadata_path = Path(metadata_path).resolve(strict=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    oracle = metadata_path.with_name("oracle.py")
    if oracle.is_symlink() or hashlib.sha256(oracle.read_bytes()).hexdigest() != metadata["oracle_sha256"]:
        raise ValueError("Original Oracle changed")
    root = Path(project).resolve(strict=True)
    relative = Path(metadata["origin_relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid source origin")
    origin = root / relative
    if any(p.is_symlink() for p in (origin, *origin.parents)) or not origin.resolve(strict=True).is_relative_to(root):
        raise ValueError("Candidate origin escapes")
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL") if k in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-I", "-B", str(oracle), str(root), metadata["case_id"], profile],
        env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15)
    value = json.loads(result.stdout, object_pairs_hook=unique)
    rows = value.get("checks")
    if (value.get("case_id") != metadata["case_id"] or value.get("profile") != profile
            or type(value.get("passed")) is not bool or result.returncode not in (0, 1)
            or (result.returncode == 0) is not value["passed"] or type(rows) is not list):
        raise ValueError("Invalid original Oracle result")
    if (any(type(r) is not dict or type(r.get("passed")) is not bool for r in rows)
            or [r.get("check") for r in rows] != metadata["expected_checks"][profile]
            or value["passed"] is not all(r["passed"] for r in rows)):
        raise ValueError("Original Oracle check identity/aggregate mismatch")
    # No private values or exception text are added to public feedback.
    output = dict(case_id=value["case_id"], profile=profile, passed=value["passed"],
        checks=[dict(check=r["check"], passed=r["passed"]) for r in rows],
        candidate_origin=str(origin), source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),
        original_stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(),
        adapter_version="historical-original-v1")
    print(json.dumps(output, allow_nan=False))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
