"""Protected generator fragment. An isolated invocation inspects one candidate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def run(case_id, filename, profile):
    if len(sys.argv) != 2 or not __debug__:
        raise ValueError("Checker requires one candidate and enabled assertions")
    root = Path(sys.argv[1]).resolve(strict=True)
    path = (root / filename).resolve(strict=True)
    if path.parent != root or (root / filename).is_symlink():
        raise ValueError("Invalid candidate origin")
    checks = sorted((name, fn) for name, fn in globals().items() if name.startswith(profile + "_"))
    if not checks:
        raise ValueError("Empty check profile")
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for name, check in checks:
        try:
            check(module)
            passed = True
        except Exception:
            passed = False
        rows.append(dict(check=name, passed=passed))
    passed = all(row["passed"] for row in rows)
    print(json.dumps(dict(case_id=case_id, profile=profile, passed=passed, checks=rows,
        candidate_origin=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        adapter_version="m21-development-v1"), allow_nan=False))
    return 0 if passed else 1
