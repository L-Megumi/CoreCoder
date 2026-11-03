"""Independent controller-only checks; hidden inputs never feed the Agent."""
# PRIVATE_M4_ORACLE_SENTINEL_83adc12b
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
profile = sys.argv[2]
sys.path.insert(0, str(root))
source = root / "loader.py"
checks = []
try:
    pipeline = importlib.import_module("pipeline")
    if profile == "f2p":
        # Compute expectations independently of the implementation/reference patch.
        for name, features in (("unseen-signed-values", [11, -4, 6]), ("empty-features", [])):
            record = {"features": features, "metadata": "retained"}
            before = copy.deepcopy(record)
            actual = pipeline.total_features(record)
            passed = type(actual) in (int, float) and actual == sum(features) and record == before
            checks.append(dict(check=name, passed=bool(passed)))
    elif profile == "p2p":
        marker = object()
        checks.append(dict(check="identity-object", passed=pipeline.legacy_identity(marker) is marker))
        checks.append(dict(check="identity-scalar", passed=pipeline.legacy_identity(-17) == -17))
    else:
        raise ValueError("Unknown check profile")
except Exception:
    # Preserve every declared check, even when the first candidate call fails.
    expected = ("unseen-signed-values", "empty-features") if profile == "f2p" else ("identity-object", "identity-scalar")
    present = {row["check"] for row in checks}
    checks.extend(dict(check=name, passed=False) for name in expected if name not in present)
passed = bool(checks) and all(row["passed"] for row in checks)
print(json.dumps(dict(schema_version=1, profile=profile, origin=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), checks=checks, passed=passed)))
raise SystemExit(0 if passed else 1)
