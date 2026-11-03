"""Controller-owned public reproduction; not copied into the candidate."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
source = root / "loader.py"
try:
    pipeline = importlib.import_module("pipeline")
    observed = pipeline.total_features({"features": [2, 3]})
    passed = type(observed) in (int, float) and observed == 5
except Exception:
    # This is a real chained Python traceback from the production candidate.
    traceback.print_exc(file=sys.stderr)
    passed = False
print(json.dumps(dict(schema_version=1, profile="public", origin=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    checks=[dict(check="public-feature-sum", passed=bool(passed))], passed=bool(passed))))
raise SystemExit(0 if passed else 1)
