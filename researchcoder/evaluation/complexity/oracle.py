"""Independent CPU oracle. Application guards are not an OS security sandbox."""

import ast
import hashlib
import json
from pathlib import Path
import sys
if __package__:
    from .reference import evaluate
else:
    from reference import evaluate

ROOT = Path(__file__).resolve().parent


def audit(project, case):
    expected = {a["path"]: a for a in case["assets"]}
    actual = {p.relative_to(project).as_posix(): p for p in project.rglob("*") if p.is_file()}
    if set(actual) != set(expected) or any(p.is_symlink() for p in project.rglob("*")):
        raise ValueError("Candidate file set is outside the declared contract")
    local = {Path(p).stem for p in expected if p.endswith(".py")}
    forbidden = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr", "delattr", "vars", "breakpoint", "input"}
    blocked = {"save", "load", "manual_seed", "set_rng_state", "set_default_dtype", "set_default_device", "set_num_threads"}
    for name, path in actual.items():
        data = path.read_bytes()
        if name not in case["writable_paths"] and hashlib.sha256(data).hexdigest() != expected[name]["faulty_sha256"]:
            raise ValueError("Read-only benchmark material changed")
        if len(data) > 1_000_000:
            raise ValueError("Candidate too large")
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(data.decode())):
            if isinstance(node, ast.Import) and any(n.name.split(".")[0] not in local | {"torch", "math"} for n in node.names):
                raise ValueError("Undeclared candidate import")
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").split(".")[0] not in local | {"torch", "math"}):
                raise ValueError("Undeclared candidate import")
            if isinstance(node, ast.Name) and node.id in forbidden:
                raise ValueError("Reflective or file execution is outside the benchmark")
            if isinstance(node, ast.Attribute):
                initializer = (node.attr == "__init__" and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "super"
                    and not node.value.args and not node.value.keywords)
                if (node.attr.startswith("__") and not initializer) or node.attr in blocked:
                    raise ValueError("Global runtime mutation is outside the benchmark")
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    while isinstance(target, ast.Attribute):
                        target = target.value
                    if isinstance(target, ast.Name) and target.id in local | {"torch", "nn", "F"}:
                        raise ValueError("Cannot replace the evaluator's Torch implementation")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    project, identity, profile = argv
    if profile not in ("discovery", "holdout", "regression"):
        raise ValueError("Unknown oracle profile")
    manifest = json.loads((ROOT / "data/cases.json").read_text())
    case = next(c for c in manifest["cases"] if c["case_id"] == identity)
    try:
        project = Path(project).resolve()
        audit(project, case)
        result = evaluate(project, identity, profile)
    except Exception as error:
        # Deliberately omit evaluator paths, private tensors and reference answers.
        result = dict(passed=False, error_type=type(error).__name__, reason="Candidate invariant checks could not complete")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
