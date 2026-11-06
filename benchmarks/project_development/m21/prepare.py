"""Materialize two original development fixtures and freeze their independent receipts. Offline."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
CASES = {
    "stream_utf8_chunks": (
        "text_stream.py",
        "Repair decode_chunks(chunks) for a single-pass stream of arbitrary UTF-8 byte chunks. "
        "Emit nonempty text incrementally as soon as complete text is available, preserve strict "
        "UTF-8 errors, including incomplete final input, and keep the existing API.",
        "UnicodeDecodeError: a valid UTF-8 euro sign straddled two read boundaries. "
        "Joining the yielded text should give the original string.",
    ),
    "stable_shard_merge": (
        "event_merge.py",
        "Repair merge_events(shards) for already time-sorted shard iterators. Merge lazily with "
        "at most one buffered head per shard. Equal timestamps preserve original shard order, "
        "then order within that shard. Preserve opaque payload objects and the generator API.",
        "TypeError: '<' not supported between instances of 'dict' and 'dict' when merging "
        "two events with equal timestamps from different shards.",
    ),
}
PROFILES = ("public", "f2p", "p2p")
VARIANTS = {
    "faulty": (False, False, True),
    "healthy": (True, True, True),
    "negative": (True, True, False),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)


def freeze_checks(case_id, filename, protected):
    protected.mkdir()
    common = (ASSETS / "check_entry.py").read_text(encoding="utf-8").strip()
    checks = {}
    for profile in PROFILES:
        definitions = (ASSETS / case_id / ("public.py" if profile == "public" else "private.py")).read_text(encoding="utf-8")
        tree = ast.parse(definitions)
        tests = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith(profile + "_")]
        names = sorted(node.name for node in tests)
        if not names or len(names) != len(set(names)):
            raise ValueError("Invalid test set")
        pieces = [ast.get_source_segment(definitions, node) for node in tests]
        checker = protected / (profile + "-entry.py")
        checker.write_bytes(("\n\n\n".join([common, *pieces,
            f"raise SystemExit(run({case_id!r}, {filename!r}, {profile!r}))"]) + "\n").encode("utf-8"))
        checks[profile] = {
            "schema_version": 1, "check_id": case_id + "-" + profile, "timeout_seconds": 20,
            "argv_template": [sys.executable, "-I", "-B", str(checker), "{project}"],
            "protected_files": [{"path": str(checker), "sha256": sha(checker), "schema_version": 1}],
            "receipt": {"kind": "checks_json", "origin_field": "candidate_origin", "origin_relative_path": filename,
                        "required_fields": ["case_id", "profile", "passed", "checks", "source_sha256", "adapter_version"],
                        "expected_checks": names, "exact_fields": [["case_id", case_id], ["profile", profile],
                        ["adapter_version", "m21-development-v1"]], "digest_field": "source_sha256",
                        "digest_relative_path": filename, "schema_version": 1},
        }
    return checks


def controls(case_id, filename, directory, checks):
    rows = []
    for variant, flags in VARIANTS.items():
        project = directory / ("source" if variant == "faulty" else variant)
        # Reference and negative controls are siblings outside the writable source.
        project.mkdir()
        target = project / filename
        shutil.copyfile(ASSETS / case_id / (variant + ".py"), target)
        digest = sha(target)
        for profile, expected in zip(PROFILES, flags):
            check = checks[profile]
            argv = [str(project) if arg == "{project}" else arg for arg in check["argv_template"]]
            execution = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                       timeout=check["timeout_seconds"], check=False)
            if execution.returncode not in (0, 1) or not execution.stdout:
                raise ValueError(f"{case_id}/{variant}/{profile}: checker crashed")
            receipt = json.loads(execution.stdout)
            valid = (execution.returncode == int(not expected) and receipt["passed"] is expected
                and receipt["source_sha256"] == digest == sha(target) and receipt["case_id"] == case_id
                and receipt["profile"] == profile and receipt["candidate_origin"] == str(target)
                and [r["check"] for r in receipt["checks"]] == check["receipt"]["expected_checks"]
                and all(type(r["passed"]) is bool for r in receipt["checks"])
                and all(r["passed"] for r in receipt["checks"]) is expected)
            if not valid:
                raise ValueError(f"{case_id}/{variant}/{profile}: unexpected evidence")
            rows.append(dict(variant=variant, profile=profile, expected=expected, exit_code=execution.returncode,
                             checks=receipt["checks"], source_sha256=digest, checker_sha256=check["protected_files"][0]["sha256"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, help="Linux ResearchCoder source tree")
    parser.add_argument("--base-config", type=Path, required=True, help="Frozen real M20 Code-first config")
    parser.add_argument("--output", type=Path, required=True, help="Fresh, exclusive output directory")
    parser.add_argument("--checks-only", action="store_true", help="Check portable fixtures without importing the Linux runtime")
    args = parser.parse_args()
    if not args.checks_only and not args.runtime:
        parser.error("--runtime required without --checks-only")
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    if base["policy"].get("repair_cycle_version") != "repair-cycle-v1" or base["expert_policy"]["initial_mode"] != "code-first-v1":
        raise ValueError("Expected the M20 frozen Code-first repair cycle")
    if not args.checks_only:
        sys.path.insert(0, str(args.runtime.resolve(strict=True)))
        from researchcoder.evaluation.projects.__main__ import load_config
        from researchcoder.evaluation.projects.workspace import capture_manifest
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for case_id, (filename, objective, log) in CASES.items():
        directory = out / case_id
        directory.mkdir()
        checks = freeze_checks(case_id, filename, directory / "protected")
        rows = controls(case_id, filename, directory, checks)
        source = directory / "source"
        target = source / filename
        entry = dict(schema_version=1, path=filename, sha256=sha(target), size_bytes=target.stat().st_size)
        config = dict(base, source_root=str(source), task=dict(project_id=case_id + "-development",
            source_revision=sha(target), manifest=dict(algorithm="all-files-v1", entries=[entry], schema_version=1),
            objective=objective, public_log=log, writable_paths=[filename], schema_version=1))
        config.update({profile + "_check": check for profile, check in checks.items()})
        write_json(directory / "config.json", config)
        if not args.checks_only:
            parsed = load_config(directory / "config.json")
            if capture_manifest(source) != parsed["task"].manifest:
                raise ValueError("Published source identity differs")
            for profile in PROFILES:
                parsed[profile + "_check"].check_protected()
        write_json(directory / "identities.json", dict(case_id=case_id, split="development", kind="original-controlled-fixture",
            source_scope="Single-file synthetic Python project, authored for M21 after M20 completion",
            case_files={p.name: sha(p) for p in sorted((ASSETS / case_id).glob("*.py"))},
            frozen_files={c["protected_files"][0]["path"]: c["protected_files"][0]["sha256"] for c in checks.values()},
            controls=rows, base_config_sha256=sha(args.base_config)))
        print(json.dumps(dict(case_id=case_id, offline_oracle_invocations=len(rows), paid_requests=0,
                              config_sha256=sha(directory / "config.json"), runtime_schema_checked=not args.checks_only)))


if __name__ == "__main__":
    main()
