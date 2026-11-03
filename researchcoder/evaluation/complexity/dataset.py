"""Private dataset, Oracle and complexity labels live outside candidate roots."""

import hashlib
from pathlib import Path
from researchcoder.evaluation.contracts import BenchmarkManifest
from .contracts import ComplexityManifest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ORACLE = ROOT / "oracle.py"
SUPPORT = ROOT / "reference.py"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def load_manifest():
    manifest = BenchmarkManifest.from_json((DATA / "cases.json").read_text(encoding="utf-8"))
    for case in manifest.cases:
        if case.oracle_sha256 != sha(ORACLE.read_bytes()) or case.oracle_support_sha256 != sha(SUPPORT.read_bytes()):
            raise ValueError("Complexity Oracle revision mismatch")
        for asset in case.assets:
            for state, expected in (("healthy", asset.healthy_sha256), ("faulty", asset.faulty_sha256)):
                path = DATA / case.case_id / state / asset.path
                if path.is_symlink() or sha(path.read_bytes()) != expected:
                    raise ValueError("Complexity fixture revision mismatch")
    return manifest


def load_complexity():
    manifest = load_manifest()
    result = ComplexityManifest.from_json((DATA / "complexity.json").read_text(encoding="utf-8"))
    if result.benchmark_manifest_sha256 != sha(manifest.to_json().encode()):
        raise ValueError("Complexity labels are not bound to the benchmark revision")
    from researchcoder.evaluation.dataset import load_manifest as original
    cases = {("complexity_benchmark", c.case_id): c for c in manifest.cases}
    cases.update({("pytorch", c.case_id): c for c in original().cases})
    for label in result.cases:
        case = cases.get((label.benchmark_subset, label.case_id))
        if case is None or label.case_sha256 != sha(case.to_json().encode()):
            raise ValueError("Labels refer to another case revision")
        assets = {a.path: a for a in case.assets}
        if set(label.source_files) != {a.path for a in case.assets if a.path.endswith(".py")}:
            raise ValueError("Complexity module inventory mismatch")
        for stage in label.stages:
            if not set(stage.patch_paths) <= set(case.writable_paths):
                raise ValueError("Reference stage exceeds candidate write authority")
        for site in label.evidence_sites:
            if site.path not in assets or site.sha256 != assets[site.path].faulty_sha256:
                raise ValueError("Complexity evidence is not backed by the declared source")
    return result


def is_complexity(case):
    return case.project_version == "complexity-v1"


def asset_root(case):
    if not is_complexity(case) or case.case_id not in {"cpx_01", "cpx_02", "cpx_03", "cpx_04", "cpx_05", "cpx_06"}:
        raise ValueError("Unknown complexity case")
    return DATA / case.case_id


def protected_paths():
    return (ORACLE, SUPPORT, DATA / "cases.json", DATA / "complexity.json", DATA / "generation.json")
