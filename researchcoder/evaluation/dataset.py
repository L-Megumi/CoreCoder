"""Controller-only benchmark loader and checked source materialization."""

import hashlib
from pathlib import Path
from .contracts import BenchmarkManifest

DATA = Path(__file__).with_name("data")
ORACLE = Path(__file__).with_name("oracles.py")


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_manifest(benchmark_subset="pytorch"):
    if benchmark_subset == "complexity_benchmark":
        from .complexity.dataset import load_manifest as complexity_manifest
        return complexity_manifest()
    if benchmark_subset != "pytorch":
        raise ValueError("Unknown benchmark subset")
    manifest = BenchmarkManifest.from_json(
        (DATA / "cases.json").read_text(encoding="utf-8")
    )
    support = sha_bytes(
        b"".join(
            p.name.encode() + p.read_bytes()
            for p in sorted((DATA / "support").iterdir())
            if p.is_file()
        )
    )
    for case in manifest.cases:
        if (
            sha_bytes(ORACLE.read_bytes()) != case.oracle_sha256
            or support != case.oracle_support_sha256
        ):
            raise ValueError("Oracle content mismatch")
        for asset in case.assets:
            for kind, expected in (
                ("faulty", asset.faulty_sha256),
                ("healthy", asset.healthy_sha256),
            ):
                if (
                    sha_bytes((DATA / case.case_id / kind / asset.path).read_bytes())
                    != expected
                ):
                    raise ValueError("Case content mismatch")
    return manifest


def case_asset_root(case):
    from .complexity.dataset import is_complexity, asset_root
    return asset_root(case) if is_complexity(case) else DATA / case.case_id


def materialize(case, destination, *, healthy=False):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    for asset in case.assets:
        source = case_asset_root(case) / ("healthy" if healthy else "faulty") / asset.path
        content = source.read_bytes()
        expected = asset.healthy_sha256 if healthy else asset.faulty_sha256
        if sha_bytes(content) != expected:
            raise ValueError("Source changed after manifest validation")
        target = destination / asset.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    return destination


def public_input(case):
    """Explicit allowlist. Never serialize BenchmarkCase into WorkerRequest/Prompt."""
    return dict(
        objective=case.objective,
        log=case.public_log,
        source_files=[a.path for a in case.assets],
    )


def source_revisions(case):
    return {
        kind: sha_bytes(
            "".join(a.path + getattr(a, kind + "_sha256") for a in case.assets).encode()
        )
        for kind in ("healthy", "faulty")
    }
