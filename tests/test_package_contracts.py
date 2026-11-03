"""The new package must ship without replacing original entry points."""

from pathlib import Path


def test_packaging_includes_both_packages_and_preserves_entrypoints():
    config = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'packages = ["corecoder", "researchcoder"]' in config
    assert 'corecoder = "corecoder.cli:main"' in config
    assert 'researchcoder = "corecoder.researchcoder:main"' in config
    from researchcoder.adapters.corecoder import CoreCoderAdapter
    from researchcoder.contracts import TaskSpec

    assert TaskSpec and CoreCoderAdapter
