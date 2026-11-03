"""Complete project snapshots adapt the legacy Scheduler without dropping files."""

from dataclasses import replace
from pathlib import Path

import pytest

from corecoder.workflow import workspace_revision
from researchcoder.evaluation.projects.workspace import ProjectSourceSnapshot, capture_manifest


def fixture(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    for path, raw in {"module.py": b"value = 1\n", "README.rst": b"Overview\n",
                      "LICENSE": b"Example\n", ".projectrc": b"mode = stable\n",
                      ".github/workflows/test.yml": b"name: tests\n",
                      "assets/icon.bin": b"\x00\x01"}.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return root


def test_complete_snapshot_preserves_bytes_and_legacy_adapter_identity(tmp_path):
    root = fixture(tmp_path)
    manifest = capture_manifest(root)
    snapshot = ProjectSourceSnapshot.create(root, tmp_path / "snapshot", manifest)
    assert snapshot.manifest == manifest
    assert snapshot.verify() == snapshot.workspace
    assert snapshot.workspace.source_manifest_sha256 == workspace_revision(root)
    assert capture_manifest(snapshot.workspace.root) == manifest
    assert snapshot.to_dict()["manifest_sha256"] == manifest.digest
    for entry in manifest.entries:
        assert (Path(snapshot.workspace.root)/entry.path).read_bytes() == (root/entry.path).read_bytes()
    (root / "module.py").write_text("value = 2\n")
    assert snapshot.verify() == snapshot.workspace
    assert (Path(snapshot.workspace.root)/"module.py").read_text() == "value = 1\n"


@pytest.mark.parametrize("name", ["README.rst", "LICENSE", ".projectrc", "assets/icon.bin"])
def test_snapshot_detects_mutation_outside_legacy_suffix_manifest(tmp_path, name):
    root = fixture(tmp_path)
    snapshot = ProjectSourceSnapshot.create(root, tmp_path / "snapshot", capture_manifest(root))
    legacy = snapshot.workspace.source_manifest_sha256
    target = Path(snapshot.workspace.root)/name
    target.chmod(0o644)
    target.write_bytes(b"changed\n")
    assert workspace_revision(snapshot.workspace.root) == legacy
    with pytest.raises(ValueError, match="Complete source snapshot changed"):
        snapshot.verify()


def test_snapshot_refuses_stale_input_and_never_overwrites(tmp_path):
    root = fixture(tmp_path)
    manifest = capture_manifest(root)
    (root / "README.rst").write_text("changed\n")
    with pytest.raises(ValueError):
        ProjectSourceSnapshot.create(root, tmp_path / "stale", manifest)
    assert not (tmp_path / "stale").exists()
    manifest = capture_manifest(root)
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("preserved")
    with pytest.raises(ValueError):
        ProjectSourceSnapshot.create(root, target, manifest)
    assert (target / "keep").read_text() == "preserved"


def test_snapshot_rechecks_legacy_identity_even_if_complete_manifest_matches(tmp_path):
    root = fixture(tmp_path)
    snapshot = ProjectSourceSnapshot.create(root, tmp_path / "snapshot", capture_manifest(root))
    forged = replace(snapshot, workspace=replace(snapshot.workspace, source_manifest_sha256="a"*64))
    with pytest.raises(ValueError, match="Legacy source snapshot revision changed"):
        forged.verify()
