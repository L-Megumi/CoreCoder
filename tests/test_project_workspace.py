"""Real repositories use exact file sets and isolated copies, including dotfiles."""

from dataclasses import replace
import os

import pytest

from researchcoder.evaluation.projects import workspace
from researchcoder.evaluation.projects.contracts import RepositoryManifest
from researchcoder.evaluation.projects.workspace import assert_allowed_changes, capture_manifest, materialize


def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name, content in {"model.py": b"value = 1\n", ".gitignore": b"__pycache__\n", "LICENSE": b"MIT\n",
                          "README.rst": b"Example\n", ".github/workflows/test.yml": b"name: tests\n",
                          "images/example.bin": b"\x00\x01\x02"}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def test_capture_and_materialize_cover_hidden_extensionless_and_binary_files(tmp_path):
    root = source(tmp_path)
    frozen = capture_manifest(root)
    assert {item.path for item in frozen.entries} == {
        "model.py", ".gitignore", "LICENSE", "README.rst", ".github/workflows/test.yml", "images/example.bin"}
    candidate = materialize(root, tmp_path / "candidate", frozen)
    assert capture_manifest(candidate) == frozen
    assert candidate != root
    (candidate / "model.py").write_text("value = 2\n")
    assert (root / "model.py").read_bytes() == b"value = 1\n"
    assert assert_allowed_changes(frozen, capture_manifest(candidate), ("model.py",)) == ("model.py",)
    assert assert_allowed_changes(frozen, frozen, ()) == ()


@pytest.mark.parametrize("change", ["extra", "missing", "content"])
def test_materialize_refuses_source_drift_before_creating_candidate(tmp_path, change):
    root = source(tmp_path)
    frozen = capture_manifest(root)
    if change == "extra":
        (root / "extra.py").write_text("new")
    elif change == "missing":
        (root / "LICENSE").unlink()
    else:
        (root / "README.rst").write_text("Changed")
    target = tmp_path / "candidate"
    with pytest.raises(ValueError, match="frozen manifest"):
        materialize(root, target, frozen)
    assert not target.exists()


def test_materialize_never_overwrites_existing_or_nested_directory(tmp_path):
    root = source(tmp_path)
    frozen = capture_manifest(root)
    target = tmp_path / "candidate"
    target.mkdir()
    (target / "existing").write_text("keep")
    for forbidden in (target, root, root / "nested", tmp_path):
        with pytest.raises(ValueError):
            materialize(root, forbidden, frozen)
    assert (target / "existing").read_text() == "keep"
    assert capture_manifest(root) == frozen


def test_source_change_during_copy_is_detected_and_no_existing_file_is_overwritten(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "file.py").write_text("original")
    frozen = capture_manifest(root)
    original = workspace._file_bytes
    calls = 0
    def change_before_copy(path, directory):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_text("changed")
        return original(path, directory)
    monkeypatch.setattr(workspace, "_file_bytes", change_before_copy)
    target = tmp_path / "candidate"
    with pytest.raises(ValueError, match="changed during materialization"):
        materialize(root, target, frozen)
    assert target.exists() and not (target / "file.py").exists()


def test_full_manifest_protects_config_and_rejects_additions_and_deletions(tmp_path):
    root = source(tmp_path)
    frozen = capture_manifest(root)
    (root / ".gitignore").write_text("changed config")
    with pytest.raises(ValueError, match="protected"):
        assert_allowed_changes(frozen, capture_manifest(root), ("model.py",))
    for entries in (frozen.entries[:-1], (*frozen.entries, replace(frozen.entries[-1], path="new.py"))):
        changed = RepositoryManifest(entries=tuple(sorted(entries, key=lambda item: item.path)))
        with pytest.raises(ValueError, match="added or deleted"):
            assert_allowed_changes(frozen, changed, ("model.py",))
    with pytest.raises(ValueError):
        assert_allowed_changes(frozen, frozen, ("model.py", "model.py"))
    with pytest.raises(ValueError):
        assert_allowed_changes(frozen, frozen, ("new.py",))
    with pytest.raises(TypeError):
        assert_allowed_changes(frozen, frozen, "model.py")


@pytest.mark.parametrize("name", [".env", "nested/.env.local", ".git/config", ".ssh/id_rsa", "secret.key"])
def test_private_paths_are_rejected_instead_of_silently_excluded(tmp_path, name):
    root = source(tmp_path)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private material")
    with pytest.raises(ValueError):
        capture_manifest(root)


def test_capture_rejects_symlink_files_directories_and_root_ancestors(tmp_path):
    root = source(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "secret").write_text("outside")
    link = root / "linked.py"
    link.symlink_to(other / "secret")
    with pytest.raises(ValueError):
        capture_manifest(root)
    link.unlink()
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        capture_manifest(root)
    link.unlink()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        capture_manifest(alias)
    sub = root / "sub"
    sub.mkdir()
    (sub / "x").write_text("inside")
    with pytest.raises(ValueError):
        capture_manifest(alias / "sub")


def test_materialize_rejects_symlink_destination_ancestor(tmp_path):
    root = source(tmp_path)
    destination = tmp_path / "destination"
    destination.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError):
        materialize(root, alias / "candidate", capture_manifest(root))
    assert not (destination / "candidate").exists()


def test_capture_rejects_fifo_without_opening_it(tmp_path):
    root = source(tmp_path)
    os.mkfifo(root / "fifo")
    with pytest.raises(ValueError, match="file type"):
        capture_manifest(root)


def test_manifest_capture_is_bounded_for_files_bytes_and_entries(tmp_path, monkeypatch):
    root = source(tmp_path)
    for constant, maximum in (("MAX_FILES", 1), ("MAX_TOTAL_BYTES", 2), ("MAX_FILE_BYTES", 2), ("MAX_ENTRIES", 1)):
        with monkeypatch.context() as patch:
            patch.setattr(workspace, constant, maximum)
            with pytest.raises(ValueError):
                capture_manifest(root)
