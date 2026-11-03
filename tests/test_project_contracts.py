"""Public real-project contracts cannot acquire private evaluation fields."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from researchcoder.evaluation.projects.contracts import ProjectRunInput, RepositoryManifest, SourceFile


def entry(path="src/model.py", size=12, sha="a" * 64):
    return SourceFile(path=path, sha256=sha, size_bytes=size)


def manifest():
    return RepositoryManifest(entries=(entry(".gitignore"), entry("LICENSE"), entry()))


def project():
    return ProjectRunInput(project_id="real-project", source_revision="a" * 40, manifest=manifest(),
                           objective="Repair the public failure", public_log="AssertionError: mismatched update",
                           writable_paths=("src/model.py",))


def test_project_contract_roundtrip_and_explicit_public_projection():
    for value in (entry(), manifest(), project()):
        assert type(value).from_json(value.to_json()) == value
        assert type(value).from_dict(value.to_dict()) == value
        with pytest.raises(FrozenInstanceError):
            value.schema_version = 2
    public = project().public_input()
    assert set(public) == {"project_id", "source_revision", "manifest_sha256", "objective", "log", "source_files", "writable_paths"}
    assert public["source_files"] == [".gitignore", "LICENSE", "src/model.py"]
    public["writable_paths"].clear()
    assert project().writable_paths == ("src/model.py",)


def test_repository_identity_is_versioned_canonical_and_sensitive_to_all_file_bytes():
    value = manifest()
    assert value.digest == hashlib.sha256(value.to_json().encode("utf-8")).hexdigest()
    assert value.digest == RepositoryManifest.from_json(value.to_json()).digest
    changed = replace(value, entries=(replace(value.entries[0], sha256="b" * 64), *value.entries[1:]))
    assert changed.digest != value.digest
    assert replace(value, entries=(replace(value.entries[0], size_bytes=13), *value.entries[1:])).digest != value.digest


@pytest.mark.parametrize("path", ["../x", "/tmp/x", "x//y", "./x", "x/../y", "x\\y", "C:x", ".git/config",
                                ".env", "sub/.env.production", ".ssh/id_ed25519", "credentials.json", "test.pem"])
def test_source_file_rejects_escape_and_known_private_paths(path):
    with pytest.raises(ValueError):
        entry(path)


def test_manifest_rejects_ambiguous_duplicate_empty_and_unbounded_inputs():
    invalid = [(), (entry(), entry()), (entry("b"), entry("a")), (entry("src"), entry("src/model.py")),
               tuple(entry(f"f{i:04}", 1_000_000) for i in range(21)),
               tuple(entry(f"f{i:04}", 0) for i in range(4001))]
    for items in invalid:
        with pytest.raises(ValueError):
            RepositoryManifest(entries=items)
    for size in (-1, 1_000_001, True, 0.5):
        with pytest.raises((TypeError, ValueError)):
            entry(size=size)
    with pytest.raises(ValueError):
        replace(manifest(), algorithm="legacy")
    with pytest.raises(ValueError):
        entry(sha="invalid")


@pytest.mark.parametrize("change", [dict(writable_paths=()), dict(writable_paths=("missing.py",)),
                                   dict(writable_paths=("src/model.py", "src/model.py")), dict(public_log=""),
                                   dict(project_id="../project"), dict(source_revision="path/branch"),
                                   dict(schema_version=2)])
def test_public_project_input_rejects_invalid_authority_and_identity(change):
    with pytest.raises((ValueError, TypeError)):
        replace(project(), **change)


def test_private_fields_runtime_objects_duplicate_json_and_nan_are_rejected():
    for private_name in ("oracle", "reference_patch", "expected_answer", "private_log", "execution_config"):
        with pytest.raises(ValueError):
            ProjectRunInput.from_dict(project().to_dict() | {private_name: "must not cross public boundary"})
    with pytest.raises(TypeError):
        replace(project(), manifest=object())
    raw = project().to_json()
    with pytest.raises(ValueError):
        ProjectRunInput.from_json(raw[:-1] + ',"schema_version":1}')
    data = entry().to_dict() | {"size_bytes": float("nan")}
    with pytest.raises(ValueError):
        SourceFile.from_json(json.dumps(data))
