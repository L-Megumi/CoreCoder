from types import SimpleNamespace
import json
import sqlite3
import pytest
from researchcoder.durable import DurableStore, CheckpointError
from researchcoder.durable.store import digest, encode
from researchcoder.durable.recovery import RecoveryCoordinator
from researchcoder.adapters.artifacts import ArtifactIO
from tests.test_orchestration_static import setup_graph


def alter(path, table, edit):
    with sqlite3.connect(path) as db:
        identity, body = db.execute(f"SELECT id,body FROM {table} LIMIT 1").fetchone()
        value = json.loads(body)
        edit(value)
        db.execute(
            f"UPDATE {table} SET body=?,sha256=? WHERE id=?",
            (encode(value), digest(value), identity),
        )


def test_unverified_shared_filesystem_is_not_automatically_enabled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "researchcoder.durable.store.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="nfs\n"),
    )
    for mode in ("auto", "wal"):
        with pytest.raises(CheckpointError):
            DurableStore(tmp_path / "db", ArtifactIO(tmp_path / "a"), journal_mode=mode)


@pytest.mark.parametrize("damage", ["budget", "workspace", "effect_path"])
def test_semantically_tampered_records_cannot_override_trusted_config(
    tmp_path, damage, monkeypatch
):
    plan, cfg, io = setup_graph(tmp_path)
    path = tmp_path / "db"
    with DurableStore(path, io) as store, RecoveryCoordinator(store) as runner:
        assert runner.execute(plan, cfg).status == "succeeded"
    outside = tmp_path / "outside"
    outside.mkdir()
    if damage == "budget":
        alter(
            path,
            "budgets",
            lambda v: v["snapshot"]["limits"].update(max_model_calls=999),
        )
    elif damage == "workspace":
        alter(
            path,
            "jobs",
            lambda v: v["workspace_versions"].update({str(outside): "a" * 64}),
        )
    else:
        alter(
            path,
            "operations",
            lambda v: v.update(tool="edit_file", path=str(outside / "private.py")),
        )
    import researchcoder.durable.recovery as module

    original_revision = module.workspace_revision
    original_hash = module.file_hash

    def revision(p):
        assert not str(p).startswith(str(outside)), (
            "Must refuse before scanning unauthorized source"
        )
        return original_revision(p)

    def file_hash(p):
        assert not str(p).startswith(str(outside)), (
            "Must refuse before reading unauthorized file"
        )
        return original_hash(p)

    monkeypatch.setattr(module, "workspace_revision", revision)
    monkeypatch.setattr(module, "file_hash", file_hash)
    with DurableStore(path, io) as store, RecoveryCoordinator(store) as runner:
        with pytest.raises(CheckpointError):
            runner.recover("job", cfg)
