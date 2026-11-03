import sqlite3
import subprocess
import sys
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.durable import DurableStore, CheckpointError


def body(status="pending", **kw):
    return dict(schema_version=1, job_id="j", status=status, **kw)


def test_reopen_references_versions_and_orphans(tmp_path):
    io = ArtifactIO(tmp_path / "artifacts")
    ref = io.put_json("j", {"original": "record"})
    with DurableStore(tmp_path / "state.sqlite", io) as store:
        store.commit(
            [
                ("jobs", "j", None, "open", body("open")),
                ("tasks", "j/r/t", None, "pending", body(output=ref.to_dict())),
            ]
        )
        assert store.filesystem["journal_mode"] == "wal"
        assert not store.orphans()
        orphan = io.put_json("j", {"uncommitted": "preserved"})
        assert store.orphans() == (orphan.storage_key,)
    with DurableStore(tmp_path / "state.sqlite", io) as store:
        assert store.validate() and store.get("tasks", "j/r/t")["version"] == 0
        assert io.read_json(ref) == {"original": "record"} and io.read_json(orphan)


def test_cas_duplicate_event_and_completion_are_atomic(tmp_path):
    with DurableStore(tmp_path / "db", ArtifactIO(tmp_path / "a")) as store:
        store.commit([("tasks", "t", None, "pending", body())], event_id="create")
        with pytest.raises(CheckpointError):
            store.commit(
                [("tasks", "t", 0, "running", body("running"))], event_id="create"
            )
        assert store.get("tasks", "t")["status"] == "pending"
        store.commit([("tasks", "t", 0, "running", body("running"))])
        with pytest.raises(CheckpointError):
            store.commit([("tasks", "t", 0, "succeeded", body("succeeded"))])
        store.commit([("tasks", "t", 1, "succeeded", body("succeeded"))])
        with pytest.raises(CheckpointError):
            store.commit([("tasks", "t", 2, "succeeded", body("succeeded"))])


@pytest.mark.parametrize("damage", ["body", "schema", "missing_artifact", "bytes"])
def test_corruption_or_incompatible_checkpoint_refused(tmp_path, damage):
    path = tmp_path / "db"
    io = ArtifactIO(tmp_path / "a")
    ref = io.put_json("j", {"safe": True})
    with DurableStore(path, io) as store:
        store.commit([("jobs", "j", None, "open", body("open", output=ref.to_dict()))])
    if damage == "missing_artifact":
        io.resolve(ref).unlink()
    elif damage == "bytes":
        path.write_bytes(b"not SQLite")
    else:
        db = sqlite3.connect(path)
        db.execute(
            "UPDATE jobs SET body='{}'" if damage == "body" else "PRAGMA user_version=9"
        )
        db.commit()
        db.close()
    with pytest.raises((CheckpointError, sqlite3.DatabaseError, OSError)):
        DurableStore(path, io)


def test_illegal_status_missing_foreign_ref_secret_and_rollback(tmp_path):
    io = ArtifactIO(tmp_path / "a")
    foreign = io.put_json("other", {})
    with DurableStore(tmp_path / "db", io) as store:
        with pytest.raises(CheckpointError):
            store.commit([("tasks", "t", None, "succeeded", body("succeeded"))])
        with pytest.raises(CheckpointError):
            store.commit(
                [("jobs", "j", None, "open", body("open", output=foreign.to_dict()))]
            )
        with pytest.raises(CheckpointError):
            store.commit(
                [("jobs", "j", None, "open", body("open", api_key="must not persist"))]
            )
        assert not store.rows("tasks") and not store.rows("jobs")


def test_single_controller_and_real_sqlite_write_lock(tmp_path):
    io = ArtifactIO(tmp_path / "a")
    path = tmp_path / "db"
    with DurableStore(path, io) as store:
        with pytest.raises(CheckpointError):
            DurableStore(path, io)
        with store.transaction():
            script = "import sqlite3,sys;d=sqlite3.connect(sys.argv[1],timeout=.05);d.execute('BEGIN IMMEDIATE')"
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(path)],
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0 and "locked" in result.stderr
    with DurableStore(path, io, journal_mode="delete") as store:
        assert store.filesystem["journal_mode"] == "delete"


def test_plan_attempt_budget_and_context_refs_survive_restart(tmp_path):
    from tests.test_orchestration_static import setup_graph
    from researchcoder.orchestration.contracts import PlanRevision
    from researchcoder.orchestration.budget import JobBudget

    plan, cfg, io = setup_graph(tmp_path)
    receipt = io.put_json("job", {"schema_version": 1, "context_receipts": []})
    budget = JobBudget(cfg.job_limits)
    budget.reserve("lost", next(iter(cfg.templates.values())).config.limits)
    budget.settle("lost", None)

    def value(**kw):
        return dict(schema_version=1, job_id="job", **kw)

    updates = [
        ("jobs", "job", None, "open", value(status="open", latest_revision="r0")),
        (
            "plans",
            "job/r0",
            None,
            "committed",
            value(status="committed", plan=plan.to_dict()),
        ),
        (
            "tasks",
            "job/r0/a",
            None,
            "pending",
            value(status="pending", fingerprint="a" * 64, context=receipt.to_dict()),
        ),
        (
            "attempts",
            "a1",
            None,
            "running",
            value(status="running", task_id="r0.a", attempt_id="a1"),
        ),
        (
            "budgets",
            "job",
            None,
            "active",
            value(status="active", snapshot=budget.snapshot()),
        ),
    ]
    with DurableStore(tmp_path / "db", io) as store:
        store.commit(updates)
    with DurableStore(tmp_path / "db", io) as store:
        assert (
            PlanRevision.from_dict(store.get("plans", "job/r0")["value"]["plan"])
            == plan
        )
        assert store.get("attempts", "a1")["value"]["attempt_id"] == "a1"
        assert store.get("budgets", "job")["value"]["snapshot"] == budget.snapshot()
        assert store.get("tasks", "job/r0/a")["value"]["context"] == receipt.to_dict()
