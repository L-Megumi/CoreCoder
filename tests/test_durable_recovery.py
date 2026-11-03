from dataclasses import replace
from pathlib import Path
import threading
import pytest
from researchcoder.durable import DurableStore, CheckpointError
from researchcoder.durable.recovery import RecoveryCoordinator
from tests.test_orchestration_static import setup_graph


def test_real_dag_checkpoint_reload_completed_result_and_context(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    from researchcoder.context.policy import ContextPolicy

    cfg = replace(
        cfg,
        templates={
            k: replace(
                t, config=replace(t.config, context_policy=ContextPolicy(role=t.role))
            )
            for k, t in cfg.templates.items()
        },
    )
    with (
        DurableStore(tmp_path / "state.db", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        outcome = runner.execute(plan, cfg)
        assert outcome.status == "succeeded"
        assert len(store.rows("tasks")) == 3 and len(store.rows("attempts")) == 3
        assert all(w["status"] == "exited" for w in store.rows("workers"))
        assert any(o["status"] == "verified" for o in store.rows("operations"))
        before = store.get("budgets", "job")["value"]["snapshot"]
        attempts = [r["id"] for r in store.rows("attempts")]
    with (
        DurableStore(tmp_path / "state.db", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        report = runner.recover("job", cfg)
        assert report["status"] == "completed" and runner.last_result == outcome
        assert [r["id"] for r in store.rows("attempts")] == attempts
        assert report["budget_after"] == before
        assert store.validate()


@pytest.mark.parametrize("change", ["policy", "model", "budget", "source"])
def test_recovery_refuses_changed_trusted_authority_or_workspace(tmp_path, change):
    plan, cfg, io = setup_graph(tmp_path)
    with (
        DurableStore(tmp_path / "db", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        assert runner.execute(plan, cfg).status == "succeeded"
    if change == "source":
        Path(cfg.templates["verify"].spec.workspace.root, "training.py").write_text(
            "loss = 999\n"
        )
    elif change == "budget":
        cfg = replace(
            cfg,
            job_limits=replace(
                cfg.job_limits, max_total_tokens=cfg.job_limits.max_total_tokens + 1
            ),
        )
    else:
        template = cfg.templates["read"]
        modified = (
            replace(
                template.config, model=replace(template.config.model, model="changed")
            )
            if change == "model"
            else replace(template.config, diagnostic_tools=("extract_log_evidence",))
        )
        cfg = replace(
            cfg, templates=dict(cfg.templates, read=replace(template, config=modified))
        )
    with (
        DurableStore(tmp_path / "db", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        if change == "source":
            assert runner.recover("job", cfg)["status"] == "needs_review"
        else:
            with pytest.raises(CheckpointError):
                runner.recover("job", cfg)


def test_real_cancelled_job_not_restarted(tmp_path):
    plan, cfg, io = setup_graph(tmp_path)
    cancel = threading.Event()
    cancel.set()
    cfg = replace(cfg, cancel_event=cancel)
    with (
        DurableStore(tmp_path / "db", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        assert runner.execute(plan, cfg).status == "cancelled"
        report = runner.recover("job", replace(cfg, cancel_event=None))
        assert (
            report["status"] == "stopped"
            and report["original_job_status"] == "cancelled"
        )
        assert not store.rows("attempts")


def test_real_autograd_dag_journal_and_heldout_oracle(tmp_path):
    from examples.multi_agent_autograd_demo import prepare_demo
    from researchcoder.context.policy import ContextPolicy

    plan, cfg, io, initial = prepare_demo(tmp_path)
    cfg = replace(
        cfg,
        templates={
            k: replace(
                t,
                config=replace(
                    t.config,
                    context_policy=ContextPolicy(role=t.role, allow_recovery=True),
                ),
            )
            for k, t in cfg.templates.items()
        },
    )
    with (
        DurableStore(tmp_path / "durable.sqlite", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        outcome = runner.execute(plan, cfg)
        assert initial["status"] == "failed" and outcome.status == "succeeded"
        assert (
            next(r for r in outcome.results if r.kind == "verify").verification.status
            == "passed"
        )
        edits = [
            o for o in store.rows("operations") if o["value"]["tool"] == "edit_file"
        ]
        assert len(edits) == 1 and edits[0]["status"] == "verified"
        assert (
            store.io.read_json(
                __import__(
                    "researchcoder.contracts", fromlist=["ArtifactRef"]
                ).ArtifactRef.from_dict(edits[0]["value"]["authority_ref"])
            )["spec"]["kind"]
            == "repair"
        )
        assert runner.recover("autograd", cfg)["status"] == "completed"


def test_verification_that_changes_source_is_not_checkpointed_as_current_success(
    tmp_path,
):
    plan, cfg, io = setup_graph(tmp_path)
    template = cfg.templates["verify"]
    command = replace(
        template.spec.verification,
        argv=(
            template.spec.verification.argv[0],
            "-c",
            "from pathlib import Path; p=Path('verification-count'); p.write_text((p.read_text() if p.exists() else '')+'run\\n'); Path('training.py').write_text('loss = 2\\n')",
        ),
    )
    changed = replace(
        template,
        spec=replace(template.spec, verification=command),
        config=replace(template.config, permitted_verifications=(command,)),
    )
    cfg = replace(cfg, templates=dict(cfg.templates, verify=changed))
    plan = replace(plan, tasks=(replace(plan.tasks[-1], dependencies=()),))
    with (
        DurableStore(tmp_path / "db", io) as store,
        RecoveryCoordinator(store) as runner,
    ):
        with pytest.raises(CheckpointError):
            runner.execute(plan, cfg)
        assert store.get("jobs", "job")["status"] != "succeeded"
        assert runner.recover("job", cfg)["status"] == "needs_review"
        counter = Path(changed.spec.workspace.root, "verification-count")
        assert counter.read_text() == "run\n"
