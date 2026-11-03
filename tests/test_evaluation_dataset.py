import os
from pathlib import Path
import pytest
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.evaluation.dataset import (
    load_manifest,
    materialize,
    DATA,
    public_input,
)
from researchcoder.evaluation.verification import verify_candidate


@pytest.mark.parametrize("case_id", [c.case_id for c in load_manifest().cases])
def test_independent_health_fault_repair_and_existing_regression(tmp_path, case_id):
    case = next(c for c in load_manifest().cases if c.case_id == case_id)
    python = os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"]
    io = ArtifactIO(tmp_path / "artifacts")
    healthy = materialize(case, tmp_path / "healthy", healthy=True)
    faulty = materialize(case, tmp_path / "faulty")
    for project, profile, status in (
        (healthy, "holdout", "passed"),
        (healthy, "regression", "passed"),
        (faulty, "discovery", "failed"),
        (faulty, "regression", "passed"),
    ):
        result = verify_candidate(
            case, project, python, profile, io, "validation", "check"
        )
        assert result.verification.status == status, (
            case_id,
            profile,
            result.to_dict(),
        )
    for path in case.writable_paths:
        (faulty / path).write_bytes((DATA / case_id / "healthy" / path).read_bytes())
    for profile in ("holdout", "regression"):
        assert (
            verify_candidate(
                case, faulty, python, profile, io, "validation", "repair"
            ).verification.status
            == "passed"
        )
    assert set(public_input(case)) == {"objective", "log", "source_files"}


@pytest.mark.parametrize(
    "case_id,content",
    [
        ("nonfinite_logsigmoid", "def positive_loss(logits):\n    return logits * 0\n"),
        (
            "optimizer_omission",
            "import torch\ndef build_optimizer(model):\n    model.backbone.requires_grad_(False)\n    return torch.optim.SGD(model.head.parameters(), lr=0.05)\n",
        ),
        (
            "autograd_detach",
            "def training_step(model,x,y,optimizer):\n    return 0.0\n",
        ),
        (
            "shape_transpose",
            'from pathlib import Path\nPath("../oracle.py").write_text("cheat")\n',
        ),
    ],
)
def test_oracle_rejects_training_goal_bypasses_and_test_mutation(
    tmp_path, case_id, content
):
    case = next(c for c in load_manifest().cases if c.case_id == case_id)
    project = materialize(case, tmp_path / "project")
    (project / "training.py").write_text(content)
    assert (
        verify_candidate(
            case,
            project,
            os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
            "holdout",
            ArtifactIO(tmp_path / "a"),
            "job",
            "test",
        ).verification.status
        == "failed"
    )
    assert not (tmp_path / "oracle.py").exists()


def test_autograd_reuses_original_oracle_and_sources_and_hashes_are_enforced(
    tmp_path, monkeypatch
):
    root = Path(__file__).parents[1]
    assert (DATA / "support/legacy.py").read_bytes() == (
        root / "benchmarks/pytorch/autograd_detach/oracle.py"
    ).read_bytes()
    assert (DATA / "autograd_detach/faulty/training.py").read_bytes() == (
        root / "benchmarks/pytorch/autograd_detach/faulty/training.py"
    ).read_bytes()
    assert {c.split for c in load_manifest().cases} == {
        "development",
        "validation",
        "holdout",
    }
    import researchcoder.evaluation.dataset as module

    monkeypatch.setattr(module, "ORACLE", tmp_path / "fake")
    (tmp_path / "fake").write_text("wrong")
    with pytest.raises(ValueError):
        load_manifest()
