import os
import pytest
from researchcoder.evaluation.dataset import load_manifest, materialize
from researchcoder.evaluation.verification import verify_candidate
from researchcoder.adapters.artifacts import ArtifactIO


@pytest.mark.parametrize(
    "case_id,source",
    [
        ("dtype_features", "def prepare_features(model,x):\n    return x.double()\n"),
        (
            "optimizer_omission",
            "import torch\ndef build_optimizer(model):\n    return torch.optim.SGD(model.parameters(),lr=0.5)\n",
        ),
        (
            "shape_transpose",
            "def predict(model,x):\n    model.weight.data.zero_()\n    model.bias.data.zero_()\n    return model(x)\n",
        ),
    ],
)
def test_oracle_rejects_hardcoded_dtype_wrong_lr_and_forward_state_mutation(
    tmp_path, case_id, source
):
    case = next(c for c in load_manifest().cases if c.case_id == case_id)
    project = materialize(case, tmp_path / "candidate")
    (project / "training.py").write_text(source)
    result = verify_candidate(
        case,
        project,
        os.environ["RESEARCHCODER_EXPERIMENT_PYTHON"],
        "holdout",
        ArtifactIO(tmp_path / "artifacts"),
        "job",
        "check",
    )
    assert result.verification.status == "failed"
