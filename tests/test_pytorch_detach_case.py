import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from examples.pytorch_detach_adapter_demo import run_demo

FIXTURES = Path(__file__).resolve().parents[1] / "benchmarks/pytorch/autograd_detach"


def torch_python():
    # This integration suite is mandatory in the declared Linux acceptance environment.
    python = os.environ.get("RESEARCHCODER_EXPERIMENT_PYTHON")
    if not python:
        raise RuntimeError("Set the existing Torch interpreter; this integration check must not silently skip")
    return python


def oracle(project, profile="joint"):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run(
        [torch_python(), "-B", str(FIXTURES / "oracle.py"), str(project), "--profile", profile],
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=35,
    )
    result = json.loads(r.stdout.splitlines()[-1])
    return r, result


@pytest.mark.parametrize(
    "version,profile,passed", [("healthy", "joint", True), ("faulty", "joint", False), ("frozen", "frozen", True)]
)
def test_independent_oracle_distinguishes_business_contracts(tmp_path, version, profile, passed):
    project = tmp_path / "candidate"
    project.mkdir()
    shutil.copyfile(FIXTURES / version / "training.py", project / "training.py")
    process, result = oracle(project, profile)
    assert (process.returncode == 0) == passed and result["passed"] == passed
    assert len(result["rows"]) == 4  # two held-out inputs, two actual optimizer steps each
    if version == "faulty":
        for row in result["rows"]:
            assert row["parameters"]["head.weight"]["gradient_ok"]
            assert row["parameters"]["head.weight"]["update_norm"] > 0
            assert row["parameters"]["backbone.0.weight"]["grad_present"] is False
    if version == "healthy":
        assert all(p["gradient_ok"] and p["update_norm"] > 0 for row in result["rows"] for p in row["parameters"].values())


@pytest.mark.parametrize(
    "cheat",
    [
        "model.backbone.requires_grad_(False)",
        "optimizer.param_groups[0]['params'] = list(model.head.parameters())",
    ],
)
def test_oracle_rejects_freezing_and_optimizer_group_shortcuts(tmp_path, cheat):
    project = tmp_path / "candidate"
    project.mkdir()
    text = (FIXTURES / "healthy/training.py").read_text()
    text = text.replace("    optimizer.zero_grad", f"    {cheat}\n    optimizer.zero_grad")
    (project / "training.py").write_text(text)
    process, result = oracle(project)
    assert process.returncode != 0 and not result["passed"]
    assert any("trainable set or optimizer" in failure for failure in result["failures"])


def test_scripted_adapter_end_to_end_keeps_oracle_and_manifest_unchanged(tmp_path):
    files = (FIXTURES / "oracle.py", FIXTURES / "case.json")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = run_demo(tmp_path / "demo", torch_python=torch_python())
    assert result["mode"] == "offline_scripted" and result["provider_requests"] == 0
    assert result["initial_status"] == "failed"
    assert result["diagnosis"]["status"] == "succeeded"
    assert result["repair"]["status"] == "succeeded"
    assert result["repair"]["verification"]["status"] == "passed"
    assert result["oracle_protected"]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    assert result["controls"]["faulty"]["status"] == "failed"
    assert result["controls"]["frozen"]["status"] == "passed"
