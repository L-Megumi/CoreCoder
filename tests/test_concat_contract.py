"""Real Torch counterexamples reject the shape-dependent fallback seen in live acceptance."""

import json
import os
import subprocess

import pytest

from corecoder.fault_injection import VERIFIER, child_environment


@pytest.mark.parametrize("dimension,fallback", [(1, False), (0, True), (1, True)])
def test_external_concat_oracle_checks_default_and_explicit_dimension_contract(tmp_path, dimension, fallback):
    python = os.getenv("RESEARCHCODER_EXPERIMENT_PYTHON")
    if not python:
        pytest.skip("Set RESEARCHCODER_EXPERIMENT_PYTHON to a Torch environment for this integration check")
    project = tmp_path / "project"
    for name in ("ultralytics", "ultralytics/nn", "ultralytics/nn/modules"):
        directory = project / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("")
    source = ("import torch\nclass Concat(torch.nn.Module):\n"
              f"    def __init__(self,dimension={dimension}):\n        super().__init__()\n        self.d=dimension\n"
              "    def forward(self,x):\n")
    if fallback:
        source += ("        if self.d==0:\n"
                   "            shapes=[tuple(t.shape) for t in x]\n"
                   "            batch_ok=all(s[1:]==shapes[0][1:] for s in shapes)\n"
                   "            channel_ok=all(s[0]==shapes[0][0] and s[2:]==shapes[0][2:] for s in shapes)\n"
                   "            if not batch_ok and channel_ok:\n                return torch.cat(x,1)\n")
    source += "        return torch.cat(x,self.d)\n"
    (project / "ultralytics/nn/modules/conv.py").write_text(source)
    verifier = tmp_path / "verifier.py"
    verifier.write_text(VERIFIER)
    expectations = tmp_path / "expect.json"
    expectations.write_text(json.dumps({"operation": "concat"}))
    result = subprocess.run([python, "-B", str(verifier), str(project), str(expectations)], cwd=project,
                            env=child_environment(project, tmp_path / "settings"), capture_output=True, text=True,
                            timeout=35, check=False)
    assert (result.returncode != 0) == fallback, result.stdout + result.stderr
    assert "independent_checks=passed" in result.stdout if not fallback else "AssertionError" in result.stderr
