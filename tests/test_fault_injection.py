"""Fixture isolation, verified controls and reproducible failure records without provider calls."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from corecoder import fault_injection as faults


@pytest.mark.parametrize("tool,args,blocked", [
    ("grep", {"pattern": "def", "path": "."}, False),
    ("glob", {"pattern": "**/*.py"}, False),
    ("grep", {"pattern": "def", "path": ".."}, True),
    ("glob", {"pattern": "../*"}, True),
    ("write_file", {"file_path": "run_experiment.py"}, True),
    ("edit_file", {"file_path": "ratio.py"}, False),
    ("write_file", {"file_path": "new.py"}, True),
])
def test_acceptance_scope(tmp_path, tool, args, blocked):
    from examples.m2d_fault_acceptance import acceptance_guard
    rejected = []
    guard = acceptance_guard({"project": str(tmp_path), "manifest": {"ratio.py": "hash", "run_experiment.py": "hash"}}, rejected)
    if blocked:
        with pytest.raises(ValueError, match="scope violation"):
            guard(tool, args)
        assert rejected
    else:
        guard(tool, args)
        assert not rejected


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "ultralytics/nn/modules").mkdir(parents=True)
    (root / "ultralytics/utils").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data/DroneVehicle.yaml").write_text('{"names": ["car", "bus"], "nc": 2}')
    (root / "ultralytics/nn/modules/conv.py").write_text(
        "import numpy as np\n"
        "self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d))\n"
        "def __init__(self, dimension=1):\n    pass\n")
    (root / "ultralytics/utils/metrics.py").write_text("def box_iou(box1, box2, eps=1e-7):\n    return eps\n")
    return root


def test_snapshot_includes_only_required_project_sources(source):
    (source / ".env").write_text("secret=not-a-real-key")
    (source / "dataset").mkdir()
    (source / "dataset/image.bin").write_bytes(b"data")
    (source / "ultralytics/weights.pt").write_bytes(b"weights")
    snapshot = faults.snapshot_sources(source)
    assert len(snapshot) == 3
    assert all(not path.endswith((".env", ".pt", ".bin")) for path in snapshot)


@pytest.mark.parametrize("content", ["absent", "needle needle"])
def test_mutation_requires_exactly_one_match_before_writing(tmp_path, content):
    path = tmp_path / "file.py"
    path.write_text(content)
    with pytest.raises(ValueError, match="exactly one"):
        faults.patch_once(path, "needle", "replacement")
    assert path.read_text() == content


def test_child_environment_is_cpu_only_and_drops_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    environment = faults.child_environment(tmp_path, tmp_path / "settings")
    assert "DEEPSEEK_API_KEY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "DEEPSEEK_API_KEY" in __import__("os").environ  # Caller configuration is preserved.


@pytest.mark.parametrize("location", ["inside", "same", "parent"])
def test_output_must_not_contain_or_be_inside_original_source(source, location):
    output = {"inside": source / "cases", "same": source, "parent": source.parent}[location]
    before = faults.snapshot_sources(source)
    with pytest.raises(ValueError, match="separate"):
        faults.prepare_cases(source, output, sys.executable)
    assert faults.snapshot_sources(source) == before


def mock_runtime(monkeypatch, healthy_passes=True):
    monkeypatch.setattr(faults.subprocess, "check_output", lambda *args, **kwargs: '{"names": ["car", "bus"], "nc": 2}')
    calls = []
    def execute(command, project, environment, timeout=35):
        calls.append(command)
        phase = (len(calls) - 1) % 3
        return {"exit_code": 0 if phase == 0 and healthy_passes else 1,
                "stdout": "iou=nan\n" if phase == 1 else "", "stderr": "RuntimeError: injected\n" if phase else "",
                "timed_out": False}
    monkeypatch.setattr(faults, "execute", execute)
    return calls


def test_preparation_keeps_oracles_outside_all_agent_projects_and_original_intact(source, tmp_path, monkeypatch):
    calls = mock_runtime(monkeypatch)
    before = faults.snapshot_sources(source)
    output = tmp_path / "suite"
    cases = faults.prepare_cases(source, output, sys.executable)
    assert len(cases) == 6 and len(calls) == 18
    assert faults.snapshot_sources(source) == before
    for case in cases:
        project = Path(case["project"])
        assert not Path(case["annotation"]).is_relative_to(project)
        assert not Path(case["command"][2]).is_relative_to(project)
        annotation = json.loads(Path(case["annotation"]).read_text())
        assert annotation["healthy_control"]["exit_code"] == 0
        assert annotation["injected_control"]["exit_code"] != 0
        assert annotation["healthy_target_sha256"] != annotation["injected_target_sha256"]
        assert annotation["injected_lines"]
        assert not (project / "evaluator").exists()
    manifest = (output / "evaluator/cases.json").read_text()
    assert '"environment"' not in manifest


def test_failing_healthy_control_stops_before_injection(source, tmp_path, monkeypatch):
    mock_runtime(monkeypatch, healthy_passes=False)
    output = tmp_path / "suite"
    with pytest.raises(ValueError, match="Healthy control failed"):
        faults.prepare_cases(source, output, sys.executable)
    healthy_config = json.loads((output / "cases/m01/project/data/DroneVehicle.yaml").read_text())
    assert healthy_config["val"] == "images/val"
    assert not (output / "evaluator/m01-annotation.json").exists()


def test_timeout_retains_available_output_without_fabricating_exit_code(tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial output", stderr=b"partial error")
    monkeypatch.setattr(faults.subprocess, "run", timeout)
    result = faults.execute([sys.executable], tmp_path, {})
    assert result["timed_out"] and result["exit_code"] is None
    assert result["stdout"] == "partial output" and result["stderr"] == "partial error"
