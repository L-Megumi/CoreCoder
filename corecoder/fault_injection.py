"""Real-source CPU fault fixtures; annotations and verification stay outside Agent projects."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import subprocess
from pathlib import Path

RECIPES = (
    ("m01", "dataset_path", "data/DroneVehicle.yaml", None, None),
    ("m02", "dataset_schema", "data/DroneVehicle.yaml", None, None),
    ("m03", "conv", "ultralytics/nn/modules/conv.py", "nn.Conv2d(c1, c2, k, s, autopad(k, p, d)", "nn.Conv2d(c1 + 1, c2, k, s, autopad(k, p, d)"),
    ("m04", "concat", "ultralytics/nn/modules/conv.py", "def __init__(self, dimension=1):", "def __init__(self, dimension=0):"),
    ("m05", "iou", "ultralytics/utils/metrics.py", "def box_iou(box1, box2, eps=1e-7):", "def box_iou(box1, box2, eps=0.0):"),
    ("m06", "import", "ultralytics/nn/modules/conv.py", "import numpy as np", "import numpy_missing_local as np"),
)

LAUNCHER = '''"""Run a CPU-sized operation from the copied project; no training or downloads."""
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
settings = json.loads(Path("scenario.json").read_text())
kind = settings["operation"]
if kind == "dataset":
    from ultralytics.data import utils as data_utils
    data_utils.check_font = lambda *args, **kwargs: None  # Disable unrelated font downloads.
    result = data_utils.check_det_dataset("data/DroneVehicle.yaml", autodownload=False)
    print("dataset_check=ok nc=" + str(result["nc"]))
elif kind in ("conv", "import"):
    from ultralytics.nn.modules.conv import Conv
    result = Conv(6, 8).eval()(torch.ones(2, 6, 8, 8))
    print("conv_shape=" + str(tuple(result.shape)))
elif kind == "concat":
    from ultralytics.nn.modules.conv import Concat
    result = Concat()([torch.ones(2, 1, 4, 4), torch.ones(2, 2, 4, 4)])
    print("concat_shape=" + str(tuple(result.shape)))
elif kind == "iou":
    from ultralytics.utils.metrics import box_iou
    result = box_iou(torch.zeros(1, 4), torch.zeros(1, 4))
    print("iou=" + str(result.item()))  # Keep a genuine frameless numerical signal.
'''

VERIFIER = '''"""Independent semantic checks, kept outside each Agent project."""
import json
import os
import sys
from pathlib import Path
for name in list(os.environ):
    if name.endswith("_API_KEY"):
        del os.environ[name]
root = Path(sys.argv[1]).resolve()
expect = json.loads(Path(sys.argv[2]).read_text())
os.chdir(root)
sys.path.insert(0, str(root))
import torch
torch.set_num_threads(1)
kind = expect["operation"]
if kind == "dataset":
    from ultralytics.data import utils as data_utils
    data_utils.check_font = lambda *args, **kwargs: None
    result = data_utils.check_det_dataset("data/DroneVehicle.yaml", autodownload=False)
    assert Path(result["val"]) == root / "sample_data/images/val"
    assert Path(result["train"]) == root / "sample_data/images/train"
    assert result["nc"] == len(expect["class_names"])
    assert list(result["names"].values()) == expect["class_names"]
elif kind in ("conv", "import"):
    from ultralytics.nn.modules.conv import Conv
    for c1, c2, batch, size in ((6, 8, 2, 8), (3, 4, 1, 5)):
        layer = Conv(c1, c2).eval()
        assert layer.conv.in_channels == c1
        assert tuple(layer.conv.weight.shape[:2]) == (c2, c1)
        inputs = torch.ones(batch, c1, size, size)
        result = layer(inputs)
        assert tuple(result.shape) == (batch, c2, size, size)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, layer.act(layer.bn(layer.conv(inputs))))
elif kind == "concat":
    from ultralytics.nn.modules.conv import Concat
    a, b = torch.ones(2, 1, 4, 4), torch.full((2, 2, 4, 4), 2.0)
    result = Concat()([a, b])
    torch.testing.assert_close(result, torch.cat([a, b], 1))
    # Equal channel sizes must still use the default channel dimension, not a shape-dependent fallback.
    a, b = torch.ones(2, 1, 4, 4), torch.full((2, 1, 4, 4), 2.0)
    torch.testing.assert_close(Concat()([a, b]), torch.cat([a, b], 1))
    a, b = torch.ones(1, 2, 3, 4), torch.full((2, 2, 3, 4), 3.0)
    torch.testing.assert_close(Concat(0)([a, b]), torch.cat([a, b], 0))
    a, b = torch.ones(2, 1, 4, 4), torch.ones(2, 2, 4, 4)
    try:
        Concat(0)([a, b])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Explicit dimension 0 must not silently fall back to dimension 1")
    a, b = torch.ones(1, 2, 3, 4), torch.full((1, 2, 5, 4), 3.0)
    torch.testing.assert_close(Concat(2)([a, b]), torch.cat([a, b], 2))
elif kind == "iou":
    from ultralytics.utils.metrics import box_iou
    zero = box_iou(torch.zeros(1, 4), torch.zeros(1, 4))
    assert torch.isfinite(zero).all() and torch.all(zero == 0)
    boxes = torch.tensor([[0., 0., 1., 1.], [1., 1., 3., 3.]])
    result = box_iou(boxes, boxes)
    assert tuple(result.shape) == (2, 2) and torch.isfinite(result).all()
    torch.testing.assert_close(result.diag(), torch.ones(2), rtol=1e-5, atol=1e-6)
    assert result[0, 1] == 0 and result[1, 0] == 0
    overlap = box_iou(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([[1., 1., 3., 3.]]))
    torch.testing.assert_close(overlap, torch.tensor([[1. / 7.]]), rtol=1e-5, atol=1e-6)
print("independent_checks=passed")
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot_sources(source):
    source = Path(source).resolve()
    files = sorted(path for path in (source / "ultralytics").rglob("*")
                   if path.is_file() and not path.is_symlink() and path.suffix in (".py", ".yaml", ".yml"))
    files.append(source / "data/DroneVehicle.yaml")
    if not files or not all(path.resolve().is_relative_to(source) for path in files):
        raise ValueError("Invalid source project or symlink escape")
    return {path.relative_to(source).as_posix(): digest(path) for path in files}


def patch_once(path, before, after):
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.count(before) != 1:
        raise ValueError(f"Source mutation requires exactly one occurrence: {path}")
    Path(path).write_text(text.replace(before, after), encoding="utf-8")


def child_environment(project, settings):
    inherited = {name: value for name, value in os.environ.items() if not name.endswith("_API_KEY")}
    return {**inherited, "PYTHONPATH": str(project), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "YOLO_CONFIG_DIR": str(settings), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def execute(command, project, environment, timeout=35):
    try:
        result = subprocess.run(command, cwd=project, env=environment, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        return {"exit_code": None, "stdout": decode(exc.stdout), "stderr": decode(exc.stderr), "timed_out": True}


def prepare_cases(source, output, python):
    """Healthy check -> actual mutation -> actual failure, before any model sees a case."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Injection output must be separate from the original experiment project")
    original = snapshot_sources(source)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "original-source-hashes.json", original)
    raw_config = subprocess.check_output([str(python), "-B", "-c",
        "import json,sys,yaml; print(json.dumps(yaml.safe_load(sys.stdin.read())))"],
        input=(source / "data/DroneVehicle.yaml").read_text(encoding="utf-8-sig"), text=True)
    config = json.loads(raw_config)
    names = config["names"]
    names = [names[key] for key in sorted(names, key=int)] if isinstance(names, dict) else names
    verifier = output / "evaluator/verify.py"
    verifier.parent.mkdir()
    verifier.write_text(VERIFIER, encoding="utf-8")
    cases = []
    for identifier, kind, target, before, after in RECIPES:
        project = output / "cases" / identifier / "project"
        project.mkdir(parents=True)
        for relative in original:
            if relative.startswith("ultralytics/"):
                destination = project / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((source / relative).read_bytes())
                if digest(destination) != original[relative]:
                    raise ValueError("Original source changed during fixture creation")
        (project / "data").mkdir()
        for name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
            if (source / name).is_file():
                (project / name).write_bytes((source / name).read_bytes())
        (project / "sample_data/images/train").mkdir(parents=True)
        (project / "sample_data/images/val").mkdir(parents=True)
        dataset = {**config, "path": str(project / "sample_data"), "train": "images/train", "val": "images/val"}
        dataset.pop("download", None)
        write_json(project / "data/DroneVehicle.yaml", dataset)  # JSON is a valid YAML subset.
        operation = "dataset" if kind.startswith("dataset_") else kind
        write_json(project / "scenario.json", {"operation": operation})
        (project / "run_experiment.py").write_text(LAUNCHER, encoding="utf-8")
        annotation = output / "evaluator" / (identifier + ".json")
        write_json(annotation, {"operation": operation, "class_names": names})
        environment = child_environment(project, output / "runtime_settings" / identifier)
        command = [str(python), "-B", str(verifier), str(project), str(annotation)]
        healthy = execute(command, project, environment)
        if healthy["exit_code"] != 0:
            write_json(output / "evaluator" / (identifier + "-preparation-error.json"), healthy)
            raise ValueError(f"Healthy control failed for {identifier}; no fault has been injected into this case")
        target_path = project / target
        healthy_hash = digest(target_path)
        healthy_lines = target_path.read_text(encoding="utf-8-sig").splitlines()
        if kind.startswith("dataset_"):
            if kind == "dataset_path":
                dataset["val"] = "images/validation_stale"
            else:
                dataset["nc"] = len(names) + 1
            write_json(target_path, dataset)
        else:
            patch_once(target_path, before, after)
        log_result = execute([str(python), "-B", "run_experiment.py"], project, environment)
        failed_check = execute(command, project, environment)
        if failed_check["exit_code"] in (0, None) or log_result["timed_out"]:
            raise ValueError(f"Injected fault was not a reproducible semantic failure: {identifier}")
        log = output / "logs" / (identifier + ".log")
        log.parent.mkdir(exist_ok=True)
        log.write_text(log_result["stdout"] + log_result["stderr"], encoding="utf-8")
        gold = {"case_id": identifier, "fault_kind": kind, "target": target,
                "healthy_target_sha256": healthy_hash, "injected_target_sha256": digest(target_path),
                "mutation_before": before, "mutation_after": after, "source_project": str(source),
                "source_origin_sha256": original.get(target), "healthy_control": healthy, "injected_control": failed_check,
                "frameless_numeric": kind == "iou", "fixture_scope": "Project-source CPU fault injection; not full training"}
        changed = []
        for tag, _, _, start, end in difflib.SequenceMatcher(
            a=healthy_lines, b=target_path.read_text(encoding="utf-8-sig").splitlines(), autojunk=False,
        ).get_opcodes():
            if tag != "equal":
                changed.extend(range(start + 1, end + 1))
        gold["injected_lines"] = changed
        write_json(output / "evaluator" / (identifier + "-annotation.json"), gold)
        manifest = snapshot_sources(project)
        manifest["run_experiment.py"] = digest(project / "run_experiment.py")
        manifest["scenario.json"] = digest(project / "scenario.json")
        cases.append({"id": identifier, "project": str(project), "log": str(log), "command": command,
                      "environment": environment, "manifest": manifest, "annotation": str(output / "evaluator" / (identifier + "-annotation.json"))})
        print(json.dumps({"prepared": identifier, "healthy_exit": healthy["exit_code"], "injected_exit": failed_check["exit_code"]}), flush=True)
    if snapshot_sources(source) != original:
        raise ValueError("Original project changed during preparation; review recorded source hashes")
    write_json(output / "evaluator/cases.json", [{key: value for key, value in case.items() if key != "environment"} for case in cases])
    return cases
