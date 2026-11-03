"""Hand-authored synthetic faults. Source files and log text are fixtures, never executed."""

from __future__ import annotations

import json
from pathlib import Path

from .llm import LLMResponse, ToolCall


def create_cases(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=False)
    cases = []
    definitions = [
        ("c01", "dataset_path", "dataset_path_unavailable", 4),
        ("c02", "dataset_path", "dataset_path_unavailable", 4),
        ("c03", "cuda_oom", "allocation_exceeds_available_memory", 3),
        ("c04", "tensor_shape", "incompatible_channels", 5),
        ("c05", "tensor_shape", "incompatible_reshape", 4),
        ("c06", "numerical_nonfinite", "nonfinite_log_zero", 3),
        ("c07", "import_dependency", "missing_dependency", 2),
    ]
    for identifier, category, cause, line in definitions:
        workspace = root / identifier
        project = workspace / "project"
        project.mkdir(parents=True)
        entry = project / "entry.py"
        paths = []
        if category == "dataset_path":
            source = (
                "from pathlib import Path\ndef check_data(path):\n    if not Path(path).is_dir():\n"
                "        raise FileNotFoundError(f\"missing path '{path}'\")\n"
            )
            current = project / "dataset/images/val"
            previous = workspace / "previous/images/val" if identifier == "c01" else current
            if identifier == "c01":
                current.mkdir(parents=True)
            config = project / "data.yaml"
            config.write_text(f"path: '{project / 'dataset'}'\ntrain: images/train\nval: images/val\n", encoding="utf-8")
            error = f"FileNotFoundError: Dataset '{config}' images not found, missing path '{previous}'"
            paths = list(dict.fromkeys([str(previous), str(current)]))
        elif identifier == "c03":
            source = "import torch\ndef train_step(batch):\n    return torch.empty((batch, 4096, 4096), device='cuda')\n"
            error = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU has 8.00 GiB total and 100 MiB free."
        elif identifier == "c04":
            source = ("import torch\nconv = torch.nn.Conv2d(3, 32, 3)\nbatch = torch.randn(8, 1, 64, 64)\n"
                      "def train_step():\n    return conv(batch)\n")
            error = "RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[8, 1, 64, 64] to have 3 channels, but got 1 channels instead"
        elif identifier == "c05":
            source = "import torch\ndef train_step(batch):\n    features = torch.zeros((batch, 1024))\n    return features.view(-1, 13, 13)\n"
            error = "RuntimeError: shape '[-1, 13, 13]' is invalid for input of size 8192"
        elif identifier == "c06":
            source = "import numpy as np\ndef loss_step(probabilities, targets):\n    return -(targets * np.log(probabilities)).sum()\n"
            error = "step=4 prob_min=0.0 target_at_zero_prob=0 loss=nan gradient=nan\n"
        else:
            source = "import torch\nimport nonexistent_runtime_ext\ndef train_step():\n    return None\n"
            error = "ModuleNotFoundError: No module named 'nonexistent_runtime_ext'"
        entry.write_text(source, encoding="utf-8")
        log = workspace / "run.log"
        if identifier == "c06":
            text = "INFO starting training\nRuntimeWarning: divide by zero encountered in log\n" + error
        else:
            text = (f'Traceback (most recent call last):\n  File "{entry}", line {line}, in '
                    + ("<module>" if identifier == "c07" else "check_data" if category == "dataset_path" else "train_step")
                    + "\n    " + source.splitlines()[line - 1].strip() + "\n" + error + "\n")
        log.write_text(text, encoding="utf-8")
        case = {
            "id": identifier, "synthetic": True, "project": str(project), "log": str(log), "workspace": str(workspace),
            "source_hints": ["entry.py"], "expected_category": category, "expected_cause_code": cause,
            "expected_locations": [{"relative_path": "entry.py", "line_no": line}], "expected_path_checks": paths,
            "notice": "Hand-authored illustrative log/source fixture; no training was executed. Gold labels are outside project.",
        }
        # The diagnosis reader is restricted to project/, so it cannot read this annotation.
        (workspace / "annotation.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
        cases.append(case)
    return cases


class GoldReplayLLM:
    """Oracle fixture round-trip for testing the pipeline/grader, never a diagnosis baseline."""

    model = "gold-annotated-replay"

    def __init__(self, case):
        self.case = case
        self.evidence = []
        self.observations = {}
        self.step = 0

    def chat(self, messages, tools=None, on_token=None):
        latest = next((message for message in reversed(messages) if message["role"] == "tool"), None)
        data = json.loads(latest["content"]) if latest else {}
        self.evidence = data.get("evidence", self.evidence)
        self.observations.update({item["id"]: item for item in data.get("observations", [])})
        step = self.step
        self.step += 1
        if step == 0:
            return self.tool("extract_log_evidence", {})
        if step == 1:
            if any(item.get("frames") for item in self.evidence):
                return self.tool("locate_source", {"evidence_ids": [item["id"] for item in self.evidence]})
            return self.tool("read_project_file", {"path": "entry.py"})
        if step == 2:
            return self.tool("read_project_file", {"path": "entry.py"})
        if step == 3 and self.case["expected_path_checks"]:
            return self.tool("inspect_paths", {"paths": self.case["expected_path_checks"]})
        source = next(item for item in self.observations.values() if item["kind"] == "file_excerpt"
                      and item["data"].get("resolved_path") == str(Path(self.case["project"]) / "entry.py"))
        evidence_ids = [item["id"] for item in self.evidence]
        observation_ids = list(self.observations)
        report = {
            "summary": "Annotated offline replay for fixture and grader validation only.",
            "historical_failure": {"description": "Failure recorded in the constructed log.", "evidence_ids": evidence_ids},
            "current_state": {"assessment": "Project files were inspected; training was not executed.", "observation_ids": observation_ids},
            "hypotheses": [{"explanation": "The fixture annotation supplies this test prediction.", "evidence_ids": evidence_ids,
                            "observation_ids": [source["id"]], "verification": "Check with a real model on independent cases."}],
            "recommended_actions": ["Review the source and verify independently."],
            "limitations": ["Gold replay uses annotations; it cannot measure model accuracy."],
            "failure_category": self.case["expected_category"], "cause_code": self.case["expected_cause_code"],
            "primary_locations": [{"observation_id": source["id"], "line_no": self.case["expected_locations"][0]["line_no"]}],
        }
        return LLMResponse(content=json.dumps(report))

    def tool(self, name, arguments):
        return LLMResponse(tool_calls=[ToolCall(f"replay_{self.step}", name, arguments)])
