"""ScriptedLLM decisions, real CPU autograd, real repair and independent checks."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.workflow import run_verification, workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.adapters.corecoder import AdapterContext, CoreCoderAdapter
from researchcoder.contracts import DiagnosticOptions, ExecutionPolicy, TaskInput, TaskSpec, VerificationSpec, WorkspaceRef


class ReferencedScript(ScriptedLLM):
    """Fixed choices; fill final IDs only from actual tool replies."""

    def __init__(self):
        super().__init__(
            [
                LLMResponse(tool_calls=[ToolCall("extract", "extract_log_evidence", {})]),
                LLMResponse(tool_calls=[ToolCall("source", "read_project_file", {"path": "training.py"})]),
                LLMResponse(content="placeholder"),
            ]
        )

    def chat(self, messages, tools=None, on_token=None):
        if len(self._turns) == 1:
            records = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
            evidence = [r["id"] for result in records for r in result.get("evidence", [])]
            observations = [r["id"] for result in records for r in result.get("observations", [])]
            report = {
                "summary": "Scripted hypothesis: detach disconnects the jointly trained backbone.",
                "historical_failure": {
                    "description": "The actual CPU probe reports missing joint-training gradients.",
                    "evidence_ids": evidence,
                },
                "current_state": {
                    "assessment": "Read training_step; backbone features are detached before the head.",
                    "observation_ids": observations,
                },
                "hypotheses": [
                    {
                        "explanation": "detach may cut the required gradient path to the backbone.",
                        "evidence_ids": evidence,
                        "observation_ids": observations,
                        "verification": "Check both component gradients and actual updates with an independent oracle.",
                    }
                ],
                "recommended_actions": ["Remove the unwanted detach under the joint-training contract."],
                "limitations": ["Scripted decisions are not a measurement of LLM diagnostic accuracy."],
            }
            self._turns[0] = LLMResponse(content=json.dumps(report))
        return super().chat(messages, tools, on_token)


def run_demo(directory=None, *, torch_python=None):
    directory = Path(tempfile.mkdtemp(prefix="researchcoder-autograd-")) if directory is None else Path(directory).resolve()
    if not directory.exists():
        directory.mkdir(parents=True)
    project = directory / "candidate"
    project.mkdir(exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1] / "benchmarks/pytorch/autograd_detach"
    oracle = fixtures / "oracle.py"
    manifest = fixtures / "case.json"
    original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (oracle, manifest)}
    shutil.copyfile(fixtures / "faulty/training.py", project / "training.py")
    python = torch_python or os.environ.get("RESEARCHCODER_EXPERIMENT_PYTHON")
    if not python:
        raise ValueError("Provide the existing CPU Torch interpreter")
    discovery = (python, "-B", str(oracle), str(project), "--seed-group", "discovery")
    initial = run_verification(discovery, project)
    if initial["status"] != "failed":
        raise ValueError("Faulty control must fail the independent gradient oracle")
    io = ArtifactIO(directory / "artifacts")
    log = io.put_bytes("autograd", initial["stdout"]["text"].encode(), media_type="text/plain")
    workspace = WorkspaceRef(workspace_id="candidate", root=str(project), source_manifest_sha256=workspace_revision(project))
    diagnose = TaskSpec(
        job_id="autograd",
        task_id="diagnose",
        kind="diagnose",
        objective="Investigate missing backbone gradients under joint training.",
        workspace=workspace,
        inputs=(TaskInput(role="raw_log", reference=log),),
        diagnostic_options=DiagnosticOptions(source_hints=("training.py",)),
    )
    adapter = CoreCoderAdapter()
    diagnosis = adapter.execute(diagnose, AdapterContext(attempt_id="diagnose-1", artifacts=io, model_factory=ReferencedScript))
    if diagnosis.status != "succeeded":
        raise ValueError("Scripted diagnosis did not complete: " + diagnosis.to_json())
    verification = VerificationSpec(
        argv=(python, "-B", str(oracle), str(project), "--seed-group", "holdout"), oracle_profile="joint-gradient-oracle"
    )

    def guard(command, root):
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in original.items()):
            raise ValueError("Protected oracle or manifest changed")

    def repair_model():
        return ScriptedLLM(
            [
                LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "training.py"})]),
                LLMResponse(
                    tool_calls=[
                        ToolCall(
                            "edit",
                            "edit_file",
                            {
                                "file_path": "training.py",
                                "old_string": "features = model.backbone(x).detach()",
                                "new_string": "features = model.backbone(x)",
                            },
                        )
                    ]
                ),
                LLMResponse(content="Removed unwanted detach; external oracle will check held-out inputs."),
            ]
        )

    repair = TaskSpec(
        job_id="autograd",
        task_id="repair",
        kind="repair",
        objective="Repair joint training; preserve the independent oracle and optimizer contract.",
        workspace=workspace,
        inputs=(TaskInput(role="diagnosis_run", reference=diagnosis.outputs[0]),),
        policy=ExecutionPolicy(writable_paths=("training.py",), allow_verification=True),
        verification=verification,
    )
    repaired = adapter.execute(
        repair,
        AdapterContext(
            attempt_id="repair-1",
            artifacts=io,
            model_factory=repair_model,
            permitted_verifications=(verification,),
            verification_guard=guard,
        ),
    )
    guard(verification.argv, project)
    if repaired.status != "succeeded" or repaired.verification.status != "passed":
        raise ValueError("Repair did not pass held-out oracle: " + repaired.to_json())
    controls = {}
    for name, profile in [("healthy", "joint"), ("faulty", "joint"), ("frozen", "frozen")]:
        control = directory / name
        control.mkdir()
        shutil.copyfile(fixtures / name / "training.py", control / "training.py")
        controls[name] = run_verification((python, "-B", str(oracle), str(control), "--profile", profile), control)
    assert [controls[n]["status"] for n in ("healthy", "faulty", "frozen")] == ["passed", "failed", "passed"]
    result = {
        "mode": "offline_scripted",
        "provider_requests": 0,
        "initial_status": initial["status"],
        "diagnosis": diagnosis.to_dict(),
        "repair": repaired.to_dict(),
        "controls": controls,
        "oracle_protected": True,
        "scope": "ScriptedLLM plumbing acceptance; real CPU gradients and file changes, not LLM accuracy.",
    }
    (directory / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir), ensure_ascii=False, indent=2))
