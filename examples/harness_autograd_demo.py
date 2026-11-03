"""Sequential isolated Harness attempts; fixed choices and real CPU autograd."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from corecoder.workflow import run_verification, workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import TaskSpec, TaskInput, WorkspaceRef, ExecutionPolicy, VerificationSpec, DiagnosticOptions
from researchcoder.runtime import ResearchHarness, ExecutionConfig
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, ScriptedCall, ProtectedFile


def call(identity, name, arguments):
    return ScriptedTurn(calls=(ScriptedCall(call_id=identity, name=name, arguments_json=json.dumps(arguments)),))


def run_demo(directory=None, *, torch_python=None):
    directory = (
        Path(tempfile.mkdtemp(prefix="researchcoder-harness-autograd-")) if directory is None else Path(directory).resolve()
    )
    directory.mkdir(parents=True, exist_ok=True)
    project = directory / "candidate"
    project.mkdir(exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1] / "benchmarks/pytorch/autograd_detach"
    shutil.copyfile(fixtures / "faulty/training.py", project / "training.py")
    python = torch_python or os.environ.get("RESEARCHCODER_EXPERIMENT_PYTHON")
    if not python:
        raise ValueError("Existing Torch interpreter required")
    oracle = fixtures / "oracle.py"
    manifest = fixtures / "case.json"
    protected = tuple(ProtectedFile(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (oracle, manifest))
    failed = run_verification((python, "-B", str(oracle), str(project), "--seed-group", "discovery"), project)
    assert failed["status"] == "failed"
    io = ArtifactIO(directory / "artifacts")
    log = io.put_bytes("autograd", failed["stdout"]["text"].encode(), media_type="text/plain")
    workspace = WorkspaceRef(workspace_id="candidate", root=str(project), source_manifest_sha256=workspace_revision(project))
    report = {
        "summary": "Scripted evidence-based hypothesis: unintended detach disconnects joint training.",
        "historical_failure": {"description": "The CPU probe reports missing backbone gradients.", "evidence_ids": ["$E0"]},
        "current_state": {"assessment": "Actual training_step detaches backbone features.", "observation_ids": ["$O0"]},
        "hypotheses": [
            {
                "explanation": "Joint training requires gradients through the backbone.",
                "evidence_ids": ["$E0"],
                "observation_ids": ["$O0"],
                "verification": "Use the independent gradient/update oracle.",
            }
        ],
        "recommended_actions": ["Remove the unwanted detach while preserving the optimizer contract."],
        "limitations": ["Fixed ScriptedLLM decisions do not measure LLM diagnostic accuracy."],
    }
    model = ModelSpec(
        turns=(
            call("extract", "extract_log_evidence", {}),
            call("source", "read_project_file", {"path": "training.py"}),
            ScriptedTurn(content=json.dumps(report)),
        )
    )
    policy = ExecutionPolicy(writable_paths=("training.py",), allow_verification=True)
    config = ExecutionConfig(
        artifact_root=str(io.root), allowed_workspaces=(str(project),), max_policy=policy, model=model, protected_files=protected
    )
    diagnosis_spec = TaskSpec(
        job_id="autograd",
        task_id="diagnose",
        kind="diagnose",
        objective="Inspect joint-training gradients",
        workspace=workspace,
        inputs=(TaskInput(role="raw_log", reference=log),),
        diagnostic_options=DiagnosticOptions(source_hints=("training.py",)),
    )
    harness = ResearchHarness()
    diagnosis = harness.execute(diagnosis_spec, config)
    assert diagnosis.status == "succeeded", diagnosis.to_json()
    verify = VerificationSpec(
        argv=(python, "-B", str(oracle), str(project), "--seed-group", "holdout"), oracle_profile="joint-gradient-oracle"
    )
    repair_model = ModelSpec(
        turns=(
            call("read", "read_file", {"file_path": "training.py"}),
            call(
                "edit",
                "edit_file",
                {
                    "file_path": "training.py",
                    "old_string": "features = model.backbone(x).detach()",
                    "new_string": "features = model.backbone(x)",
                },
            ),
            ScriptedTurn(content="Edit complete."),
        )
    )
    repair_spec = TaskSpec(
        job_id="autograd",
        task_id="repair",
        kind="repair",
        objective="Repair the jointly trained backbone",
        workspace=workspace,
        inputs=(TaskInput(role="diagnosis_run", reference=diagnosis.outputs[0]),),
        policy=policy,
        verification=verify,
    )
    repair = harness.execute(repair_spec, replace(config, model=repair_model, permitted_verifications=(verify,)))
    assert repair.status == "succeeded" and repair.verification.status == "passed", repair.to_json()
    close_trace = harness.close_job("autograd")
    records = [harness.records[result.attempt_id] for result in (diagnosis, repair)]
    assert records[0]["worker_pid"] != records[1]["worker_pid"]
    result = {
        "mode": "offline_scripted",
        "provider_requests": 0,
        "initial_status": failed["status"],
        "diagnosis": diagnosis.to_dict(),
        "repair": repair.to_dict(),
        "records": records,
        "closed_job": harness.repository.job("autograd").status,
        "job_close_trace": close_trace.to_dict(),
        "scope": "Sequential isolated workers, not DAG or Multi-Agent scheduling.",
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir), indent=2))
