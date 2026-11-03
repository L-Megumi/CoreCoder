"""Real specialist processes, real tools and held-out CPU oracle; fixed offline choices."""

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
from researchcoder.runtime import ExecutionConfig
from researchcoder.runtime.protocol import ModelSpec, ScriptedTurn, ScriptedCall, RuntimeLimits, ProtectedFile
from researchcoder.orchestration import (
    DAGScheduler,
    SchedulerConfig,
    TaskTemplate,
    PlanRevision,
    OrchestrationTask,
    TaskDependency,
)
from researchcoder.orchestration.workspaces import SourceSnapshot
from researchcoder.orchestration.budget import JobLimits


def call(identity, name, arguments=None):
    return ScriptedTurn(calls=(ScriptedCall(call_id=identity, name=name, arguments_json=json.dumps(arguments or {})),))


def report():
    return ScriptedTurn(
        content=json.dumps(
            {
                "summary": "Unintended detach interrupts jointly trained backbone gradients; confirm with independent oracle.",
                "historical_failure": {
                    "description": "The executed CPU oracle reports missing backbone gradients.",
                    "evidence_ids": ["$E0"],
                },
                "current_state": {
                    "assessment": "Actual training source inspected for joint-gradient flow.",
                    "observation_ids": ["$O0"],
                },
                "hypotheses": [
                    {
                        "explanation": "Detached backbone features prevent joint training.",
                        "evidence_ids": ["$E0"],
                        "observation_ids": ["$O0"],
                        "verification": "Independent held-out gradient and parameter-update oracle.",
                    }
                ],
                "recommended_actions": ["Remove unwanted detach while keeping oracle, optimizer and joint-training contract."],
                "limitations": ["Fixed offline model choices; not measured LLM diagnosis accuracy."],
            }
        )
    )


def prepare_demo(directory, *, torch_python=None, max_workers=2, bad_patch=False):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    project = directory / "candidate"
    project.mkdir(exist_ok=False)
    fixtures = Path(__file__).resolve().parents[1] / "benchmarks/pytorch/autograd_detach"
    shutil.copyfile(fixtures / "faulty/training.py", project / "training.py")
    python = torch_python or os.environ.get("RESEARCHCODER_EXPERIMENT_PYTHON")
    if not python:
        raise ValueError("Existing CPU Torch interpreter required")
    oracle = fixtures / "oracle.py"
    initial = run_verification((python, "-B", str(oracle), str(project), "--seed-group", "discovery"), project)
    if initial["status"] != "failed":
        raise ValueError("Expected reproducible initial fault")
    io = ArtifactIO(directory / "artifacts")
    log = io.put_bytes("autograd", initial["stdout"]["text"].encode(), media_type="text/plain")
    snapshot = SourceSnapshot.create(project, directory / "source-snapshot")
    workspace = WorkspaceRef(workspace_id="candidate", root=str(project), source_manifest_sha256=workspace_revision(project))
    readonly = ExecutionPolicy(allowed_tool_names=("read_file",))
    write = ExecutionPolicy(allowed_tool_names=("read_file", "edit_file"), writable_paths=("training.py",))
    verify_policy = ExecutionPolicy(allowed_tool_names=(), allow_verification=True)
    protected = tuple(
        ProtectedFile(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (oracle, fixtures / "case.json")
    )
    limits = RuntimeLimits(max_model_calls=8, max_tool_calls=8, max_rounds=8, max_total_tokens=80000, max_runtime_seconds=30)
    base = ExecutionConfig(
        artifact_root=str(io.root), allowed_workspaces=(str(project),), max_policy=write, limits=limits, protected_files=protected
    )
    diag = TaskSpec(
        job_id="autograd",
        task_id="base",
        kind="diagnose",
        objective="Inspect gradient flow",
        workspace=workspace,
        inputs=(TaskInput(role="raw_log", reference=log),),
        policy=readonly,
        diagnostic_options=DiagnosticOptions(source_hints=("training.py",)),
    )
    models = {
        "log": ModelSpec(
            turns=(
                call("e", "extract_log_evidence"),
                call("l", "locate_source", {"evidence_ids": ["$E0"]}),
                call("r", "read_project_file", {"path": "training.py"}),
                report(),
            )
        ),
        "code": ModelSpec(
            turns=(
                call("e", "extract_log_evidence"),
                call("r", "read_project_file", {"path": "training.py"}),
                call(
                    "s",
                    "resolve_project_symbol",
                    {"symbol": "training_step", "source_observation_id": "$O0", "evidence_ids": ["$E0"]},
                ),
                report(),
            )
        ),
        "diagnose": ModelSpec(
            turns=(call("e", "extract_log_evidence"), call("r", "read_project_file", {"path": "training.py"}), report())
        ),
        "repair": ModelSpec(
            turns=(
                call("r", "read_file", {"file_path": "training.py"}),
                call(
                    "w",
                    "edit_file",
                    {
                        "file_path": "training.py",
                        "old_string": "features = model.backbone(x).detach()",
                        "new_string": "features = model.backbone(x).clone().detach()"
                        if bad_patch
                        else "features = model.backbone(x)",
                    },
                ),
                ScriptedTurn(content="Authorized edit complete; verification is independent."),
            )
        ),
    }
    templates = {
        "log": TaskTemplate("log_analysis", diag, replace(base, model=models["log"])),
        "code": TaskTemplate("code_analysis", diag, replace(base, model=models["code"])),
        "diagnose": TaskTemplate("diagnosis", diag, replace(base, model=models["diagnose"]), "candidate"),
    }
    repair = TaskSpec(
        job_id="autograd",
        task_id="repair",
        kind="repair",
        objective="Repair joint training",
        workspace=workspace,
        inputs=(TaskInput(role="diagnosis_run", reference=log),),
        policy=write,
    )
    templates["repair"] = TaskTemplate("repair", repair, replace(base, model=models["repair"]), "candidate")
    command = VerificationSpec(
        argv=(python, "-B", str(oracle), str(project), "--seed-group", "holdout"), oracle_profile="joint-gradient-oracle"
    )
    verify = TaskSpec(
        job_id="autograd",
        task_id="verify",
        kind="verify",
        objective="Independent held-out oracle",
        workspace=workspace,
        policy=verify_policy,
        verification=command,
    )
    templates["verify"] = TaskTemplate(
        "verification",
        verify,
        replace(base, model=ModelSpec(), max_policy=verify_policy, permitted_verifications=(command,)),
        "candidate",
    )
    tasks = (
        OrchestrationTask(task_id="log", template_id="log", role="log_analysis"),
        OrchestrationTask(task_id="code", template_id="code", role="code_analysis"),
        OrchestrationTask(
            task_id="diagnose",
            template_id="diagnose",
            role="diagnosis",
            dependencies=(
                TaskDependency(task_id="log", input_role="evidence"),
                TaskDependency(task_id="code", input_role="evidence"),
            ),
        ),
        OrchestrationTask(
            task_id="repair",
            template_id="repair",
            role="repair",
            dependencies=(TaskDependency(task_id="diagnose", input_role="diagnosis_run"),),
        ),
        OrchestrationTask(
            task_id="verify", template_id="verify", role="verification", dependencies=(TaskDependency(task_id="repair"),)
        ),
    )
    plan = PlanRevision(job_id="autograd", revision_id="r0", tasks=tasks)
    config = SchedulerConfig(
        templates=templates,
        artifact_root=str(io.root),
        snapshot=snapshot,
        max_workers=max_workers,
        role_profiles=True,
        job_limits=JobLimits(max_model_calls=100, max_tool_calls=100, max_total_tokens=1500000, max_runtime_seconds=150),
    )
    return plan, config, io, initial


def run_demo(directory=None, *, torch_python=None, max_workers=2):
    directory = Path(tempfile.mkdtemp(prefix="researchcoder-multi-autograd-")) if directory is None else Path(directory)
    plan, config, io, initial = prepare_demo(directory, torch_python=torch_python, max_workers=max_workers)
    scheduler = DAGScheduler()
    result = scheduler.execute(plan, config)
    summary = io.read_json(result.summary_ref)
    output = dict(
        mode="offline_scripted",
        provider_requests=0,
        initial_status=initial["status"],
        result=result.to_dict(),
        execution=summary,
        scope="Real independent specialist processes/tools and oracle; no model accuracy claim.",
    )
    (directory / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output_dir, max_workers=args.max_workers), indent=2))
