"""Trusted code builds the same authority on restart; never execute a DB-supplied client/config."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from examples.multi_agent_autograd_demo import call, report
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import (
    ArtifactRef,
    TaskSpec,
    TaskInput,
    WorkspaceRef,
    ExecutionPolicy,
    VerificationSpec,
    DiagnosticOptions,
)
from researchcoder.runtime import ExecutionConfig
from researchcoder.runtime.protocol import (
    RuntimeLimits,
    ModelSpec,
    ScriptedTurn,
    ProtectedFile,
)
from researchcoder.orchestration import (
    PlanRevision,
    OrchestrationTask,
    TaskDependency,
    TaskTemplate,
    SchedulerConfig,
)
from researchcoder.orchestration.workspaces import SourceSnapshot
from researchcoder.orchestration.budget import JobLimits

SOURCE = b"value = 0\n"
LOG = b"WARNING: backbone gradients missing\n"
ORACLE = """import pathlib,runpy,sys,time
project,marker,expected,delay=sys.argv[1:]
with pathlib.Path(marker).open('a') as f:
 f.write('verification invoked\\n');f.flush();__import__('os').fsync(f.fileno())
time.sleep(float(delay))
value=runpy.run_path(str(pathlib.Path(project)/'training.py'))['value']
assert value==int(expected),(value,expected)
print('independent functional oracle passed')
"""


def trusted_case(directory, mode, *, initialize=False):
    directory = Path(directory).resolve()
    root = directory / "candidate"
    snapshot = directory / "snapshot"
    original_hash = hashlib.sha256(SOURCE).hexdigest()
    version = hashlib.sha256(
        json.dumps(
            {"training.py": original_hash}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    io = ArtifactIO(directory / "artifacts")
    if initialize:
        root.mkdir(parents=True)
        (root / "training.py").write_bytes(SOURCE)
        (directory / "oracle.py").write_text(ORACLE)
        io.put_bytes(
            "recovery", LOG, artifact_id="original-log", media_type="text/plain"
        )
        SourceSnapshot.create(root, snapshot)
    log = ArtifactRef(
        job_id="recovery",
        artifact_id="original-log",
        storage_key="recovery/original-log.blob",
        sha256=hashlib.sha256(LOG).hexdigest(),
        size_bytes=len(LOG),
        media_type="text/plain",
    )
    workspace = WorkspaceRef(
        workspace_id="candidate", root=str(root), source_manifest_sha256=version
    )
    snap = SourceSnapshot(
        WorkspaceRef(
            workspace_id="source-snapshot",
            root=str(snapshot),
            source_manifest_sha256=version,
        ),
        (("training.py", original_hash),),
    )
    write = ExecutionPolicy(
        allowed_tool_names=("read_file", "edit_file"), writable_paths=("training.py",)
    )
    verify_policy = ExecutionPolicy(allowed_tool_names=(), allow_verification=True)
    protected = (
        ProtectedFile(
            path=str(directory / "oracle.py"),
            sha256=hashlib.sha256(ORACLE.encode()).hexdigest(),
        ),
    )
    limits = RuntimeLimits(
        max_model_calls=8,
        max_tool_calls=8,
        max_rounds=8,
        max_total_tokens=80000,
        max_runtime_seconds=15,
    )
    base = ExecutionConfig(
        artifact_root=str(io.root),
        allowed_workspaces=(str(root),),
        max_policy=write,
        limits=limits,
        protected_files=protected,
    )
    diagonal = ModelSpec(
        turns=(
            call("e", "extract_log_evidence"),
            call("r", "read_project_file", {"path": "training.py"}),
            report(),
        )
    )
    diag = TaskSpec(
        job_id="recovery",
        task_id="base",
        kind="diagnose",
        objective="Inspect bounded training source",
        workspace=workspace,
        inputs=(TaskInput(role="raw_log", reference=log),),
        policy=ExecutionPolicy(allowed_tool_names=("read_file",)),
        diagnostic_options=DiagnosticOptions(source_hints=("training.py",)),
    )
    templates = {
        name: TaskTemplate(role, diag, replace(base, model=diagonal), place)
        for name, role, place in (
            ("log", "log_analysis", "snapshot"),
            ("code", "code_analysis", "snapshot"),
            ("diagnose", "diagnosis", "candidate"),
        )
    }
    repair = TaskSpec(
        job_id="recovery",
        task_id="repair",
        kind="repair",
        objective="Apply exactly the authorized one-line repair",
        workspace=workspace,
        inputs=(TaskInput(role="diagnosis_run", reference=log),),
        policy=write,
    )
    model = ModelSpec(
        turns=(
            call("r", "read_file", {"file_path": "training.py"}),
            call(
                "w",
                "edit_file",
                {
                    "file_path": "training.py",
                    "old_string": "value = 0",
                    "new_string": "value = 1 # PATCH_APPLIED",
                },
            ),
            ScriptedTurn(content="Repair complete, independent verification remains."),
        )
    )
    templates["repair"] = TaskTemplate(
        "repair", repair, replace(base, model=model), "candidate"
    )
    second = ModelSpec(
        turns=(
            call("r", "read_file", {"file_path": "training.py"}),
            call(
                "w",
                "edit_file",
                {
                    "file_path": "training.py",
                    "old_string": "value = 1 # PATCH_APPLIED",
                    "new_string": "value = 2 # PATCH_APPLIED",
                },
            ),
            ScriptedTurn(content="Second authorized revision complete."),
        )
    )
    templates["repair_second"] = TaskTemplate(
        "repair", repair, replace(base, model=second), "candidate"
    )
    expected = (
        2 if mode in ("revision_before", "revision_after", "verification_failed") else 1
    )
    delay = 3 if mode == "verification_running" else 0
    command = VerificationSpec(
        argv=(
            sys.executable,
            "-B",
            str(directory / "oracle.py"),
            str(root),
            str(directory / "verify-count.txt"),
            str(expected),
            str(delay),
        ),
        timeout_seconds=8,
    )
    verify = TaskSpec(
        job_id="recovery",
        task_id="verify",
        kind="verify",
        objective="Independent functional counter oracle",
        workspace=workspace,
        policy=verify_policy,
        verification=command,
    )
    templates["verify"] = TaskTemplate(
        "verification",
        verify,
        replace(base, max_policy=verify_policy, permitted_verifications=(command,)),
        "candidate",
    )
    tasks = (
        OrchestrationTask(task_id="log", role="log_analysis", template_id="log"),
        OrchestrationTask(task_id="code", role="code_analysis", template_id="code"),
        OrchestrationTask(
            task_id="diagnose",
            role="diagnosis",
            template_id="diagnose",
            dependencies=(
                TaskDependency(task_id="log", input_role="evidence"),
                TaskDependency(task_id="code", input_role="evidence"),
            ),
        ),
        OrchestrationTask(
            task_id="repair",
            role="repair",
            template_id="repair",
            dependencies=(
                TaskDependency(task_id="diagnose", input_role="diagnosis_run"),
            ),
        ),
        OrchestrationTask(
            task_id="verify",
            role="verification",
            template_id="verify",
            dependencies=(TaskDependency(task_id="repair"),),
        ),
    )
    plan = PlanRevision(job_id="recovery", revision_id="r0", tasks=tasks)
    job_limits = JobLimits(
        max_model_calls=1 if mode == "budget_exhausted" else 160,
        max_tool_calls=160,
        max_total_tokens=2000000,
        max_runtime_seconds=120,
    )
    cfg = SchedulerConfig(
        templates=templates,
        artifact_root=str(io.root),
        snapshot=snap,
        max_workers=2,
        role_profiles=True,
        job_limits=job_limits,
    )
    return plan, cfg, io
