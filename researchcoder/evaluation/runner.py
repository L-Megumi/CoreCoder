"""Controller evaluation driver over the original Harness/Scheduler/Recovery.

No token loop, planner, verifier subprocess implementation or remote judge here.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time
import uuid

from corecoder.workflow import run_verification, workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.contracts import (
    TaskSpec,
    TaskInput,
    WorkspaceRef,
    ExecutionPolicy,
    BudgetSpec,
    DiagnosticOptions,
    ArtifactRef,
)
from researchcoder.runtime import ResearchHarness, ExecutionConfig
from researchcoder.runtime.protocol import (
    ModelSpec,
    ScriptedTurn,
    ScriptedCall,
    CodingTaskSpec,
)
from researchcoder.context.policy import ContextPolicy
from researchcoder.orchestration import (
    PlanRevision,
    OrchestrationTask,
    TaskDependency,
    TaskTemplate,
    SchedulerConfig,
)
from researchcoder.orchestration.workspaces import SourceSnapshot
from researchcoder.orchestration.budget import JobBudget
from researchcoder.durable import DurableStore
from researchcoder.durable.recovery import RecoveryCoordinator
from .contracts import EvaluationConfig, EvaluationRun, EvaluationResult, TrialSpec
from .dataset import (
    load_manifest,
    materialize,
    public_input,
    source_revisions,
    sha_bytes,
    case_asset_root,
)
from .verification import (
    oracle_command,
    protected_inputs,
    check_protected,
    verify_candidate,
)
from .metrics import compute, summarize
from .admission import BatchAdmission

PROMPT_VERSION = "benchmark-instructions-v1"
FINAL_INSTRUCTION = "Explain the root cause and name faulty source files in the final summary. Preserve public training behavior and restrict edits to allowed candidate files. Never inspect evaluator files."


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        value.to_json()
        if hasattr(value, "to_json")
        else json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def planned_trials(cases, config):
    rows = []
    for repeat in range(config.repeats):
        for index, case in enumerate(cases):
            modes = list(config.modes)
            if config.rotate_order:
                offset = (index + repeat) % len(modes)
                modes = modes[offset:] + modes[:offset]
            for mode in modes:
                ordinal = len(rows)
                rows.append(
                    TrialSpec(
                        trial_id="trial-" + str(ordinal).zfill(4),
                        case_id=case.case_id,
                        mode=mode,
                        repeat=repeat,
                        ordinal=ordinal,
                    )
                )
    if len(rows) > config.max_trials:
        raise ValueError("Planned experiment exceeds approved trial bound")
    return tuple(rows)


def _call(identity, name, args=None):
    return ScriptedTurn(
        calls=(
            ScriptedCall(
                call_id=identity, name=name, arguments_json=json.dumps(args or {})
            ),
        )
    )


def _offline_report(case):
    assessment = case.reference_cause + " Location: " + ", ".join(case.fault_locations)
    return ScriptedTurn(
        content=json.dumps(
            dict(
                summary=assessment,
                historical_failure=dict(
                    description="The public log reports a candidate training invariant failure.",
                    evidence_ids=["$E0"],
                ),
                current_state=dict(assessment=assessment, observation_ids=["$O0"]),
                hypotheses=[
                    dict(
                        explanation=assessment,
                        evidence_ids=["$E0"],
                        observation_ids=["$O0"],
                        verification="Independent external checks",
                    )
                ],
                recommended_actions=[
                    "Repair the public training contract within authorized files."
                ],
                limitations=[
                    "Fixed scripted diagnosis and repair, not measured model accuracy."
                ],
            )
        )
    )


def offline_models(case, behavior):
    """Evaluator-owned fixed actions are a model stub, never live prompt few-shots."""
    from .complexity.dataset import is_complexity
    if behavior == "diagnosis_fail":
        refusal = ModelSpec(turns=(ScriptedTurn(content="Insufficient evidence; unable to diagnose"),))
        return refusal, refusal
    inspected = tuple(dict.fromkeys((*case.fault_locations, "entry.py"))) if is_complexity(case) else tuple(a.path for a in case.assets)
    reads = tuple(_call("read-" + str(i), "read_project_file", dict(path=path)) for i, path in enumerate(inspected))
    diagnosis = ModelSpec(
        turns=(_call("extract", "extract_log_evidence"),)
        + reads
        + (_offline_report(case),)
    )
    patch_reads = tuple(
        _call("read-" + str(i), "read_file", dict(file_path=a.path))
        for i, a in enumerate(case.assets) if a.path.endswith(".py")
    )
    edits = []
    if behavior in ("correct", "wrong_patch"):
        for i, path in enumerate(case.writable_paths):
            old = (case_asset_root(case) / "faulty" / path).read_text()
            new = (
                (case_asset_root(case) / "healthy" / path).read_text()
                if behavior == "correct"
                else old + "\n# attempted repair remains incorrect\n"
            )
            if old == new:
                continue
            edits.append(
                _call(
                    "edit-" + str(i),
                    "edit_file",
                    dict(file_path=path, old_string=old, new_string=new),
                )
            )
    final = (
        "Unable to safely finish this task."
        if behavior == "refuse"
        else case.reference_cause + " Location: " + ", ".join(case.fault_locations)
    )
    repair = ModelSpec(
        turns=patch_reads + tuple(edits) + (ScriptedTurn(content=final),)
    )
    return diagnosis, repair


def _budget(limits):
    return BudgetSpec(
        max_rounds=limits.max_rounds,
        max_tool_calls=limits.max_tool_calls,
        max_context_tokens=limits.max_context_tokens,
    )


def rejected_tool_events(records, artifacts):
    """Controller-verified Trace metadata, distinct from unchanged file hashes.

    A denied invocation has no file side effect, but must still stop a batch
    whose protocol requires stopping on permission violations.
    """
    rejected = []
    for record in records:
        if not record.get("trace"):
            raise ValueError("Missing execution trace for policy review")
        trace = artifacts.read_json(ArtifactRef.from_dict(record["trace"]))
        for event in trace if isinstance(trace, list) else trace["events"]:
            if event["event_type"] == "tool.rejected":
                if (event["job_id"], event["task_id"], event["attempt_id"]) != (
                    record["job_id"], record["task_id"], record["attempt_id"]
                ):
                    raise ValueError("Foreign tool rejection trace")
                rejected.append({key: event[key] for key in (
                    "job_id", "task_id", "attempt_id", "event_id", "event_type", "tool", "failure_kind"
                )})
    return rejected


def build_workflow(case, trial, directory, config, torch_python, *, cancel_event=None):
    from .complexity.dataset import is_complexity
    directory = Path(directory)
    project = materialize(case, directory / "candidate")
    job_id = "eval-" + uuid.uuid4().hex
    io = ArtifactIO(directory / "artifacts")
    public = public_input(case)
    log = io.put_bytes(job_id, public["log"].encode(), media_type="text/plain")
    workspace = WorkspaceRef(
        workspace_id="candidate",
        root=str(project),
        source_manifest_sha256=workspace_revision(project),
    )
    snapshot = SourceSnapshot.create(project, directory / "source-snapshot")
    write = ExecutionPolicy(
        allowed_tool_names=("read_file", "edit_file"),
        writable_paths=case.writable_paths,
    )
    read = ExecutionPolicy(allowed_tool_names=("read_file",))
    verify = ExecutionPolicy(allowed_tool_names=(), allow_verification=True)
    protected = protected_inputs(case)
    diagnose_model, repair_model = (
        offline_models(case, config.scripted_behavior)
        if config.model.provider == "scripted"
        else (config.model, config.model)
    )
    strategy = config.context_strategy or (
        "evidence" if trial.mode == "evidence" else None
    )
    policy = (
        ContextPolicy(strategy=strategy, role="diagnosis", allow_recovery=False)
        if strategy
        else None
    )
    base = ExecutionConfig(
        artifact_root=str(io.root),
        allowed_workspaces=(str(project),),
        max_policy=write,
        limits=config.task_limits,
        model=config.model,
        model_options=config.model_options,
        protected_files=protected,
        context_policy=policy,
        cancel_event=cancel_event,
        record_requests=is_complexity(case),
    )
    inventory = "\nPublic candidate file inventory (paths only): " + json.dumps(public["source_files"]) if is_complexity(case) else ""
    objective = case.objective + inventory + "\n" + FINAL_INSTRUCTION
    # Complex-case feedback must not query the final scoring inputs. Legacy
    # benchmark behavior stays compatible; new protocols can explicitly select
    # discovery for their Level 1 anchors through trusted controller config.
    command = oracle_command(case, project, torch_python, "discovery" if is_complexity(case) else "holdout")
    if trial.mode == "single":
        limits = replace(
            config.task_limits,
            max_model_calls=config.job_limits.max_model_calls,
            max_tool_calls=min(100, config.job_limits.max_tool_calls),
            max_rounds=min(30, config.job_limits.max_model_calls),
            max_runtime_seconds=config.job_limits.max_runtime_seconds,
            max_total_tokens=config.job_limits.max_total_tokens,
        )
        spec = CodingTaskSpec(
            job_id=job_id,
            task_id="single",
            kind="coding",
            objective=objective,
            workspace=workspace,
            inputs=(TaskInput(role="raw_log", reference=log),),
            policy=replace(write, allow_verification=True),
            verification=command,
            budget=_budget(limits),
        )
        cfg = replace(
            base,
            model=repair_model,
            limits=limits,
            max_policy=spec.policy,
            permitted_verifications=(command,),
            system_instructions=FINAL_INSTRUCTION,
        )
        return dict(
            job_id=job_id,
            io=io,
            project=project,
            spec=spec,
            config=cfg,
            protected=protected,
        )
    diag = TaskSpec(
        job_id=job_id,
        task_id="diagnosis",
        kind="diagnose",
        objective=objective,
        workspace=workspace,
        inputs=(TaskInput(role="raw_log", reference=log),),
        policy=read,
        budget=_budget(config.task_limits),
        diagnostic_options=DiagnosticOptions(
            source_hints=("entry.py",) if is_complexity(case) else tuple(a.path for a in case.assets)
        ),
    )
    repair = TaskSpec(
        job_id=job_id,
        task_id="repair",
        kind="repair",
        objective=objective,
        workspace=workspace,
        inputs=(TaskInput(role="diagnosis_run", reference=log),),
        policy=write,
        budget=_budget(config.task_limits),
    )
    verification = TaskSpec(
        job_id=job_id,
        task_id="verify",
        kind="verify",
        objective="Authorized independent validation",
        workspace=workspace,
        policy=verify,
        verification=command,
        budget=_budget(config.task_limits),
    )
    templates = {
        role: TaskTemplate(
            role,
            diag,
            replace(base, model=diagnose_model),
            "candidate" if role == "diagnosis" else "snapshot",
        )
        for role in ("log_analysis", "code_analysis", "diagnosis")
    }
    templates["repair"] = TaskTemplate(
        "repair", repair, replace(base, model=repair_model), "candidate"
    )
    templates["verification"] = TaskTemplate(
        "verification",
        verification,
        replace(
            base,
            model=ModelSpec(),
            model_options=None,
            max_policy=verify,
            permitted_verifications=(command,),
        ),
        "candidate",
    )
    tasks = (
        OrchestrationTask(
            task_id="log",
            template_id="log_analysis",
            role="log_analysis",
            objective=objective,
        ),
        OrchestrationTask(
            task_id="code",
            template_id="code_analysis",
            role="code_analysis",
            objective=objective,
        ),
        OrchestrationTask(
            task_id="diagnose",
            template_id="diagnosis",
            role="diagnosis",
            objective=objective,
            dependencies=(
                TaskDependency(task_id="log", input_role="evidence"),
                TaskDependency(task_id="code", input_role="evidence"),
            ),
        ),
        OrchestrationTask(
            task_id="repair",
            template_id="repair",
            role="repair",
            objective=objective,
            dependencies=(
                TaskDependency(task_id="diagnose", input_role="diagnosis_run"),
            ),
        ),
        OrchestrationTask(
            task_id="verify",
            template_id="verification",
            role="verification",
            dependencies=(TaskDependency(task_id="repair"),),
        ),
    )
    # Same graph in sequential and parallel modes; only execution concurrency differs.
    plan = PlanRevision(job_id=job_id, revision_id="r0", tasks=tasks)
    cfg = SchedulerConfig(
        templates=templates,
        artifact_root=str(io.root),
        snapshot=snapshot,
        max_workers=1 if trial.mode == "sequential" else 2,
        job_limits=config.job_limits,
        role_profiles=True,
        cancel_event=cancel_event,
    )
    built = dict(
        job_id=job_id,
        io=io,
        project=project,
        plan=plan,
        config=cfg,
        protected=protected,
    )
    if config.harness_strategy != 'fixed-v1':
        from .optimization import configure
        built = configure(built,case,config,torch_python)
    return built


def _assessment(result, io):
    if result is None:
        return ""
    # Grade a model's stated conclusion, not every file it happened to inspect.
    return result.summary


def _policy_intact(case, project):
    expected = {a.path for a in case.assets}
    actual = {
        p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file()
    }
    if actual != expected or any(p.is_symlink() for p in project.rglob("*")):
        return False
    return all(
        sha_bytes((project / a.path).read_bytes()) == a.faulty_sha256
        for a in case.assets
        if a.path not in case.writable_paths
    )


class BatchStopped(RuntimeError):
    """Evaluation admission stopped; no new Worker is started for this sample."""


class EvaluationRunner:
    def __init__(self):
        self.last_run = None

    def run(
        self,
        directory,
        config=EvaluationConfig(),
        *,
        torch_python,
        cancel_event=None,
        exporter=None,
    ):
        config = EvaluationConfig.from_json(config.to_json())
        directory = Path(directory).resolve()
        source = Path(__file__).resolve().parents[2]
        if directory.is_relative_to(source):
            raise ValueError(
                "Evaluation output must be outside the source installation"
            )
        directory.mkdir(parents=True, exist_ok=False)
        manifest = load_manifest(config.benchmark_subset)
        cases = tuple(
            c
            for c in manifest.cases
            if c.split in config.splits
            and (not config.case_ids or c.case_id in config.case_ids)
        )
        if not cases or set(config.case_ids) - {c.case_id for c in cases}:
            raise ValueError("Selected cases/splits unavailable")
        trials = planned_trials(cases, config)
        environment = run_verification(
            (
                str(torch_python),
                "-B",
                "-c",
                'import json,sys,torch; print(json.dumps({"python":sys.version.split()[0],"torch":torch.__version__,"device":"cpu"}))',
            ),
            directory,
            30,
        )
        if environment["status"] != "passed":
            raise ValueError("Configured CPU Torch environment is unavailable")
        environment = json.loads(environment["stdout"]["text"])
        environment["controller_python"] = sys.version.split()[0]
        run = EvaluationRun(
            run_id="evaluation-" + uuid.uuid4().hex,
            manifest_sha256=sha_bytes(manifest.to_json().encode()),
            config_sha256=sha_bytes(config.to_json().encode()),
            environment_sha256=sha_bytes(
                json.dumps(environment, sort_keys=True).encode()
            ),
            trials=trials,
        )
        self.last_run = run
        write_json(directory / "run.json", run)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "config.json", config)
        write_json(directory / "environment.json", environment)
        code_root = Path(__file__).resolve().parents[2]
        complexity_sha = None
        if config.benchmark_subset == "complexity_benchmark":
            from .complexity.dataset import load_complexity
            dimensions = load_complexity()
            complexity_sha = sha_bytes(dimensions.to_json().encode())
            write_json(directory / "complexity-manifest.json", dimensions)
        modules = [
            "corecoder/prompt.py",
            "corecoder/agent.py",
            "corecoder/diagnose.py",
            "corecoder/diagnostic_policy.py",
            "corecoder/citation_scope.py",
            "corecoder/workflow.py",
            "researchcoder/orchestration/profiles.py",
            "researchcoder/orchestration/scheduler.py",
            "researchcoder/orchestration/planning.py",
            "researchcoder/orchestration/baseline.py",
            "researchcoder/runtime/budget.py",
            "researchcoder/context/compression.py",
            "researchcoder/context/engine.py",
            "researchcoder/orchestration/evidence.py",
            "researchcoder/evaluation/admission.py",
            "researchcoder/evaluation/runner.py",
            "researchcoder/evaluation/metrics.py",
            "researchcoder/evaluation/oracles.py",
            "researchcoder/runtime/request_records.py",
            "researchcoder/runtime/protocol.py",
            "researchcoder/runtime/harness.py",
            "researchcoder/runtime/worker.py",
        ]
        if complexity_sha:
            modules += ["researchcoder/evaluation/complexity/" + name + ".py"
                        for name in ("analysis", "contracts", "dataset", "feedback", "oracle", "protocol", "reference")]
        if config.harness_strategy != 'fixed-v1':
            modules += ['researchcoder/evaluation/optimization.py','researchcoder/runtime/optimization.py',
                        'researchcoder/orchestration/remediation.py','corecoder/diagnostic_output.py',
                        'researchcoder/verification/public_training.py']
        write_json(
            directory / "versions.json",
            dict(
                prompt_version=PROMPT_VERSION,
                final_instruction=FINAL_INSTRUCTION,
                source_revisions={c.case_id: source_revisions(c) for c in cases},
                code={p: sha_bytes((code_root / p).read_bytes()) for p in modules},
                mode_semantics="Identical model/output limit and JobLimits, candidate read/write union and final oracles. Multi uses role APIs; single uses original coding tools.",
                **({"complexity_manifest_sha256": complexity_sha} if complexity_sha else {}),
            ),
        )
        preflight = {}
        for case in cases:
            if cancel_event is not None and cancel_event.is_set():
                preflight[case.case_id] = {}
                continue
            controls = directory / "preflight" / case.case_id
            healthy = materialize(case, controls / "healthy", healthy=True)
            faulty = materialize(case, controls / "faulty")
            io = ArtifactIO(controls / "artifacts")
            checks = {}
            for name, project, profile in (
                ("healthy", healthy, "holdout"),
                ("healthy_regression", healthy, "regression"),
                ("fault", faulty, "holdout"),
                ("regression", faulty, "regression"),
            ):
                checks[name] = verify_candidate(
                    case,
                    project,
                    str(torch_python),
                    profile,
                    io,
                    "preflight-" + case.case_id,
                    name,
                ).to_dict()
            preflight[case.case_id] = dict(
                valid=all(
                    checks[n]["verification"]["status"] == s
                    for n, s in (
                        ("healthy", "passed"),
                        ("healthy_regression", "passed"),
                        ("fault", "failed"),
                        ("regression", "passed"),
                    )
                ),
                fault_failed=checks["fault"]["verification"]["status"] == "failed",
                regression_passed=checks["regression"]["verification"]["status"]
                == "passed",
                checks=checks,
            )
        write_json(directory / "preflight.json", preflight)
        admission = BatchAdmission(config)
        for trial in trials:
            case = next(c for c in cases if c.case_id == trial.case_id)
            try:
                stop_reason = admission.admit()
                if stop_reason:
                    raise BatchStopped(stop_reason)
                result = self._trial(
                    directory,
                    run,
                    trial,
                    case,
                    config,
                    str(torch_python),
                    preflight[case.case_id],
                    cancel_event,
                )
            except Exception as exc:
                # Setup/collection failures are explicit samples, never silently dropped.
                io = ArtifactIO(directory / "trials" / trial.trial_id / "artifacts")
                job_id = "eval-" + uuid.uuid4().hex
                raw = dict(
                    schema_version=1,
                    run_id=run.run_id,
                    trial=trial.to_dict(),
                    job_id=job_id,
                    preflight={},
                    records=[],
                    workflow_status="incomplete",
                    workflow_completed=False,
                    status="cancelled"
                    if isinstance(exc, BatchStopped)
                    else "incomplete",
                    failure_kind=admission.reason
                    if isinstance(exc, BatchStopped)
                    else type(exc).__name__,
                    policy_intact=False,
                    wall_seconds=0,
                    setup_failure=True,
                )
                result = self._record(
                    directory, run, trial, case, config, io, job_id, raw
                )
            if not admission.reason:
                result_io = ArtifactIO(
                    directory / "trials" / trial.trial_id / "artifacts"
                )
                admission.settle(result, result_io.read_json(result.raw))
            # Results are committed locally before optional observability. Export cannot change them.
            if exporter is not None:
                try:
                    exported = exporter.export_result(result, directory)
                except Exception as exc:
                    exported = dict(
                        status="export_error", error_type=type(exc).__name__
                    )
                write_json(directory / "exports" / (trial.trial_id + ".json"), exported)
        write_json(directory / "batch-admission.json", admission.snapshot())
        return recompute(directory)

    def _trial(
        self, directory, run, trial, case, config, python, preflight, cancel_event
    ):
        started = time.monotonic()
        target = directory / "trials" / trial.trial_id
        built = build_workflow(
            case, trial, target, config, python, cancel_event=cancel_event
        )
        io = built["io"]
        job_id = built["job_id"]
        project = built["project"]
        raw = dict(
            schema_version=1,
            run_id=run.run_id,
            trial=trial.to_dict(),
            job_id=job_id,
            preflight={k: v for k, v in preflight.items() if k != "checks"},
            records=[],
            workflow_status="incomplete",
            workflow_completed=False,
            assessment="",
            fail_to_pass=None,
            pass_to_pass=None,
            status="incomplete",
            failure_kind=None,
            policy_intact=False,
            replanning_count=0,
            recovery_success=None,
            duplicate_side_effects=None,
            workspace_before=workspace_revision(project),
            manifest_sha256=run.manifest_sha256,
            config_sha256=run.config_sha256,
        )
        all_results = []
        try:
            if cancel_event is not None and cancel_event.is_set():
                raw.update(status="cancelled", failure_kind="interrupted")
            elif not preflight.get("valid"):
                raw.update(failure_kind="benchmark_preflight_failed")
            else:
                if config.max_feedback_rounds or config.harness_strategy != 'fixed-v1':
                    from .complexity.feedback import execute_feedback
                    values, all_results = execute_feedback(built, case, config, trial.mode)
                    raw.update(values)
                elif trial.mode == "single":
                    harness = ResearchHarness()
                    job_budget = JobBudget(config.job_limits)
                    job_budget.reserve("single", built["config"].limits)
                    execution = harness.execute(built["spec"], built["config"])
                    record = harness.records[execution.attempt_id]
                    job_budget.settle("single", record)
                    raw.update(
                        records=[record],
                        budget=job_budget.snapshot(),
                        assessment=execution.summary,
                        workflow_status=execution.status,
                        workflow_completed=execution.workflow_completed,
                        failure_kind=execution.failure_kind,
                    )
                    all_results = [execution]
                    harness.close_job(job_id)
                else:
                    with (
                        DurableStore(target / "state.sqlite", io) as store,
                        RecoveryCoordinator(store) as coordinator,
                    ):
                        execution = coordinator.execute(built["plan"], built["config"])
                        summary = io.read_json(execution.summary_ref)
                        raw.update(
                            records=list(summary["records"].values()),
                            budget=summary["budget"],
                            workflow_status=execution.status,
                            workflow_completed=execution.status == "succeeded"
                            and all(r.workflow_completed for r in execution.results),
                            scheduler_result=execution.to_dict(),
                            failure_kind=next(
                                (
                                    r.failure_kind
                                    for r in execution.results
                                    if r.failure_kind
                                ),
                                None,
                            ),
                        )
                        diagnosis = next(
                            (
                                r
                                for r in execution.results
                                if r.task_id == "r0.diagnose"
                            ),
                            None,
                        )
                        raw["assessment"] = _assessment(diagnosis, io)
                        all_results = list(execution.results)
                raw["task_results"] = [r.to_dict() for r in all_results]
                raw["policy_violations"] = rejected_tool_events(raw["records"], io)
                check_protected(built["protected"])
                raw["policy_intact"] = _policy_intact(case, project)
                if cancel_event is not None and cancel_event.is_set():
                    raw.update(status="cancelled", failure_kind="interrupted")
                elif raw["policy_intact"]:
                    for key, profile in (
                        ("fail_to_pass", "holdout"),
                        ("pass_to_pass", "regression"),
                    ):
                        oracle = verify_candidate(
                            case,
                            project,
                            python,
                            profile,
                            io,
                            job_id,
                            "evaluation-" + profile,
                        )
                        raw[key] = (
                            oracle.verification.to_dict()
                            if oracle.verification
                            else None
                        )
                    raw["status"] = raw["workflow_status"]
                else:
                    raw.update(
                        status="needs_review",
                        failure_kind="unauthorized_workspace_change",
                    )
                check_protected(built["protected"])
        except Exception as exc:
            raw.update(
                status="needs_review"
                if type(exc).__name__ == "CheckpointError"
                else "incomplete",
                failure_kind=type(exc).__name__,
            )
        raw["wall_seconds"] = time.monotonic() - started
        raw["workspace_after"] = workspace_revision(project)
        if config.benchmark_subset == "complexity_benchmark":
            from .complexity.dataset import load_complexity
            from .complexity.analysis import trial_diagnostics
            label = next(c for c in load_complexity().cases if c.case_id == case.case_id)
            raw["complexity_diagnostics"] = trial_diagnostics(raw, io, label)
        return self._record(directory, run, trial, case, config, io, job_id, raw)

    def _record(self, directory, run, trial, case, config, io, job_id, raw):
        metrics = compute(raw, case, config)
        status = (
            "succeeded"
            if metrics.end_to_end_success
            else "needs_review"
            if metrics.needs_review
            else raw["status"]
            if raw["status"] in ("failed", "incomplete", "cancelled", "needs_review")
            else "failed"
        )
        raw["evaluation_status"] = status
        if status != "succeeded" and raw.get("failure_kind") is None:
            raw["failure_kind"] = (
                "workflow_incomplete"
                if not raw.get("workflow_completed")
                else "oracle_failed"
            )
        raw_ref = io.put_json(job_id, raw)
        from researchcoder.contracts import VerificationResult

        result = EvaluationResult(
            run_id=run.run_id,
            trial=trial,
            job_id=job_id,
            status=status,
            failure_kind=raw["failure_kind"],
            raw=raw_ref,
            metrics=metrics,
            fail_to_pass_verification=VerificationResult.from_dict(raw["fail_to_pass"])
            if raw.get("fail_to_pass")
            else None,
            pass_to_pass_verification=VerificationResult.from_dict(raw["pass_to_pass"])
            if raw.get("pass_to_pass")
            else None,
        )
        result_ref = io.put_json(job_id, result.to_dict())
        write_json(directory / "records" / (trial.trial_id + ".json"), result_ref)
        return result


def recompute(directory):
    directory = Path(directory).resolve()
    run = EvaluationRun.from_json((directory / "run.json").read_text())
    from .contracts import BenchmarkManifest

    manifest = BenchmarkManifest.from_json((directory / "manifest.json").read_text())
    config = EvaluationConfig.from_json((directory / "config.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    if (
        sha_bytes(manifest.to_json().encode()) != run.manifest_sha256
        or sha_bytes(config.to_json().encode()) != run.config_sha256
        or sha_bytes(json.dumps(environment, sort_keys=True).encode())
        != run.environment_sha256
    ):
        raise ValueError("Evaluation identity mismatch")
    results = []
    for trial in run.trials:
        record = directory / "records" / (trial.trial_id + ".json")
        if not record.exists():
            continue
        io = ArtifactIO(directory / "trials" / trial.trial_id / "artifacts")
        result = EvaluationResult.from_dict(
            io.read_json(ArtifactRef.from_json(record.read_text()))
        )
        raw = io.read_json(result.raw)
        case = next(c for c in manifest.cases if c.case_id == trial.case_id)
        if (
            raw["run_id"] != run.run_id
            or raw["trial"] != trial.to_dict()
            or raw["job_id"] != result.job_id
            or result.trial != trial
        ):
            raise ValueError("Foreign raw result")
        metrics = compute(raw, case, config)
        status = (
            "succeeded"
            if metrics.end_to_end_success
            else "needs_review"
            if metrics.needs_review
            else raw["status"]
            if raw["status"] in ("failed", "incomplete", "cancelled", "needs_review")
            else "failed"
        )
        results.append(replace(result, metrics=metrics, status=status))
    summary = summarize(run, results)
    summary["scorer_sha256"] = sha_bytes(
        Path(__file__).with_name("metrics.py").read_bytes()
    )
    summary["original_scorer_sha256"] = json.loads(
        (directory / "versions.json").read_text()
    )["code"]["researchcoder/evaluation/metrics.py"]
    if config.benchmark_subset == "complexity_benchmark":
        from .complexity.analysis import enrich_summary
        summary = enrich_summary(directory, run, manifest, results, summary)
    return summary
