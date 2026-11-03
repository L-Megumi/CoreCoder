"""Bounded evaluator feedback rounds over existing Harness and DAG Scheduler.

Not another Agent/Token Loop, scheduler, learned planner or recoverable trial
checkpoint. Default legacy evaluation still uses its DurableRecovery path.
"""

from dataclasses import replace
import hashlib
import json
import time
from pathlib import Path
from corecoder.workflow import workspace_revision
from researchcoder.contracts import TaskInput, ArtifactRef
from researchcoder.runtime import ResearchHarness
from researchcoder.runtime.protocol import ModelSpec
from researchcoder.runtime.budget import BudgetExceeded
from researchcoder.orchestration import DAGScheduler
from researchcoder.orchestration.budget import JobBudget
from researchcoder.orchestration.workspaces import SourceSnapshot
from researchcoder.runtime.tracing import TraceSink
from .dataset import DATA, is_complexity, load_complexity


def repair_model(case, behavior, project, generation, default):
    """Only the evaluator's ScriptedLLM fixture sees reference/stage answers."""
    from researchcoder.evaluation.runner import _call
    from researchcoder.runtime.protocol import ScriptedTurn
    if not is_complexity(case) or behavior not in ("staged_correct", "repeat_patch"):
        return default
    label = next(x for x in load_complexity().cases if x.case_id == case.case_id and x.benchmark_subset == "complexity_benchmark")
    stage = label.stages[min(generation, len(label.stages) - 1)] if behavior == "staged_correct" else label.stages[0]
    turns = [_call("read-" + str(i), "read_file", dict(file_path=path)) for i, path in enumerate(stage.patch_paths)]
    for i, path in enumerate(stage.patch_paths):
        old = (project / path).read_text()
        if behavior == "repeat_patch":
            old = (DATA / case.case_id / "faulty" / path).read_text()
        new = (DATA / case.case_id / "healthy" / path).read_text()
        if old != new:
            turns.append(_call("patch-" + str(i), "edit_file", dict(file_path=path, old_string=old, new_string=new)))
    turns.append(ScriptedTurn(content=case.reference_cause + " Location: " + ", ".join(case.fault_locations)))
    return ModelSpec(turns=tuple(turns))


def diagnosis_model(case, behavior, project, generation, default):
    if not generation or not is_complexity(case) or behavior not in ("staged_correct", "repeat_patch"):
        return default
    from researchcoder.runtime.protocol import ScriptedTurn
    inspected = tuple(dict.fromkeys((*case.fault_locations, "entry.py")))
    remaining = [p for p in case.fault_locations if (project / p).read_bytes() != (DATA / case.case_id / "healthy" / p).read_bytes()]
    report = json.loads(default.turns[-1].content)
    summary = "Previous edits are historical. Current independently read reference behavior remains unmet in: " + ", ".join(remaining)
    ids = ["$O" + str(inspected.index(p)) for p in remaining] or ["$O0"]
    report["summary"] = summary
    report["historical_failure"]["description"] = "The historical public log records training invariant failures."
    report["current_state"] = dict(assessment=summary, observation_ids=ids)
    report["hypotheses"][0].update(explanation=summary, observation_ids=ids)
    return replace(default, turns=default.turns[:-1] + (ScriptedTurn(content=json.dumps(report)),))


def single_limits(limits, budget, seconds):
    snapshot = budget.snapshot()
    left = {key: getattr(budget.limits, "max_" + key) - snapshot["charged"][key]
            - sum(p[key] for p in snapshot["pending"].values()) for key in ("model_calls", "tool_calls", "total_tokens")}
    if min(left.values()) < 1 or seconds < .1 or left["total_tokens"] <= limits.max_output_tokens:
        raise BudgetExceeded("Shared feedback allowance exhausted")
    return replace(limits, max_model_calls=min(limits.max_model_calls, left["model_calls"]),
        max_tool_calls=min(limits.max_tool_calls, left["tool_calls"]),
        max_rounds=min(limits.max_rounds, left["model_calls"]),
        max_runtime_seconds=min(limits.max_runtime_seconds, seconds),
        max_context_tokens=min(limits.max_context_tokens, left["total_tokens"]),
        max_total_tokens=min(limits.max_total_tokens, left["total_tokens"]))


def execute_feedback(built, case, config, mode):
    from researchcoder.evaluation.runner import _assessment, _budget, _policy_intact, rejected_tool_events
    io, job_id, project = built["io"], built["job_id"], built["project"]
    budget = JobBudget(config.job_limits)
    started = time.monotonic()
    trace = TraceSink(job_id, "evaluation-feedback", "controller")
    trace.parent_id = trace.emit("evaluation.feedback.started")
    harness = ResearchHarness() if mode == "single" else None
    records, results, rounds = [], [], []
    feedback = None
    remediation_refs = []
    seen = set()
    final_status, final_completed, failure, assessment = "incomplete", False, "unknown", ""
    original_cfg = built["config"]
    previous_revision_id = None
    previous_plan_generation = -1
    for generation in range(config.max_feedback_rounds + 1):
        seconds = config.job_limits.max_runtime_seconds - (time.monotonic() - started)
        if seconds < .1:
            final_status, failure = "failed", "timeout"
            break
        if original_cfg.cancel_event is not None and original_cfg.cancel_event.is_set():
            final_status, failure = "cancelled", "interrupted"
            break
        revision = "r" + str(generation)
        trace.emit("evaluation.feedback.round", revision_id=revision, count=generation)
        if mode == "single":
            cfg = original_cfg
            try:
                limits = single_limits(cfg.limits, budget, seconds)
                budget.reserve(revision + ".single", limits)
            except BudgetExceeded:
                final_status, final_completed, failure = "incomplete", False, "budget_exhausted"
                break
            spec = replace(built["spec"], task_id=revision + ".single",
                workspace=replace(built["spec"].workspace, source_manifest_sha256=workspace_revision(project)), budget=_budget(limits))
            if feedback is not None:
                spec = replace(spec, inputs=tuple(i for i in spec.inputs if i.role != "raw_log") + (TaskInput(role="raw_log", reference=feedback),))
            model = cfg.model
            if config.model.provider == "scripted":
                model = repair_model(case, config.scripted_behavior, project, generation, model)
            cfg = replace(cfg, model=model, limits=limits)
            key = revision + ".single"
            result = harness.execute(spec, cfg)
            record = harness.records[result.attempt_id]
            budget.settle(key, record)
            current_results, current_records = [result], [record]
            final_status, final_completed, failure, assessment = result.status, result.workflow_completed, result.failure_kind, result.summary
        else:
            cfg = replace(original_cfg, job_limits=replace(config.job_limits, max_runtime_seconds=seconds))
            if generation:
                snapshot = SourceSnapshot.create(project, Path(project).parent / ("source-snapshot-" + revision))
                cfg = replace(cfg, snapshot=snapshot)
            templates = {}
            for key, template in cfg.templates.items():
                spec = template.spec
                if generation and config.harness_strategy != 'fixed-v1':
                    # The previous authorized repair produced this candidate.
                    # Bind the new trusted templates to its verified snapshot;
                    # otherwise remediation's unchanged-workspace guard would
                    # correctly reject the stale initial template revision.
                    spec = replace(spec, workspace=replace(spec.workspace,
                        source_manifest_sha256=cfg.snapshot.workspace.source_manifest_sha256))
                if feedback is not None and spec.kind == "diagnose":
                    spec = replace(spec, inputs=tuple(i for i in spec.inputs if i.role != "raw_log") + (TaskInput(role="raw_log", reference=feedback),))
                worker_cfg = template.config
                if template.role in ("log_analysis", "code_analysis", "diagnosis") and config.model.provider == "scripted":
                    worker_cfg = replace(worker_cfg, model=diagnosis_model(case, config.scripted_behavior, project, generation, worker_cfg.model))
                if template.role == "repair" and config.model.provider == "scripted":
                    worker_cfg = replace(worker_cfg, model=repair_model(case, config.scripted_behavior, project, generation, worker_cfg.model))
                templates[key] = replace(template, spec=spec, config=worker_cfg)
            cfg = replace(cfg, templates=templates)
            plan = replace(built["plan"], revision_id=revision, generation=previous_plan_generation + 1,
                parent_revision_id=previous_revision_id,
                tasks=tuple(replace(t, retry_of=previous_revision_id + "." + t.task_id if previous_revision_id else None) for t in built["plan"].tasks))
            plan_ref = io.put_json(job_id, plan.to_dict())
            trace.emit("evaluation.feedback.plan", revision_id=revision, artifact_id=plan_ref.artifact_id)
            from researchcoder.runtime.optimization import budget_remediation_enabled
            if budget_remediation_enabled(config.harness_strategy):
                from researchcoder.orchestration.planning import ReplanningController
                from researchcoder.orchestration.remediation import RemediationPolicy
                remedy = ReplanningController().execute_authorized(plan,cfg,policy=RemediationPolicy(),budget=budget)
                execution=remedy.latest
                remediation_refs.append(remedy.summary_ref.to_dict())
                current_results=list({r.attempt_id:r for run in remedy.runs for r in run.results}.values())
                current_records=list(remedy.records)
            else:
                execution = DAGScheduler().execute(plan, cfg, budget=budget)
                summary = io.read_json(execution.summary_ref)
                current_results, current_records = list(execution.results), list(summary["records"].values())
            final_status = execution.status
            final_completed = execution.status == "succeeded" and all(r.workflow_completed for r in execution.results)
            failure = next((r.failure_kind for r in execution.results if r.failure_kind), None) or next((s.reason for s in execution.states if s.reason), None)
            diagnosis = next((r for r in execution.results if r.kind=='diagnose' and r.task_id.endswith('.diagnose')), None)
            assessment = _assessment(diagnosis, io)
            # A bounded expert remediation may have created an intervening
            # PlanRevision. Feedback must descend from that actual revision,
            # not an invented rN ancestry. Legacy no-remediation values match.
            previous_revision_id = execution.revision_id
            previous_plan_generation = io.read_json(execution.summary_ref)['plan']['generation']
        records.extend(current_records)
        results.extend(current_results)
        verification = next((r.verification for r in reversed(current_results) if r.verification is not None), None)
        round_value = dict(generation=generation, revision_id=revision, status=final_status,
            completed=final_completed, failure_kind=failure, workspace_revision=workspace_revision(project),
            result_refs=[r.outputs[0].to_dict() for r in current_results if r.outputs],
            verification=verification.to_dict() if verification else None, budget=budget.snapshot())
        if mode != 'single' and config.harness_strategy != 'fixed-v1':
            round_value.update(revision_id=execution.revision_id, feedback_round_id=revision, plan_generation=previous_plan_generation)
        rounds.append(round_value)
        io.put_json(job_id, dict(schema_version=1, feedback_round=round_value))
        if final_status == "succeeded" and final_completed:
            break
        if rejected_tool_events(current_records, io) or not _policy_intact(case, project):
            failure = "tool_error"
            break
        if config.stop_on_unknown_usage and any(r["budget"].get("unknown_requests", 0) or r["budget"].get("pending") for r in current_records):
            break
        # Only a real, ordinary verification failure allows another attempt.
        if verification is None or verification.status != "failed" or failure not in (None, "verification_failed"):
            break
        signature = (workspace_revision(project), hashlib.sha256(verification.stdout.text.encode()).hexdigest())
        if signature in seen:
            trace.emit("evaluation.feedback.stopped", reason="no_new_evidence")
            break
        seen.add(signature)
        if generation == config.max_feedback_rounds:
            break
        data = case.public_log + "\nWARNING Previous authorized validation failed; investigate current source, not historical snippets.\n" + verification.stdout.text
        feedback = io.put_bytes(job_id, data.encode(), media_type="text/plain")
    if harness is not None and records:
        harness.close_job(job_id)
    trace_ref = io.put_json(job_id, dict(schema_version=1, events=trace.events))
    value = dict(records=records, budget=budget.snapshot(), task_results=[r.to_dict() for r in results],
        workflow_status=final_status, workflow_completed=final_completed, failure_kind=failure, assessment=assessment,
        replanning_count=max(0, len(rounds) - 1) if mode != "single" else 0,
        feedback_rounds=rounds, feedback_retries=max(0, len(rounds) - 1), feedback_trace=trace_ref.to_dict(),
        feedback_semantics="Comparable verification feedback rounds; fixed graph revisions are not learned replanning or RSI")
    if config.harness_strategy != 'fixed-v1':
        value['harness_optimization'] = dict(strategy=config.harness_strategy,remediation_refs=remediation_refs,
            remediation_count=sum(len(io.read_json(ArtifactRef.from_dict(r))['remedies']) for r in remediation_refs))
    return value, results
