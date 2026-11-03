"""One bounded read-only consultation over the existing Scheduler and JobBudget.

This module receives only the public task and public verification result. Final
Oracle profiles, answers and checker paths are intentionally not parameters.
Successful expert reports are hypotheses, not a validated Single-Agent diagnosis
or a repair outcome. The original CodingTaskSpec route remains the sole writer.
"""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
import time
from typing import Literal

from corecoder.workflow import workspace_revision
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.context.policy import ContextPolicy
from researchcoder.contracts import (
    ArtifactRef, BudgetSpec, Contract, DiagnosticOptions, ExecutionPolicy, TaskInput,
    TaskResult, TaskSpec, VerificationResult, digest, identifier,
)
from researchcoder.evaluation.runner import rejected_tool_events
from researchcoder.orchestration.budget import JobBudget
from researchcoder.orchestration.contracts import OrchestrationTask, PlanRevision, SchedulerResult
from researchcoder.orchestration.evidence import EvidenceBundle, EvidenceContextCapacityError
from researchcoder.orchestration.profiles import PROFILES
from researchcoder.orchestration.scheduler import DAGScheduler, SchedulerConfig, TaskTemplate
from researchcoder.runtime import ExecutionConfig
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.protocol import ModelSpec, ProtectedFile, model_request_kwargs
from researchcoder.runtime.tracing import TraceSink
from .contracts import ProjectRunInput, RepositoryManifest
from .routing import ExpertPolicy, PublicRoutingFeatures, RoutingDecision, decide, extract_public_features
from .workspace import ProjectSourceSnapshot, assert_allowed_changes, capture_manifest
from .verification import ProjectCheck, validate_receipt


@dataclass(frozen=True, kw_only=True)
class InitialConsultationReceipt(Contract):
    """Controller-only authority from an actual public Verifier task.

    The command/check configuration never enters an expert's model context.
    This is not a writer attempt, a model help request or a fabricated failure.
    """
    job_id: str
    task_id: str
    attempt_id: str
    result_ref: ArtifactRef
    record_ref: ArtifactRef
    source_manifest_sha256: str
    check: ProjectCheck
    version: Literal["code-first-v1"] = "code-first-v1"
    schema_version: int = 1

    def validate(self):
        for value in (self.job_id, self.task_id, self.attempt_id):
            identifier(value)
        digest(self.source_manifest_sha256)
        if self.task_id != "preflight-public" or any(
                ref.job_id != self.job_id for ref in (self.result_ref, self.record_ref)):
            raise ValueError("Initial consultation requires the same-Job public preflight")


def validate_initial_consultation(receipt, *, candidate, manifest, public_failure, io, policy, features):
    """Reproduce the public provenance before any expert dispatch or snapshot."""
    if (type(receipt) is not InitialConsultationReceipt or type(policy) is not ExpertPolicy
            or policy.initial_mode != "code-first-v1"
            or type(manifest) is not RepositoryManifest or type(public_failure) is not VerificationResult
            or type(io) is not ArtifactIO or type(features) is not PublicRoutingFeatures):
        raise ValueError("Typed initial consultation provenance and enabled policy required")
    receipt.validate()
    root = Path(candidate).resolve()
    if receipt.source_manifest_sha256 != manifest.digest or capture_manifest(root) != manifest:
        raise ValueError("Initial consultation source changed")
    receipt.check.check_protected()
    result = TaskResult.from_dict(io.read_json(receipt.result_ref))
    record = io.read_json(receipt.record_ref)
    identity = (receipt.job_id, receipt.task_id, receipt.attempt_id)
    if (tuple(getattr(result, key) for key in ("job_id", "task_id", "attempt_id")) != identity
            or type(record) is not dict
            or tuple(record.get(key) for key in ("job_id", "task_id", "attempt_id")) != identity
            or result.kind != "verify" or result.status != "failed" or result.workflow_completed is not False
            or result.failure_kind != "verification_failed"
            or result.verification != public_failure or public_failure.spec != receipt.check.command(root)
            or (public_failure.job_id, public_failure.task_id, public_failure.attempt_id) != identity
            or public_failure.status != "failed" or public_failure.exit_code != 1
            or public_failure.stdout.truncated or public_failure.stderr.truncated
            or public_failure.checked_workspace_revision != workspace_revision(root)
            or features.workflow_completed != result.workflow_completed
            or features.failure_kind != result.failure_kind or features.verification_status != public_failure.status
            or features.episodes_used != 0 or features.prior_failure_signatures):
        raise ValueError("Initial consultation does not match its real public Verifier")
    parsed = validate_receipt(public_failure, receipt.check.receipt, root)
    if parsed["passed"] is not False:
        raise ValueError("Initial consultation requires a valid failing public receipt")
    from .handoff import _known_accounting
    budget = record.get("budget", {})
    if (not _known_accounting(record) or budget.get("model_calls") != 0
            or budget.get("tool_calls") != 1 or budget.get("stop_kind") is not None
            or budget.get("last_failure_kind") is not None or record.get("status") != "failed"
            or record.get("failure_kind") != "verification_failed"):
        raise ValueError("Initial Verifier execution was not settled and complete")
    if type(record.get("trace")) is not dict:
        raise ValueError("Initial execution trace is missing")
    reference = ArtifactRef.from_dict(record["trace"])
    if reference.job_id != receipt.job_id:
        raise ValueError("Foreign initial execution trace")
    trace_record = io.read_json(reference)
    if type(trace_record) is not dict:
        raise ValueError("Initial execution trace is not an object")
    events = trace_record.get("events")
    if (type(events) is not list or not events or any(type(e) is not dict or
            tuple(e.get(key) for key in ("job_id", "task_id", "attempt_id")) != identity for e in events)
            or len({e.get("event_id") for e in events}) != len(events)):
        raise ValueError("Invalid initial execution trace")
    starts = [e for e in events if e.get("event_type") == "verification.command.start"]
    ends = [e for e in events if e.get("event_type") == "verification.command.end"]
    results = [e for e in events if e.get("event_type") == "worker.result"]
    if (len(starts) != 1 or len(ends) != 1 or ends[0].get("parent_id") != starts[0].get("event_id")
            or ends[0].get("status") != "failed" or events.index(starts[0]) >= events.index(ends[0])
            or len(results) != 1 or results[0].get("status") != "failed"
            or results[0].get("failure_kind") != "verification_failed"
            or events.index(ends[0]) >= events.index(results[0])
            or any(e.get("event_type") in ("model.request.start", "tool.call.start") for e in events)):
        raise ValueError("Initial public command completion is not proven")
    return receipt


@dataclass(frozen=True)
class ConsultationExecution:
    run: SchedulerResult | None
    records: tuple[dict, ...]
    bundle: EvidenceBundle | None
    context: str
    context_ref: ArtifactRef | None
    selected_producers: tuple[str, ...]
    stop_reason: str | None
    safe_to_continue: bool
    snapshot: ProjectSourceSnapshot | None
    budget_before: dict
    budget_after: dict
    decision: RoutingDecision
    trace_ref: ArtifactRef
    elapsed_seconds: float
    schedule: dict | None = None

    def to_dict(self):
        value = dict(schema_version=1, run=self.run.to_dict() if self.run else None,
            records=list(self.records), bundle=self.bundle.to_dict() if self.bundle else None,
            context_ref=self.context_ref.to_dict() if self.context_ref else None,
            selected_producers=list(self.selected_producers), stop_reason=self.stop_reason,
            safe_to_continue=self.safe_to_continue, snapshot=self.snapshot.to_dict() if self.snapshot else None,
            budget_before=self.budget_before, budget_after=self.budget_after,
            decision=self.decision.to_dict(), trace_ref=self.trace_ref.to_dict(), elapsed_seconds=self.elapsed_seconds,
            report_semantics="expert_hypotheses_not_single_validated_diagnosis",
            handoff_semantics="upstream_data_requires_candidate_source_read_before_edit")
        if self.schedule is not None:
            value["schedule"] = self.schedule
        return value


def _unknown(snapshot):
    return any(s.get("usage_unknown", True) or not s.get("accounting_complete", False)
               for s in snapshot["settlements"].values())


def _policy_violation(results, records, io):
    if records and rejected_tool_events(records, io):
        return True
    for result in results:
        for reference in result.outputs:
            raw = io.read_json(reference)
            if not isinstance(raw, dict):
                continue
            if any(event.get("policy_rejection") for event in raw.get("tool_events", [])):
                return True
            # Direct file/path requests outside the authorized root are refused
            # by the original tools. Historical traceback location misses are
            # distinct and are not automatically classified as policy violations.
            if any(item.get("kind") in ("file_excerpt", "path_status")
                   and item.get("data", {}).get("status") in ("outside_project", "outside_allowed_roots")
                   for item in raw.get("observations", [])):
                return True
    return False


class _StopOnUnsafe:
    """Controller cancellation signal; already running Workers are supervised."""

    def __init__(self, external, budget, scheduler, io):
        self.external, self.budget, self.scheduler, self.io = external, budget, scheduler, io
        self.reason = None
        self._lock = threading.RLock()
        self._reviewed = set()

    def is_set(self):
        with self._lock:
            if self.reason is not None:
                return True
            if self.external is not None and self.external.is_set():
                self.reason = "interrupted"
            elif _unknown(self.budget.snapshot()):
                self.reason = "accounting_unknown"
            else:
                # Scheduler-owned completed records are immutable after publication.
                # Checking them here stops the peer after a collected policy error,
                # rather than starting another Worker or a subsequent writer.
                for entry in list(self.scheduler.completed.values()):
                    result = entry["result"]
                    if result.attempt_id in self._reviewed:
                        continue
                    try:
                        violation = _policy_violation((result,), (entry["record"],), self.io)
                    except (ValueError, TypeError, OSError, KeyError):
                        self.reason = "source_changed"
                        break
                    self._reviewed.add(result.attempt_id)
                    if violation:
                        self.reason = "policy_violation"
                        break
                    if result.failure_kind in ("model_error", "timeout", "interrupted"):
                        self.reason = result.failure_kind
                        break
                    if result.failure_kind in ("reference_stale", "input_invalid", "tool_error"):
                        self.reason = "source_changed"
                        break
            return self.reason is not None


def execute_consultation(*, job_id, candidate, manifest, public_task, public_failure, io, budget,
        model, decision, policy, features, remaining_seconds, episode_id="experts-1", public_log=None,
        model_options=None, cancel_event=None, role_models=None, worker_command=None, handoff=None,
        initial_receipt=None):
    """Execute a controller-authorized Code-first expert episode, without writes.

    ``manifest`` identifies the current candidate; ``public_task.manifest`` may
    identify its original baseline. The public VerificationResult must refer to
    the current legacy revision. ``features``/``decision`` are checked again;
    neither can grant a new tool, write path, model configuration or budget.
    ``role_models`` is solely an offline ScriptedLLM fixture hook.
    """
    typed = ((manifest, RepositoryManifest), (public_task, ProjectRunInput),
             (public_failure, VerificationResult), (io, ArtifactIO), (budget, JobBudget),
             (model, ModelSpec), (decision, RoutingDecision), (policy, ExpertPolicy),
             (features, PublicRoutingFeatures))
    if any(type(value) is not expected for value, expected in typed):
        raise TypeError("Typed public consultation inputs and controller dependencies required")
    identifier(job_id)
    identifier(episode_id)
    if len(episode_id) > 45:
        raise ValueError("Consultation revision identity too long")
    if handoff is not None:
        from .handoff import validate_handoff
        validate_handoff(handoff, candidate=candidate, manifest=manifest,
            public_failure=public_failure, public_task=public_task, io=io)
    if initial_receipt is not None:
        if handoff is not None:
            raise ValueError("A preflight consultation cannot be a writer handoff")
        if manifest != public_task.manifest:
            raise ValueError("Initial consultation requires the original candidate source")
        validate_initial_consultation(initial_receipt, candidate=candidate, manifest=manifest,
            public_failure=public_failure, io=io, policy=policy, features=features)
    if decision != decide(features, policy, handoff=handoff, initial=initial_receipt is not None) or decision.action != "consult_experts":
        raise ValueError("Consultation requires a matching affirmative public routing decision")
    model_request_kwargs(model, model_options)
    overrides = role_models or {}
    if (not isinstance(overrides, dict) or set(overrides) - set(decision.roles)
            or any(type(value) is not ModelSpec or value.provider != "scripted" for value in overrides.values())
            or overrides and model.provider != "scripted"):
        raise ValueError("Role models are explicit ScriptedLLM fixtures only")
    if type(remaining_seconds) not in (float, int) or not 0 <= remaining_seconds <= 3600:
        raise ValueError("Bounded remaining Job time required")
    if public_log is None:
        public_log = public_task.public_log
    # Rebuild the public identity, retaining trusted scalar observations solely
    # to compare the projection. Actual budget admission is checked below.
    observed = extract_public_features(public_log=public_log,
        source_files=tuple(entry.path for entry in manifest.entries),
        verification_status=public_failure.status, workflow_completed=features.workflow_completed,
        failure_kind=features.failure_kind, verification_stdout=public_failure.stdout.text,
        verification_stderr=public_failure.stderr.text,
        remaining_model_calls=features.remaining_model_calls, remaining_tool_calls=features.remaining_tool_calls,
        remaining_total_tokens=features.remaining_total_tokens, remaining_seconds=features.remaining_seconds,
        remaining_retries=features.remaining_retries, prior_failure_signatures=features.prior_failure_signatures,
        episodes_used=features.episodes_used, usage_unknown=features.usage_unknown,
        budget_pending=features.budget_pending, policy_intact=features.policy_intact,
        source_intact=features.source_intact, cancelled=features.cancelled)
    if observed != features:
        raise ValueError("Routing features do not match the supplied public evidence")
    if public_failure.job_id != job_id or public_failure.status != "failed" or public_failure.exit_code != 1:
        raise ValueError("A normal same-Job public verification failure is required")
    if public_failure.stdout.truncated or public_failure.stderr.truncated:
        raise ValueError("Truncated public verification cannot authorize expert execution")

    started = time.monotonic()
    before = budget.snapshot()
    trace = TraceSink(job_id, "project-consultation", episode_id)
    trace.parent_id = trace.emit("consultation.requested", revision_id=episode_id, count=len(decision.roles))
    run = snapshot = bundle = context_ref = None
    records, selected, context = (), (), ""
    effective_workers = min(len(decision.roles), policy.max_workers or len(decision.roles))
    schedule = (dict(version=policy.schedule_version, max_workers=policy.max_workers,
        effective_workers=effective_workers, role_order=list(decision.roles),
        expert_limits=policy.expert_limits.to_dict(),
        admitted_expert_seconds=policy.expert_seconds(len(decision.roles)))
        if policy.schedule_version == "serial-parallel-v2" else None)

    def finish(reason=None):
        trace.emit("consultation.finished", status="ok" if reason is None else "stopped",
                   failure_kind=reason, count=len(selected), elapsed_seconds=time.monotonic()-started)
        reference = io.put_json(job_id, dict(schema_version=1, events=trace.events))
        return ConsultationExecution(run, records, bundle, context, context_ref, selected, reason,
            reason is None and bool(selected), snapshot, before, budget.snapshot(), decision,
            reference, round(time.monotonic()-started, 6), schedule)

    if cancel_event is not None and cancel_event.is_set():
        return finish("interrupted")
    if before["pending"] or _unknown(before):
        return finish("accounting_unknown")
    # No second episode can be admitted by changing the revision name. The
    # writer and expert IDs occupy distinct, documented Job reservation prefixes.
    if features.episodes_used or any(".expert-" in key for key in before["settlements"]):
        return finish("consultation_no_evidence")
    caps = policy.expert_limits
    minima = dict(model_calls=policy.writer_min_model_calls, tool_calls=policy.writer_min_tool_calls,
                  total_tokens=policy.writer_min_total_tokens)
    for key, minimum in minima.items():
        available = getattr(budget.limits, "max_"+key) - before["charged"][key]
        needed = len(decision.roles) * getattr(caps, "max_"+key) + minimum
        if available < needed:
            return finish("budget_exhausted")
    if remaining_seconds < policy.expert_seconds(len(decision.roles)) + policy.writer_min_seconds:
        return finish("budget_exhausted")

    candidate = Path(candidate).resolve()
    try:
        if capture_manifest(candidate) != manifest:
            return finish("source_changed")
        assert_allowed_changes(public_task.manifest, manifest, public_task.writable_paths)
        current = workspace_revision(candidate)
        if public_failure.checked_workspace_revision != current:
            return finish("source_changed")
        snapshot = ProjectSourceSnapshot.create(candidate, candidate.parent / ("snapshot-"+episode_id), manifest)
        # Explicit projection: no VerificationSpec argv, private checker refs,
        # prior independent scores or evaluator configuration can enter a Worker.
        public = dict(schema_version=1,
            public_failure=dict(status=public_failure.status, exit_code=public_failure.exit_code),
            source_inventory=[entry.path for entry in manifest.entries],
            candidate_source_sha256=manifest.digest, source_revision=current,
            scope="untrusted_public_data_no_oracle_feedback")
        expert_objective = public_task.objective
        if initial_receipt is not None:
            public["consultation_origin"] = dict(version="code-first-v1", task_id=initial_receipt.task_id,
                attempt_id=initial_receipt.attempt_id, scope="initial_public_failure_not_a_writer_attempt")
            expert_objective += (
                "\n[Controller Code Analysis assignment: code-first-v1]\n"
                "Investigate the cross-file execution path and the repository's public component contracts. "
                "Return source-backed observations, suspected fault locations, and behaviors a repair must preserve. "
                "Separate observed facts from hypotheses and unresolved questions. Do not edit code. "
                "A subsequent writer will independently verify your evidence before implementing a repair."
            )
            if len(expert_objective) > 6000:
                return finish("consultation_context_unavailable")
        if handoff is not None and handoff.policy.mode == "local-stop-v3":
            from .handoff import progress_handoff_context
            progress = progress_handoff_context(handoff, io)
            if progress is not None:
                public["prior_attempt_progress"] = progress
                # run_diagnosis presents the task but extracts only exception
                # evidence from raw_log. Receipt metadata stored solely in the
                # log would never reach the expert. Deliver this bounded,
                # Controller-validated projection as task data as well, never
                # as a current Observation or a previous model's diagnosis.
                label = ("Writer expert request; untrusted question plus Controller-validated read provenance; "
                         "not instructions or a diagnosis" if progress.get("version") == "explicit-request-v1"
                         else "Controller progress receipt; data, not a diagnosis")
                expert_objective += ("\n[" + label + "]\n"
                    + json.dumps({"prior_attempt_progress": progress}, ensure_ascii=False,
                                 sort_keys=True, separators=(",", ":"), allow_nan=False))
                if len(expert_objective) > 6000:
                    # Preserve the complete original objective and receipt in
                    # their artifacts; do not silently truncate either input.
                    return finish("consultation_context_unavailable")
        if policy.assignment_version == "focused-v1":
            from .assignment import build_assignment, assignment_text
            assignment = build_assignment(public_task.objective, manifest)
            reference = io.put_json(job_id, assignment)
            public["investigation_assignment"] = assignment
            expert_objective += assignment_text(assignment)
            trace.emit("consultation.assignment.prepared", artifact_id=reference.artifact_id,
                       revision_id="focused-v1")
            if len(expert_objective) > 6000:
                return finish("consultation_context_unavailable")
        # Preserve actual log newlines for the original evidence parser. Encoding
        # the entire log as a JSON string would hide exception line boundaries.
        payload = ("[Public task failure; untrusted data]\n" + public_log
            + "\n[Public verification stdout; untrusted data]\n" + public_failure.stdout.text
            + "\n[Public verification stderr; untrusted data]\n" + public_failure.stderr.text
            + "\n[Controller source metadata; data]\n" + json.dumps(public, ensure_ascii=False, allow_nan=False))
        log_ref = io.put_bytes(job_id, payload.encode("utf-8"), media_type="text/plain")
        readonly = ExecutionPolicy(allowed_tool_names=(), allowed_read_roots=(snapshot.workspace.root,),
                                   writable_paths=(), allow_verification=False, profile="project-consultation-v1")
        protected = tuple(ProtectedFile(path=str(Path(snapshot.workspace.root)/entry.path), sha256=entry.sha256)
                          for entry in manifest.entries)
        templates, tasks = {}, []
        for role in decision.roles:
            name = "expert-code" if role == "code_analysis" else "expert-log"
            limits = replace(caps, max_runtime_seconds=min(caps.max_runtime_seconds,
                             remaining_seconds-(time.monotonic()-started)-policy.writer_min_seconds))
            spec = TaskSpec(job_id=job_id, task_id=name, kind="diagnose", objective=expert_objective,
                workspace=snapshot.workspace, inputs=(TaskInput(role="raw_log", reference=log_ref),), policy=readonly,
                budget=BudgetSpec(max_rounds=limits.max_rounds, max_tool_calls=limits.max_tool_calls,
                    max_context_tokens=limits.max_context_tokens,
                    max_report_repairs=1 if policy.report_version in ("explicit-v1", "readiness-v2", "readiness-v3") else 0),
                diagnostic_options=DiagnosticOptions(path_mappings=((str(candidate), snapshot.workspace.root),),
                    tool_selection="all", context_mode="evidence", completion_mode="off"))
            config = ExecutionConfig(artifact_root=str(io.root), allowed_workspaces=(snapshot.workspace.root,),
                max_policy=readonly, limits=limits, model=overrides.get(role, model), model_options=model_options,
                protected_files=protected, diagnostic_tools=PROFILES[role].diagnostic_tools,
                record_requests=True, worker_command=worker_command,
                context_policy=(ContextPolicy(role=role, working_set_version="active-v1")
                                if policy.working_set_version == "active-v1" else None),
                optimization=RuntimeOptimization(compact_reports=True,
                    report_reserve_calls=2 if policy.report_version in ("explicit-v1", "readiness-v3") else 1,
                    version=("evidence-ready-v3" if policy.report_version == "readiness-v3" else
                             "evidence-ready-v2" if policy.report_version == "readiness-v2" else
                             "explicit-report-v1" if policy.report_version == "explicit-v1" else "failure-aware-v1")))
            templates[name] = TaskTemplate(role, spec, config)
            tasks.append(OrchestrationTask(task_id=name, template_id=name, role=role, objective=spec.objective,
                                           workspace_revision=current))
        plan = PlanRevision(job_id=job_id, revision_id=episode_id, tasks=tuple(tasks))
        scheduler = DAGScheduler()
        stop = _StopOnUnsafe(cancel_event, budget, scheduler, io)
        seconds = remaining_seconds-(time.monotonic()-started)-policy.writer_min_seconds
        if seconds < .1:
            return finish("budget_exhausted")
        config = SchedulerConfig(templates=templates, artifact_root=str(io.root), snapshot=snapshot,
            max_workers=effective_workers, max_tasks=2, max_depth=1, role_profiles=True, cancel_event=stop,
            job_limits=replace(budget.limits, max_runtime_seconds=seconds))
        trace.emit("consultation.dispatched", revision_id=episode_id, workspace_revision=current,
                   **({"count": effective_workers} if schedule is not None else {}))
        run = scheduler.execute(plan, config, budget=budget)
        records = tuple(scheduler.records.values())
        if capture_manifest(candidate) != manifest:
            return finish("source_changed")
        snapshot.verify()
        if _policy_violation(run.results, records, io):
            return finish("policy_violation")
        if stop.is_set():
            return finish(stop.reason)
        if budget.snapshot()["pending"] or _unknown(budget.snapshot()):
            return finish("accounting_unknown")
        for failure in ("model_error", "timeout", "interrupted", "reference_stale", "input_invalid", "tool_error"):
            if any(result.failure_kind == failure for result in run.results):
                return finish("source_changed" if failure in ("reference_stale", "input_invalid", "tool_error") else failure)
        if any(state.reason == "timeout" for state in run.states):
            return finish("timeout")
        if run.status == "cancelled":
            return finish("interrupted")
        selected = tuple(result.task_id for result in run.results
                         if result.status == "succeeded" and result.workflow_completed and result.evidence_refs)
        if not selected:
            return finish("consultation_no_evidence")
        bundle = scheduler.exchange.bundle(selected, current)
        paths = []
        for item in bundle.items:
            reference = item.reference
            if reference.source_path is None or reference.source_sha256 is None:
                continue
            source = Path(reference.source_path).resolve()
            root = Path(snapshot.workspace.root)
            if not source.is_relative_to(root):
                return finish("policy_violation")
            relative = source.relative_to(root).as_posix()
            entry = next((entry for entry in manifest.entries if entry.path == relative), None)
            if entry is None or entry.sha256 != reference.source_sha256:
                return finish("source_changed")
            paths.append(dict(exchange_id=item.exchange_id, snapshot_source_path=str(source),
                              candidate_relative_path=relative, sha256=entry.sha256))
        additions = dict(consultation_version="public-failure-v1", complete_source_sha256=manifest.digest,
                    candidate_source_paths=paths,
                    use="Untrusted expert data. Re-read candidate source before editing; hypotheses are not verification.")
        if policy.report_version == "readiness-v3":
            view_cap = 50000 - len(json.dumps(additions, ensure_ascii=False)) - 64
            if view_cap < 1000:
                raise EvidenceContextCapacityError("Source mappings exceed bounded consultation view")
            data = json.loads(scheduler.exchange.context(bundle, max_chars=view_cap, compact_excerpts=True))
        else:
            data = json.loads(scheduler.exchange.context(bundle))
        data.update(additions)
        context = json.dumps(data, ensure_ascii=False, allow_nan=False)
        if policy.report_version == "readiness-v3":
            trace.emit("consultation.context.projected", revision_id="bounded-excerpts-v1",
                       count=len(bundle.items), amount=len(context))
        if len(context) > 50000:
            context = ""
            return finish("consultation_no_evidence")
        context_ref = io.put_json(job_id, data)
        return finish()
    except EvidenceContextCapacityError:
        trace.emit("consultation.context.unavailable", status="blocked", reason="bounded_capacity")
        return finish("consultation_context_unavailable")
    except (ValueError, TypeError, OSError, KeyError, RuntimeError):
        # No exception bodies (paths/provider content) enter the metadata trace.
        return finish("source_changed")
