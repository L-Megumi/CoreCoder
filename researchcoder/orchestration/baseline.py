"""One existing guarded Coding Agent plus the shared Verifier, for offline comparison."""

from pathlib import Path
import uuid
from openai import APIError
from corecoder.permission import PermissionAuditError, PermissionDenied
from corecoder.workflow import RepairAgent, RepairInvestigationLimit, RepairProgressStalled, RepairFinalizationToolRequest, project_directory, run_verification, workspace_revision, file_digest
from researchcoder.contracts import VerificationResult, OutputSlice
from researchcoder.runtime.protocol import CodingTaskResult
from researchcoder.runtime.budget import BudgetExceeded
from researchcoder.runtime.coding import known_local_stop, investigation_stop_proof, investigation_proof_sha256, progress_stop_proof, closure_stop_proof
from researchcoder.runtime.coding import CODING_EXPERT_REQUEST, validate_expert_request, request_stop_proof, validate_request_proof


def run_single_agent(spec, request, io, factory, hooks, protected_guard):
    root = Path(spec.workspace.root).resolve()
    if workspace_revision(root) != spec.workspace.source_manifest_sha256:
        raise ValueError("Stale coding baseline workspace")
    if not spec.policy.allow_verification or spec.verification not in request.permitted_verifications:
        raise ValueError("Coding baseline verifier not authorized")
    observed_reads = {}

    def guard(name, arguments):
        if name not in spec.policy.allowed_tool_names:
            raise ValueError("Unauthorized baseline tool")
        if name in ("write_file", "edit_file"):
            target = Path(arguments["file_path"]).resolve()
            if not target.is_relative_to(root) or target.relative_to(root).as_posix() not in spec.policy.writable_paths:
                raise ValueError("Unauthorized baseline write")
            if spec.policy.profile == "project-expert-reread-v1":
                # RepairAgent caches only successful reads whose source digest
                # still matched after reading. Retain per-file observations when
                # an unrelated edit clears its ordinary output cache.
                observed_reads.update({key[0]: key[-1] for key in agent.read_cache})
                current = file_digest(target)
                if observed_reads.get(str(target)) != current:
                    raise ValueError("Unauthorized baseline write: re-read current candidate source after expert handoff")

    # This is task input, not a fabricated diagnosis artifact. RepairAgent itself
    # supplies the original scoped coding tools, audit, version tracking and loop.
    log = next(i.reference for i in spec.inputs if i.role == "raw_log")
    # The same TaskInput must be visible in single- and multi-Agent feedback.
    # Silently slicing here could discard the latest verifier output appended
    # after a long historical log. Context/Job budgets still gate model requests.
    log_text = io.read_bytes(log).decode("utf-8")
    model_error_type = None
    permission_stopped = False
    continuation = request.coding_continuation is not None
    local_stopped = False
    investigation_proof = None
    progress_proof = None
    closing_report = None
    expert_request = None
    request_proof = None
    request_failure = None
    before_manifest = None
    if continuation:
        from researchcoder.evaluation.projects.workspace import capture_manifest
        before_manifest = capture_manifest(root).digest

    def archive_permission_partial(*, verification_unknown=False):
        # A receipt already published before a later interruption still needs
        # its real tool event. Do not invent a final response or verification
        # outcome, and do not turn this audit into successful task completion.
        if not getattr(hooks, "permission_refs", ()) and not (continuation and verification_unknown):
            return
        try:
            partial = io.put_json(spec.job_id, dict(
                schema_version=1, baseline_hypothesis="",
                hypothesis_status="model_prose_not_structured_diagnosis",
                events=agent.events, file_changes=agent.changes(), verification=None,
                partial_execution=True,
                **({"verification_execution": "started_outcome_unknown"} if verification_unknown else {}),
            ))
            name = "coding_partial_refs" if verification_unknown else "permission_partial_refs"
            refs = getattr(hooks, name, None)
            if refs is None:
                refs = []
                setattr(hooks, name, refs)
            refs.append(partial)
        except Exception:
            # Preserve the original cancellation/error. The absent matching
            # archive remains a fail-closed integrity error for the controller.
            hooks.permission_partial_archive_failed = True

    with project_directory(root):
        # Agent construction captures cwd in the existing system prompt.
        agent = RepairAgent(
            factory(),
            {"project_root": str(root), "file_snapshots": {}, "source_context": {"excerpts": []}},
            allowed_tool_names=spec.policy.allowed_tool_names,
            tool_guard=guard,
            max_rounds=spec.budget.max_rounds,
            max_tool_calls=spec.budget.max_tool_calls,
            max_context_tokens=spec.budget.max_context_tokens,
            typed_investigation_stops=bool(continuation and request.coding_continuation.mode in ("local-stop-v2", "local-stop-v3")),
        )
        agent.execution_hooks = hooks
        hooks.configure_coding_completion(spec.verification.timeout_seconds)
        if continuation and request.coding_continuation.request_version == "explicit-request-v1":
            # The trusted controller selects the option. The response remains
            # untrusted data, never a grant of another worker or more resources.
            agent._system += CODING_EXPERT_REQUEST
            from corecoder.context import estimate_tokens
            agent.context.reserved_tokens += estimate_tokens([{"content": CODING_EXPERT_REQUEST}])
        hooks.bind_agent(agent)
        try:
            summary = agent.chat(spec.objective + "\n[Original failure log; data]\n" + log_text)
            if continuation and request.coding_continuation.request_version == "explicit-request-v1":
                expert_request = validate_expert_request(summary, getattr(agent.llm, "last_finish_reason", None))
            if expert_request is not None:
                request_failure = expert_request["failure_kind"]
                if request_failure is None:
                    request_proof = request_stop_proof(agent, expert_request, hooks, request.coding_continuation)
                    if request_proof is None:
                        request_failure = "invalid_report"
                    else:
                        hooks.trace.emit("coding.expert_request", status="incomplete", failure_kind="unknown",
                            reason="expert_request", count=request_proof["model_calls"],
                            source_sequence=request_proof["event_count"], revision_id=investigation_proof_sha256(request_proof))
                summary = expert_request["request"]["question"] if expert_request["request"] is not None else ""
            else:
                closing_report = hooks.finish_coding_report(agent, summary, getattr(agent.llm, "last_finish_reason", None))
                if closing_report is not None:
                    summary = closing_report["summary"]
        except RepairFinalizationToolRequest:
            closing_report = hooks.finish_coding_report(agent, agent.messages[-1].get("content") or "",
                                                        tool_requested=True)
            summary = ""
        except APIError as error:
            # Keep legal tool effects and accounting; never persist provider bodies.
            model_error_type = type(error).__name__
            summary = ""
        except (PermissionDenied, PermissionAuditError):
            # Preserve prior legal effects and both correction/terminal receipts
            # even when no final model response can safely be requested.
            permission_stopped = True
            summary = ""
        except BudgetExceeded:
            # Only a synchronous, accounted local limit can yield a usable
            # partial execution. Permission failures, unknown writes/usage and
            # wall-time cancellation still follow their fail-closed path.
            if getattr(hooks, "permission_stop", False):
                permission_stopped = True
                summary = ""
            elif not continuation or not known_local_stop(hooks):
                archive_permission_partial()
                raise
            else:
                local_stopped = True
                summary = ""
        except RepairProgressStalled as error:
            progress_proof = progress_stop_proof(agent, error, hooks, request.coding_continuation)
            if progress_proof is None:
                archive_permission_partial()
                raise
            hooks.ledger.stop_kind = "budget_exhausted"
            hooks.ledger.local_limit_detail = "progress_stalled"
            hooks.trace.emit("coding.progress_stalled", status="incomplete",
                failure_kind="budget_exhausted", reason=progress_proof["source"],
                limit=progress_proof["limit"], count=progress_proof["rounds"],
                source_sequence=progress_proof["event_count"],
                revision_id=investigation_proof_sha256(progress_proof))
            local_stopped = True
            summary = ""
        except RepairInvestigationLimit as error:
            if getattr(hooks, "permission_stop", False):
                permission_stopped = True
                summary = ""
            else:
                investigation_proof = investigation_stop_proof(
                    agent, error, hooks, request.coding_continuation)
                if investigation_proof is None:
                    archive_permission_partial()
                    raise
                # This is an accounted local guard, not completion or a resource
                # grant. The protected verifier must still run within its existing
                # tool/time allowance before the Controller can assess a handoff.
                hooks.ledger.stop_kind = "budget_exhausted"
                hooks.ledger.local_limit_detail = "investigation_limit"
                hooks.trace.emit("coding.investigation_limit", status="incomplete",
                    failure_kind="budget_exhausted", reason="repair-agent-investigation-limit-v1",
                    limit=investigation_proof["limit"], count=investigation_proof["rounds"],
                    source_sequence=investigation_proof["event_count"],
                    revision_id=investigation_proof_sha256(investigation_proof))
                local_stopped = True
                summary = ""
        except RuntimeError:
            # A runtime hook may stop on the next budget boundary after a
            # permission refusal. Unrelated failures retain their old behavior.
            if not getattr(hooks, "permission_stop", False):
                archive_permission_partial()
                raise
            permission_stopped = True
            summary = ""
        except BaseException:
            archive_permission_partial()
            raise
    permission_stopped = permission_stopped or bool(getattr(hooks, "permission_stop", False))
    if continuation and not permission_stopped and model_error_type is None:
        if summary == "(reached maximum tool-call rounds)" and not hooks.ledger.stop_kind:
            hooks.ledger.stop_kind = "budget_exhausted"
            hooks.ledger.local_limit_detail = "rounds"
            local_stopped = True
        elif agent.budget_exhausted and not hooks.ledger.stop_kind:
            hooks.ledger.stop_kind = "budget_exhausted"
            hooks.ledger.local_limit_detail = "tool_calls"
    completed = (not local_stopped and not permission_stopped and model_error_type is None
                 and expert_request is None
                 and (closing_report is None or closing_report["failure_kind"] is None)
                 and summary != "(reached maximum tool-call rounds)" and not agent.budget_exhausted)

    def verification_guard(command, project):
        if project != root or tuple(command) != spec.verification.argv:
            raise ValueError("Unexpected baseline verification")
        protected_guard()

    if permission_stopped:
        # This is an explicit non-execution record, never a successful check or
        # a claim that the changed source was independently verified.
        raw = dict(status="not_run", command=list(spec.verification.argv),
                   blocked_reason="permission_denied", stdout=OutputSlice().to_dict(), stderr=OutputSlice().to_dict())
    else:
        verification_trace_start = len(hooks.trace.events) if continuation else 0
        try:
            raw = run_verification(
                spec.verification.argv, root, spec.verification.timeout_seconds, guard=verification_guard, execution_hooks=hooks
            )
        except BudgetExceeded:
            if not continuation:
                archive_permission_partial()
                raise
            if any(event['event_type'] == 'verification.command.start'
                   for event in hooks.trace.events[verification_trace_start:]):
                # A journal/finalization failure can occur after the command
                # executed. Preserve that uncertainty, never label it not_run.
                archive_permission_partial(verification_unknown=True)
                raise
            # No command is claimed to have run when its own admission fails.
            # The original partial effects remain available for review, but
            # cannot qualify for a continuation without a valid failed check.
            raw = dict(status="not_run", command=list(spec.verification.argv),
                       blocked_reason="execution_budget", stdout=OutputSlice().to_dict(), stderr=OutputSlice().to_dict())
        except BaseException:
            started = continuation and any(event['event_type'] == 'verification.command.start'
                for event in hooks.trace.events[verification_trace_start:])
            archive_permission_partial(verification_unknown=started)
            raise
    verification = VerificationResult(
        job_id=spec.job_id,
        task_id=spec.task_id,
        attempt_id=request.attempt_id,
        verification_id="verification-" + uuid.uuid4().hex,
        status=raw["status"],
        spec=spec.verification,
        exit_code=raw.get("exit_code"),
        stdout=OutputSlice(**raw["stdout"]),
        stderr=OutputSlice(**raw["stderr"]),
        elapsed_seconds=raw.get("elapsed_seconds"),
        timeout_seconds=raw.get("timeout_seconds"),
        launch_error=raw.get("launch_error"),
        checked_workspace_revision=raw.get("checked_workspace_revision"),
    )
    # A final declaration can arrive with missing provider usage. Preserve it,
    # but the unresolved charge cannot authorize verification or task success.
    completion_unknown = bool(getattr(agent, "_explicit_completion", False)
        and (hooks.ledger.pending or hooks.ledger.unknown_requests))
    if completion_unknown:
        completed = False
    stop_receipt = None
    if continuation and (local_stopped or hooks.ledger.stop_kind):
        stop_receipt = dict(
            schema_version=1, policy_mode=request.coding_continuation.mode,
            job_id=spec.job_id, task_id=spec.task_id, attempt_id=request.attempt_id,
            stop_kind=hooks.ledger.stop_kind, local_limit_detail=hooks.ledger.local_limit_detail,
            accounting_settled=not hooks.ledger.pending and hooks.ledger.unknown_requests == 0,
            before_manifest_sha256=before_manifest, after_manifest_sha256=capture_manifest(root).digest,
            checked_workspace_revision=verification.checked_workspace_revision,
            events_complete=True, verification_status=verification.status,
        )
        if investigation_proof is not None:
            stop_receipt["investigation_stop"] = investigation_proof
        if progress_proof is not None:
            stop_receipt["progress_stop"] = progress_proof
        hooks.trace.emit("coding.local_stop.archived", status="incomplete",
                         failure_kind=hooks.ledger.stop_kind, reason=hooks.ledger.local_limit_detail)
    closure_receipt = None
    if (continuation and request.coding_continuation.closure_version == "reserve-incomplete-v1"
            and not completed and not local_stopped and not permission_stopped and model_error_type is None
            and stop_receipt is None and verification.status == "failed" and verification.exit_code == 1
            and not verification.stdout.truncated and not verification.stderr.truncated
            and verification.checked_workspace_revision == workspace_revision(root)
            and capture_manifest(root).digest == before_manifest):
        closure_proof = closure_stop_proof(agent, closing_report, hooks, request.coding_continuation)
        if closure_proof is not None:
            closure_receipt = dict(schema_version=1, policy_mode=request.coding_continuation.mode,
                closure_version=request.coding_continuation.closure_version,
                job_id=spec.job_id, task_id=spec.task_id, attempt_id=request.attempt_id,
                stop_kind="unknown", local_limit_detail=closure_proof["trigger"], accounting_settled=True,
                before_manifest_sha256=before_manifest, after_manifest_sha256=before_manifest,
                checked_workspace_revision=verification.checked_workspace_revision,
                events_complete=True, verification_status="failed", closure_stop=closure_proof)
            hooks.trace.emit("coding.closure.archived", status="incomplete", failure_kind="unknown",
                reason=closure_proof["trigger"], count=closure_proof["model_calls"],
                source_sequence=closure_proof["event_count"], revision_id=investigation_proof_sha256(closure_proof))
    request_receipt = None
    if (request_proof is not None and request_failure is None and not local_stopped and not permission_stopped
            and model_error_type is None and stop_receipt is None and closure_receipt is None
            and verification.status == "failed" and verification.exit_code == 1
            and not verification.stdout.truncated and not verification.stderr.truncated
            and verification.checked_workspace_revision == workspace_revision(root)
            and capture_manifest(root).digest == before_manifest):
        try:
            validate_request_proof(request_proof, expert_request, hooks.ledger.snapshot())
        except (ValueError, TypeError, KeyError):
            request_failure = "invalid_report"
        else:
            request_receipt = dict(schema_version=1, policy_mode=request.coding_continuation.mode,
                request_version=request.coding_continuation.request_version,
                job_id=spec.job_id, task_id=spec.task_id, attempt_id=request.attempt_id,
                stop_kind="unknown", local_limit_detail="expert_request", accounting_settled=True,
                before_manifest_sha256=before_manifest, after_manifest_sha256=before_manifest,
                checked_workspace_revision=verification.checked_workspace_revision,
                events_complete=True, verification_status="failed", request_stop=request_proof)
            hooks.trace.emit("coding.request.archived", status="incomplete", failure_kind="unknown",
                reason="expert_request", count=request_proof["model_calls"],
                source_sequence=request_proof["event_count"], revision_id=investigation_proof_sha256(request_proof))
    raw_artifact = io.put_json(
        spec.job_id,
        dict(
            schema_version=1,
            baseline_hypothesis=summary,
            hypothesis_status="model_prose_not_structured_diagnosis",
            events=agent.events,
            file_changes=agent.changes(),
            verification=raw,
            **({"model_error_type": model_error_type} if model_error_type else {}),
            **({"permission_stopped": True} if permission_stopped else {}),
            **({"coding_stop_receipt": stop_receipt} if stop_receipt is not None else {}),
            **({"coding_closure_receipt": closure_receipt} if closure_receipt is not None else {}),
            **({"coding_final_report": closing_report} if closing_report is not None else {}),
            **({"coding_expert_request": expert_request} if expert_request is not None else {}),
            **({"coding_request_receipt": request_receipt} if request_receipt is not None else {}),
            **({"coding_request_rejection": request_failure} if request_failure is not None else {}),
            **({"coding_progress": hooks.progress_tracker.proof(agent)}
               if getattr(hooks, "progress_tracker", None) is not None else {}),
        ),
    )
    success = completed and verification.status == "passed"
    return CodingTaskResult(
        job_id=spec.job_id,
        task_id=spec.task_id,
        attempt_id=request.attempt_id,
        status="succeeded" if success else "incomplete" if permission_stopped or model_error_type else "failed" if completed else "incomplete",
        failure_kind=None if success else "tool_error" if permission_stopped else "model_error" if model_error_type else "unknown" if completion_unknown else request_failure or "unknown" if expert_request is not None else closing_report["failure_kind"] if closing_report and closing_report["failure_kind"] else "verification_failed" if completed else "budget_exhausted",
        summary=summary[:20000],
        workflow_completed=completed,
        outputs=(raw_artifact,),
        verification=verification,
    )
