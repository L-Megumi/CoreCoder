"""Synchronous task boundary over the existing diagnosis, repair and verifier."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import APIError

from corecoder.diagnose import run_diagnosis
from corecoder.workflow import load_handoff, run_repair, run_verification, workspace_revision
from researchcoder.contracts import (
    OutputSlice,
    TaskResult,
    TaskSpec,
    UsageSummary,
    VerificationResult,
    VerificationSpec,
    identifier,
)

from .artifacts import ArtifactIO


@dataclass
class AdapterContext:
    """Process-local dependencies. Never serialized into TaskSpec or artifacts."""

    attempt_id: str
    artifacts: ArtifactIO
    model_factory: Callable | None = None
    mode: str = "injected"
    permitted_verifications: tuple[VerificationSpec, ...] = ()
    verification_guard: Callable | None = None
    tool_guard: Callable | None = None
    execution_hooks: object | None = None

    def __post_init__(self):
        identifier(self.attempt_id)
        if self.mode not in ("live", "injected"):
            raise ValueError("Invalid execution mode")
        if not isinstance(self.artifacts, ArtifactIO):
            raise TypeError("ArtifactIO required")


def _slice(value):
    value = value or {}
    return OutputSlice(
        text=value.get("text", ""),
        truncated=value.get("truncated", False),
        original_chars=value.get("original_chars", len(value.get("text", ""))),
        omitted_chars=value.get("omitted_chars", 0),
    )


def _verification(spec, context, raw):
    return VerificationResult(
        job_id=spec.job_id,
        task_id=spec.task_id,
        attempt_id=context.attempt_id,
        verification_id="verification-" + uuid.uuid4().hex,
        status=raw["status"],
        spec=spec.verification,
        exit_code=raw.get("exit_code"),
        stdout=_slice(raw.get("stdout")),
        stderr=_slice(raw.get("stderr")),
        elapsed_seconds=raw.get("elapsed_seconds"),
        timeout_seconds=raw.get("timeout_seconds"),
        launch_error=raw.get("launch_error"),
        trigger=raw.get("trigger", "standalone"),
        checked_workspace_revision=raw.get("checked_workspace_revision"),
    )


class CoreCoderAdapter:
    def execute(self, spec: TaskSpec, context: AdapterContext) -> TaskResult:
        if not isinstance(spec, TaskSpec) or not isinstance(context, AdapterContext):
            raise TypeError("TaskSpec and AdapterContext required")
        root = Path(spec.workspace.root).resolve()
        phase = "input_invalid"
        raw = None
        try:
            phase = "reference_stale"
            if workspace_revision(root) != spec.workspace.source_manifest_sha256:
                raise ValueError("Workspace source revision changed")
            for item in spec.inputs:
                if hasattr(item.reference, "artifact"):
                    context.artifacts.read_record(item.reference)
                else:
                    context.artifacts.read_bytes(item.reference)
            phase = "input_invalid"

            def verification_guard(command, project):
                if (
                    not spec.policy.allow_verification
                    or spec.verification not in context.permitted_verifications
                    or tuple(command) != spec.verification.argv
                    or project != root
                ):
                    raise ValueError("Verification command not authorized by controller")
                if context.verification_guard:
                    context.verification_guard(command, project)

            if spec.verification is not None:
                # Reject before invoking a model or permitting edits, then check again at execution.
                verification_guard(spec.verification.argv, root)

            def tool_guard(name, arguments):
                if name not in spec.policy.allowed_tool_names:
                    raise ValueError("Tool not permitted")
                if name in ("grep", "glob"):
                    target = Path(arguments.get("path", ".")).expanduser().resolve()
                    if not target.is_relative_to(root):
                        raise ValueError("Search target not permitted")
                    if name == "glob" and (Path(arguments["pattern"]).is_absolute() or ".." in Path(arguments["pattern"]).parts):
                        raise ValueError("Glob pattern outside project")
                if name in ("write_file", "edit_file"):
                    target = Path(arguments["file_path"]).expanduser().resolve()
                    if not target.is_relative_to(root) or target.relative_to(root).as_posix() not in spec.policy.writable_paths:
                        raise ValueError("Write target not permitted")
                if context.tool_guard:
                    context.tool_guard(name, arguments)

            inputs = {item.role: item.reference for item in spec.inputs}
            b = spec.budget
            phase = "model_error"
            if spec.kind == "diagnose":
                if context.model_factory is None:
                    raise ValueError("Model factory required")
                options = spec.diagnostic_options
                raw = run_diagnosis(
                    context.artifacts.resolve(inputs["raw_log"]),
                    root,
                    context.model_factory(),
                    max_rounds=b.max_rounds,
                    max_tool_calls=b.max_tool_calls,
                    max_context_tokens=b.max_context_tokens,
                    max_report_repairs=b.max_report_repairs,
                    path_mappings=dict(options.path_mappings),
                    allowed_path_roots=spec.policy.allowed_read_roots,
                    source_hints=options.source_hints,
                    tool_selection=options.tool_selection,
                    context_mode=options.context_mode,
                    completion_mode=options.completion_mode,
                    findings=options.findings,
                    task=spec.objective,
                    mode=context.mode,
                    execution_hooks=context.execution_hooks,
                    compact_report=bool(getattr(getattr(context.execution_hooks, 'optimization', None), 'compact_reports', False)),
                )
            elif spec.kind == "repair":
                phase = "reference_stale"
                handoff = load_handoff(context.artifacts.resolve(inputs["diagnosis_run"]), root)
                phase = "model_error"
                if context.model_factory is None:
                    raise ValueError("Model factory required")
                raw = run_repair(
                    handoff,
                    spec.objective,
                    context.model_factory(),
                    verify_command=list(spec.verification.argv) if spec.verification else None,
                    verification_timeout=spec.verification.timeout_seconds if spec.verification else 30,
                    verification_guard=verification_guard if spec.verification else None,
                    max_rounds=b.max_rounds,
                    max_tool_calls=b.max_tool_calls,
                    max_context_tokens=b.max_context_tokens,
                    max_investigation_rounds=b.max_investigation_rounds,
                    mode=context.mode,
                    allowed_tool_names=spec.policy.allowed_tool_names,
                    tool_guard=tool_guard,
                    execution_hooks=context.execution_hooks,
                )
            else:
                phase = "tool_error"
                result = run_verification(
                    spec.verification.argv, root, spec.verification.timeout_seconds, guard=verification_guard,
                    execution_hooks=context.execution_hooks,
                )
                result["trigger"] = "standalone"
                raw = {
                    "schema_version": 1,
                    "status": result["status"],
                    "verification": result,
                    "project_root": str(root),
                    "metrics": {},
                    "limitations": ["Explicit command checks only."],
                }
        except KeyboardInterrupt:
            raw = {
                "status": "interrupted",
                "adapter_failure": "interrupted",
                "limitations": [
                    "Interrupted inner calls may not expose their in-memory partial investigation. No automatic retry."
                ],
            }
        except (APIError, ValueError, TypeError, OSError, RuntimeError) as exc:
            # Provider/guard messages may contain secrets; record category, never exception bodies.
            raw = {
                "status": "incomplete",
                "adapter_failure": phase,
                "error_type": type(exc).__name__,
                "limitations": ["Adapter preflight or delegated execution did not complete. No automatic retry."],
            }
        return self._result(spec, context, raw)

    def _result(self, spec, context, raw):
        artifact = context.artifacts.put_json(spec.job_id, raw)
        refs = context.artifacts.index_evidence(
            artifact,
            task_id=spec.task_id,
            attempt_id=context.attempt_id,
            workspace_revision=spec.workspace.source_manifest_sha256,
        )
        verification = _verification(spec, context, raw["verification"]) if "verification" in raw else None
        old_status = raw["status"]
        failure = raw.get("adapter_failure")
        output_failure = raw.get("output_failure_kind")
        if old_status == "interrupted":
            status, failure = "cancelled", "interrupted"
        elif (
            spec.kind == "diagnose"
            and old_status == "completed"
            or spec.kind == "repair"
            and old_status in ("verified", "completed_unverified")
            or spec.kind == "verify"
            and old_status == "passed"
        ):
            status = "succeeded"
        elif old_status in ("verification_failed", "failed", "timeout", "launch_error"):
            status = "failed"
            failure = old_status if old_status in ("timeout", "launch_error") else "verification_failed"
            if verification and verification.status in ("timeout", "launch_error"):
                failure = verification.status
        else:
            status = "incomplete"
            stop = raw.get("metrics", {}).get("stop_reason")
            failure = failure or (
                "budget_exhausted"
                if stop in ("round_limit", "tool_request_limit", "investigation_limit")
                else "model_error"
                if stop == "model_api_error"
                else "unknown"
            )
            if output_failure in ('response_length_exceeded', 'invalid_json', 'invalid_report', 'model_error'):
                failure = output_failure
        if verification and verification.status == "launch_error" and status != "succeeded":
            failure = "launch_error"
        metrics = raw.get("metrics", {})
        complete_usage = bool(metrics.get("provider_usage_complete", False))
        changes_ref = context.artifacts.put_json(spec.job_id, raw["file_changes"]) if "file_changes" in raw else None
        summary = (raw.get("diagnosis") or {}).get("summary", raw.get("model_summary", ""))
        return TaskResult(
            job_id=spec.job_id,
            task_id=spec.task_id,
            attempt_id=context.attempt_id,
            kind=spec.kind,
            status=status,
            failure_kind=failure,
            summary=summary[:20000],
            outputs=(artifact,),
            evidence_refs=refs,
            verification=verification,
            file_changes_ref=changes_ref,
            workflow_completed=raw.get("workflow_completed", status == "succeeded"),
            usage=UsageSummary(
                prompt_tokens=metrics.get("prompt_tokens") if complete_usage else None,
                completion_tokens=metrics.get("completion_tokens") if complete_usage else None,
                usage_complete=complete_usage,
                llm_calls=metrics.get("llm_calls", 0),
                tool_calls=metrics.get("tool_calls", 0),
                elapsed_seconds=metrics.get("elapsed_seconds"),
                mode=context.mode,
            ),
            limitations=tuple(raw.get("limitations", [])) + ("No OS sandbox, global scheduling or task recovery in Phase 1.",),
        )
