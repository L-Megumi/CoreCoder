"""Only versioned values cross the worker boundary."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit
from researchcoder.contracts import Contract, TaskSpec, TaskResult, VerificationSpec, ArtifactRef, identifier, digest, text
from researchcoder.context.policy import ContextPolicy
from researchcoder.durable.journal import JournalEndpoint
from .optimization import RuntimeOptimization
from .permissions import PermissionRecoverySpec
from .coding import CodingContinuationPolicy


@dataclass(frozen=True, kw_only=True)
class RuntimeLimits(Contract):
    max_model_calls: int = 24
    max_tool_calls: int = 24
    max_rounds: int = 12
    max_runtime_seconds: float = 60.0
    max_context_tokens: int = 128000
    max_output_tokens: int = 2048
    max_total_tokens: int = 1000000
    schema_version: int = 1

    def validate(self):
        for value in (
            self.max_model_calls,
            self.max_tool_calls,
            self.max_rounds,
            self.max_context_tokens,
            self.max_output_tokens,
            self.max_total_tokens,
        ):
            if value < 1:
                raise ValueError("Limits must be positive")
        if not 0.05 <= self.max_runtime_seconds <= 3600:
            raise ValueError("Invalid runtime duration")
        if self.max_output_tokens >= self.max_context_tokens:
            raise ValueError("Output reserve must fit context")
        if max(self.max_model_calls, self.max_tool_calls, self.max_rounds) > 1000:
            raise ValueError("Runtime counter limit too large")
        if self.max_context_tokens > 1000000 or self.max_total_tokens > 100000000:
            raise ValueError("Runtime token limit too large")


@dataclass(frozen=True, kw_only=True)
class ScriptedCall(Contract):
    call_id: str
    name: str
    arguments_json: str = "{}"
    schema_version: int = 1

    def validate(self):
        identifier(self.call_id)
        identifier(self.name)
        import json

        if len(self.arguments_json) > 20000 or not isinstance(json.loads(self.arguments_json), dict):
            raise ValueError("Expected bounded argument object")


@dataclass(frozen=True, kw_only=True)
class ScriptedTurn(Contract):
    content: str = ""
    calls: tuple[ScriptedCall, ...] = ()
    prompt_tokens: int = 0
    completion_tokens: int = 0
    schema_version: int = 1

    def validate(self):
        if len(self.content) > 20000 or min(self.prompt_tokens, self.completion_tokens) < 0:
            raise ValueError("Invalid scripted turn")


@dataclass(frozen=True, kw_only=True)
class ModelSpec(Contract):
    provider: Literal["scripted", "openai"] = "scripted"
    model: str = "scripted-runtime"
    turns: tuple[ScriptedTurn, ...] = ()
    base_url: str | None = None
    credential_env: Literal["CORECODER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"] = "OPENAI_API_KEY"
    schema_version: int = 1

    def validate(self):
        text(self.model, 120)
        if self.model.startswith("sk-") or any(c.isspace() for c in self.model):
            raise ValueError("Invalid model label")
        if len(self.turns) > 100:
            raise ValueError("Too many scripted turns")
        if self.provider == "openai" and self.turns:
            raise ValueError("Live profile cannot contain scripted data")
        if self.base_url:
            u = urlsplit(self.base_url)
            if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment:
                raise ValueError("Endpoint must not contain credentials/query")


@dataclass(frozen=True, kw_only=True)
class ProtectedFile(Contract):
    path: str
    sha256: str
    schema_version: int = 1

    def validate(self):
        if not self.path.startswith("/") or "\x00" in self.path:
            raise ValueError("Expected absolute protected path")
        digest(self.sha256)


@dataclass(frozen=True, kw_only=True)
class ModelRequestOptions(Contract):
    """Explicit trusted DeepSeek non-thinking profile; no arbitrary provider JSON."""

    thinking: Literal["disabled"] = "disabled"
    schema_version: int = 1


def model_request_kwargs(model, options):
    if options is None:
        return {}
    if type(options) is not ModelRequestOptions or model.provider != "openai" or urlsplit(model.base_url or "").hostname != "api.deepseek.com":
        raise ValueError("Non-thinking profile requires explicit DeepSeek OpenAI-format endpoint")
    return {"extra_body": {"thinking": {"type": options.thinking}}}


@dataclass(frozen=True, kw_only=True)
class PlanningTaskSpec(TaskSpec):
    """Separate control-task wire kind; Phase 1 TaskSpec still rejects kind=plan."""

    kind: Literal["plan"] = "plan"

    def validate(self):
        super().validate()
        if self.policy.allowed_tool_names or self.policy.writable_paths or self.policy.allow_verification or self.verification:
            raise ValueError("Planning is a tool-free control task")


@dataclass(frozen=True, kw_only=True)
class PlanningTaskResult(TaskResult):
    kind: Literal["plan"] = "plan"

    def validate(self):
        super().validate()
        if self.verification or self.file_changes_ref or self.evidence_refs:
            raise ValueError("A plan is not a source observation or a verified patch")


@dataclass(frozen=True, kw_only=True)
class CodingTaskSpec(TaskSpec):
    """Explicit single-Agent benchmark route, not a Phase 1 repair handoff."""

    kind: Literal["coding"] = "coding"

    def validate(self):
        super().validate()
        if 'raw_log' not in {i.role for i in self.inputs} or not self.verification or not self.policy.writable_paths:
            raise ValueError("Coding baseline requires original log, bounded writes and independent verification")


@dataclass(frozen=True, kw_only=True)
class CodingTaskResult(TaskResult):
    kind: Literal["coding"] = "coding"

    def validate(self):
        super().validate()
        if self.status == 'succeeded' and (self.verification is None or self.verification.status != 'passed'):
            raise ValueError("Coding baseline success requires actual independent verification")


@dataclass(frozen=True, kw_only=True)
class WorkerRequest(Contract):
    spec: TaskSpec | PlanningTaskSpec | CodingTaskSpec
    attempt_id: str
    artifact_root: str
    trace_path: str
    parent_event_id: str
    limits: RuntimeLimits = field(default_factory=RuntimeLimits)
    model: ModelSpec = field(default_factory=ModelSpec)
    model_options: ModelRequestOptions | None = None
    permitted_verifications: tuple[VerificationSpec, ...] = ()
    protected_files: tuple[ProtectedFile, ...] = ()
    diagnostic_tools: tuple[str, ...] = (
        "extract_log_evidence",
        "locate_source",
        "read_project_file",
        "inspect_paths",
        "search_project_source",
        "resolve_project_symbol",
    )
    artifact_namespace: str | None = None
    system_instructions: str = ""
    evidence_context: str = ""
    context_policy: ContextPolicy | None = None
    evidence_bundle: ArtifactRef | None = None
    journal_endpoint: JournalEndpoint | None = None
    record_requests: bool = False
    optimization: RuntimeOptimization | None = None
    permission_recovery: PermissionRecoverySpec | None = None
    coding_continuation: CodingContinuationPolicy | None = None
    schema_version: int = 1

    def to_dict(self):
        value = super().to_dict()
        if not self.record_requests:
            value.pop("record_requests")
        if self.optimization is None:
            value.pop("optimization")
        if self.permission_recovery is None:
            value.pop("permission_recovery")
        if self.coding_continuation is None:
            value.pop("coding_continuation")
        else:
            value["coding_continuation"] = self.coding_continuation.to_dict()
        if isinstance(value.get('context_policy'), dict) and value['context_policy'].get('coding_source_mode') == 'off':
            value['context_policy'].pop('coding_source_mode')
        if isinstance(value.get('context_policy'), dict) and value['context_policy'].get('repair_state_version') == 'off':
            value['context_policy'].pop('repair_state_version')
        if isinstance(value.get('context_policy'), dict) and value['context_policy'].get('expert_followup_version') == 'off':
            value['context_policy'].pop('expert_followup_version')
        return value

    def validate(self):
        model_request_kwargs(self.model, self.model_options)
        if self.coding_continuation is not None and self.spec.kind != 'coding':
            raise ValueError("Local Coding stop policy requires a coding task")
        if self.permission_recovery is not None:
            if self.spec.kind != 'coding':
                raise ValueError("Public-check parameter recovery is limited to the coding Worker")
            protected = {p.path for p in self.protected_files}
            if self.spec.verification not in self.permitted_verifications:
                raise ValueError('Recovery targets require the permitted current public verification')
            arguments = set(self.spec.verification.argv[1:])
            if not set(self.permission_recovery.public_check_paths) <= protected & arguments:
                raise ValueError("Recovery targets must be protected public command inputs")
        if self.journal_endpoint is not None and (self.journal_endpoint.job_id,self.journal_endpoint.task_id,self.journal_endpoint.attempt_id)!=(self.spec.job_id,self.spec.task_id,self.attempt_id):
            raise ValueError("Foreign journal endpoint")
        if self.evidence_bundle is not None and (self.context_policy is None or self.evidence_bundle.job_id != self.spec.job_id):
            raise ValueError("Context policy and same-Job bundle required")
        if self.artifact_namespace is not None:
            identifier(self.artifact_namespace)
        if len(self.system_instructions) > 6000 or len(self.evidence_context) > 50000:
            raise ValueError("Role context exceeds protocol bounds")
        if self.evidence_context:
            import json

            if not isinstance(json.loads(self.evidence_context), dict):
                raise ValueError("Structured evidence context required")
        identifier(self.attempt_id)
        identifier(self.parent_event_id)
        for path in (self.artifact_root, self.trace_path):
            if not path.startswith("/") or "\x00" in path:
                raise ValueError("Expected controller absolute path")
        if not set(self.diagnostic_tools) <= {
            "extract_log_evidence",
            "locate_source",
            "read_project_file",
            "inspect_paths",
            "search_project_source",
            "resolve_project_symbol",
        }:
            raise ValueError("Unsupported diagnostic capability")
        import os
        import re

        raw = self.to_json()
        if re.search(r"\bsk-[A-Za-z0-9_-]{12,}", raw):
            raise ValueError("Worker request must not contain credential material")
        for name in ("CORECODER_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
            secret = os.environ.get(name)
            if secret and len(secret) >= 8 and secret in raw:
                raise ValueError("Worker request must not contain configured credentials")


@dataclass(frozen=True, kw_only=True)
class WorkerResponse(Contract):
    result: TaskResult | PlanningTaskResult | CodingTaskResult
    budget_json: str
    schema_version: int = 1

    def validate(self):
        import json

        if len(self.budget_json) > 500000 or not isinstance(json.loads(self.budget_json), dict):
            raise ValueError("Invalid budget snapshot")
