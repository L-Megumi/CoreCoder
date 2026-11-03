"""Deterministic public-feedback routing, without a model, executor or Oracle.

The controller supplies explicit observations and remaining responsibility limits.
A decision is an auditable request, never a grant of tool or filesystem authority.
Execution must recheck budgets, the source version and trusted role permissions.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Literal

from researchcoder.contracts import Contract, digest, identifier
from researchcoder.runtime.protocol import RuntimeLimits
from .contracts import MAX_FILES, source_path


EXPERT_LIMITS = RuntimeLimits(max_model_calls=4, max_tool_calls=6, max_rounds=4,
    max_total_tokens=80000, max_runtime_seconds=45, max_context_tokens=16000, max_output_tokens=2048)

# Opt-in capacity for actual cross-file reading; legacy v1 admission is unchanged.
READINESS_EXPERT_LIMITS = RuntimeLimits(max_model_calls=6, max_tool_calls=10, max_rounds=6,
    max_total_tokens=80000, max_runtime_seconds=45, max_context_tokens=16000, max_output_tokens=2048)

# Opt-in wall-time profile for streamed providers; call/tool/token caps stay identical.
STREAMING_EXPERT_LIMITS = replace(READINESS_EXPERT_LIMITS, max_runtime_seconds=180)

# This text is part of the policy identity. Change the version and digest when
# changing a threshold, precedence or feature interpretation.
ROUTING_RULES = """public-failure-v1
Only explicit public log, public verifier output, source filenames, public failure
signatures, controller safety observations and remaining Job allowance are inputs.
Stop on cancellation, permission violation, source change, pending responsibility,
unknown usage, incomplete workflow, invalid public verification, nonbusiness failure,
or no remaining writer retry. Public success stops further Agent execution.
Off mode never requests experts and leaves legacy Single budget admission unchanged.
At most one consultation episode is permitted.
Code is always first. Log is additionally requested only when one individual public
text has at least two exact Python traceback headings and at least one exact chained
exception cause/context separator. Duplicate copies across texts do not add counts.
Admission reserves the sum of all selected expert model/tool/token maxima plus the
explicit writer minima. Parallel expert time reserves the largest role duration plus
the writer minimum. If two experts do not fit, try Code alone. If Code does not fit
but a writer does, keep Single; otherwise stop. Once consultation is used, continue
Single only for a new public failure signature, otherwise stop. No unknown write is
replayed. No project identifier, hidden case label, reference patch or final score is
a feature. A decision does not confer permissions or establish repair success.
"""
RULES_SHA256 = hashlib.sha256(ROUTING_RULES.encode("utf-8")).hexdigest()
V2_ROUTING_RULES = ROUTING_RULES + """\nserial-parallel-v2 / local-stop-v1
For v2 scheduling both modes admit the SUM of all expert time limits plus writer
time, with identical role selection. Only Scheduler.max_workers differs.
A Controller-validated local-stop receipt can admit an incomplete Coding attempt
without changing its status. It must bind known accounting, known file effects,
the current source and an actually executed protected public failure. It cannot
permit unknown effects, repeat a handoff, or replace an unavailable writer retry.
If experts cannot be admitted after a local stop, stop; do not reset Single limits.
"""
V2_RULES_SHA256 = hashlib.sha256(V2_ROUTING_RULES.encode("utf-8")).hexdigest()
V3_ROUTING_RULES = V2_ROUTING_RULES + """\nlocal-stop-v3 / progress-v1
The Controller may additionally admit a progress-stalled local stop after three
completed tool rounds without novel successful evidence or effective file edits.
The stopped attempt remains incomplete. A reproduced progress receipt, known
accounting, known candidate effects and an actually failed protected public check
are required. Expert and writer allowances still come from the original Job.
"""
V3_RULES_SHA256 = hashlib.sha256(V3_ROUTING_RULES.encode("utf-8")).hexdigest()
CLOSURE_ROUTING_RULES = V3_ROUTING_RULES + """\nreserve-incomplete-v1
A separately validated closure receipt may admit a valid model-declared incomplete
report after a Controller round/model-call reserve. Keep the original unknown
failure classification; this is not unknown Usage or successful completion.
Require zero writes, unchanged source, fully settled known Usage, complete trace
and an actually failed protected public check. Only the existing one consultation
episode and writer retry may use the original Job's remaining allowance.
"""
CLOSURE_RULES_SHA256 = hashlib.sha256(CLOSURE_ROUTING_RULES.encode("utf-8")).hexdigest()
REQUEST_ROUTING_RULES = CLOSURE_ROUTING_RULES + """\nexplicit-request-v1
A voluntary writer request may pause before a local limit. Admission requires a
Controller-reproduced same-Job request receipt, successful observed source reads,
zero writes, unchanged source, known settled Usage and an actual protected public
failure. The original attempt remains incomplete/unknown; this is not budget
exhaustion. A bounded untrusted question requests the existing Code expert only.
Use the same expert and writer reservation floors, one-episode and retry limits.
No additional budget, role, tool or path authority follows from the request.
"""
REQUEST_RULES_SHA256 = hashlib.sha256(REQUEST_ROUTING_RULES.encode("utf-8")).hexdigest()
INITIAL_ROUTING_RULES = ROUTING_RULES + """\ncode-first-v1
An explicitly selected initial consultation may use the real preflight-public
Verifier failure before any Coding attempt. Its original workflow_completed
value is retained; an independently validated immutable initial receipt proves
the actually completed protected command, current source and settled execution.
Only the existing Code expert may run. Original Job allowance, writer reserve,
single episode and unknown/permission/source stops remain in force.
"""
INITIAL_RULES_SHA256 = hashlib.sha256(INITIAL_ROUTING_RULES.encode("utf-8")).hexdigest()
_TRACEBACK = re.compile(r"(?m)^\s*Traceback \(most recent call last\):\s*$")
_CHAIN = re.compile(r"(?m)^\s*(?:The above exception was the direct cause of the following exception:|"
                    r"During handling of the above exception, another exception occurred:)\s*$")


def _sha(value):
    if isinstance(value, Contract):
        raw = value.to_json()
    else:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bounded_text(value, maximum):
    if type(value) is not str or len(value) > maximum or "\x00" in value:
        raise ValueError("Expected bounded public text")


@dataclass(frozen=True, kw_only=True)
class ExpertPolicy(Contract):
    mode: Literal["off", "public-failure-v1"] = "off"
    max_episodes: int = 1
    expert_limits: RuntimeLimits = EXPERT_LIMITS
    writer_min_model_calls: int = 3
    writer_min_tool_calls: int = 3
    writer_min_total_tokens: int = 20000
    writer_min_seconds: float = 15.0
    schedule_version: Literal["legacy-v1", "serial-parallel-v2"] = "legacy-v1"
    max_workers: int | None = None
    report_version: Literal["legacy-v1", "explicit-v1", "readiness-v2", "readiness-v3"] = "legacy-v1"
    working_set_version: Literal["legacy-v1", "active-v1"] = "legacy-v1"
    initial_mode: Literal["off", "code-first-v1"] = "off"
    assignment_version: Literal["off", "focused-v1"] = "off"
    latency_profile: Literal["legacy-v1", "streaming-v1"] = "legacy-v1"
    schema_version: int = 1

    def validate(self):
        if self.latency_profile != "legacy-v1" and (self.mode != "public-failure-v1" or self.report_version not in ("readiness-v2", "readiness-v3")):
            raise ValueError("Streaming latency requires enabled evidence-ready experts")
        if self.assignment_version != "off" and (self.mode != "public-failure-v1" or self.report_version not in ("readiness-v2", "readiness-v3")):
            raise ValueError("Focused assignment requires enabled evidence-ready experts")
        if self.initial_mode != "off" and self.mode != "public-failure-v1":
            raise ValueError("Initial consultation requires enabled expert policy")
        if self.report_version in ("explicit-v1", "readiness-v2", "readiness-v3") and (self.expert_limits.max_model_calls < 2
                or self.expert_limits.max_rounds < 2):
            raise ValueError("Explicit report and bounded correction need existing request/round allowance")
        if (self.schedule_version == "legacy-v1" and self.max_workers is not None
                or self.schedule_version == "serial-parallel-v2" and self.max_workers not in (1, 2)):
            raise ValueError("Explicit v2 expert concurrency must be one or two")
        if not 0 <= self.max_episodes <= 1:
            raise ValueError("At most one bounded expert episode")
        cap = READINESS_EXPERT_LIMITS if self.report_version in ("readiness-v2", "readiness-v3") else EXPERT_LIMITS
        if self.latency_profile == "streaming-v1":
            cap = STREAMING_EXPERT_LIMITS
        for name in ("max_model_calls", "max_tool_calls", "max_rounds", "max_total_tokens",
                     "max_runtime_seconds", "max_context_tokens", "max_output_tokens"):
            if getattr(self.expert_limits, name) > getattr(cap, name):
                raise ValueError("Expert policy exceeds the reviewed readiness-v2 cap" if self.report_version in ("readiness-v2", "readiness-v3")
                                 else "Expert policy exceeds the reviewed v1 cap")
        if self.expert_limits.max_context_tokens < 1000:
            raise ValueError("Expert context must satisfy the existing TaskSpec minimum")
        if min(self.writer_min_model_calls, self.writer_min_tool_calls, self.writer_min_total_tokens) < 1:
            raise ValueError("Explicit positive writer reserve required")
        if not 0.1 <= self.writer_min_seconds <= 3600:
            raise ValueError("Explicit bounded writer time reserve required")

    @property
    def sha256(self):
        return _sha(self)

    def to_dict(self):
        value = super().to_dict()
        if self.latency_profile == "legacy-v1":
            value.pop("latency_profile")
        if self.assignment_version == "off":
            value.pop("assignment_version")
        if self.initial_mode == "off":
            value.pop("initial_mode")
        if self.report_version == "legacy-v1":
            value.pop("report_version")
        if self.working_set_version == "legacy-v1":
            value.pop("working_set_version")
        if self.schedule_version == "legacy-v1":
            value.pop("schedule_version")
            value.pop("max_workers")
        return value

    def expert_seconds(self, count):
        return (count if self.schedule_version == "serial-parallel-v2" else bool(count)) * self.expert_limits.max_runtime_seconds


@dataclass(frozen=True, kw_only=True)
class PublicRoutingFeatures(Contract):
    public_input_sha256: str
    failure_signature: str
    verification_status: Literal["passed", "failed", "unavailable"]
    workflow_completed: bool
    failure_kind: str | None
    source_file_count: int
    mentioned_source_file_count: int
    traceback_count: int
    cause_chain_count: int
    chained_traceback: bool
    prior_failure_signatures: tuple[str, ...]
    remaining_model_calls: int
    remaining_tool_calls: int
    remaining_total_tokens: int
    remaining_seconds: float
    remaining_retries: int
    episodes_used: int = 0
    usage_unknown: bool = False
    budget_pending: bool = False
    policy_intact: bool = True
    source_intact: bool = True
    cancelled: bool = False
    feature_version: Literal["public-routing-features-v1"] = "public-routing-features-v1"
    schema_version: int = 1

    def validate(self):
        digest(self.public_input_sha256)
        digest(self.failure_signature)
        if self.failure_kind is not None:
            identifier(self.failure_kind)
        if not 1 <= self.source_file_count <= MAX_FILES or not 0 <= self.mentioned_source_file_count <= self.source_file_count:
            raise ValueError("Invalid public source inventory counts")
        if min(self.traceback_count, self.cause_chain_count, self.remaining_model_calls, self.remaining_tool_calls,
               self.remaining_total_tokens, self.remaining_seconds, self.remaining_retries, self.episodes_used) < 0:
            raise ValueError("Negative public observation or remaining allowance")
        if self.chained_traceback and (self.traceback_count < 2 or self.cause_chain_count < 1):
            raise ValueError("Chained traceback feature lacks required observations")
        if len(self.prior_failure_signatures) > 32 or self.remaining_retries > 32 or self.episodes_used > 32:
            raise ValueError("Unbounded public failure history")
        for signature in self.prior_failure_signatures:
            digest(signature)

    @property
    def sha256(self):
        return _sha(self)


@dataclass(frozen=True, kw_only=True)
class RoutingDecision(Contract):
    action: Literal["keep_single", "consult_experts", "stop"]
    roles: tuple[Literal["code_analysis", "log_analysis"], ...]
    reason_codes: tuple[str, ...]
    policy_sha256: str
    features_sha256: str
    rules_sha256: str = RULES_SHA256
    schema_version: int = 1

    def validate(self):
        for value in (self.policy_sha256, self.features_sha256, self.rules_sha256):
            digest(value)
        if self.rules_sha256 not in (RULES_SHA256, V2_RULES_SHA256, V3_RULES_SHA256, CLOSURE_RULES_SHA256, REQUEST_RULES_SHA256, INITIAL_RULES_SHA256):
            raise ValueError("Unknown routing rule identity")
        if self.action == "consult_experts":
            if self.roles not in (("code_analysis",), ("code_analysis", "log_analysis")):
                raise ValueError("Consultation must select bounded Code-first roles")
        elif self.roles:
            raise ValueError("Nonconsulting decision cannot start expert roles")
        if not self.reason_codes or len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("Unique decision reasons required")
        for reason in self.reason_codes:
            identifier(reason)


def extract_public_features(*, public_log, source_files, verification_status, workflow_completed,
        failure_kind, remaining_model_calls, remaining_tool_calls, remaining_total_tokens,
        remaining_seconds, remaining_retries, verification_stdout="", verification_stderr="",
        prior_failure_signatures=(), episodes_used=0, usage_unknown=False, budget_pending=False,
        policy_intact=True, source_intact=True, cancelled=False):
    """Extract only supplied public data; never read a Trial, path, model or Oracle.

    Remaining values must already subtract charged and pending Job responsibility.
    The separate pending/unknown flags conservatively prevent further dispatch.
    Prior signatures are hashes of *public* verification text from earlier attempts.
    """
    for value, maximum in ((public_log, 200000), (verification_stdout, 100000), (verification_stderr, 100000)):
        _bounded_text(value, maximum)
    if not isinstance(source_files, (tuple, list)) or not 1 <= len(source_files) <= MAX_FILES:
        raise ValueError("Explicit bounded public source filenames required")
    for name in source_files:
        if type(name) is not str:
            raise TypeError("Public source filenames must be strings")
        source_path(name)
    if len(set(source_files)) != len(source_files):
        raise ValueError("Duplicate public source filenames")
    files = tuple(sorted(source_files))
    texts = (public_log, verification_stdout, verification_stderr)
    traces = tuple(len(_TRACEBACK.findall(value)) for value in texts)
    chains = tuple(len(_CHAIN.findall(value)) for value in texts)
    public = dict(public_log=public_log, source_files=files, verification_status=verification_status,
                  verification_stdout=verification_stdout, verification_stderr=verification_stderr)
    # Do not include project IDs, source hashes, paths outside the inventory or
    # timestamps in the novelty signal. An unchanged failure cannot buy retries.
    failure = dict(status=verification_status, stdout=verification_stdout, stderr=verification_stderr)
    return PublicRoutingFeatures(public_input_sha256=_sha(public), failure_signature=_sha(failure),
        verification_status=verification_status, workflow_completed=workflow_completed, failure_kind=failure_kind,
        source_file_count=len(files), mentioned_source_file_count=sum(any(name in value for value in texts) for name in files),
        traceback_count=max(traces), cause_chain_count=max(chains),
        chained_traceback=any(t >= 2 and c >= 1 for t, c in zip(traces, chains)),
        prior_failure_signatures=prior_failure_signatures, remaining_model_calls=remaining_model_calls,
        remaining_tool_calls=remaining_tool_calls, remaining_total_tokens=remaining_total_tokens,
        remaining_seconds=remaining_seconds, remaining_retries=remaining_retries, episodes_used=episodes_used,
        usage_unknown=usage_unknown, budget_pending=budget_pending, policy_intact=policy_intact,
        source_intact=source_intact, cancelled=cancelled)


def decide(features, policy=ExpertPolicy(), *, handoff=None, initial=False):
    """Recomputable admission proposal; no mutation, budget reservation or execution."""
    if type(features) is not PublicRoutingFeatures or type(policy) is not ExpertPolicy:
        raise TypeError("Typed public features and expert policy required")
    if type(initial) is not bool or initial and handoff is not None:
        raise ValueError("Initial consultation and writer handoff are distinct origins")
    if handoff is not None:
        from .handoff import LocalStopReceipt
        if type(handoff) is not LocalStopReceipt:
            raise TypeError("Typed Controller local-stop receipt required")
    local = handoff is not None and handoff.eligible
    version = V2_RULES_SHA256 if handoff is not None or policy.schedule_version == "serial-parallel-v2" else RULES_SHA256
    if handoff is not None and handoff.policy.mode == "local-stop-v3":
        version = V3_RULES_SHA256
    closure = local and handoff.reason_codes == ("known_closure_stop",)
    requested = local and handoff.reason_codes == ("known_explicit_request",)
    if handoff is not None and handoff.policy.closure_version == "reserve-incomplete-v1":
        version = CLOSURE_RULES_SHA256
    if requested:
        version = REQUEST_RULES_SHA256
    if initial:
        version = INITIAL_RULES_SHA256

    def result(action, *reasons, roles=()):
        return RoutingDecision(action=action, roles=roles, reason_codes=reasons,
            policy_sha256=policy.sha256, features_sha256=features.sha256, rules_sha256=version)

    if initial and policy.initial_mode != "code-first-v1":
        return result("stop", "initial_consultation_disabled")

    for unsafe, reason in ((features.cancelled, "interrupted"), (not features.policy_intact, "policy_violation"),
            (not features.source_intact, "source_changed"), (features.budget_pending, "accounting_pending"),
            (features.usage_unknown, "accounting_unknown"),
            (not features.workflow_completed and not local and not initial, "incomplete_workflow")):
        if unsafe:
            return result("stop", reason)
    if features.verification_status == "passed":
        return result("stop", "public_check_passed")
    if features.verification_status != "failed":
        return result("stop", "invalid_public_verification")
    if local and (features.workflow_completed or features.failure_kind != ("unknown" if closure or requested else "budget_exhausted")):
        return result("stop", "invalid_local_stop")
    if features.failure_kind != "verification_failed" and not local:
        return result("stop", "nonbusiness_failure")
    if features.remaining_retries == 0:
        return result("stop", "retry_limit")
    if policy.mode == "off":
        # Leave original Single budget admission to its existing controller.
        # Enabling no experts must not impose this new strategy's writer floor.
        return result("stop" if local else "keep_single", "expert_policy_off")

    def fits(experts):
        limits = policy.expert_limits
        return (features.remaining_model_calls >= policy.writer_min_model_calls + experts * limits.max_model_calls
            and features.remaining_tool_calls >= policy.writer_min_tool_calls + experts * limits.max_tool_calls
            and features.remaining_total_tokens >= policy.writer_min_total_tokens + experts * limits.max_total_tokens
            and features.remaining_seconds >= policy.writer_min_seconds + policy.expert_seconds(experts))

    if not fits(0):
        return result("stop", "writer_reserve_unavailable")
    if features.episodes_used >= policy.max_episodes:
        if local or initial:
            return result("stop", "expert_episode_limit")
        if features.failure_signature in features.prior_failure_signatures:
            return result("stop", "repeated_public_failure")
        return result("keep_single", "expert_episode_limit")
    if requested or initial:
        return (result("consult_experts", "initial_public_failure" if initial else "explicit_expert_request", "code_first", roles=("code_analysis",))
                if fits(1) else result("stop", "expert_allocation_unavailable"))
    if features.chained_traceback and fits(2):
        return result("consult_experts", "public_failure", "chained_traceback", roles=("code_analysis", "log_analysis"))
    if fits(1):
        reasons = ("public_failure", "code_first", "log_allocation_unavailable") if features.chained_traceback else ("public_failure", "code_first")
        return result("consult_experts", *reasons, roles=("code_analysis",))
    return result("stop" if local else "keep_single", "expert_allocation_unavailable")
