"""Versioned evaluation values; private answers never become TaskSpec inputs."""

from dataclasses import dataclass, field
from typing import Literal
from researchcoder.contracts import (
    Contract,
    ArtifactRef,
    VerificationResult,
    identifier,
    digest,
    relative,
    text,
)
from researchcoder.runtime.protocol import (
    ModelSpec,
    RuntimeLimits,
    ModelRequestOptions,
    model_request_kwargs,
)
from researchcoder.orchestration.budget import JobLimits

Mode = Literal["single", "sequential", "parallel", "evidence"]
Split = Literal["development", "validation", "holdout"]


@dataclass(frozen=True, kw_only=True)
class SourceAsset(Contract):
    path: str
    faulty_sha256: str
    healthy_sha256: str
    schema_version: int = 1

    def validate(self):
        relative(self.path)
        digest(self.faulty_sha256)
        digest(self.healthy_sha256)


@dataclass(frozen=True, kw_only=True)
class BenchmarkCase(Contract):
    case_id: str
    category: str
    split: Split
    project_version: str
    objective: str
    public_log: str
    assets: tuple[SourceAsset, ...]
    writable_paths: tuple[str, ...]
    reference_cause: str
    cause_concepts: tuple[tuple[str, ...], ...]
    fault_locations: tuple[str, ...]
    acceptable_repair: str
    oracle_sha256: str
    oracle_support_sha256: str
    dependency_profile: str = "pytorch-cpu"
    timeout_seconds: int = 30
    fail_to_pass_profile: str = "holdout"
    pass_to_pass_profile: str = "regression"
    schema_version: int = 1

    def validate(self):
        for value in (
            self.case_id,
            self.category,
            self.project_version,
            self.dependency_profile,
        ):
            identifier(value)
        text(self.objective)
        text(self.public_log, 20000)
        text(self.reference_cause)
        text(self.acceptable_repair)
        if (
            not self.assets
            or len(self.assets) > 16
            or len({a.path for a in self.assets}) != len(self.assets)
        ):
            raise ValueError("Unique candidate sources required")
        paths = {a.path for a in self.assets}
        if (
            not self.writable_paths
            or not set(self.writable_paths) <= paths
            or not set(self.fault_locations) <= paths
        ):
            raise ValueError("Writes and expected locations must be candidate files")
        if (
            not self.fault_locations
            or not self.cause_concepts
            or any(not g or any(not s.strip() for s in g) for g in self.cause_concepts)
        ):
            raise ValueError("Independent cause/location rubric required")
        for p in (*self.writable_paths, *self.fault_locations):
            relative(p)
        digest(self.oracle_sha256)
        digest(self.oracle_support_sha256)
        if not 1 <= self.timeout_seconds <= 120:
            raise ValueError("Bounded oracle timeout required")
        if (
            self.fail_to_pass_profile != "holdout"
            or self.pass_to_pass_profile != "regression"
        ):
            raise ValueError("Separate fixed oracle profiles required")


@dataclass(frozen=True, kw_only=True)
class BenchmarkManifest(Contract):
    benchmark_id: str
    version: str
    cases: tuple[BenchmarkCase, ...]
    schema_version: int = 1

    def validate(self):
        identifier(self.benchmark_id)
        identifier(self.version)
        if not self.cases or len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError("Nonempty unique benchmark cases required")
        if {c.split for c in self.cases} != {"development", "validation", "holdout"}:
            raise ValueError(
                "Development, validation and held-out identities must be explicit"
            )


@dataclass(frozen=True, kw_only=True)
class PriceSpec(Contract):
    model: str
    input_per_million: float
    output_per_million: float
    source_url: str
    effective_date: str
    currency: Literal["USD", "CNY"] = "USD"
    billing_basis: Literal["uniform_rate_confirmed", "planning_only"] = "planning_only"
    schema_version: int = 1

    def validate(self):
        text(self.model, 120)
        if min(self.input_per_million, self.output_per_million) < 0:
            raise ValueError("Negative price")
        if not self.source_url.startswith("https://") or len(self.effective_date) != 10:
            raise ValueError("Versioned reliable pricing provenance required")


@dataclass(frozen=True, kw_only=True)
class EvaluationConfig(Contract):
    harness_strategy: Literal['fixed-v1','budget-v1','compact-v1','constraints-v1','combined-v1'] = 'fixed-v1'
    benchmark_subset: Literal["pytorch", "complexity_benchmark"] = "pytorch"
    max_feedback_rounds: int = 0
    model: ModelSpec = field(default_factory=ModelSpec)
    model_options: ModelRequestOptions | None = None
    modes: tuple[Mode, ...] = ("single", "sequential", "parallel", "evidence")
    splits: tuple[Split, ...] = ("development", "validation", "holdout")
    case_ids: tuple[str, ...] = ()
    repeats: int = 1
    rotate_order: bool = True
    context_strategy: Literal["original", "truncate", "summary", "evidence"] | None = (
        None
    )
    job_limits: JobLimits = field(
        default_factory=lambda: JobLimits(
            max_model_calls=40,
            max_tool_calls=64,
            max_total_tokens=1000000,
            max_runtime_seconds=180,
        )
    )
    task_limits: RuntimeLimits = field(
        default_factory=lambda: RuntimeLimits(
            max_model_calls=8,
            max_tool_calls=8,
            max_rounds=8,
            max_total_tokens=80000,
            max_runtime_seconds=40,
        )
    )
    price: PriceSpec | None = None
    max_trials: int = 128
    max_run_model_requests: int | None = None
    max_run_token_charge: int | None = None
    stop_after_failures: int | None = None
    stop_on_unknown_usage: bool = False
    stop_after_consecutive_api_failures: int | None = None
    stop_on_safety_violation: bool = False
    allow_live: bool = False
    scripted_behavior: Literal["correct", "no_patch", "wrong_patch", "refuse", "diagnosis_fail", "staged_correct", "repeat_patch"] = (
        "correct"
    )
    schema_version: int = 1

    def to_dict(self):
        value = super().to_dict()
        if self.harness_strategy == 'fixed-v1':
            value.pop('harness_strategy')
        if self.benchmark_subset == "pytorch":
            value.pop("benchmark_subset")
        if not self.max_feedback_rounds:
            value.pop("max_feedback_rounds")
        # Preserve canonical JSON identities of pre-6C2 runs when these optional
        # stop policies are unused. Old raw records must remain recomputable.
        if self.stop_after_consecutive_api_failures is None:
            value.pop("stop_after_consecutive_api_failures")
        if not self.stop_on_safety_violation:
            value.pop("stop_on_safety_violation")
        return value

    def validate(self):
        if not 0 <= self.max_feedback_rounds <= 2:
            raise ValueError("Bounded shared feedback rounds required")
        if self.scripted_behavior in ("staged_correct", "repeat_patch") and (
            self.max_feedback_rounds < 1 or self.benchmark_subset != "complexity_benchmark"
        ):
            raise ValueError("Staged scripted profiles require the complexity subset and explicit feedback rounds")
        model_request_kwargs(self.model, self.model_options)
        if (
            not self.modes
            or len(set(self.modes)) != len(self.modes)
            or not self.splits
            or len(set(self.splits)) != len(self.splits)
        ):
            raise ValueError("Unique modes/splits required")
        if not 1 <= self.repeats <= 100 or not 1 <= self.max_trials <= 2000:
            raise ValueError("Bounded trial count required")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Duplicate case selection")
        for c in self.case_ids:
            identifier(c)
        if self.model.provider != "scripted" and not self.allow_live:
            raise ValueError("Live model requires explicit trusted opt-in")
        if self.model.turns:
            raise ValueError(
                "Benchmark offline actions are built per case, not arbitrary config turns"
            )
        if self.price and self.price.model != self.model.model:
            raise ValueError("Pricing model mismatch")
        for limit in (
            self.max_run_model_requests,
            self.max_run_token_charge,
            self.stop_after_failures,
            self.stop_after_consecutive_api_failures,
        ):
            if limit is not None and limit < 1:
                raise ValueError("Positive batch admission limits required")
        if self.model.provider != "scripted" and self.scripted_behavior != "correct":
            raise ValueError(
                "Scripted behavior switches do not control live model behavior"
            )


@dataclass(frozen=True, kw_only=True)
class TrialSpec(Contract):
    trial_id: str
    case_id: str
    mode: Mode
    repeat: int
    ordinal: int
    schema_version: int = 1

    def validate(self):
        identifier(self.trial_id)
        identifier(self.case_id)
        if min(self.repeat, self.ordinal) < 0:
            raise ValueError("Invalid trial ordering")


@dataclass(frozen=True, kw_only=True)
class EvaluationRun(Contract):
    run_id: str
    manifest_sha256: str
    config_sha256: str
    environment_sha256: str
    trials: tuple[TrialSpec, ...]
    schema_version: int = 1

    def validate(self):
        identifier(self.run_id)
        for v in (self.manifest_sha256, self.config_sha256, self.environment_sha256):
            digest(v)
        if not self.trials or len({t.trial_id for t in self.trials}) != len(
            self.trials
        ):
            raise ValueError("Unique planned denominator required")


@dataclass(frozen=True, kw_only=True)
class EvaluationMetrics(Contract):
    end_to_end_success: bool = False
    diagnosis_correct: bool = False
    localization_correct: bool = False
    fail_to_pass: bool | None = None
    pass_to_pass: bool | None = None
    wall_seconds: float = 0.0
    model_requests: int | None = None
    tool_calls: int | None = None
    actual_prompt_tokens: int | None = None
    actual_completion_tokens: int | None = None
    estimated_input_tokens: int | None = None
    unknown_requests: int | None = None
    usage_complete: bool = False
    cost: float | None = None
    evidence_retention: float | None = None
    replanning_count: int = 0
    recovery_success: bool | None = None
    needs_review: bool = False
    duplicate_side_effects: int | None = None
    schema_version: int = 1

    def validate(self):
        for key in (
            "wall_seconds",
            "model_requests",
            "tool_calls",
            "actual_prompt_tokens",
            "actual_completion_tokens",
            "estimated_input_tokens",
            "unknown_requests",
            "cost",
            "replanning_count",
            "duplicate_side_effects",
        ):
            value = getattr(self, key)
            if value is not None and value < 0:
                raise ValueError("Negative metric")
        if self.usage_complete and (
            self.actual_prompt_tokens is None
            or self.actual_completion_tokens is None
            or self.unknown_requests not in (0,)
        ):
            raise ValueError("Complete usage needs actual counts")
        if self.cost is not None and not self.usage_complete:
            raise ValueError("Unknown usage cannot produce cost")
        if (
            self.evidence_retention is not None
            and not 0 <= self.evidence_retention <= 1
        ):
            raise ValueError("Invalid retention")
        if self.end_to_end_success and (
            self.fail_to_pass is not True
            or self.pass_to_pass is not True
            or self.needs_review
        ):
            raise ValueError("Workflow success requires both independent checks")


@dataclass(frozen=True, kw_only=True)
class EvaluationResult(Contract):
    run_id: str
    trial: TrialSpec
    job_id: str
    status: Literal["succeeded", "failed", "incomplete", "cancelled", "needs_review"]
    failure_kind: str | None
    raw: ArtifactRef
    metrics: EvaluationMetrics
    fail_to_pass_verification: VerificationResult | None = None
    pass_to_pass_verification: VerificationResult | None = None
    schema_version: int = 1

    def validate(self):
        identifier(self.run_id)
        identifier(self.job_id)
        if self.raw.job_id != self.job_id:
            raise ValueError("Foreign raw result")
        for v in (self.fail_to_pass_verification, self.pass_to_pass_verification):
            if v is not None and v.job_id != self.job_id:
                raise ValueError("Foreign oracle result")
        if self.status == "succeeded" and not self.metrics.end_to_end_success:
            raise ValueError("Do not promote a partial workflow to success")
        if self.metrics.end_to_end_success and self.status != "succeeded":
            raise ValueError("Success/status mismatch")
