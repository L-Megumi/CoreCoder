"""Explicit common engineering profile; no live model or adaptive routing here."""

from researchcoder.evaluation.contracts import EvaluationConfig
from dataclasses import replace
from researchcoder.runtime.protocol import RuntimeLimits
from researchcoder.orchestration.budget import JobLimits


def offline_config(**changes):
    base = EvaluationConfig(
        benchmark_subset="complexity_benchmark", splits=("development",),
        task_limits=RuntimeLimits(max_model_calls=12, max_tool_calls=16, max_rounds=12,
            max_runtime_seconds=60, max_context_tokens=32000, max_output_tokens=2048, max_total_tokens=150000),
        job_limits=JobLimits(max_model_calls=80, max_tool_calls=128, max_total_tokens=1000000, max_runtime_seconds=240),
        stop_on_safety_violation=True,
    )
    config = replace(base, **changes)
    if config.model.provider != "scripted":
        raise ValueError("Offline engineering profile cannot start paid model calls")
    return config
