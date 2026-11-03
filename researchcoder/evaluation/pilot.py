"""Generate a costed proposal only. This module never creates a model client."""

from .contracts import EvaluationConfig, PriceSpec
from researchcoder.runtime.protocol import ModelSpec, RuntimeLimits, ModelRequestOptions
from researchcoder.orchestration.budget import JobLimits

PRICE_SOURCE = "https://api-docs.deepseek.com/quick_start/pricing/"


def proposal():
    # Planning snapshot checked 2026-10-25. Peak cache-miss is an upper-rate
    # scenario, not an invoice: current Runtime does not retain cache/time tiers.
    config = EvaluationConfig(
        model=ModelSpec(
            provider="openai",
            model="deepseek-flash",
            base_url="https://api.deepseek.com",
            credential_env="DEEPSEEK_API_KEY",
        ),
        model_options=ModelRequestOptions(),
        allow_live=True,
        case_ids=("autograd_detach", "shape_transpose", "dtype_features"),
        max_trials=12,
        task_limits=RuntimeLimits(
            max_model_calls=6,
            max_tool_calls=6,
            max_rounds=6,
            max_context_tokens=16000,
            max_output_tokens=1024,
            max_total_tokens=50000,
            max_runtime_seconds=40,
        ),
        job_limits=JobLimits(
            max_model_calls=24,
            max_tool_calls=32,
            max_total_tokens=300000,
            max_runtime_seconds=120,
        ),
        max_run_model_requests=288,
        max_run_token_charge=3600000,
        stop_after_failures=1,
        stop_on_unknown_usage=True,
        price=PriceSpec(
            model="deepseek-flash",
            input_per_million=0.3,
            output_per_million=1.2,
            source_url=PRICE_SOURCE,
            effective_date="2026-10-25",
            billing_basis="planning_only",
        ),
    )
    calls = 12 * config.job_limits.max_model_calls
    return dict(
        schema_version=1,
        executed=False,
        requires_authorization=["model/API", "USD budget", "12-trial scope"],
        config=config.to_dict(),
        provider_requests_executed=0,
        max_admitted_model_requests=calls,
        planning_estimate=dict(
            input_tokens=[150000, 600000],
            output_tokens=[12000, 60000],
            off_peak_cache_miss_usd=[
                0.15 * 0.15 + 0.012 * 0.6,
                0.6 * 0.15 + 0.06 * 0.6,
            ],
            peak_cache_miss_usd=[0.15 * 0.3 + 0.012 * 1.2, 0.6 * 0.3 + 0.06 * 1.2],
            conservative_allocation_cost_scenario_usd=(
                3.6 * 0.3 + (calls * 1024 / 1000000) * 1.2
            ),
            basis="Scenario, not a strict spend ceiling. Input estimates and final in-flight request may exceed estimates. Cache hits/time bands and hidden reasoning accounting require pilot verification.",
        ),
        limits=dict(
            output_tokens_per_request=1024,
            job_deadline_seconds=120,
            max_trials=12,
            stop_on_unknown_usage=True,
            stop_after_failures=1,
            heldout_cases_used=False,
        ),
        preflight="Explicitly disable thinking via trusted request options, Mock-tested only. Confirm real tool-call compatibility and usage fields in authorized pilot; full reasoning-content history is not implemented.",
        pricing_source=PRICE_SOURCE,
        pricing_checked_date="2026-10-25",
    )


if __name__ == "__main__":
    import json

    print(json.dumps(proposal(), indent=2))
