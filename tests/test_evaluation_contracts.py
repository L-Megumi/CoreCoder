from dataclasses import replace
import pytest
from researchcoder.evaluation.contracts import (
    EvaluationConfig,
    EvaluationMetrics,
    EvaluationRun,
    TrialSpec,
    PriceSpec,
    SourceAsset,
    BenchmarkCase,
    BenchmarkManifest,
)
from researchcoder.runtime.protocol import ModelSpec


def sample(case_id="example", split="development"):
    return BenchmarkCase(
        case_id=case_id,
        category="gradient",
        split=split,
        project_version="v1",
        objective="Joint training",
        public_log="WARNING: training failed",
        assets=(
            SourceAsset(
                path="training.py", faulty_sha256="a" * 64, healthy_sha256="b" * 64
            ),
        ),
        writable_paths=("training.py",),
        reference_cause="detached backbone",
        cause_concepts=(("detach",),),
        fault_locations=("training.py",),
        acceptable_repair="restore gradients",
        oracle_sha256="c" * 64,
        oracle_support_sha256="d" * 64,
    )


@pytest.mark.parametrize(
    "value",
    [
        EvaluationConfig(),
        EvaluationMetrics(),
        TrialSpec(trial_id="t", case_id="c", mode="single", repeat=0, ordinal=0),
        sample(),
        BenchmarkManifest(
            benchmark_id="suite",
            version="v1",
            cases=(sample(), sample("b", "validation"), sample("c", "holdout")),
        ),
    ],
)
def test_evaluation_contract_roundtrip(value):
    assert type(value).from_json(value.to_json()) == value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"modes": ("single", "single")},
        {"repeats": 0},
        {"max_trials": 0},
        {"model": ModelSpec(provider="openai", model="live")},
        {"case_ids": ("../secret",)},
    ],
)
def test_evaluation_config_rejects_unsafe_or_ambiguous_input(kwargs):
    with pytest.raises((ValueError, TypeError)):
        EvaluationConfig(**kwargs)


def test_metric_missing_usage_is_not_zero_cost_or_success():
    assert EvaluationMetrics().cost is None
    for kwargs in (
        {"cost": 0.0},
        {"usage_complete": True},
        {"end_to_end_success": True},
        {"evidence_retention": 1.1},
    ):
        with pytest.raises(ValueError):
            EvaluationMetrics(**kwargs)
    with pytest.raises(ValueError):
        replace(sample(), writable_paths=("oracle.py",))
    with pytest.raises(ValueError):
        BenchmarkManifest(benchmark_id="suite", version="v1", cases=(sample(),))
    with pytest.raises(ValueError):
        EvaluationConfig.from_dict(dict(EvaluationConfig().to_dict(), schema_version=2))


def test_run_denominator_and_pricing_provenance():
    trial = TrialSpec(trial_id="t", case_id="c", mode="single", repeat=0, ordinal=0)
    run = EvaluationRun(
        run_id="run",
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        trials=(trial,),
    )
    assert EvaluationRun.from_json(run.to_json()) == run
    with pytest.raises(ValueError):
        replace(run, trials=(trial, trial))
    price = PriceSpec(
        model="other",
        input_per_million=1,
        output_per_million=2,
        source_url="https://example.org/price",
        effective_date="2026-10-25",
    )
    with pytest.raises(ValueError):
        EvaluationConfig(price=price)
