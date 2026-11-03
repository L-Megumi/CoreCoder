import pytest
from researchcoder.evaluation.contracts import (
    EvaluationMetrics,
    EvaluationResult,
    EvaluationRun,
    TrialSpec,
    PriceSpec,
)
from researchcoder.contracts import ArtifactRef
from researchcoder.evaluation.metrics import (
    summarize,
    usage_metrics,
    diagnosis_scores,
    compute,
    recovery_metrics,
)
from researchcoder.evaluation.dataset import load_manifest
from researchcoder.evaluation.contracts import EvaluationConfig
from researchcoder.runtime.context_audit import evidence_signatures, signature


def trial(index):
    return TrialSpec(
        trial_id="t" + str(index),
        case_id="case",
        mode="single",
        repeat=index,
        ordinal=index,
    )


def row(index, metrics=EvaluationMetrics(), status="failed"):
    ref = ArtifactRef(
        job_id="job",
        artifact_id="a",
        storage_key="job/a.blob",
        sha256="a" * 64,
        size_bytes=1,
    )
    return EvaluationResult(
        run_id="run",
        trial=trial(index),
        job_id="job",
        status=status,
        failure_kind=None,
        raw=ref,
        metrics=metrics,
    )


def test_all_planned_failed_cancelled_and_missing_samples_stay_in_denominator():
    run = EvaluationRun(
        run_id="run",
        manifest_sha256="a" * 64,
        config_sha256="b" * 64,
        environment_sha256="c" * 64,
        trials=tuple(trial(i) for i in range(4)),
    )
    good = EvaluationMetrics(
        end_to_end_success=True, fail_to_pass=True, pass_to_pass=True
    )
    summary = summarize(
        run, [row(0, good, "succeeded"), row(1), row(2, status="cancelled")]
    )["modes"]["single"]
    assert summary["rates"]["end_to_end_success"]["rate"] == 0.25
    assert summary["not_recorded"] == 1 and summary["statuses"]["cancelled"] == 1
    assert summary["actual_prompt_tokens"] is None and summary["cost"] is None
    with pytest.raises(ValueError):
        summarize(run, [row(0), row(0)])


def test_unknown_partial_usage_and_reliable_actual_only_cost():
    b = dict(
        model_calls=2,
        tool_calls=3,
        unknown_requests=1,
        estimated_input_tokens=50,
        known_prompt_tokens=7,
        known_completion_tokens=4,
        pending={},
        context_audits=[],
    )
    price = PriceSpec(
        model="m",
        input_per_million=1,
        output_per_million=2,
        source_url="https://example.org/pricing",
        effective_date="2026-10-25",
        billing_basis="uniform_rate_confirmed",
    )
    unknown = usage_metrics([dict(budget=b)], (), price, live=True)
    assert (
        unknown["actual_prompt_tokens"] is None
        and unknown["unknown_requests"] == 1
        and unknown["cost"] is None
    )
    known = usage_metrics(
        [dict(budget=dict(b, unknown_requests=0))], (), price, live=True
    )
    assert known["actual_prompt_tokens"] == 7 and known["cost"] == pytest.approx(
        15 / 1000000
    )
    assert usage_metrics([dict(budget=b)], (), price)["cost"] is None
    assert usage_metrics([{}], ())["model_requests"] is None


def test_true_verifier_pass_does_not_complete_incomplete_workflow():
    case = load_manifest().cases[0]
    raw = dict(
        preflight=dict(fault_failed=True, regression_passed=True),
        fail_to_pass=dict(status="passed"),
        pass_to_pass=dict(status="passed"),
        policy_intact=True,
        workflow_status="incomplete",
        workflow_completed=False,
    )
    m = compute(raw, case, EvaluationConfig())
    assert m.fail_to_pass and m.pass_to_pass and not m.end_to_end_success


def test_diagnosis_is_stated_conclusion_and_localization_is_file_level():
    case = load_manifest().cases[-1]
    assert diagnosis_scores(case.reference_cause + " Location: ops.py", case) == (
        True,
        True,
    )
    assert diagnosis_scores("I read ops.py but did not determine the cause", case) == (
        False,
        True,
    )
    assert diagnosis_scores("", case) == (False, False)


def test_hash_only_context_evidence_audit_and_measured_retention():
    text = "WARNING: private training signal"
    messages = [dict(role="user", content=text)]
    before = evidence_signatures(messages)
    assert before == [signature(text)] and text not in str(before)
    assert (
        evidence_signatures(
            [
                dict(
                    role="tool",
                    content='prefix\n{"evidence":[{"message":"' + text + '"}]}',
                )
            ]
        )
        == before
    )
    b = dict(
        model_calls=1,
        tool_calls=1,
        unknown_requests=0,
        estimated_input_tokens=30,
        known_prompt_tokens=10,
        known_completion_tokens=4,
        pending={},
        context_audits=[dict(before_evidence=before, after_evidence=[])],
    )
    assert usage_metrics([dict(budget=b)], (text,))["evidence_retention"] == 0
    assert usage_metrics([dict(budget=b)], ("unseen",))["evidence_retention"] is None


def test_recovery_unmeasured_side_effects_are_not_reported_as_zero():
    summary = recovery_metrics([dict(status="needs_review", result_status=None)])
    assert summary["duplicate_side_effects"] is None and summary["needs_review"] == 1
    assert (
        recovery_metrics(
            [dict(status="completed", result_status="succeeded", duplicate_effects=0)]
        )["business_succeeded"]
        == 1
    )


def test_batch_admission_uses_original_job_charge_and_preserves_unknown():
    from types import SimpleNamespace
    from researchcoder.evaluation.admission import BatchAdmission

    cfg = EvaluationConfig(max_run_model_requests=40, max_run_token_charge=1000000)
    batch = BatchAdmission(cfg)
    assert batch.admit() is None
    result = SimpleNamespace(
        status="succeeded", metrics=SimpleNamespace(usage_complete=False)
    )
    batch.settle(result, {})
    assert batch.model_charge == 40 and batch.token_charge == 1000000
    assert batch.admit() == "batch_token_allocation_exhausted"
    stopped = BatchAdmission(EvaluationConfig(stop_on_unknown_usage=True))
    stopped.settle(
        result, dict(budget=dict(charged=dict(model_calls=2, total_tokens=3000)))
    )
    assert stopped.admit() == "batch_usage_unknown"


def test_passed_oracle_for_old_source_is_not_current_repair_success():
    case = load_manifest().cases[0]
    raw = dict(
        preflight=dict(fault_failed=True, regression_passed=True),
        workflow_status="succeeded",
        workflow_completed=True,
        policy_intact=True,
        workspace_after="b" * 64,
        fail_to_pass=dict(status="passed", checked_workspace_revision="a" * 64),
        pass_to_pass=dict(status="passed", checked_workspace_revision="b" * 64),
    )
    metrics = compute(raw, case, EvaluationConfig())
    assert (
        not metrics.end_to_end_success
        and metrics.needs_review
        and metrics.fail_to_pass is None
    )
    assert metrics.pass_to_pass is True
