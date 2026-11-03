"""Pure offline scoring; missing observations remain explicit and in denominators."""

import math
import re
import statistics
from .contracts import EvaluationMetrics
from researchcoder.runtime.context_audit import signature


def diagnosis_scores(assessment, case):
    value = assessment.lower()
    # Deterministic lexical rubric, not a semantic/model judge accuracy claim.
    root = bool(value) and all(
        any(term.lower() in value for term in group) for group in case.cause_concepts
    )
    located = all(
        re.search(r"(?<!\w)" + re.escape(path.lower()) + r"(?![\w.])", value)
        for path in case.fault_locations
    )
    return root, bool(located)


def usage_metrics(records, expected_evidence, price=None, *, live=False):
    records = list(records)
    budgets = [r.get("budget", {}) for r in records]
    accounted = bool(budgets) and all(
        all(
            type(b.get(k)) is int and b[k] >= 0
            for k in (
                "model_calls",
                "tool_calls",
                "unknown_requests",
                "estimated_input_tokens",
                "known_prompt_tokens",
                "known_completion_tokens",
            )
        )
        for b in budgets
    )
    if not accounted:
        return dict(
            model_requests=None,
            tool_calls=None,
            actual_prompt_tokens=None,
            actual_completion_tokens=None,
            estimated_input_tokens=None,
            unknown_requests=None,
            usage_complete=False,
            cost=None,
            evidence_retention=None,
        )
    count = sum(b["model_calls"] for b in budgets)
    unknown = sum(b["unknown_requests"] + len(b.get("pending", {})) for b in budgets)
    prompt = sum(b["known_prompt_tokens"] for b in budgets)
    completion = sum(b["known_completion_tokens"] for b in budgets)
    complete = unknown == 0
    cost = (
        (prompt * price.input_per_million + completion * price.output_per_million)
        / 1000000
        if complete
        and live
        and price is not None
        and price.billing_basis == "uniform_rate_confirmed"
        else None
    )
    expected = {signature(s) for s in expected_evidence}
    opportunities = retained = 0
    for b in budgets:
        for audit in b.get("context_audits", []):
            if (
                audit.get("before_evidence") is None
                or audit.get("after_evidence") is None
            ):
                continue
            present = expected & set(audit["before_evidence"])
            opportunities += len(present)
            retained += len(present & set(audit["after_evidence"]))
    return dict(
        model_requests=count,
        tool_calls=sum(b["tool_calls"] for b in budgets),
        actual_prompt_tokens=prompt if complete else None,
        actual_completion_tokens=completion if complete else None,
        estimated_input_tokens=sum(b["estimated_input_tokens"] for b in budgets),
        unknown_requests=unknown,
        usage_complete=complete,
        cost=cost,
        evidence_retention=retained / opportunities
        if opportunities and all(b.get("context_audit_complete", True) for b in budgets)
        else None,
    )


def compute(raw, case, config):
    root, location = diagnosis_scores(raw.get("assessment", ""), case)
    before = raw.get("preflight", {})
    fail_to_pass = (
        None
        if raw.get("fail_to_pass") is None
        else bool(
            before.get("fault_failed") and raw["fail_to_pass"]["status"] == "passed"
        )
    )
    pass_to_pass = (
        None
        if raw.get("pass_to_pass") is None
        else bool(
            before.get("regression_passed")
            and raw["pass_to_pass"]["status"] == "passed"
        )
    )
    review = raw.get("status") == "needs_review"
    if raw.get("workspace_after"):
        if (
            raw.get("fail_to_pass")
            and raw["fail_to_pass"].get("checked_workspace_revision")
            != raw["workspace_after"]
        ):
            fail_to_pass = None
            review = True
        if (
            raw.get("pass_to_pass")
            and raw["pass_to_pass"].get("checked_workspace_revision")
            != raw["workspace_after"]
        ):
            pass_to_pass = None
            review = True
    usage = usage_metrics(
        raw.get("records", []),
        (case.public_log,),
        config.price,
        live=config.model.provider == "openai",
    )
    job = raw.get("budget", {})
    if job.get("pending") or any(
        not s.get("accounting_complete", False)
        for s in job.get("settlements", {}).values()
    ):
        usage.update(
            model_requests=None,
            tool_calls=None,
            actual_prompt_tokens=None,
            actual_completion_tokens=None,
            unknown_requests=None,
            usage_complete=False,
            cost=None,
        )
    return EvaluationMetrics(
        end_to_end_success=bool(
            raw.get("workflow_status") == "succeeded"
            and raw.get("workflow_completed")
            and fail_to_pass is True
            and pass_to_pass is True
            and raw.get("policy_intact")
            and not review
        ),
        diagnosis_correct=root,
        localization_correct=location,
        fail_to_pass=fail_to_pass,
        pass_to_pass=pass_to_pass,
        wall_seconds=raw.get("wall_seconds", 0),
        needs_review=review,
        replanning_count=raw.get("replanning_count", 0),
        recovery_success=raw.get("recovery_success"),
        duplicate_side_effects=raw.get("duplicate_side_effects"),
        **usage,
    )


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [max(0, center - radius), min(1, center + radius)]


def summarize(run, results):
    by_id = {}
    for result in results:
        identity = result.trial.trial_id
        if identity in by_id or result.run_id != run.run_id:
            raise ValueError("Duplicate/foreign evaluation result")
        by_id[identity] = result
    planned = {t.trial_id: t for t in run.trials}
    if set(by_id) - set(planned) or any(by_id[k].trial != planned[k] for k in by_id):
        raise ValueError("Trial does not belong to planned denominator")
    output = {}
    for mode in dict.fromkeys(t.mode for t in run.trials):
        trials = [t for t in run.trials if t.mode == mode]
        rows = [by_id[t.trial_id] for t in trials if t.trial_id in by_id]
        metrics = [r.metrics for r in rows]

        def rate(key):
            passed = sum(getattr(m, key) is True for m in metrics)
            known = sum(getattr(m, key) is not None for m in metrics)
            return dict(
                passed=passed,
                denominator=len(trials),
                rate=passed / len(trials),
                observed=known,
                wilson_iid_interval=wilson(passed, len(trials)),
            )

        cases = {}
        for case_id in dict.fromkeys(t.case_id for t in trials):
            ct = [t for t in trials if t.case_id == case_id]
            cr = [by_id[t.trial_id] for t in ct if t.trial_id in by_id]
            values = [
                int(by_id[t.trial_id].metrics.end_to_end_success)
                if t.trial_id in by_id
                else 0
                for t in ct
            ]
            cases[case_id] = dict(
                planned=len(ct),
                recorded=len(cr),
                statuses=[r.status for r in cr],
                end_to_end_rate=statistics.mean(values),
                repeat_stddev=statistics.stdev(values) if len(values) > 1 else None,
            )
        complete_usage = len(rows) == len(trials) and all(
            m.usage_complete for m in metrics
        )
        output[mode] = dict(
            planned=len(trials),
            recorded=len(rows),
            not_recorded=len(trials) - len(rows),
            statuses={
                s: sum(r.status == s for r in rows)
                for s in (
                    "succeeded",
                    "failed",
                    "incomplete",
                    "cancelled",
                    "needs_review",
                )
            },
            rates={
                k: rate(k)
                for k in (
                    "end_to_end_success",
                    "diagnosis_correct",
                    "localization_correct",
                    "fail_to_pass",
                    "pass_to_pass",
                )
            },
            macro_case_repair_rate=statistics.mean(
                c["end_to_end_rate"] for c in cases.values()
            ),
            per_case=cases,
            mean_wall_seconds=statistics.mean(m.wall_seconds for m in metrics)
            if metrics
            else None,
            model_requests=sum(m.model_requests for m in metrics)
            if len(rows) == len(trials)
            and all(m.model_requests is not None for m in metrics)
            else None,
            tool_calls=sum(m.tool_calls for m in metrics)
            if len(rows) == len(trials)
            and all(m.tool_calls is not None for m in metrics)
            else None,
            actual_prompt_tokens=sum(m.actual_prompt_tokens for m in metrics)
            if complete_usage
            else None,
            actual_completion_tokens=sum(m.actual_completion_tokens for m in metrics)
            if complete_usage
            else None,
            estimated_input_tokens_observed=sum(
                m.estimated_input_tokens
                for m in metrics
                if m.estimated_input_tokens is not None
            ),
            estimated_usage_missing_trials=sum(
                m.estimated_input_tokens is None for m in metrics
            )
            + len(trials)
            - len(rows),
            unknown_requests_observed=sum(
                m.unknown_requests for m in metrics if m.unknown_requests is not None
            ),
            usage_unknown_trials=sum(not m.usage_complete for m in metrics)
            + len(trials)
            - len(rows),
            cost=sum(m.cost for m in metrics)
            if len(rows) == len(trials) and all(m.cost is not None for m in metrics)
            else None,
            evidence_retention_mean=statistics.mean(
                m.evidence_retention
                for m in metrics
                if m.evidence_retention is not None
            )
            if any(m.evidence_retention is not None for m in metrics)
            else None,
            evidence_retention_observed_trials=sum(
                m.evidence_retention is not None for m in metrics
            ),
            replanning_count=sum(m.replanning_count for m in metrics),
            recovery_observed_trials=sum(
                m.recovery_success is not None for m in metrics
            ),
            recovery_successes=sum(m.recovery_success is True for m in metrics),
            needs_review=sum(m.needs_review for m in metrics),
            duplicate_side_effects_observed=sum(
                m.duplicate_side_effects
                for m in metrics
                if m.duplicate_side_effects is not None
            ),
            duplicate_side_effects_measured_trials=sum(
                m.duplicate_side_effects is not None for m in metrics
            ),
        )
    return dict(
        schema_version=1,
        run_id=run.run_id,
        planned=len(run.trials),
        recorded=len(results),
        modes=output,
        scope="Fixed scripted actions test engineering, not LLM accuracy. All planned trials stay in denominators; missing oracle is not passed. Root cause uses lexical rubric, localization is file-level. Wilson intervals assume IID and are descriptive only; repeated cases are correlated.",
    )


def recovery_metrics(records):
    """Consume measured Phase 5 records without inventing physical side-effect counts."""
    rows = list(records)
    known = [r for r in rows if r.get("duplicate_effects") is not None]
    return dict(
        cases=len(rows),
        business_succeeded=sum(r.get("result_status") == "succeeded" for r in rows),
        needs_review=sum(r.get("status") == "needs_review" for r in rows),
        duplicate_side_effects=sum(r["duplicate_effects"] for r in known)
        if len(known) == len(rows) and rows
        else None,
        measured_cases=len(known),
    )
