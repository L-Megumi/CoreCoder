"""Batch admission over original JobBudget charges, not another task ledger."""


class BatchAdmission:
    def __init__(self, config):
        self.config = config
        self.model_charge = self.token_charge = self.failures = 0
        self.consecutive_api_failures = 0
        self.reason = None

    def admit(self):
        c = self.config
        if self.reason:
            return self.reason
        if (
            c.max_run_model_requests is not None
            and self.model_charge + c.job_limits.max_model_calls
            > c.max_run_model_requests
        ):
            self.reason = "batch_model_allocation_exhausted"
        if (
            c.max_run_token_charge is not None
            and self.token_charge + c.job_limits.max_total_tokens
            > c.max_run_token_charge
        ):
            self.reason = "batch_token_allocation_exhausted"
        return self.reason

    def settle(self, result, raw):
        charged = raw.get("budget", {}).get("charged")
        if charged is None:
            # A lost response is not free. Retain the full original Job allowance.
            self.model_charge += self.config.job_limits.max_model_calls
            self.token_charge += self.config.job_limits.max_total_tokens
        else:
            self.model_charge += charged["model_calls"]
            self.token_charge += charged["total_tokens"]
        self.failures += result.status != "succeeded"
        if (
            self.config.stop_after_failures is not None
            and self.failures >= self.config.stop_after_failures
        ):
            self.reason = "batch_failure_limit"
        if self.config.stop_on_unknown_usage and not result.metrics.usage_complete:
            self.reason = "batch_usage_unknown"
        api_failed = raw.get("failure_kind") == "model_error" or any(
            r.get("failure_kind") == "model_error" for r in raw.get("task_results", [])
        )
        self.consecutive_api_failures = self.consecutive_api_failures + 1 if api_failed else 0
        if (self.config.stop_after_consecutive_api_failures is not None
                and self.consecutive_api_failures >= self.config.stop_after_consecutive_api_failures):
            self.reason = "batch_consecutive_api_failures"
        if self.config.stop_on_safety_violation and (
            raw.get("policy_intact") is not True or result.status == "needs_review"
            or bool(raw.get("policy_violations"))
        ):
            self.reason = "batch_safety_violation"

    def snapshot(self):
        return dict(
            model_charge=self.model_charge,
            token_charge=self.token_charge,
            failures=self.failures,
            consecutive_api_failures=self.consecutive_api_failures,
            stop_reason=self.reason,
            scope="Sum of original JobBudget charges and lost-job allowances, not actual invoice or a strict monetary cap",
        )
