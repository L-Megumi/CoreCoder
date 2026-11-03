"""Versioned allocation inside the existing JobBudget, not a second executor."""
from dataclasses import replace
from researchcoder.runtime.budget import BudgetExceeded

VERSION = "repair-cycle-v1"
MIN_CALLS, MIN_TOOLS, MIN_TOKENS, MIN_SECONDS = 4, 3, 20000, 70
INSTRUCTION = (
    "\n[Controller repair-cycle-v1]\n"
    "Treat expert explanations as hypotheses and check decisive assumptions against current source. "
    "Use accepted observations instead of repeating broad investigation. Preserve budget for a minimal "
    "evidence-supported edit, public verification and one bounded correction. After a failed public check, "
    "use the retained failure to reassess the previous patch; do not blindly reapply it. "
    "Hidden evaluation checks are unavailable. If evidence is insufficient, finish incomplete."
)


def task_limits(limits, generation):
    """Caller has already clipped all dimensions to the same Job's allowance."""
    if generation not in (0, 1):
        raise BudgetExceeded("Repair cycle permits one correction")
    if generation == 0:
        calls = min(9, limits.max_model_calls - MIN_CALLS)
        tools = limits.max_tool_calls - MIN_TOOLS
        tokens = limits.max_total_tokens - MIN_TOKENS
        seconds = limits.max_runtime_seconds - MIN_SECONDS
        if calls < 4 or tools < 3 or tokens <= limits.max_output_tokens or seconds < 1:
            raise BudgetExceeded("Insufficient first-attempt and correction allowance")
        return replace(limits, max_model_calls=calls, max_rounds=min(limits.max_rounds, calls),
            max_tool_calls=tools, max_total_tokens=tokens,
            max_context_tokens=min(limits.max_context_tokens, tokens), max_runtime_seconds=seconds)
    if (limits.max_model_calls < MIN_CALLS or limits.max_tool_calls < MIN_TOOLS
            or limits.max_total_tokens < MIN_TOKENS or limits.max_runtime_seconds < MIN_SECONDS):
        raise BudgetExceeded("Insufficient known correction allowance")
    calls = min(6, limits.max_model_calls)
    return replace(limits, max_model_calls=calls, max_rounds=min(limits.max_rounds, calls),
        max_runtime_seconds=min(120, limits.max_runtime_seconds))


def provider_accounting_unknown(result, transports):
    """Missing counters stay unknown; cancellation must not raise a TypeError."""
    if any(r.get("status") != "known" for r in transports):
        return True
    requests, unknown = result.get("model_requests"), result.get("unknown_requests")
    if type(requests) is not int or requests < 0 or type(unknown) is not int or unknown < 0:
        return True
    return bool(unknown or len(transports) != requests
                or (requests > 0 and result.get("usage_complete") is not True))
