import pytest

from researchcoder.runtime.budget import BudgetExceeded, BudgetLedger, RuntimeHooks
from researchcoder.runtime.optimization import RuntimeOptimization
from researchcoder.runtime.protocol import RuntimeLimits
from researchcoder.runtime.tracing import TraceSink


def writer(phase, *, calls=8):
    trace = TraceSink("job", "task", "attempt")
    limits = RuntimeLimits(max_rounds=6, max_model_calls=calls)
    hooks = RuntimeHooks(BudgetLedger(limits, trace), trace)
    hooks.optimization = RuntimeOptimization(version="explicit-report-v1", compact_reports=True, coding_phase_version=phase)
    hooks._coding_completion = True
    return hooks


@pytest.mark.parametrize("phase, penultimate", [
    ("corrective-edit-window-v1", None), ("repair-cycle-v1", "round_reserve"), ("off", "round_reserve"),
])
def test_admitted_round_counter_and_legacy_boundary(phase, penultimate):
    hooks = writer(phase)
    for _ in range(5):
        hooks.before_round()
    assert hooks.coding_report_reason([], []) == penultimate
    hooks.before_round()
    expected = "last_admitted_round" if phase == "corrective-edit-window-v1" else "round_reserve"
    assert hooks.coding_report_reason([], []) == expected
    with pytest.raises(BudgetExceeded):
        hooks.before_round()


def test_last_model_call_and_unknown_execution_still_take_precedence():
    hooks = writer("corrective-edit-window-v1", calls=6)
    for _ in range(5):
        hooks.before_round()
        reservation = hooks.ledger.reserve_model(20)
        hooks.ledger.settle(reservation, prompt=10, completion=3)
    assert hooks.coding_report_reason([], []) == "model_call_reserve"
    hooks.permission_stop = True
    with pytest.raises(BudgetExceeded):
        hooks.coding_report_reason([], [])
