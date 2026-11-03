import pytest
from benchmarks.recovery.run_faults import CASES, run_case


@pytest.mark.parametrize("mode", CASES)
def test_real_crash_reconciliation_and_no_duplicate_effect(tmp_path, mode):
    result = run_case(tmp_path / mode, mode)
    assert result["expected_disposition_matched"] and result["duplicate_effects"] == 0
    assert not result["supervised_processes_remaining"]
    if mode in ("parallel_reads", "artifact_uncommitted"):
        assert result["budget_after"]["charged"]["model_calls"] >= 16
        assert any(
            s["usage_unknown"] for s in result["budget_after"]["settlements"].values()
        )
    if mode == "unknown_usage":
        assert result["budget_before"] == result["budget_after"]
