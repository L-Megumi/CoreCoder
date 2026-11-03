import json
from types import SimpleNamespace

import pytest

from corecoder.controlled_experiment import RequestLedger, product_hashes, schedule_runs, score_localization, summarize


def test_schedule_is_paired_complete_and_reproducible():
    schedule = schedule_runs([f"m0{i}" for i in range(1, 7)])
    assert len(schedule) == 48
    assert schedule == schedule_runs([f"m0{i}" for i in range(1, 7)])
    assert len({row["run_id"] for row in schedule}) == 48
    for case in {row["case_id"] for row in schedule}:
        for repeat in (1, 2):
            assert {row["arm"] for row in schedule if row["case_id"] == case and row["repeat"] == repeat} == {
                "baseline", "tools", "context", "both"}


@pytest.mark.parametrize("cases,repeats", [([], 2), (["a", "a"], 2), (["a"], 0), (["a"], 4)])
def test_schedule_rejects_invalid_inputs(cases, repeats):
    with pytest.raises(ValueError):
        schedule_runs(cases, repeats)


def test_transport_budget_counts_failures_before_sending_and_never_resets(tmp_path):
    calls = []
    path = tmp_path / "ledger.json"
    def transport(params, max_retries):
        calls.append(max_retries)
        assert json.loads(path.read_text())["events"][-1]["status"] == "started"
        raise RuntimeError("secret transport error must not persist")
    model = SimpleNamespace(_call_with_retry=transport)
    ledger = RequestLedger(path, 2, 1)
    ledger.attach(model, "r1")
    with pytest.raises(RuntimeError, match="secret transport"):
        model._call_with_retry({"model": "test"})
    with pytest.raises(RuntimeError, match="budget exhausted"):
        model._call_with_retry({"model": "test"})
    assert calls == [1]
    assert "secret transport" not in path.read_text()
    assert ledger.events[0]["status"] == "request_error"
    with pytest.raises(ValueError, match="never be reset"):
        RequestLedger(path, 2, 1)


def test_stream_usage_includes_cache_and_actual_model(tmp_path):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=64,
                            prompt_cache_miss_tokens=36)
    model = SimpleNamespace(_call_with_retry=lambda params, max_retries: iter([
        SimpleNamespace(model="actual-version", usage=None), SimpleNamespace(model="actual-version", usage=usage)]))
    ledger = RequestLedger(tmp_path / "ledger.json", 1, 1)
    ledger.attach(model, "r1")
    assert len(list(model._call_with_retry({"model": "alias", "tools": [{"name": "test"}]}))) == 2
    event = ledger.events[0]
    assert event["response_model"] == "actual-version"
    assert event["usage_received"] and event["status"] == "completed"
    assert event["prompt_tokens"] == 100 and event["prompt_cache_hit_tokens"] == 64


def test_failed_stream_preserves_partial_usage(tmp_path):
    def stream():
        yield SimpleNamespace(model="test", usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3))
        raise RuntimeError("do not persist body")
    model = SimpleNamespace(_call_with_retry=lambda params, max_retries: stream())
    ledger = RequestLedger(tmp_path / "ledger.json", 1, 1)
    ledger.attach(model, "r1")
    with pytest.raises(RuntimeError):
        list(model._call_with_retry({"model": "test"}))
    assert ledger.events[0]["status"] == "stream_error"
    assert ledger.events[0]["prompt_tokens"] == 12


def test_global_budget_cannot_be_borrowed_by_new_run(tmp_path):
    ledger = RequestLedger(tmp_path / "ledger.json", 1, 2)
    for run in ("r1", "r2"):
        model = SimpleNamespace(_call_with_retry=lambda params, max_retries: iter([]))
        ledger.attach(model, run)
        if run == "r1":
            list(model._call_with_retry({"model": "test"}))
        else:
            with pytest.raises(RuntimeError, match="budget exhausted"):
                model._call_with_retry({"model": "test"})
    assert len(ledger.events) == 1


def test_coverage_requires_visible_lines_current_citation_and_valid_report():
    annotation = {"injected_target_sha256": "sha", "injected_lines": [11]}
    report = {"status": "completed", "current_observation_ids": ["o1"],
              "observations": [{"id": "o1", "kind": "file_excerpt", "data": {
                  "file_sha256": "sha", "start_line": 10, "end_line": 99, "content": "first\nsecond\n"}}],
              "diagnosis": {"hypotheses": [{"observation_ids": ["o1"]}]}}
    assert score_localization(report, annotation)["valid_hypothesis_cites_current_injected_line"]
    report["current_observation_ids"] = []
    assert not score_localization(report, annotation)["valid_hypothesis_cites_current_injected_line"]
    report["status"] = "incomplete"
    assert not score_localization(report, annotation)["valid_report"]
    annotation["injected_lines"] = [50]
    assert not score_localization(report, annotation)["injected_line_observed"]


def test_summary_keeps_failed_runs_and_pair_costs():
    def row(arm, tokens, valid):
        return {"case_id": "m01", "repeat": 1, "arm": arm, "source_unchanged": True,
                "usage_complete": True, "scores": {"valid_report": valid, "injected_line_observed": True,
                    "valid_hypothesis_cites_current_injected_line": valid}, "request_attempts": 2,
                "prompt_tokens": tokens, "completion_tokens": 10, "tool_calls": 1, "tool_errors": 0,
                "tool_schema_chars": 200, "elapsed_seconds": 1}
    result = summarize([row("baseline", 100, True), row("both", 50, False)])
    assert result["arms"]["both"]["runs"] == 1 and result["arms"]["both"]["valid_report"] == 0
    assert result["paired_against_baseline"]["both"]["prompt_tokens"]["total_relative_change"] == -0.5


def test_product_hashes_accepts_relative_and_absolute_roots(tmp_path, monkeypatch):
    (tmp_path / "corecoder").mkdir()
    (tmp_path / "examples").mkdir()
    (tmp_path / "corecoder/a.py").write_text("a=1\n")
    (tmp_path / "examples/b.py").write_text("b=2\n")
    monkeypatch.chdir(tmp_path)
    assert product_hashes(".") == product_hashes(tmp_path)
    assert set(product_hashes(".")) == {"corecoder/a.py", "examples/b.py"}
