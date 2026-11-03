"""Source search contracts, budgets, evidence gates and actual investigation flow."""

import json
from pathlib import Path

import pytest

from corecoder import source_search
from corecoder.completion import CompletionChecklist
from corecoder.diagnose import validate_diagnosis
from corecoder.diagnostic_policy import EvidenceToolPolicy
from corecoder.diagnostic_tools import Investigation
from corecoder.evidence_context import EvidenceContextManager, observation_view, project_tool_result
from corecoder.observation_state import current_observation_ids
from corecoder.report_findings import validate_findings
from corecoder.source_search_demo import run_demo


@pytest.fixture
def state(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "train.py").write_text("def compute_loss(total, count):\n    loss = total / count\n    return loss\n", encoding="utf-8")
    log = tmp_path / "failure.log"
    log.write_text("step=1 loss=nan\n", encoding="utf-8")
    state = Investigation(log, root)
    state.extract_log_evidence()
    return state


def search(state, **kwargs):
    args = {"query": "loss", "evidence_ids": [state.evidence[0]["id"]], **kwargs}
    result = json.loads(state.execute("search_project_source", args))
    assert "error" not in result, result
    return result["observations"][:-1], result["observations"][-1]


def test_candidates_link_evidence_lines_function_and_hash_without_executing(state):
    (state.root / "train.py").write_text("raise RuntimeError('must not run')\ndef compute_loss():\n    return loss\n")
    before = (state.root / "train.py").read_bytes()
    candidates, summary = search(state)
    assert candidates[0]["data"]["hit_line"] == 2
    assert candidates[1]["data"]["function"]["name"] == "compute_loss"
    assert candidates[0]["evidence_ids"] == [state.evidence[0]["id"]]
    assert len(candidates[0]["data"]["file_sha256"]) == 64
    assert summary["data"]["candidate_ids"] == [c["id"] for c in candidates]
    assert not summary["data"]["partial"]
    assert (state.root / "train.py").read_bytes() == before


def test_literal_query_not_regex_and_case_insensitive(state):
    (state.root / "train.py").write_text("pattern = '.*'\nLOSS = 3\n")
    candidates, _ = search(state, query=".*")
    assert len(candidates) == 1 and candidates[0]["data"]["hit_line"] == 1
    candidates, _ = search(state)
    assert len(candidates) == 1 and candidates[0]["data"]["hit_line"] == 2


def test_scope_skips_hidden_dependencies_and_allows_source_data_directory(state):
    for name in (".private", "node_modules", "venv", "data"):
        directory = state.root / name
        directory.mkdir()
        (directory / "secret.py").write_text("loss = 1\n")
    candidates, _ = search(state)
    assert {Path(c["data"]["relative_path"]).as_posix() for c in candidates} == {"train.py", "data/secret.py"}
    candidates, _ = search(state, paths=["data"])
    assert len(candidates) == 1
    candidates, _ = search(state, paths=["train.py", "train.py"])
    assert len(candidates) == 3


def test_symlinks_are_not_followed(state, tmp_path):
    other = tmp_path / "other.py"
    other.write_text("loss = 'private'\n")
    link = state.root / "link.py"
    try:
        link.symlink_to(other)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    candidates, _ = search(state)
    assert all(c["data"]["relative_path"] == "train.py" for c in candidates)
    result = json.loads(state.execute("search_project_source", {"query": "loss", "evidence_ids": [state.evidence[0]["id"]], "paths": ["link.py"]}))
    assert "error" in result


@pytest.mark.parametrize("change", [
    {"query": ""}, {"query": "x"}, {"query": "a\nb"}, {"query": "a\x00b"}, {"query": 3},
    {"evidence_ids": []}, {"evidence_ids": ["unknown"]}, {"max_results": True}, {"max_results": 21},
    {"paths": []}, {"paths": ["../"]}, {"paths": [".private"]}, {"paths": "train.py"},
])
def test_invalid_requests_are_audited(state, change):
    result = json.loads(state.execute("search_project_source", {"query": "loss", "evidence_ids": [state.evidence[0]["id"]], **change}))
    assert "error" in result
    assert state.events[-1]["status"] == "error"
    assert not state.observations


def test_unexposed_evidence_and_search_before_extraction_are_rejected(state):
    state.exposed_evidence_ids.clear()
    result = json.loads(state.execute("search_project_source", {"query": "loss", "evidence_ids": [state.evidence[0]["id"]]}))
    assert "error" in result
    state.evidence = None
    assert EvidenceToolPolicy().select(state)[0] == ["extract_log_evidence"]


@pytest.mark.parametrize("limit,value,reason", [("files", 0, "file_limit"), ("entries", 1, "entry_limit"),
                                              ("file_bytes", 3, "too_large"), ("bytes", 3, "byte_limit")])
def test_scan_limits_report_partial_and_stay_bounded(state, monkeypatch, limit, value, reason):
    monkeypatch.setitem(source_search.SEARCH_LIMITS, limit, value)
    _, summary = search(state)
    assert summary["data"]["partial"]
    assert reason in summary["data"]["reasons"]
    assert summary["data"]["stats"]["bytes_read"] <= source_search.SEARCH_LIMITS["bytes"]
    assert summary["data"]["stats"]["entries"] <= source_search.SEARCH_LIMITS["entries"]


def test_result_limit_and_missing_scope_are_explicit(state):
    candidates, summary = search(state, max_results=1)
    assert len(candidates) == 1 and summary["data"]["reasons"] == {"result_limit": 1}
    candidates, summary = search(state, paths=["absent"])
    assert not candidates and summary["data"]["reasons"] == {"missing_scope": 1}


def test_no_matches_are_scoped_unknown_not_proof_of_absence(state):
    _, summary = search(state, query="unseen")
    assert summary["data"]["status"] == "searched" and summary["data"]["matches_returned"] == 0
    plan = CompletionChecklist(state).status()
    assert plan["ready_for_report"] and plan["blocked_count"] == 1
    assert "unknown" in plan["checks"][0]["blocked_reason"]


def test_checklist_requires_search_then_actual_same_version_candidate_read(state):
    plan = CompletionChecklist(state)
    assert plan.guidance()["next_checks"][0]["next_action"]["arguments"]["query"] == "loss"
    assert "search_project_source" in EvidenceToolPolicy().select(state)[0]
    candidates, _ = search(state)
    assert plan.guidance()["next_checks"][0]["kind"] == "candidate_read"
    state.execute("read_project_file", {"path": "train.py", "start_line": 3})
    assert plan.status()["ready_for_report"]  # At least one candidate fully inspected, not every keyword hit.
    (state.root / "train.py").write_text("loss = 5\n")
    state.execute("read_project_file", {"path": "train.py"})
    assert all(c["id"] not in current_observation_ids(state) for c in candidates)
    assert plan.guidance()["next_checks"][0]["kind"] == "source_search"
    assert all(c["id"] in state.observations for c in candidates)


def test_source_hints_use_direct_reads_instead_of_forcing_search(state):
    plan = CompletionChecklist(state, source_hints=["train.py"])
    assert [c["kind"] for c in plan.status()["checks"]] == ["source_hint_read"]
    state.execute("read_project_file", {"path": "train.py"})
    assert plan.status()["ready_for_report"]


def test_search_projection_and_memory_keep_limits_query_and_candidates(state):
    candidates, summary = search(state, max_results=1)
    projected = json.loads(project_tool_result(json.dumps({"observations": [*candidates, summary]}, indent=2)))
    data = projected["observations"][-1]["data"]
    assert data["query"] == "loss" and data["partial"] and data["candidate_ids"]
    assert observation_view(candidates[0])["data"]["hit_line"] == 1
    memory = json.loads(EvidenceContextManager(state).build_memory(12000))
    search_row = next(c for c in memory["observations"] if c["kind"] == "source_search")
    assert search_row["data"]["reasons"] == {"result_limit": 1}
    assert search_row["snapshot_current"]


@pytest.mark.parametrize("selection,context", [("all", "baseline"), ("guided", "baseline"),
                                               ("all", "evidence"), ("guided", "evidence")])
def test_frameless_investigation_runs_real_tools_without_api(tmp_path, selection, context):
    result = run_demo(tmp_path / "demo", tool_selection=selection, context_mode=context)
    assert result["status"] == "completed", result["failure_reason"]
    assert [e["tool"] for e in result["tool_events"]] == ["extract_log_evidence", "search_project_source", "read_project_file"]
    assert result["investigation_checklist"]["ready_for_report"]
    assert result["metrics"]["llm_calls"] == 4
    assert not result["evidence"][0].get("frames")


def test_candidate_only_cannot_satisfy_report_current_state(state):
    candidates, _ = search(state)
    report = {"summary": "candidate", "historical_failure": {"description": "nan", "evidence_ids": [state.evidence[0]["id"]]},
              "current_state": {"assessment": "candidate", "observation_ids": [candidates[0]["id"]]},
              "hypotheses": [], "recommended_actions": ["inspect"], "limitations": ["unknown"]}
    with pytest.raises(ValueError, match="No usable current"):
        validate_diagnosis(json.dumps(report), state)


def test_candidate_cannot_be_claimed_as_primary_location_even_after_another_read(state):
    candidates, _ = search(state)
    state.execute("read_project_file", {"path": "train.py"})
    with pytest.raises(ValueError, match="observed source line"):
        validate_findings({"failure_category": "unknown", "cause_code": "unknown", "primary_locations": [
            {"observation_id": candidates[0]["id"], "line_no": 1},
        ]}, state)


def test_partial_search_without_hits_is_blocked_unknown(state, monkeypatch):
    monkeypatch.setitem(source_search.SEARCH_LIMITS, "files", 0)
    search(state)
    check = CompletionChecklist(state).status()["checks"][0]
    assert check["status"] == "blocked" and "unscanned" in check["blocked_reason"]


def test_depth_limit_and_decode_failure_are_explicit(state, monkeypatch):
    monkeypatch.setitem(source_search.SEARCH_LIMITS, "depth", 1)
    (state.root / "nested").mkdir()
    (state.root / "nested" / "hidden.py").write_text("loss = 1\n")
    (state.root / "invalid.py").write_bytes(b"\xff\xfe")
    _, summary = search(state)
    assert summary["data"]["reasons"] == {"decode_error": 1, "depth_limit": 1}


def test_long_hit_line_is_blocked_instead_of_repeated_identical_reads(state):
    (state.root / "train.py").write_text("loss = '" + "x" * 15000 + "'\n")
    search(state)
    plan = CompletionChecklist(state)
    request = plan.guidance()["next_checks"][0]["next_action"]
    state.execute(request["tool"], request["arguments"])
    status = plan.status()
    assert status["ready_for_report"] and status["blocked_count"] == 1
    assert "character limit" in status["checks"][-1]["blocked_reason"]


def test_bounded_search_errors_still_consume_tool_budget(state):
    state.max_tool_calls = 1
    state.execute("search_project_source", {"query": "x", "evidence_ids": []})
    result = json.loads(state.execute("search_project_source", {"query": "loss", "evidence_ids": [state.evidence[0]["id"]]}))
    assert "budget exhausted" in result["error"]["message"]
