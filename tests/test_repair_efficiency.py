"""Repair progress, current source reuse, safe history trimming and budget isolation."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
from openai import APIError

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.repair_efficiency import RepairContextManager, plan_case_requests, source_context
from corecoder.workflow import RepairAgent, load_handoff, project_directory, run_repair
from corecoder.workflow_demo import prepare_demo


@pytest.fixture
def case(tmp_path):
    return prepare_demo(tmp_path / "case")


def read_call(identifier="read"):
    return ToolCall(identifier, "read_file", {"file_path": "ratio.py"})


def edit_call():
    return ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})


def test_handoff_carries_actual_read_source_and_runtime_invalidates_after_edit(case):
    _, project, report, _, _ = case
    handoff = load_handoff(report, project)
    assert any("return a // b" in item["content"] for item in handoff["source_context"]["excerpts"])
    agent = RepairAgent(ScriptedLLM([]), handoff)
    with project_directory(project):
        before = json.loads(agent._full_messages()[-1]["content"].split("\n", 1)[1])
        assert before["source_excerpts_revalidated_now"]
        agent._exec_tool(edit_call())
        after = json.loads(agent._full_messages()[-1]["content"].split("\n", 1)[1])
    assert after["invalidated_source_observation_ids"]
    assert not any("a // b" in item["content"] for item in after["source_excerpts_revalidated_now"])
    assert after["effective_file_edits"] == 1


def test_handoff_projection_excludes_stale_and_incomplete_lines():
    observations = {
        "old": {"id": "old", "kind": "file_excerpt", "data": {"file_sha256": "old", "resolved_path": "/p/config.py",
                "start_line": 1, "end_line": 1, "content": "stale"}},
        "new": {"id": "new", "kind": "file_excerpt", "data": {"file_sha256": "new", "resolved_path": "/p/config.py",
                "start_line": 4, "end_line": 9, "char_truncated": True, "content": "complete\ncut"}},
    }
    result = source_context(observations, {"new"}, {"hypotheses": []})
    assert len(result["excerpts"]) == 1
    assert result["excerpts"][0]["content"] == "complete"
    assert result["excerpts"][0]["start_line"] == result["excerpts"][0]["end_line"] == 4
    assert result["excerpts"][0]["omitted_chars"] == 4


def test_source_projection_bounds_payload_and_keeps_failure_line():
    content = "\n".join(f"line {i} " + "x" * 100 for i in range(50))
    item = {"id": "obs", "kind": "source_location", "data": {"file_sha256": "hash", "resolved_path": "/p/conv.py",
            "current_line_no": 25, "snippet": {"start_line": 1, "end_line": 50, "content": content}}}
    result = source_context({"obs": item}, {"obs"}, {"hypotheses": [{"observation_ids": ["obs"]}]})
    excerpt = result["excerpts"][0]
    assert excerpt["start_line"] <= 25 <= excerpt["end_line"]
    assert len(excerpt["content"]) <= 1400 and excerpt["omitted_chars"] > 0
    assert excerpt["content"].splitlines() == content.splitlines()[excerpt["start_line"] - 1:excerpt["end_line"]]
    assert not source_context({"obs": item}, {"obs"}, {}, max_chars=10)["excerpts"]


def test_duplicate_reads_are_reused_but_external_changes_are_not(case, monkeypatch):
    _, project, report, _, _ = case
    agent = RepairAgent(ScriptedLLM([]), load_handoff(report, project))
    tool = agent._tool_by_name["read_file"]
    execute, calls = tool.execute, []
    def counted(file_path, offset=1, limit=2000):
        calls.append(file_path)
        return execute(file_path, offset, limit)
    monkeypatch.setattr(tool, "execute", counted)
    with project_directory(project):
        agent._exec_tool(read_call())
        reused = agent._exec_tool(ToolCall("alias", "read_file", {"file_path": str(project / "ratio.py"), "offset": 1, "limit": 2000}))
        assert "reused" in reused and len(calls) == 1
        (project / "ratio.py").write_text("value = 9\n")
        result = agent._exec_tool(read_call("changed"))
    assert "value = 9" in result and len(calls) == 2
    assert agent.events[1]["reused_from_event"] == 1
    assert agent.events[2]["reused_from_event"] is None


def test_guard_is_checked_even_for_an_unchanged_cached_read(case):
    _, project, report, _, _ = case
    allowed = [True]
    def guard(name, args):
        if not allowed[0]:
            raise ValueError("read now prohibited")
    agent = RepairAgent(ScriptedLLM([]), load_handoff(report, project), tool_guard=guard)
    with project_directory(project):
        agent._exec_tool(read_call())
        allowed[0] = False
        result = agent._exec_tool(read_call("blocked"))
    assert "prohibited" in result and agent.read_cache_hits == 0


def test_read_racing_with_a_change_is_not_cached_under_the_old_hash(case, monkeypatch):
    _, project, report, _, _ = case
    agent = RepairAgent(ScriptedLLM([]), load_handoff(report, project))
    path = project / "ratio.py"
    original = path.read_bytes()
    tool = agent._tool_by_name["read_file"]
    execute = tool.execute
    def raced(file_path, offset=1, limit=2000):
        path.write_text("value = 3\n")
        return execute(file_path, offset, limit)
    monkeypatch.setattr(tool, "execute", raced)
    with project_directory(project):
        agent._exec_tool(read_call())
        path.write_bytes(original)
        monkeypatch.setattr(tool, "execute", execute)
        result = agent._exec_tool(read_call("after"))
    assert "a // b" in result and agent.read_cache_hits == 0


def test_edit_and_shell_actions_invalidate_read_reuse(case, monkeypatch):
    _, project, report, _, _ = case
    agent = RepairAgent(ScriptedLLM([]), load_handoff(report, project))
    monkeypatch.setattr(agent._tool_by_name["bash"], "execute", lambda command: "shell action")
    with project_directory(project):
        agent._exec_tool(read_call())
        agent._exec_tool(ToolCall("shell", "bash", {"command": "placeholder"}))
        agent._exec_tool(read_call("after_shell"))
        agent._exec_tool(edit_call())
        result = agent._exec_tool(read_call("after_edit"))
    assert "a / b" in result and agent.read_cache_hits == 0


def test_read_loop_stops_without_claiming_a_repair(case):
    _, project, report, command, _ = case
    model = ScriptedLLM([LLMResponse(tool_calls=[read_call(str(i))]) for i in range(10)])
    result = run_repair(load_handoff(report, project), "Repair", model, verify_command=command, max_investigation_rounds=3)
    assert result["status"] == "incomplete" and not result["workflow_completed"]
    assert result["metrics"]["llm_calls"] == 3
    assert result["metrics"]["stop_reason"] == "investigation_limit"
    assert result["metrics"]["read_cache_hits"] == 2
    assert not result["file_changes"] and result["verification"]["status"] == "not_run"


def test_effective_edit_resets_investigation_rounds_and_can_finish(case):
    _, project, report, command, _ = case
    model = ScriptedLLM([
        LLMResponse(tool_calls=[read_call()]), LLMResponse(tool_calls=[edit_call()]),
        LLMResponse(tool_calls=[read_call("new")]), LLMResponse(content="Ready for verification"),
    ])
    result = run_repair(load_handoff(report, project), "Repair", model, verify_command=command, max_investigation_rounds=2)
    assert result["status"] == "verified" and result["workflow_completed"]
    assert result["metrics"]["effective_file_edits"] == 1


def test_noop_edit_does_not_count_as_progress(case):
    _, project, report, _, _ = case
    same = ToolCall("same", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a // b"})
    model = ScriptedLLM([LLMResponse(tool_calls=[same]) for _ in range(5)])
    result = run_repair(load_handoff(report, project), "Repair", model, max_investigation_rounds=2)
    assert result["status"] == "incomplete"
    assert result["metrics"]["effective_file_edits"] == 0
    assert result["metrics"]["llm_calls"] == 2


def test_shell_change_to_observed_source_counts_as_progress(case, monkeypatch):
    _, project, report, _, _ = case
    agent = RepairAgent(ScriptedLLM([]), load_handoff(report, project), max_investigation_rounds=2)
    def shell(command):
        path = project / "ratio.py"
        path.write_text(path.read_text().replace("a // b", "a / b"))
        return "changed"
    monkeypatch.setattr(agent._tool_by_name["bash"], "execute", shell)
    with project_directory(project):
        agent._exec_tool(read_call())
        agent._full_messages()
        assert agent.investigation_rounds == 1
        agent._exec_tool(ToolCall("shell", "bash", {"command": "scripted change"}))
        agent._full_messages()
    assert agent.mutation_count == 1 and agent.investigation_rounds == 0


@pytest.mark.parametrize("max_rounds", [1, 12])
def test_partial_patch_is_verified_without_upgrading_incomplete_workflow(case, max_rounds):
    _, project, report, command, _ = case
    class FailingAfterEdit(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if not self._turns:
                raise RuntimeError("scripted provider budget exhausted")
            return super().chat(*args, **kwargs)
    result = run_repair(load_handoff(report, project), "Repair", FailingAfterEdit([LLMResponse(tool_calls=[edit_call()])]),
                        verify_command=command, max_rounds=max_rounds)
    assert result["status"] == "incomplete" and not result["workflow_completed"]
    assert result["verification"]["status"] == "passed" and result["verification"]["exit_code"] == 0
    assert result["verification"]["trigger"] == "partial_file_changes"


def test_api_failure_after_patch_is_redacted_and_actual_patch_is_checked(case):
    _, project, report, command, _ = case
    class FailedProvider(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if not self._turns:
                raise APIError("sk-must-not-be-saved", request=SimpleNamespace(), body=None)
            return super().chat(*args, **kwargs)
    result = run_repair(load_handoff(report, project), "Repair", FailedProvider([LLMResponse(tool_calls=[edit_call()])]),
                        verify_command=command)
    assert result["status"] == "incomplete" and result["verification"]["status"] == "passed"
    assert "sk-must-not-be-saved" not in json.dumps(result)
    assert result["metrics"]["stop_reason"] == "model_api_error"


def test_keyboard_interrupt_does_not_launch_verification_after_a_patch(case):
    _, project, report, _, _ = case
    class Interrupted(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if not self._turns:
                raise KeyboardInterrupt
            return super().chat(*args, **kwargs)
    marker = project / "must_not_exist"
    result = run_repair(load_handoff(report, project), "Repair", Interrupted([LLMResponse(tool_calls=[edit_call()])]),
                        verify_command=[sys.executable, "-c", "from pathlib import Path; Path('must_not_exist').touch()"])
    assert result["status"] == "interrupted" and not marker.exists()
    assert result["verification"]["status"] == "not_run"


def test_deterministic_compression_preserves_task_and_whole_parallel_tool_groups():
    first = {"role": "user", "content": "Keep this task, evidence ID ev1 and exact source hash abc."}
    messages = [first]
    for i in range(5):
        messages.append(LLMResponse(tool_calls=[ToolCall(f"a{i}", "read_file", {}), ToolCall(f"b{i}", "grep", {})]).message)
        messages.extend({"role": "tool", "tool_call_id": f"{letter}{i}", "content": "x" * 1500} for letter in "ab")
    original = copy.deepcopy(messages)
    class ForbiddenSummary:
        def chat(self, **kwargs):
            pytest.fail("Compression must not contact a model")
    manager = RepairContextManager(1500)
    assert manager.maybe_compress(messages, ForbiddenSummary())
    assert messages[0] == first and len(messages) < len(original)
    expected = {call["id"] for item in messages for call in item.get("tool_calls", [])}
    actual = {item["tool_call_id"] for item in messages if item["role"] == "tool"}
    assert actual == expected and manager.events[0]["dropped_messages"] > 0


@pytest.mark.parametrize("value", [0, True, 31])
def test_invalid_investigation_limits_are_rejected(case, value):
    _, project, report, _, _ = case
    with pytest.raises(ValueError, match="investigation"):
        run_repair(load_handoff(report, project), "Repair", ScriptedLLM([]), max_investigation_rounds=value)


def test_reserved_budgets_protect_later_cases_and_count_all_prior_phases():
    counts = {"m01": 6, "m02": 8, "m03": 6, "m04": 7, "m05": 14, "m06": 5}
    active = ["m01", "m02", "m04", "m05", "m06"]
    plan = plan_case_requests(active, counts, 72, 20)
    assert sum(plan.values()) == 26 and min(plan.values()) >= 5
    assert plan == plan_case_requests(list(reversed(active)), counts, 72, 20)
    assert all(counts[cid] + value <= 20 for cid, value in plan.items())
    assert plan_case_requests(active, counts, 46, 20) == dict.fromkeys(sorted(active), 0)


def test_capped_cases_cannot_take_or_lose_other_cases_reserved_requests():
    plan = plan_case_requests(["a", "b", "c"], {"a": 9}, 24, 10)
    assert plan == {"a": 1, "b": 7, "c": 7}
    assert plan_case_requests(["a"], {}, 72, 14) == {"a": 14}


@pytest.mark.parametrize("ids,counts,total,cap", [(["a", "a"], {}, 4, 4), (["a"], {"b": 5}, 4, 4),
                                               (["a"], {}, True, 4), (["a"], {"a": -1}, 4, 4)])
def test_invalid_or_exhausted_ledger_cannot_reset_budget(ids, counts, total, cap):
    with pytest.raises(ValueError):
        plan_case_requests(ids, counts, total, cap)


def test_case_runner_enforces_reserved_allowance_before_provider_request(case, monkeypatch):
    from corecoder.fault_injection import child_environment, digest, write_json
    from examples import m2d_fault_acceptance as acceptance
    workspace, project, report, _, _ = case
    (project / "run_experiment.py").write_text("# protected launcher\n")
    (project / "scenario.json").write_text("{}")
    verifier = workspace / "verify.py"
    verifier.write_text("import sys; sys.path.insert(0, sys.argv[1]); from ratio import ratio; assert ratio(5, 2) == 2.5\n")
    expect = workspace / "expect.json"
    expect.write_text("{}")
    annotation = workspace / "annotation.json"
    write_json(annotation, {"fault_kind": "ratio", "injected_target_sha256": digest(project / "ratio.py"),
                            "injected_lines": [2], "healthy_control": {"exit_code": 0}, "injected_control": {"exit_code": 1}})
    prior_session = workspace / "prior-session.json"
    write_json(prior_session, {"turns": [{"artifact": str(report)}]})
    previous = {"session": str(prior_session), "diagnosis_status": "completed", "repair_status": "incomplete",
                "file_changes": [], "requests": [{"number": 1}], "responses": [{"prompt_tokens": 2, "completion_tokens": 1}],
                "elapsed_seconds": 0}
    fixture = {"id": "case", "project": str(project), "annotation": str(annotation), "log": str(workspace / "failure.log"),
               "command": [sys.executable, "-B", str(verifier), str(project), str(expect)],
               "environment": child_environment(project, workspace / "settings"),
               "manifest": {name: digest(project / name) for name in ("ratio.py", "run_experiment.py", "scenario.json")}}
    actual_requests = []
    class Provider:
        model = "offline-counting-provider"
        def _call_with_retry(self, params, max_retries=1):
            actual_requests.append(params)
        def chat(self, messages, tools=None, on_token=None):
            self._call_with_retry({})
            return LLMResponse(tool_calls=[read_call(str(len(actual_requests)))], prompt_tokens=10, completion_tokens=1)
    monkeypatch.setattr(acceptance, "build_coding_llm", lambda config: Provider())
    monkeypatch.setattr(acceptance, "_coding_config", lambda: None)
    monkeypatch.setattr(acceptance, "build_deepseek_llm", lambda: pytest.fail("Do not repeat diagnosis"))
    total = list(previous["requests"])
    row = acceptance.run_case(fixture, workspace / "acceptance", total, previous=previous, phase="reserved", phase_allowance=2)
    assert len(actual_requests) == 2 and len(total) == 3
    assert row["phase_request_attempts"] == 2 and row["phase_request_allowance"] == 2
    assert row["repair_status"] == "incomplete" and "reserved request budget" in row["repair_failure_reason"]
    assert row["protected_files_unchanged"] and not row["file_changes"]
    ledger = json.loads((workspace / "acceptance/reserved/case/request-ledger.json").read_text())
    assert len(ledger["requests"]) == 3 and ledger["phase_allowance"] == 2
    zero = acceptance.run_case(fixture, workspace / "acceptance", list(previous["requests"]), previous=previous,
                               phase="zero", phase_allowance=0)
    assert zero["phase_request_attempts"] == 0 and len(actual_requests) == 2


@pytest.mark.parametrize("existing", ["summary.json", "budget-plan-initial.json", "runs"])
def test_existing_phase_cannot_reset_ledger_or_contact_model(tmp_path, monkeypatch, existing):
    from examples import m2d_fault_acceptance as acceptance
    summary = tmp_path / existing
    if existing == "runs":
        summary.mkdir()
    else:
        summary.write_text('{"request_attempts":72}')
    monkeypatch.setattr(acceptance, "build_deepseek_llm", lambda: pytest.fail("No API calls allowed"))
    with pytest.raises(ValueError, match="already recorded"):
        acceptance.main(["--source", str(tmp_path), "--experiment-python", sys.executable,
                         "--output-dir", str(tmp_path), "--prepared", "--live"])
    assert summary.is_dir() if existing == "runs" else summary.read_text() == '{"request_attempts":72}'


def test_offline_demo_uses_real_verification_and_keeps_incomplete_status(tmp_path):
    from corecoder.repair_efficiency_demo import run_demo
    summary = run_demo(tmp_path / "demo")
    assert summary["provider_request_attempts"] == 0
    assert summary["cases"]["handoff-edit"]["status"] == "verified"
    assert summary["cases"]["partial-patch"]["verification"] == "passed"
    assert not summary["cases"]["partial-patch"]["workflow_completed"]
    assert sum(summary["historical_budget_replay"]["reserved_new_requests"].values()) == 26
