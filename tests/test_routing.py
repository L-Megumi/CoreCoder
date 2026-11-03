"""Human-intent routing, mode isolation and real conversation transitions."""

import json
from pathlib import Path

import pytest
from openai import APIError

from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.researchcoder import main
from corecoder.routed_cli import submit
from corecoder.routed_session import RoutedSession
from corecoder.routing_demo import run_demo
from corecoder.task_router import select_route
from corecoder.tools import ALL_TOOLS
from corecoder.workflow_demo import DiagnosisDemoLLM, prepare_demo


@pytest.mark.parametrize("task,log,report,mode", [
    ("帮我写一个排序函数", False, False, "code"),
    ("实现一个日志分析器", True, True, "code"),
    ("create a logger", True, False, "code"),
    ("fix the sorting function", False, False, "code"),
    ("修复这个函数", False, False, "code"),
    ("排查这次训练报错", True, False, "diagnose"),
    ("分析日志中的 loss=nan", True, False, "diagnose"),
    ("debug this error", True, False, "diagnose"),
    ("修复这个报错", True, False, "diagnose"),
    ("修复这个报错", True, True, "repair"),
    ("请根据诊断修复", True, True, "repair"),
    ("please fix this error", False, True, "repair"),
    ("不要修复，解释原因", True, True, "review"),
    ("这个问题应该怎么修复？", True, True, "review"),
    ("修复方案是什么？", True, True, "review"),
    ("修复之前只分析原因，不改代码", True, True, "review"),
    ("explain the report", False, True, "review"),
    ("don't modify files; explain the report", False, True, "review"),
    ("先不要修改，分析日志", True, False, "diagnose"),
    ("排查训练异常", False, False, "needs_input"),
    ("那继续吧。", True, True, "needs_input"),
    ("continue", False, True, "needs_input"),
])
def test_human_task_routing(task, log, report, mode):
    assert select_route(task, has_log=log, has_report=report).mode == mode


@pytest.mark.parametrize("mode,log,report,expected", [
    ("code", True, True, "code"), ("diagnose", True, True, "diagnose"),
    ("repair", False, True, "repair"), ("review", False, True, "review"),
    ("repair", True, False, "needs_input"), ("diagnose", False, True, "needs_input"),
    ("review", True, False, "needs_input"),
])
def test_explicit_mode_overrides_require_inputs(mode, log, report, expected):
    assert select_route("Inspect the project", mode=mode, has_log=log, has_report=report).mode == expected


@pytest.mark.parametrize("task,mode", [("", "auto"), ("x" * 6001, "auto"), ("abc\x00", "auto"), ("task", "unknown")])
def test_invalid_routing_requests(task, mode):
    with pytest.raises(ValueError):
        select_route(task, mode=mode)


def test_readonly_request_cannot_silently_enter_explicit_repair():
    assert select_route("不要修改，只分析", mode="repair", has_report=True).mode == "needs_input"


def test_scoped_read_write_instruction_is_not_misread_as_read_only():
    assert select_route("只读写 metrics.py，修复缩放函数，保留原校验。", mode="repair", has_report=True).mode == "repair"


@pytest.mark.parametrize("task", [
    "修复项目源码，不改输入或校验。", "修复功能，不要修改测试。",
    "修复代码，不要修改 run_experiment.py。", "fix the implementation; do not modify the tests.",
    "修复函数，不要执行命令，控制器负责验证。",
])
def test_scoped_preservation_and_no_shell_requirements_allow_explicit_repair(task):
    assert select_route(task, mode="repair", has_report=True).mode == "repair"


@pytest.mark.parametrize("task", ["不要修改任何文件", "don't modify files", "不改代码，只分析问题"])
def test_global_readonly_requests_still_block_repair(task):
    assert select_route(task, mode="repair", has_report=True).mode == "needs_input"


@pytest.fixture
def case(tmp_path):
    return prepare_demo(tmp_path / "case")


def session_for(case, **kwargs):
    _, root, _, _, _ = case
    return RoutedSession(root, coding_factory=lambda: ScriptedLLM([]), diagnosis_factory=DiagnosisDemoLLM, **kwargs)


def test_missing_log_and_ambiguous_continue_do_not_build_models(case):
    session = session_for(case)
    assert session.handle("排查这个报错")["status"] == "needs_input"
    assert session.handle("继续")["status"] == "needs_input"
    assert not list(session.directory.glob("turn-*.json"))
    assert session.coding_agent is None
    assert json.loads((session.directory / "session.json").read_text())["phase"] == "needs_input"


def test_diagnosis_passes_human_task_and_sets_repair_context_with_readonly_tools(case):
    workspace, root, _, _, _ = case
    session = session_for(case, log_file=workspace / "failure.log")
    before = (root / "ratio.py").read_bytes()
    response = session.handle("排查报错，核对比例计算")
    assert response["status"] == "completed", response
    result = json.loads(Path(response["artifact"]).read_text())
    assert result["task"] == "排查报错，核对比例计算"
    assert session.report_file == Path(response["artifact"])
    assert all(event["tool"] in {"extract_log_evidence", "locate_source", "read_project_file", "inspect_paths", "search_project_source"}
               for event in result["tool_events"])
    assert (root / "ratio.py").read_bytes() == before
    assert session.coding_agent is None


def test_binding_new_log_clears_old_handoff_and_failed_diagnosis_cannot_restore_it(case):
    workspace, _, report, _, _ = case
    session = session_for(case, report_file=report)
    session.bind_log(workspace / "failure.log")
    assert session.report_file is None
    session.diagnosis_factory = lambda: ScriptedLLM([LLMResponse(content="finished") for _ in range(8)])
    result = session.handle("排查日志")
    assert result["status"] == "incomplete"
    assert session.report_file is None
    assert Path(result["artifact"]).is_file()


def test_two_code_turns_keep_history_and_all_original_tools(case):
    _, root, _, _, _ = case
    seen = []
    class Model(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            seen.append((list(messages), {item["function"]["name"] for item in tools}))
            return super().chat(messages, tools, on_token)
    model = Model([LLMResponse(content="first answer"), LLMResponse(content="second answer")])
    session = session_for(case)
    session.coding_factory = lambda: model
    session.handle("解释这个项目")
    session.handle("继续解释 ratio 函数")
    assert any(item.get("content") == "first answer" for item in seen[-1][0])
    assert all(names == {tool.name for tool in ALL_TOOLS} for _, names in seen)
    assert Path.cwd() != root


def test_report_review_has_no_execution_tools_and_does_not_consume_handoff(case):
    _, root, report, _, _ = case
    seen = []
    class Model(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            seen.append(tools)
            return super().chat(messages, tools, on_token)
    session = session_for(case, report_file=report)
    session.coding_factory = lambda: Model([LLMResponse(content="解释原因，尚未修复。")])
    before = (root / "ratio.py").read_bytes()
    result = session.handle("这个问题怎么修复？")
    assert result["mode"] == "review" and result["status"] == "completed"
    assert seen == [None] and session.report_file == report
    assert (root / "ratio.py").read_bytes() == before


def test_review_tool_request_is_rejected_without_execution(case):
    _, root, report, _, _ = case
    session = session_for(case, report_file=report)
    session.coding_factory = lambda: ScriptedLLM([LLMResponse(tool_calls=[ToolCall("bad", "write_file", {"file_path": "bad.txt", "content": "bad"})])])
    result = session.handle("解释报告")
    assert result["status"] == "incomplete" and "no tools" in result["text"]
    assert not (root / "bad.txt").exists()


def test_explicit_repair_runs_original_command_and_consumes_report(case):
    _, root, report, command, _ = case
    session = session_for(case, report_file=report, verify_command=command)
    session.coding_factory = lambda: ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(content="已修改，等待复验。"),
    ])
    result = session.handle("修复比例计算")
    assert result["status"] == "verified", result
    assert "a / b" in (root / "ratio.py").read_text()
    assert session.report_file is None and session.coding_agent is None
    assert session.handle("继续")["status"] == "needs_input"


def test_stale_report_rejected_before_model_factory_and_cleared(case):
    _, root, report, _, _ = case
    session = session_for(case, report_file=report)
    (root / "ratio.py").write_text("changed = True\n")
    session.coding_factory = lambda: pytest.fail("must reject before model construction")
    result = session.handle("修复比例计算")
    assert result["status"] == "incomplete" and "changed since diagnosis" in result["text"]
    assert session.report_file is None


def test_failed_verification_is_not_reported_as_success(case):
    _, _, report, command, _ = case
    session = session_for(case, report_file=report, verify_command=command)
    session.coding_factory = lambda: ScriptedLLM([LLMResponse(content="All fixed.")])
    result = session.handle("修复比例计算")
    assert result["status"] == "verification_failed"
    assert session.last_result["verification"]["exit_code"] != 0


def test_no_verification_command_keeps_repair_unverified(case):
    _, _, report, _, _ = case
    session = session_for(case, report_file=report)
    session.coding_factory = lambda: ScriptedLLM([LLMResponse(content="All fixed.")])
    assert session.handle("修复比例计算")["status"] == "completed_unverified"


def test_provider_errors_redacted_and_interruption_saved(case):
    session = session_for(case)
    def fail():
        raise APIError("secret-body", request=None, body={"key": "secret-body"})
    session.coding_factory = fail
    result = session.handle("解释项目")
    assert result["status"] == "incomplete"
    assert "secret-body" not in (session.directory / "session.json").read_text()
    def interrupt():
        raise KeyboardInterrupt
    session.coding_factory = interrupt
    assert session.handle("解释项目")["status"] == "interrupted"
    assert json.loads((session.directory / "session.json").read_text())["turns"][-1]["status"] == "interrupted"


def test_commands_update_bindings_and_select_one_turn_mode(case, capsys):
    workspace, _, report, command, _ = case
    session = session_for(case)
    submit(session, f'/report "{report}"')
    submit(session, "/verify " + json.dumps(command))
    submit(session, "/status")
    assert session.report_file == report and session.verify_command == command
    submit(session, f'/log "{workspace / "failure.log"}"')
    assert session.report_file is None
    response = submit(session, "/diagnose 排查异常")
    assert response["mode"] == "diagnose" and response["status"] == "completed"
    with pytest.raises(ValueError, match="Unknown"):
        submit(session, "/unknown")
    with pytest.raises(ValueError):
        submit(session, "/verify python -m pytest")
    assert capsys.readouterr().out


def test_dry_run_makes_no_session_files_or_model_requests(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("corecoder.routed_cli.load_dotenv", lambda *args, **kwargs: pytest.fail("no credential loading needed"))
    assert main(["run", "排查报错", "--project", str(tmp_path), "--log", "not-read.log", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "diagnose"
    assert list(tmp_path.iterdir()) == []


def test_dry_run_and_execution_use_same_explicit_prefix(tmp_path, capsys):
    assert main(["run", "/code 解释函数", "--project", str(tmp_path), "--log", "not-read.log", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "code"


def test_interactive_chat_uses_same_controller_and_exits_without_api(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder.routed_cli.load_dotenv", lambda *args, **kwargs: None)
    lines = iter(["排查报错", "/status", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(lines))
    assert main(["chat", "--project", str(tmp_path)]) == 0
    record = json.loads(next(tmp_path.glob(".tmp/session-*/session.json")).read_text())
    assert record["turns"][0]["status"] == "needs_input"


@pytest.mark.parametrize("command", [[], ["python", ""], "python -m pytest", [True]])
def test_invalid_verification_not_bound(case, command):
    session = session_for(case)
    with pytest.raises(ValueError):
        session.set_verification(command)


def test_new_session_directory_cannot_overwrite_existing_files(case):
    _, root, _, _, _ = case
    existing = root / "existing"
    existing.mkdir()
    (existing / "session.json").write_text("keep")
    with pytest.raises(FileExistsError):
        session_for(case, session_dir=existing)
    assert (existing / "session.json").read_text() == "keep"


def test_complete_multi_turn_workflow_executes_files_and_verification(tmp_path):
    result = run_demo(tmp_path / "demo")
    assert result["routes"] == ["code", "diagnose", "review", "repair", "code"]
    assert result["statuses"] == ["completed", "completed", "completed", "verified", "completed"]
    assert result["initial_exit_code"] != 0
    session = json.loads(Path(result["session"]).read_text())
    assert [turn["from_phase"] for turn in session["turns"]] == ["idle", "code", "diagnose", "review", "repair"]
    assert (Path(result["workspace"]) / "project/README.md").exists()


def test_one_shot_run_retains_general_coding_tools(tmp_path, monkeypatch):
    monkeypatch.setattr("corecoder.routed_cli.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("corecoder.routed_cli.build_coding_llm", lambda config: ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("write", "write_file", {"file_path": "new.txt", "content": "written"})]),
        LLMResponse(content="written"),
    ]))
    assert main(["run", "帮我写一个说明文件", "--project", str(tmp_path)]) == 0
    assert (tmp_path / "new.txt").read_text() == "written"


def test_code_api_failure_after_edit_preserves_tool_audit(case):
    _, root, _, _, _ = case
    class FailingModel:
        model = "offline-failure"
        calls = 0
        def chat(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=[ToolCall("write", "write_file", {"file_path": "partial.txt", "content": "keep"})])
            raise APIError("secret-body", request=None, body={"key": "secret-body"})
    session = session_for(case)
    session.coding_factory = FailingModel
    response = session.handle("帮我写一个文件")
    assert response["status"] == "incomplete" and (root / "partial.txt").read_text() == "keep"
    artifact = Path(response["artifact"]).read_text()
    assert "secret-body" not in artifact
    assert json.loads(artifact)["tool_events"][0]["tool"] == "write_file"
