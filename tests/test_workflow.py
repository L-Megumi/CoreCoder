"""Preserved coding capabilities and diagnosis-to-repair handoff with actual verification."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import APIError

from corecoder.config import Config
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall
from corecoder.researchcoder import main
from corecoder.tools import ALL_TOOLS
from corecoder.workflow import RepairAgent, build_coding_llm, load_handoff, project_directory, run_repair
from corecoder.workflow_demo import prepare_demo, run_demo


@pytest.fixture(autouse=True)
def no_launcher_env(monkeypatch):
    monkeypatch.setattr("corecoder.researchcoder.load_dotenv", lambda *args, **kwargs: None)


@pytest.fixture
def case(tmp_path):
    return prepare_demo(tmp_path / "case")


def quiet_model(summary="No verification has been performed."):
    return ScriptedLLM([LLMResponse(content=summary)])


def test_offline_demo_executes_real_failure_edit_and_verification(tmp_path):
    result = run_demo(tmp_path / "workflow")
    assert result["initial_verification_exit_code"] != 0
    assert result["diagnosis_status"] == "completed"
    assert result["repair_status"] == "verified"
    assert result["verification_exit_code"] == 0
    root = Path(result["workspace"])
    assert "return a / b" in (root / "project/ratio.py").read_text()
    repair = json.loads((root / "repair.json").read_text())
    assert repair["mode"] == "offline_scripted"
    assert repair["metrics"]["provider_usage_complete"] is False
    assert repair["metrics"]["tool_calls"] == 2
    assert repair["file_changes"][0]["before_sha256"] != repair["file_changes"][0]["after_sha256"]


def test_unified_code_mode_retains_original_tool_set_and_can_write_edit_run(tmp_path, monkeypatch):
    root = tmp_path / "code-project"
    root.mkdir()
    monkeypatch.setenv("CORECODER_API_KEY", "offline-key")
    seen = []
    class CodingModel(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            seen.append({tool["function"]["name"] for tool in tools})
            return super().chat(messages, tools, on_token)
    model = CodingModel([
        LLMResponse(tool_calls=[ToolCall("write", "write_file", {"file_path": "value.py", "content": "answer = 1\n"})]),
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "value.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "value.py", "old_string": "answer = 1", "new_string": "answer = 2"})]),
        LLMResponse(tool_calls=[ToolCall("run", "bash", {"command": f'"{sys.executable}" -c "from value import answer; assert answer == 2"'})]),
        LLMResponse(content="Completed code change and executed verification."),
    ])
    monkeypatch.setattr("corecoder.researchcoder.build_coding_llm", lambda config: model)
    before = Path.cwd()
    assert main(["code", "--project", str(root), "-p", "Change answer and verify."]) == 0
    assert Path.cwd() == before
    assert (root / "value.py").read_text() == "answer = 2\n"
    assert seen and all(names == {tool.name for tool in ALL_TOOLS} for names in seen)


def test_diagnosis_dispatch_preserves_readonly_entry(monkeypatch):
    calls = []
    monkeypatch.setattr("corecoder.diagnose.main", lambda args: calls.append(args) or 0)
    assert main(["diagnose", "error.log", "--project", "/project"]) == 0
    assert calls == [["error.log", "--project", "/project"]]


@pytest.mark.parametrize("damage", ["incomplete", "unknown_reference", "duplicate_id"])
def test_invalid_diagnosis_cannot_be_used_for_repair(case, damage):
    _, project, report, _, _ = case
    value = json.loads(report.read_text())
    if damage == "incomplete":
        value["status"] = "incomplete"
    elif damage == "unknown_reference":
        value["diagnosis"]["current_state"]["observation_ids"] = ["obs_invented"]
    else:
        value["observations"].append(value["observations"][0])
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_handoff(report, project)


def test_wrong_project_rejected_without_editing(case, tmp_path):
    _, _, report, _, _ = case
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="different project"):
        load_handoff(report, other)


def test_changed_source_rejected_before_model_call(case):
    _, project, report, _, _ = case
    (project / "ratio.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="changed since diagnosis"):
        load_handoff(report, project)


def test_source_rechecked_when_handoff_is_used_later(case):
    _, project, report, _, _ = case
    handoff = load_handoff(report, project)
    (project / "ratio.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="changed since diagnosis"):
        run_repair(handoff, "Repair", ScriptedLLM([]))


def test_changed_report_rejected_after_preparation(case):
    _, project, report, _, _ = case
    handoff = load_handoff(report, project)
    value = json.loads(report.read_text())
    value["diagnosis"]["summary"] = "Different report"
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact changed"):
        run_repair(handoff, "Repair", ScriptedLLM([]))


def test_model_claim_of_success_is_unverified_without_command(case):
    _, project, report, _, _ = case
    result = run_repair(load_handoff(report, project), "Investigate the repair", quiet_model("Everything passed."))
    assert result["status"] == "completed_unverified"
    assert result["verification"]["status"] == "not_run"


def test_real_failing_command_overrides_model_success_claim(case):
    _, project, report, command, _ = case
    result = run_repair(load_handoff(report, project), "Repair", quiet_model("Fixed and all tests passed."), verify_command=command)
    assert result["status"] == "verification_failed"
    assert result["verification"]["exit_code"] != 0
    assert "AssertionError" in result["verification"]["stderr"]["text"]


def test_verification_timeout_is_reported(case):
    _, project, report, _, _ = case
    result = run_repair(load_handoff(report, project), "Inspect", quiet_model(), verification_timeout=1,
                        verify_command=[sys.executable, "-c", "import time; time.sleep(3)"])
    assert result["status"] == "verification_failed"
    assert result["verification"]["status"] == "timeout"


def test_repair_budget_cannot_be_bypassed_and_partial_audit_is_kept(case):
    _, project, report, command, _ = case
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "ratio.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "//", "new_string": "/"})]),
        LLMResponse(content="Finished"),
    ])
    result = run_repair(load_handoff(report, project), "Repair", model, max_tool_calls=1, verify_command=command)
    assert result["status"] == "incomplete"
    assert result["metrics"]["tool_calls"] == 2
    assert result["metrics"]["tool_errors"] == 1
    assert result["verification"]["status"] == "not_run"
    assert "//" in (project / "ratio.py").read_text()


def test_repair_file_tool_cannot_write_outside_selected_project(case):
    workspace, project, report, _, _ = case
    handoff = load_handoff(report, project)
    agent = RepairAgent(quiet_model(), handoff)
    with project_directory(project):
        response = agent._exec_tool(ToolCall("bad", "write_file", {"file_path": "../outside.py", "content": "bad"}))
    assert "outside" in response
    assert not (workspace / "outside.py").exists()
    assert agent.events[0]["status"] == "error"


def test_project_directory_restored_after_failure(tmp_path):
    previous = Path.cwd()
    with pytest.raises(RuntimeError), project_directory(tmp_path):
        raise RuntimeError("test")
    assert Path.cwd() == previous


def test_deepseek_coding_adapter_has_no_diagnosis_json_constraint(monkeypatch):
    config = Config(model="deepseek-flash", api_key="offline-key")
    llm = build_coding_llm(config)
    assert str(llm.client.base_url).rstrip("/") == "https://api.deepseek.com"
    assert "response_format" not in llm.extra
    assert llm.extra["extra_body"]["thinking"]["type"] == "disabled"
    assert llm.client.max_retries == 0


def test_existing_output_rejected_before_model_or_mutation(case, monkeypatch, tmp_path):
    _, project, report, _, _ = case
    output = tmp_path / "existing.json"
    output.write_text("keep")
    monkeypatch.setattr("corecoder.researchcoder.build_coding_llm", lambda config: pytest.fail("No model should be built"))
    with pytest.raises(SystemExit) as exc:
        main(["repair", str(report), "--project", str(project), "--instruction", "Repair", "--output", str(output)])
    assert exc.value.code == 2
    assert output.read_text() == "keep"


@pytest.mark.parametrize("kwargs", [{"max_rounds": 0}, {"max_tool_calls": True}, {"verify_command": []}, {"verification_timeout": 0}])
def test_invalid_repair_limits_are_rejected_before_execution(case, kwargs):
    _, project, report, _, _ = case
    with pytest.raises(ValueError):
        run_repair(load_handoff(report, project), "Repair", ScriptedLLM([]), **kwargs)


def test_interrupted_repair_preserves_completed_edits_and_restores_cwd(case):
    _, project, report, _, _ = case
    class InterruptedModel(ScriptedLLM):
        def chat(self, *args, **kwargs):
            if not self._turns:
                raise KeyboardInterrupt
            return super().chat(*args, **kwargs)
    model = InterruptedModel([LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {
        "file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b",
    })])])
    previous = Path.cwd()
    result = run_repair(load_handoff(report, project), "Repair", model)
    assert result["status"] == "interrupted"
    assert result["file_changes"]
    assert result["tool_events"][0]["status"] == "ok"
    assert Path.cwd() == previous


@pytest.mark.parametrize("explicit_model", [False, True])
def test_unified_code_deepseek_defaults_and_explicit_model_precedence(tmp_path, monkeypatch, explicit_model):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-key")
    captured = []
    monkeypatch.setattr("corecoder.researchcoder.build_coding_llm", lambda config: captured.append(config) or quiet_model())
    arguments = ["code", "--project", str(tmp_path), "-p", "Inspect project"]
    if explicit_model:
        arguments += ["--model", "custom-model"]
    assert main(arguments) == 0
    assert captured[0].model == ("custom-model" if explicit_model else "deepseek-flash")


def test_unified_repair_cli_writes_verification_audit(case, monkeypatch, tmp_path):
    _, project, report, command, _ = case
    output = tmp_path / "repair-result.json"
    monkeypatch.setenv("CORECODER_API_KEY", "offline-key")
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(content="The verification command will check the edit."),
    ])
    monkeypatch.setattr("corecoder.researchcoder.build_coding_llm", lambda config: model)
    assert main(["repair", str(report), "--project", str(project), "--instruction", "Repair integer division",
                 "--output", str(output), "--verify", *command]) == 0
    result = json.loads(output.read_text())
    assert result["status"] == "verified"
    assert result["verification"]["exit_code"] == 0
    assert result["handoff"]["report_path"] == str(report)


def test_repair_api_error_body_is_not_saved(case):
    _, project, report, _, _ = case
    class FailingModel:
        model = "offline-api-failure"
        def chat(self, **kwargs):
            raise APIError("Secret sk-must-not-be-saved", request=SimpleNamespace(), body=None)
    result = run_repair(load_handoff(report, project), "Inspect", FailingModel())
    assert result["status"] == "incomplete"
    assert "sk-must-not-be-saved" not in json.dumps(result)


def test_coding_cli_api_error_does_not_print_credential_echo(capsys):
    from corecoder.cli import _run_once
    class FailingAgent:
        def chat(self, *args, **kwargs):
            raise APIError("Secret sk-must-not-appear", request=SimpleNamespace(), body=None)
    with pytest.raises(SystemExit) as exc:
        _run_once(FailingAgent(), "Inspect")
    assert exc.value.code == 1
    assert "sk-must-not-appear" not in capsys.readouterr().out


def test_restricted_tools_agree_with_prompt_and_can_recover(case):
    _, project, report, command, _ = case
    names = {"read_file", "edit_file", "write_file", "grep", "glob"}
    class Model(ScriptedLLM):
        def chat(self, messages, tools=None, on_token=None):
            assert {item["function"]["name"] for item in tools} == names
            assert "**bash**" not in messages[0]["content"]
            assert "**grep**" in messages[0]["content"]
            return super().chat(messages, tools, on_token)
    model = Model([
        LLMResponse(tool_calls=[ToolCall("bad", "bash", {"command": "must never execute"})]),
        LLMResponse(tool_calls=[ToolCall("search", "grep", {"path": ".", "pattern": "//"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(content="The controller will verify the change."),
    ])
    result = run_repair(load_handoff(report, project), "Repair", model, verify_command=command, allowed_tool_names=names)
    assert result["status"] == "verified"
    assert result["tool_events"][0]["status"] == "error"
    assert result["tool_events"][1]["status"] == "ok"


def test_runtime_guard_blocks_before_mutation_and_model_can_finish(case):
    _, project, report, _, _ = case
    before = (project / "ratio.py").read_bytes()
    def guard(name, arguments):
        raise ValueError("protected fixture")
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(content="Blocked; no changes performed."),
    ])
    result = run_repair(load_handoff(report, project), "Repair", model, tool_guard=guard)
    assert result["status"] == "completed_unverified"
    assert "protected fixture" in result["tool_events"][0]["result"]
    assert (project / "ratio.py").read_bytes() == before


@pytest.mark.parametrize("names", [(), ("agent",), ("unknown",)])
def test_invalid_repair_tool_selection(case, names):
    _, project, report, _, _ = case
    with pytest.raises(ValueError, match="tool selection"):
        run_repair(load_handoff(report, project), "Repair", quiet_model(), allowed_tool_names=names)
