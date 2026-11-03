"""End-to-end offline Agent Loop, report validation and DeepSeek configuration."""

import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from dotenv import dotenv_values
from openai import APIError

from corecoder.diagnose import (
    _write_outputs,
    build_deepseek_llm,
    configure_deepseek,
    main,
    render_markdown,
    run_diagnosis,
    validate_diagnosis,
)
from corecoder.diagnosis_demo import DemoLLM, create_demo_case, run_demo
from corecoder.llm import LLMResponse, ScriptedLLM, ToolCall


@pytest.fixture
def completed(tmp_path):
    return run_demo(directory=tmp_path / "demo")


def state_from(result):
    return SimpleNamespace(
        evidence=result["evidence"],
        observations={item["id"]: item for item in result["observations"]},
    )


def test_offline_demo_exercises_real_tools_and_cites_their_observations(completed):
    assert completed["status"] == "completed"
    assert completed["mode"] == "offline_scripted"
    assert completed["metrics"]["llm_calls"] == 5
    assert completed["metrics"]["tool_calls"] == 4
    assert completed["metrics"]["provider_usage_complete"] is False
    assert completed["metrics"]["prompt_tokens"] == completed["metrics"]["completion_tokens"] == 0
    assert [event["tool"] for event in completed["tool_events"]] == [
        "extract_log_evidence", "locate_source", "read_project_file", "inspect_paths",
    ]
    paths = [item["data"]["status"] for item in completed["observations"] if item["kind"] == "path_status"]
    assert paths == ["exists", "missing"]
    assert completed["evidence"][1]["cause_id"] == completed["evidence"][0]["id"]
    known = {item["id"] for item in completed["observations"]}
    assert set(completed["diagnosis"]["current_state"]["observation_ids"]) <= known
    assert completed["validation"]["semantic_correctness"] == "not_automatically_verified"


@pytest.mark.parametrize("field", ["historical", "current", "hypothesis"])
def test_unknown_report_references_are_rejected(completed, field):
    report = copy.deepcopy(completed["diagnosis"])
    if field == "historical":
        report["historical_failure"]["evidence_ids"] = ["ev_invented"]
    elif field == "current":
        report["current_state"]["observation_ids"] = ["obs_invented"]
    else:
        report["hypotheses"][0]["evidence_ids"] = ["ev_invented"]
    with pytest.raises(ValueError, match="unknown"):
        validate_diagnosis(json.dumps(report), state_from(completed))


@pytest.mark.parametrize("change", ["missing_key", "empty_action", "extra_key", "uncited_hypothesis", "missing_limitations"])
def test_report_contract_rejects_incomplete_or_unbounded_format(completed, change):
    report = copy.deepcopy(completed["diagnosis"])
    if change == "missing_key":
        del report["current_state"]
    elif change == "empty_action":
        report["recommended_actions"] = [""]
    elif change == "extra_key":
        report["accuracy"] = "100%"
    elif change == "uncited_hypothesis":
        report["hypotheses"][0]["evidence_ids"] = []
        report["hypotheses"][0]["observation_ids"] = []
    else:
        report["limitations"] = []
    with pytest.raises(ValueError):
        validate_diagnosis(json.dumps(report), state_from(completed))


def test_report_cannot_claim_completion_without_any_investigation(completed):
    report = json.dumps(completed["diagnosis"])
    with pytest.raises(ValueError, match="No log evidence"):
        validate_diagnosis(report, SimpleNamespace(evidence=[], observations={}))
    with pytest.raises(ValueError, match="No usable current"):
        validate_diagnosis(report, SimpleNamespace(evidence=completed["evidence"], observations={}))


def test_single_json_code_fence_is_accepted(completed):
    fence = chr(96) * 3
    report = json.dumps(completed["diagnosis"])
    assert validate_diagnosis(fence + "json\n" + report + "\n" + fence, state_from(completed)) == completed["diagnosis"]


def test_unstructured_model_reply_is_incomplete_not_a_valid_diagnosis(tmp_path):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")
    result = run_diagnosis(log, root, ScriptedLLM([LLMResponse(content="I think the dataset is missing.")]))
    assert result["status"] == "incomplete"
    assert result["diagnosis"] is None
    assert result["failure_reason"]
    assert result["metrics"]["llm_calls"] == 1


def test_general_shell_tool_is_unavailable(tmp_path):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")
    marker = root / "should-not-exist.txt"
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall(id="bad", name="bash", arguments={"command": f"touch {marker}"})]),
        LLMResponse(content="No evidence was obtained."),
    ])
    result = run_diagnosis(log, root, llm)
    assert result["status"] == "incomplete"
    assert result["metrics"]["tool_calls"] == 1
    assert not marker.exists()


def test_round_limit_and_tool_budget_preserve_partial_audit(tmp_path):
    first = run_demo(directory=tmp_path / "first", max_rounds=1)
    assert first["status"] == "incomplete"
    assert "round limit" in first["failure_reason"]
    assert first["evidence"]
    second = run_demo(directory=tmp_path / "second", max_tool_calls=1)
    assert second["status"] == "incomplete"
    assert second["metrics"]["tool_errors"] == 1
    assert second["tool_events"][-1]["status"] == "error"


def test_provider_error_does_not_persist_credential_echo(tmp_path):
    _, root, log, _, _ = create_demo_case(tmp_path / "case")

    class FailingLLM:
        model = "provider-error-test"

        def chat(self, **kwargs):
            raise APIError("Invalid secret sk-should-not-appear", request=SimpleNamespace(), body=None)

    result = run_diagnosis(log, root, FailingLLM())
    assert result["status"] == "incomplete"
    assert "APIError" in result["failure_reason"]
    assert "sk-should-not-appear" not in json.dumps(result)


@pytest.mark.parametrize("argument,value", [("max_rounds", 0), ("max_rounds", True), ("max_context_tokens", 10)])
def test_invalid_execution_limits_are_rejected(tmp_path, argument, value):
    _, root, log, current, previous = create_demo_case(tmp_path / "case")
    with pytest.raises(ValueError):
        run_diagnosis(log, root, DemoLLM(root, current, previous), **{argument: value})


def test_deepseek_adapter_disables_thinking_and_uses_dedicated_key(monkeypatch):
    monkeypatch.setattr("corecoder.diagnose.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-deepseek-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.delenv("CORECODER_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    llm = build_deepseek_llm()
    assert llm.model == "deepseek-flash"
    assert llm.client.api_key == "offline-deepseek-key"
    assert str(llm.client.base_url).rstrip("/") == "https://api.deepseek.com"
    assert llm.extra["extra_body"] == {"thinking": {"type": "disabled"}}
    assert llm.extra["response_format"] == {"type": "json_object"}
    assert llm.client.max_retries == 0


def test_missing_deepseek_key_does_not_fall_back_to_an_unrelated_openai_key(monkeypatch):
    monkeypatch.setattr("corecoder.diagnose.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("CORECODER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_deepseek_llm()


def test_configuration_preserves_other_fields_and_quotes_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("EXISTING=value\n", encoding="utf-8")
    configure_deepseek(env, "offline-key-with-'quote")
    values = dotenv_values(env)
    assert values["EXISTING"] == "value"
    assert values["DEEPSEEK_API_KEY"] == "offline-key-with-'quote"
    assert values["CORECODER_MODEL"] == "deepseek-flash"
    if os.name != "nt":
        assert env.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("key", ["", "line\nbreak", "null\x00key"])
def test_invalid_key_does_not_create_config(tmp_path, key):
    env = tmp_path / ".env"
    with pytest.raises(ValueError):
        configure_deepseek(env, key)
    assert not env.exists()


def test_interactive_configuration_does_not_call_provider_or_print_key(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("corecoder.diagnose.sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("corecoder.diagnose.getpass.getpass", lambda prompt: "offline-hidden-key")
    monkeypatch.setattr("corecoder.diagnose.build_deepseek_llm", lambda **kw: pytest.fail("Unexpected provider call"))
    assert main(["--configure-deepseek"]) == 0
    assert dotenv_values(tmp_path / ".env")["DEEPSEEK_API_KEY"] == "offline-hidden-key"
    assert "offline-hidden-key" not in capsys.readouterr().out


def test_configuration_refuses_noninteractive_key_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("corecoder.diagnose.sys.stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("corecoder.diagnose.getpass.getpass", lambda prompt: pytest.fail("Key entry must not start"))
    with pytest.raises(SystemExit) as exc:
        main(["--configure-deepseek"])
    assert exc.value.code == 2
    assert not (tmp_path / ".env").exists()


def test_markdown_includes_citation_index_and_mode(completed):
    text = render_markdown(completed)
    assert "offline_scripted" in text
    assert "引用索引" in text
    assert completed["evidence"][0]["id"] in text
    assert "结构与引用存在性校验不等于诊断结论正确" in text


def test_output_cannot_overwrite_log_or_observed_source(completed, tmp_path):
    for destination in [completed["log_file"], next(
        item["data"]["resolved_path"] for item in completed["observations"] if item["kind"] == "source_location"
    )]:
        args = SimpleNamespace(output=type(tmp_path)(destination), markdown=None)
        with pytest.raises(ValueError, match="overwrite"):
            _write_outputs(args, completed)
    args = SimpleNamespace(output=tmp_path / "same.json", markdown=tmp_path / "same.json")
    with pytest.raises(ValueError, match="different"):
        _write_outputs(args, completed)


def test_cli_offline_demo_writes_json_and_markdown_without_credentials(tmp_path):
    output = tmp_path / "diagnosis.json"
    markdown = tmp_path / "diagnosis.md"
    env = {key: value for key, value in os.environ.items() if key not in ("DEEPSEEK_API_KEY", "CORECODER_API_KEY")}
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.diagnose", "--demo", "--output", str(output), "--markdown", str(markdown)],
        env=env, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["mode"] == "offline_scripted"
    assert "offline_scripted" in markdown.read_text(encoding="utf-8")


def test_cli_usage_errors(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "corecoder.diagnose"], capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert result.stderr


class ReportVariantLLM(DemoLLM):
    """Exercise a real investigation, then return malformed or corrected reports."""

    def __init__(self, *args, variant="extra_field", repair="valid"):
        super().__init__(*args)
        self.variant = variant
        self.repair = repair
        self.good_content = None

    def chat(self, messages, tools=None, on_token=None):
        if self.good_content is not None:
            assert tools is None
            assert messages[-1]["role"] == "user"
            if self.repair == "invalid":
                return LLMResponse(content="Still invalid.")
            if self.repair == "tool":
                return self._tool(9, "inspect_paths", {"paths": [str(self.current)]})
            if self.repair == "api_error":
                raise APIError("Secret must not leak", request=SimpleNamespace(), body=None)
            return LLMResponse(content=self.good_content, prompt_tokens=7, completion_tokens=3)
        response = super().chat(messages, tools, on_token)
        if response.tool_calls:
            return response
        self.good_content = response.content
        report = json.loads(response.content)
        if self.variant == "preamble":
            response.content = "诊断已完成。以下是报告：\n" + response.content
        elif self.variant == "unknown_reference":
            report["historical_failure"]["evidence_ids"] = ["ev_invented"]
            response.content = json.dumps(report)
        else:
            report["hypotheses_note"] = None
            response.content = json.dumps(report)
        return response


@pytest.mark.parametrize("variant", ["preamble", "extra_field", "unknown_reference"])
def test_invalid_report_can_be_corrected_once_without_repeating_investigation(tmp_path, variant):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")
    llm = ReportVariantLLM(root, current, previous, variant=variant)
    result = run_diagnosis(log, root, llm, allowed_path_roots=[workspace])
    assert result["status"] == "completed"
    assert result["metrics"]["llm_calls"] == 6
    assert result["metrics"]["report_repairs"] == 1
    assert result["metrics"]["tool_calls"] == len(result["tool_events"]) == 4
    assert result["metrics"]["prompt_tokens"] == 7
    assert result["metrics"]["completion_tokens"] == 3
    assert len(result["report_attempts"]) == 2
    assert result["report_attempts"][0]["validation_error"]
    assert result["report_attempts"][1]["validation_error"] is None
    assert "ev_invented" not in result["diagnosis"]["historical_failure"]["evidence_ids"]


@pytest.mark.parametrize("repair", ["invalid", "tool", "api_error"])
def test_failed_report_correction_is_bounded_and_preserves_audit(tmp_path, repair):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")
    llm = ReportVariantLLM(root, current, previous, repair=repair)
    result = run_diagnosis(log, root, llm, allowed_path_roots=[workspace])
    assert result["status"] == "incomplete"
    assert result["metrics"]["llm_calls"] == 6
    assert result["metrics"]["report_repairs"] == 1
    assert len(result["tool_events"]) == 4
    assert result["evidence"] and result["observations"] and result["report_attempts"]
    if repair == "api_error":
        assert "Secret must not leak" not in json.dumps(result)
    if repair == "tool":
        assert "no repair tools were executed" in result["failure_reason"]


def test_report_correction_can_be_disabled(tmp_path):
    workspace, root, log, current, previous = create_demo_case(tmp_path / "case")
    result = run_diagnosis(log, root, ReportVariantLLM(root, current, previous),
                           allowed_path_roots=[workspace], max_report_repairs=0)
    assert result["status"] == "incomplete"
    assert result["metrics"]["llm_calls"] == 5
    assert result["metrics"]["report_repairs"] == 0
    assert len(result["report_attempts"]) == 1


@pytest.mark.parametrize("limit", [-1, 2, True])
def test_invalid_report_repair_budget_is_rejected(tmp_path, limit):
    _, root, log, current, previous = create_demo_case(tmp_path / "case")
    with pytest.raises(ValueError, match="max_report_repairs"):
        run_diagnosis(log, root, DemoLLM(root, current, previous), max_report_repairs=limit)
