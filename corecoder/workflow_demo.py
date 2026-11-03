"""Offline real-file diagnosis -> edit -> verification demo; model decisions are scripted."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .diagnose import run_diagnosis
from .llm import LLMResponse, ScriptedLLM, ToolCall
from .workflow import load_handoff, run_repair


class DiagnosisDemoLLM:
    model = "scripted-workflow-diagnosis"

    def __init__(self):
        self.step = 0
        self.evidence = []

    def chat(self, messages, tools=None, on_token=None):
        latest = next((item for item in reversed(messages) if item["role"] == "tool"), None)
        data = json.loads(latest["content"]) if latest else {}
        step = self.step
        self.step += 1
        if step == 0:
            return LLMResponse(tool_calls=[ToolCall("extract", "extract_log_evidence", {})])
        if step == 1:
            self.evidence = data["evidence"]
            return LLMResponse(tool_calls=[ToolCall("locate", "locate_source", {"evidence_ids": [item["id"] for item in self.evidence]})])
        if step == 2:
            return LLMResponse(tool_calls=[ToolCall("read", "read_project_file", {"path": "ratio.py"})])
        evidence = self.evidence[0]["id"]
        observation = data["observations"][0]["id"]
        return LLMResponse(content=json.dumps({
            "summary": "整数除法可能导致比例计算不符合预期。",
            "historical_failure": {"description": "断言要求 ratio(5,2) 等于 2.5，但实际不符。", "evidence_ids": [evidence]},
            "current_state": {"assessment": "当前实现使用 // 整数除法。", "observation_ids": [observation]},
            "hypotheses": [{"explanation": "比例计算需要浮点除法。", "evidence_ids": [evidence],
                            "observation_ids": [observation], "verification": "修改后重新执行原断言脚本。"}],
            "recommended_actions": ["核对语义后将 // 改为 /，执行 check_ratio.py。"],
            "limitations": ["脚本模型仅演示流程；尚未修改源码或验证修复。"],
        }, ensure_ascii=False))


def prepare_demo(directory=None):
    workspace = Path(tempfile.mkdtemp(prefix="researchcoder-workflow-")) if directory is None else Path(directory).resolve()
    if directory is not None:
        workspace.mkdir(parents=True, exist_ok=False)
    project = workspace / "project"
    project.mkdir()
    (project / "ratio.py").write_text("def ratio(a, b):\n    return a // b\n", encoding="utf-8")
    (project / "check_ratio.py").write_text("from ratio import ratio\nassert ratio(5, 2) == 2.5\n", encoding="utf-8")
    command = [sys.executable, "check_ratio.py"]
    failed = subprocess.run(command, check=False, cwd=project, capture_output=True, text=True, timeout=10)
    if failed.returncode == 0:
        raise ValueError("Demo fixture unexpectedly passed before repair")
    log = workspace / "failure.log"
    log.write_text(failed.stdout + failed.stderr, encoding="utf-8")
    diagnosis = run_diagnosis(log, project, DiagnosisDemoLLM(), completion_mode="checklist", source_hints=["ratio.py"],
                              mode="offline_scripted")
    report = workspace / "diagnosis.json"
    report.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    return workspace, project, report, command, failed.returncode


def run_demo(directory=None):
    workspace, project, report, command, initial_exit = prepare_demo(directory)
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "ratio.py"})]),
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
        LLMResponse(content="已修改比例计算；验证由工作流执行原检查脚本。"),
    ], model="scripted-workflow-repair")
    repair = run_repair(load_handoff(report, project), "修复比例计算，保留原检查脚本。", model,
                        verify_command=command, mode="offline_scripted")
    (workspace / "repair.json").write_text(json.dumps(repair, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"mode": "offline_scripted", "workspace": str(workspace), "initial_verification_exit_code": initial_exit,
              "diagnosis_status": "completed", "repair_status": repair["status"],
              "verification_exit_code": repair["verification"].get("exit_code"),
              "scope": "Scripted model choices; actual traceback, actual file edit, actual verification command"}
    (workspace / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="researchcoder demo")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = run_demo(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["repair_status"] == "verified" else 2
