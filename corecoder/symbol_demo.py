"""Offline frameless-NaN diagnosis through a real imported function and verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .diagnose import run_diagnosis
from .llm import LLMResponse, ScriptedLLM, ToolCall
from .workflow import load_handoff, run_repair


class SymbolDemoLLM:
    model = "scripted-symbol-diagnosis"

    def __init__(self):
        self.step = 0
        self.evidence_id = None

    def chat(self, messages, tools=None, on_token=None):
        latest = next((item for item in reversed(messages) if item["role"] == "tool"), None)
        data = json.loads(latest["content"]) if latest else {}
        step, self.step = self.step, self.step + 1
        if step == 0:
            return LLMResponse(tool_calls=[ToolCall("extract", "extract_log_evidence", {})])
        if step == 1:
            self.evidence_id = data["evidence"][0]["id"]
            return LLMResponse(tool_calls=[ToolCall("search", "search_project_source", {
                "query": "loss", "evidence_ids": [self.evidence_id],
            })])
        if step == 2:
            return LLMResponse(tool_calls=[ToolCall("caller", "read_project_file", {"path": "train.py"})])
        if step == 3:
            source_id = data["observations"][0]["id"]
            return LLMResponse(tool_calls=[ToolCall("definition", "resolve_project_symbol", {
                "symbol": "calc", "source_observation_id": source_id, "evidence_ids": [self.evidence_id],
            })])
        identifier = next(item["id"] for item in data["observations"] if item["data"].get("definition_kind") == "function")
        return LLMResponse(content=json.dumps({
            "summary": "无堆栈日志记录 NaN；沿 calc 的导入定位到 metrics.safe_ratio。",
            "historical_failure": {"description": "日志记录 loss=nan。", "evidence_ids": [self.evidence_id]},
            "current_state": {"assessment": "实际函数默认 eps=0.0，分母为零时返回 NaN。", "observation_ids": [identifier]},
            "hypotheses": [{"explanation": "稳定项为零可能导致退化输入产生 NaN。", "evidence_ids": [self.evidence_id],
                            "observation_ids": [identifier], "verification": "恢复正稳定项后执行原有限性与正常输入断言。"}],
            "recommended_actions": ["核对函数语义后修复稳定项，保留原断言。"],
            "limitations": ["脚本模型仅演示静态关联与控制流程，不证明通用根因准确率。"],
        }, ensure_ascii=False))


def run_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    project = directory / "project"
    project.mkdir()
    (project / "metrics.py").write_text("def safe_ratio(a,b,eps=0.0):\n    d=b+eps\n    return a/d if d else float('nan')\n")
    (project / "train.py").write_text("from metrics import safe_ratio as calc\nprint('loss='+str(calc(0,0)))\n")
    (project / "check.py").write_text("import math\nfrom metrics import safe_ratio\nassert math.isfinite(safe_ratio(0,0))\n"
                                    "assert safe_ratio(0,0)==0\nassert math.isclose(safe_ratio(1,2),0.5,rel_tol=1e-6)\n")
    command = [sys.executable, "-B", "check.py"]
    failed = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=10, check=False)
    logged = subprocess.run([sys.executable, "-B", "train.py"], cwd=project, capture_output=True, text=True, timeout=10, check=False)
    assert failed.returncode != 0 and logged.stdout.strip() == "loss=nan"
    log = directory / "failure.log"
    log.write_text(logged.stdout)
    diagnosis = run_diagnosis(log, project, SymbolDemoLLM(), tool_selection="guided", context_mode="evidence",
                              completion_mode="checklist", mode="offline_scripted")
    report = directory / "diagnosis.json"
    report.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")
    model = ScriptedLLM([
        LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "metrics.py", "old_string": "eps=0.0", "new_string": "eps=1e-7"})]),
        LLMResponse(content="Ready for the original verification command"),
    ])
    repair = run_repair(load_handoff(report, project), "Fix the numerical fault; preserve the original assertions.", model,
                        verify_command=command, mode="offline_scripted")
    (directory / "repair.json").write_text(json.dumps(repair, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "offline_scripted", "initial_exit_code": failed.returncode, "diagnosis_status": diagnosis["status"],
               "diagnosis_tools": [event["tool"] for event in diagnosis["tool_events"]], "repair_status": repair["status"],
               "verification_exit_code": repair["verification"].get("exit_code"), "provider_requests": 0,
               "scope": "Scripted model decisions; real NaN log, static import lookup, file edit and original checks"}
    assert repair["status"] == "verified"
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(run_demo(parser.parse_args(argv).output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
