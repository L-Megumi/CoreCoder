"""Offline frameless-log investigation; scripted choices, real read-only tools."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .diagnose import run_diagnosis
from .llm import LLMResponse, ToolCall


class SearchDemoLLM:
    model = "scripted-source-search"

    def __init__(self):
        self.step = 0

    def chat(self, messages, tools=None, **kwargs):
        step = self.step
        self.step += 1
        if step == 0:
            return LLMResponse(tool_calls=[ToolCall("extract", "extract_log_evidence", {})])
        data = json.loads(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))
        if step == 1:
            self.evidence_id = data["evidence"][0]["id"]
            return LLMResponse(tool_calls=[ToolCall("search", "search_project_source", {
                "query": "loss", "evidence_ids": [self.evidence_id],
            })])
        if step == 2:
            candidate = next(item["data"] for item in data["observations"] if item["kind"] == "source_candidate")
            return LLMResponse(tool_calls=[ToolCall("read", "read_project_file", {
                "path": candidate["relative_path"], "start_line": max(1, candidate["hit_line"] - 3), "max_lines": 20,
            })])
        observation = data["observations"][0]["id"]
        return LLMResponse(content=json.dumps({
            "summary": "日志出现非有限 loss，已搜索并读取相关候选源码。",
            "historical_failure": {"description": "运行输出 loss=nan，没有 traceback。", "evidence_ids": [self.evidence_id]},
            "current_state": {"assessment": "已读取候选实现，其中显式返回 float('nan')。", "observation_ids": [observation]},
            "hypotheses": [{"explanation": "该示例中的非有限返回值可解释日志；搜索命中本身不能确认实际项目根因。",
                            "evidence_ids": [self.evidence_id], "observation_ids": [observation],
                            "verification": "核对实际调用路径和损失输入，再运行针对性检查。"}],
            "recommended_actions": ["核查 loss 计算和输入数值，不自动修改或启动训练。"],
            "limitations": ["固定脚本模型只验证调查流程，不代表真实模型诊断效果。"],
        }, ensure_ascii=False))


def run_demo(directory, *, tool_selection="guided", context_mode="evidence"):
    workspace = Path(directory).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    project = workspace / "project"
    project.mkdir()
    (project / "loss.py").write_text("def compute_loss():\n    return float('nan')\n", encoding="utf-8")
    command = [sys.executable, "-c", "from loss import compute_loss; print(f'step=1 loss={compute_loss()}')"]
    execution = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=10, check=True)
    log = workspace / "failure.log"
    log.write_text(execution.stdout, encoding="utf-8")
    result = run_diagnosis(log, project, SearchDemoLLM(), tool_selection=tool_selection, context_mode=context_mode,
                           completion_mode="checklist", mode="offline_scripted")
    (workspace / "diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_demo(args.output_dir)
    print(json.dumps({"status": result["status"], "mode": result["mode"],
                      "tools": [event["tool"] for event in result["tool_events"]]}, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
