"""A labeled scripted demo that exercises the same real diagnostic tools."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .diagnose import run_diagnosis
from .llm import LLMResponse, ToolCall


def create_demo_case(directory=None):
    workspace = Path(tempfile.mkdtemp(prefix="researchcoder-diagnosis-")) if directory is None else Path(directory).resolve()
    if directory is not None:
        workspace.mkdir(parents=True, exist_ok=False)
    root = workspace / "project"
    root.mkdir()
    current = root / "dataset/images/val"
    current.mkdir(parents=True)
    previous = workspace / "previous/dataset/images/val"
    source = root / "train.py"
    source.write_text(
        'def check_data():\n    raise FileNotFoundError("missing dataset")\n\ndef train():\n    check_data()\n',
        encoding="utf-8",
    )
    config = root / "data.yaml"
    config.write_text(f"path: {root / 'dataset'}\nval: images/val\n", encoding="utf-8")
    message = f"Dataset '{config}' images not found, missing path '{previous}'"
    text = (
        f'Traceback (most recent call last):\n  File "{source}", line 2, in check_data\n'
        '    raise FileNotFoundError("missing dataset")\nFileNotFoundError:\n'
        + message
        + "\n\nThe above exception was the direct cause of the following exception:\n\n"
        + f'Traceback (most recent call last):\n  File "{source}", line 5, in train\n'
        + '    check_data()\nRuntimeError: training failed\n' + message + "\n"
    )
    log = workspace / "historical-failure.log"
    log.write_text(text, encoding="utf-8")
    return workspace, root, log, current, previous


class DemoLLM:
    """Fixed tool sequence; final references and path states come from actual tool replies."""

    model = "scripted-diagnosis-demo"

    def __init__(self, root, current, previous):
        self.root = root
        self.current = current
        self.previous = previous
        self.step = 0
        self.evidence = []
        self.config_observation = None

    def chat(self, messages, tools=None, on_token=None):
        step = self.step
        self.step += 1
        latest = next((message for message in reversed(messages) if message["role"] == "tool"), None)
        data = json.loads(latest["content"]) if latest else {}
        if "error" in data:
            return LLMResponse(content="工具请求未完成，无法形成离线演示报告。")
        if step == 0:
            return self._tool(step, "extract_log_evidence", {})
        if step == 1:
            self.evidence = data["evidence"]
            return self._tool(step, "locate_source", {"evidence_ids": [item["id"] for item in self.evidence]})
        if step == 2:
            return self._tool(step, "read_project_file", {"path": str(self.root / "data.yaml")})
        if step == 3:
            self.config_observation = data["observations"][0]["id"]
            return self._tool(step, "inspect_paths", {"paths": [str(self.current), str(self.previous)]})
        observations = data["observations"]
        current, previous = observations
        root_id = self.evidence[0]["id"]
        references = [self.config_observation, current["id"], previous["id"]]
        report = {
            "summary": "离线脚本演示：历史验证集路径不可用，当前配置与历史路径不同。",
            "historical_failure": {
                "description": "历史日志记录 FileNotFoundError，随后包装为 RuntimeError。",
                "evidence_ids": [item["id"] for item in self.evidence],
            },
            "current_state": {
                "assessment": f"当前验证集路径检查为 {current['data']['status']}，"
                              f"历史路径检查为 {previous['data']['status']}；没有运行训练。",
                "observation_ids": references,
            },
            "hypotheses": [{
                "explanation": "历史数据路径可能不适用于当时的机器或挂载状态；当前配置已经使用另一路径。",
                "evidence_ids": [root_id], "observation_ids": references,
                "verification": "核对报错时的配置与数据挂载记录，再检查实际图像和标注，必要时人工重跑数据检查。",
            }],
            "recommended_actions": ["确认正式训练所使用的配置文件。", "验证图像和标注完整性后，由用户运行训练验证。"],
            "limitations": ["这是脚本演示，不是 DeepSeek 诊断效果测试。", "当前目录存在不代表数据完整，也不能还原历史挂载状态。"],
        }
        content = json.dumps(report, ensure_ascii=False)
        if on_token:
            on_token(content)
        return LLMResponse(content=content)

    @staticmethod
    def _tool(step, name, arguments):
        return LLMResponse(tool_calls=[ToolCall(id=f"demo_{step}", name=name, arguments=arguments)])


def run_demo(*, max_rounds=8, max_tool_calls=16, max_report_repairs=1, directory=None,
             tool_selection="all", context_mode="baseline", max_context_tokens=128_000, completion_mode="off"):
    workspace, root, log, current, previous = create_demo_case(directory)
    result = run_diagnosis(
        log, root, DemoLLM(root, current, previous), allowed_path_roots=[workspace],
        max_rounds=max_rounds, max_tool_calls=max_tool_calls, mode="offline_scripted",
        max_report_repairs=max_report_repairs,
        tool_selection=tool_selection, context_mode=context_mode, max_context_tokens=max_context_tokens,
        completion_mode=completion_mode,
    )
    result["demo_workspace"] = str(workspace)
    result["demo_notice"] = "Scripted tool-flow demonstration; no model API was called."
    return result
