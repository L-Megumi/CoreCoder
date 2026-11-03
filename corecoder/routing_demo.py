"""Offline multi-turn coding -> diagnosis -> review -> repair -> coding demonstration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .llm import LLMResponse, ScriptedLLM, ToolCall
from .routed_session import RoutedSession
from .workflow_demo import DiagnosisDemoLLM


def run_demo(directory):
    workspace = Path(directory).resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    root = workspace / "project"
    root.mkdir()
    (root / "ratio.py").write_text("def ratio(a, b):\n    return a // b\n", encoding="utf-8")
    (root / "check_ratio.py").write_text("from ratio import ratio\nassert ratio(5, 2) == 2.5\n", encoding="utf-8")
    command = [sys.executable, "check_ratio.py"]
    before = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
    if before.returncode == 0:
        raise ValueError("Demo must fail before repair")
    log = workspace / "failure.log"
    log.write_text(before.stdout + before.stderr, encoding="utf-8")
    models = iter([
        ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("write", "write_file", {"file_path": "README.md", "content": "# Ratio example\n"})]),
            LLMResponse(content="已编写项目说明。"),
        ]),
        ScriptedLLM([LLMResponse(content="报告中 // 是整数除法；当前尚未修改，建议核对比例语义后复验。")]),
        ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b"})]),
            LLMResponse(content="已修改比例实现，等待执行原断言脚本。"),
        ]),
        ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("read", "read_file", {"file_path": "ratio.py"})]),
            LLMResponse(content="已重新读取修改后的比例实现。"),
        ]),
    ])
    session = RoutedSession(root, session_dir=workspace / "session", log_file=log, verify_command=command,
                            coding_factory=lambda: next(models), diagnosis_factory=DiagnosisDemoLLM,
                            execution_mode="offline_scripted")
    responses = [session.handle(task) for task in (
        "帮我写一个项目说明", "排查这次运行报错", "解释报告，为什么会失败？", "修复比例计算", "解释修改后的代码",
    )]
    # The bound historical log remains available; the last task asks to read code.
    if responses[-1]["mode"] != "code":
        raise ValueError("Final coding task was misrouted")
    result = {"mode": "offline_scripted", "workspace": str(workspace), "initial_exit_code": before.returncode,
              "routes": [item["mode"] for item in responses], "statuses": [item["status"] for item in responses],
              "session": str(session.directory / "session.json"),
              "scope": "Scripted model choices; real log, file tools, report preflight and original verification command"}
    (workspace / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_demo(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["statuses"] == ["completed", "completed", "completed", "verified", "completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
