"""Offline failed edit -> fresh read-only diagnosis -> explicitly requested repair."""

import argparse
import json
from pathlib import Path

from .llm import LLMResponse, ScriptedLLM, ToolCall
from .routed_session import RoutedSession
from .workflow_demo import DiagnosisDemoLLM, prepare_demo


class FeedbackDemoLLM(DiagnosisDemoLLM):
    def chat(self, messages, tools=None, on_token=None):
        response = super().chat(messages, tools, on_token)
        if not response.tool_calls:
            report = json.loads(response.content)
            report["summary"] = "上次增加常量的修改没有通过原断言，需要重新核查除法语义。"
            report["current_state"]["assessment"] = "当前实现仍使用整数除法，并增加了 1。"
            report["hypotheses"][0]["explanation"] = "增加常量不能普遍恢复浮点比例；应结合原断言核查整数除法。"
            response.content = json.dumps(report, ensure_ascii=False)
        return response


def run_demo(directory):
    workspace, root, report, command, initial_exit = prepare_demo(directory)
    models = iter([
        ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("wrong", "edit_file", {"file_path": "ratio.py", "old_string": "a // b", "new_string": "a // b + 1"})]),
            LLMResponse(content="尝试了增加常量的修改，交由原命令验证。"),
        ]),
        ScriptedLLM([
            LLMResponse(tool_calls=[ToolCall("fix", "edit_file", {"file_path": "ratio.py", "old_string": "a // b + 1", "new_string": "a / b"})]),
            LLMResponse(content="已移除常量补偿并使用浮点除法，等待原命令验证。"),
        ]),
    ])
    session = RoutedSession(root, report_file=report, verify_command=command, session_dir=workspace / "session",
                            coding_factory=lambda: next(models), diagnosis_factory=FeedbackDemoLLM,
                            execution_mode="offline_scripted", max_feedback_rounds=1)
    first = session.handle("修复比例计算")
    if first["status"] != "verification_failed" or not session.report_file:
        raise ValueError("Expected a failed first repair and a completed read-only feedback diagnosis")
    # This is a second explicit human task, not an automatic model-triggered edit.
    second = session.handle("修复新诊断发现的问题：移除常量补偿，按比例语义修复除法，保留原断言。")
    result = {"mode": "offline_scripted", "workspace": str(workspace), "initial_exit_code": initial_exit,
              "first_repair_status": first["status"], "feedback_status": session.feedback_records[0]["status"],
              "second_repair_status": second["status"], "feedback_attempts": session.feedback_attempts,
              "session": str(session.directory / "session.json"),
              "scope": "Scripted model decisions; actual failed edit, captured output, fresh source reads and original assertion rerun. "
                       "The second edit is a separate explicit task, not an autonomous repair retry."}
    (Path(workspace) / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_demo(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["second_repair_status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
