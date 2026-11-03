"""Offline repair-efficiency acceptance with real files and scripted model choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import LLMResponse, ScriptedLLM, ToolCall
from .repair_efficiency import plan_case_requests
from .workflow import load_handoff, run_repair
from .workflow_demo import prepare_demo


def run_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in ("handoff-edit", "repeated-reads", "partial-patch"):
        workspace, project, report, command, _ = prepare_demo(directory / name)
        edit = LLMResponse(tool_calls=[ToolCall("edit", "edit_file", {
            "file_path": "ratio.py", "old_string": "a // b", "new_string": "a / b",
        })])
        if name == "handoff-edit":
            class HandoffModel(ScriptedLLM):
                def chat(self, messages, tools=None, on_token=None):
                    if len(self._turns) == 2:
                        memory = json.loads(messages[-1]["content"].split("\n", 1)[1])
                        if not any("a // b" in item["content"] for item in memory["source_excerpts_revalidated_now"]):
                            raise ValueError("Actual observed source was not preserved")
                    return super().chat(messages, tools, on_token)
            model = HandoffModel([edit, LLMResponse(content="Ready for controller verification")])
        elif name == "repeated-reads":
            model = ScriptedLLM([LLMResponse(tool_calls=[ToolCall(str(i), "read_file", {"file_path": "ratio.py"})])
                                 for i in range(10)])
        else:
            class InterruptedProvider(ScriptedLLM):
                def chat(self, *args, **kwargs):
                    if not self._turns:
                        raise RuntimeError("Scripted request budget exhausted after patch")
                    return super().chat(*args, **kwargs)
            model = InterruptedProvider([edit])
        result = run_repair(load_handoff(report, project), "Fix ratio; preserve the original assertion script.", model,
                            verify_command=command, mode="offline_scripted", max_investigation_rounds=3)
        (workspace / "repair.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        results[name] = {"status": result["status"], "verification": result["verification"]["status"],
                         "workflow_completed": result["workflow_completed"], "metrics": result["metrics"]}
    counts = {"m01": 6, "m02": 8, "m03": 6, "m04": 7, "m05": 14, "m06": 5}
    summary = {"mode": "offline_scripted", "provider_request_attempts": 0, "cases": results,
               "historical_budget_replay": {"prior_attempts": sum(counts.values()), "total_limit": 72,
                                           "reserved_new_requests": plan_case_requests(
                                               [cid for cid in counts if cid != "m03"], counts, 72, 20),
                                           "scope": "Allocation arithmetic only; historical cases were not rerun"},
               "scope": "Scripted decisions, actual source/version checks, file edits, stop conditions and verification. "
                        "No live model accuracy, token savings or speed improvement is measured."}
    assert results["handoff-edit"]["status"] == "verified"
    assert results["repeated-reads"]["status"] == "incomplete"
    assert results["repeated-reads"]["metrics"]["read_cache_hits"] == 2
    assert results["partial-patch"]["status"] == "incomplete" and results["partial-patch"]["verification"] == "passed"
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_demo(args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
