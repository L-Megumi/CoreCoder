"""Explicit, bounded DeepSeek acceptance on an independent two-fault metrics fixture."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError

from corecoder.diagnose import build_deepseek_llm
from corecoder.routed_cli import _coding_config
from corecoder.routed_session import RoutedSession
from corecoder.workflow import build_coding_llm


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Required: authorize actual provider calls")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    parser.add_argument("--max-requests", type=int, default=20, choices=range(1, 31))
    parser.add_argument("--direct", action="store_true", help="Ignore proxy environment only within this acceptance process")
    args = parser.parse_args(argv)
    if not args.live:
        parser.error("This acceptance makes actual API calls; pass --live or use corecoder.feedback_demo offline")
    load_dotenv(args.env_file, override=False)
    if args.direct:
        for key in list(os.environ):
            if key.lower().endswith("_proxy"):
                del os.environ[key]
    workspace = args.output_dir.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    root = workspace / "project"
    root.mkdir()
    source = root / "metrics.py"
    source.write_text("def mean_loss(total, count):\n    return total // count\n\ndef scale_loss(value, factor):\n    return value + factor\n", encoding="utf-8")
    check = root / "check_metrics.py"
    check.write_text("from metrics import mean_loss, scale_loss\nassert mean_loss(5, 2) == 2.5\nassert scale_loss(2.0, 3.0) == 6.0\n", encoding="utf-8")
    check_hash = hashlib.sha256(check.read_bytes()).hexdigest()
    command = [sys.executable, "check_metrics.py"]
    before = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
    log = workspace / "initial-failure.log"
    log.write_text(before.stdout + before.stderr, encoding="utf-8")
    assert before.returncode != 0
    started = time.monotonic()
    requests = []
    responses = []

    def model_factory(diagnosis):
        label = "diagnose" if diagnosis else "repair"
        model = build_deepseek_llm() if diagnosis else build_coding_llm(_coding_config())
        original_request, original_chat = model._call_with_retry, model.chat
        def bounded_request(params, max_retries=1):
            if len(requests) >= args.max_requests or time.monotonic() - started > 300:
                raise RuntimeError("Live acceptance request/time budget exhausted")
            requests.append({"number": len(requests) + 1, "phase": label})
            print(json.dumps({"request_attempt": len(requests), "phase": label}), flush=True)
            return original_request(params, max_retries=1)
        def measured_chat(*call_args, **kwargs):
            response = original_chat(*call_args, **kwargs)
            responses.append({"phase": label, "prompt_tokens": response.prompt_tokens,
                              "completion_tokens": response.completion_tokens})
            return response
        model._call_with_retry, model.chat = bounded_request, measured_chat
        return model

    session = RoutedSession(root, log_file=log, session_dir=workspace / "session", source_hints=["metrics.py"],
                            verify_command=command, coding_factory=lambda: model_factory(False),
                            diagnosis_factory=lambda: model_factory(True), max_feedback_rounds=1, execution_mode="live")
    stages = []
    error = None
    try:
        diagnosis = session.handle("只读诊断当前断言失败，读取 metrics.py，核查当前失败的均值计算及验证方法。", mode="diagnose")
        stages.append(diagnosis)
        if diagnosis["status"] == "completed":
            first = session.handle(
                "本轮仅修复 metrics.py 中 mean_loss 的浮点均值问题。scale_loss 留待单独任务，保持该函数和 check_metrics.py 不变。"
                "先读源码，仅用 read_file/edit_file 修改目标函数；不执行命令，原验证命令由控制器运行。", mode="repair")
            stages.append(first)
            if first["status"] == "verification_failed" and session.report_file:
                stages.append(session.handle(
                    "修复新反馈诊断发现的 scale_loss 问题：其语义是 value 乘 factor。保留已修复的 mean_loss 和全部原断言。"
                    "只读写 metrics.py，仅用 read_file/edit_file，不执行命令；验证由控制器运行。", mode="repair"))
    except (APIError, ValueError, TypeError, OSError, RuntimeError) as exc:
        error = type(exc).__name__  # Never expose provider error bodies or credentials.
    unchanged = hashlib.sha256(check.read_bytes()).hexdigest() == check_hash
    final_check = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10, check=False)
    (workspace / "final-check.log").write_text(final_check.stdout + final_check.stderr, encoding="utf-8")
    statuses = [stage["status"] for stage in stages]
    feedback = session.feedback_records[0]["status"] if session.feedback_records else None
    completed = (statuses == ["completed", "verification_failed", "verified"] and feedback == "completed"
                 and unchanged and final_check.returncode == 0)
    summary = {"status": "passed" if completed else "incomplete", "mode": "live", "error_type": error,
               "stage_statuses": statuses, "feedback_status": feedback, "initial_exit_code": before.returncode,
               "final_exit_code": final_check.returncode, "original_check_unchanged": unchanged,
               "request_attempts": len(requests), "max_requests": args.max_requests, "responses": responses,
               "prompt_tokens": sum(r["prompt_tokens"] for r in responses),
               "completion_tokens": sum(r["completion_tokens"] for r in responses),
               "usage_complete": len(requests) == len(responses) and bool(responses)
                                 and all(r["prompt_tokens"] or r["completion_tokens"] for r in responses),
               "elapsed_seconds": round(time.monotonic() - started, 3), "session": str(session.directory / "session.json"),
               "scope": "One controlled two-fault fixture with actual model calls. First task intentionally addresses only mean_loss, "
                        "leaving scale_loss for feedback and a second explicit task. Not a real-world dataset or repair-success benchmark."}
    (workspace / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
