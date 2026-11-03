"""One-shot routing and an interactive conversation with explicit mode overrides."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .diagnose import build_deepseek_llm
from .routed_session import RoutedSession
from .task_router import select_route
from .workflow import build_coding_llm

HELP = """普通任务自动路由；/code 任务、/diagnose 任务、/repair 任务、/review 问题指定本轮模式。
/log 路径：绑定日志并清除旧修复上下文；/report 路径：绑定现有有效报告。
/verify [\"python\",\"-m\",\"pytest\",\"-q\"]：设置修复后的验证命令；/verify null 清除。
/status 查看状态；/help 帮助；/quit 退出。相对路径以项目目录为基准。
验证失败会在预算内自动只读重查；诊断不会自动转入修复；通用编程保留原工具。"""


def _coding_config(model=None, base_url=None):
    config = Config.from_env()
    if model:
        config.model = model
    elif not os.getenv("CORECODER_MODEL") and os.getenv("DEEPSEEK_API_KEY") and not (
        os.getenv("CORECODER_API_KEY") or os.getenv("OPENAI_API_KEY")
    ):
        config.model = "deepseek-flash"
    if base_url:
        config.base_url = base_url
    return config


def submit(session, text, *, mode="auto"):
    text = text.strip()
    if text == "/help":
        print(HELP)
        return None
    if text == "/status":
        snapshot = session.snapshot()
        print(json.dumps({key: snapshot[key] for key in ("project_root", "phase", "log_file", "active_report", "verify_command")}, ensure_ascii=False, indent=2))
        return None
    if text.startswith("/"):
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        if command in ("/log", "/report"):
            if not argument:
                raise ValueError("Provide a literal path after /log or /report")
            if len(argument) >= 2 and argument[0] == argument[-1] and argument[0] in ("'", '"'):
                argument = argument[1:-1]
            (session.bind_log if command == "/log" else session.bind_report)(argument)
            print("已更新绑定；使用 /status 查看当前上下文。")
            return None
        if command == "/verify":
            session.set_verification(json.loads(argument))
            print("已更新修复验证命令。")
            return None
        if command not in ("/code", "/diagnose", "/repair", "/review"):
            raise ValueError("Unknown session command; use /help")
        mode, text = command[1:], argument
    result = session.handle(text, mode=mode, on_route=lambda route: print(f"[{route.mode}] {route.reason}", flush=True))
    print(result["text"])
    if result["artifact"]:
        print("执行记录：" + result["artifact"])
    return result


def main(argv=None, *, interactive=False):
    parser = argparse.ArgumentParser(prog="researchcoder chat" if interactive else "researchcoder run", description=__doc__, allow_abbrev=False)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--log", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--session-dir", type=Path)
    parser.add_argument("--mode", choices=("auto", "code", "diagnose", "repair", "review"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Show routing only; no session files or API calls")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--map-root", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--allow-path-root", type=Path, action="append", default=[])
    parser.add_argument("--source-hint", action="append", default=[])
    parser.add_argument("--max-investigation-rounds", type=int, default=8, choices=range(1, 31))
    parser.add_argument("--max-feedback-rounds", type=int, default=2, choices=range(6),
                        help="Bound follow-up read-only investigations after failed repair verification (default 2; 0 captures only)")
    parser.add_argument("--verify", nargs=argparse.REMAINDER, help="Verification command; place last; runs only after repair")
    args = parser.parse_args(argv)
    if not args.project.resolve().is_dir():
        parser.error("--project must be an existing directory")
    if args.log is not None and args.report is not None:
        parser.error("Bind either --log or --report")
    if not args.task and (not interactive or args.dry_run):
        parser.error("Provide a task")
    if args.dry_run:
        task, selected_mode = args.task.strip(), args.mode
        if task.startswith("/"):
            prefix, _, task = task.partition(" ")
            if prefix not in ("/code", "/diagnose", "/repair", "/review"):
                parser.error("--dry-run previews tasks, not binding commands")
            selected_mode = prefix[1:]
        print(json.dumps(asdict(select_route(task, mode=selected_mode, has_log=args.log is not None,
                                              has_report=args.report is not None)), ensure_ascii=False, indent=2))
        return 0
    mappings = {}
    for item in args.map_root:
        old, separator, new = item.partition("=")
        if not separator or not old or not new:
            parser.error("--map-root requires OLD=NEW")
        mappings[old] = new
    load_dotenv(Path.cwd() / ".env", override=False)
    # Factories remain lazy: missing inputs and binding commands never call a model.
    session = RoutedSession(
        args.project, session_dir=args.session_dir, log_file=args.log, report_file=args.report, verify_command=args.verify,
        coding_factory=lambda: build_coding_llm(_coding_config(args.model, args.base_url)),
        diagnosis_factory=lambda: build_deepseek_llm(model=args.model, base_url=args.base_url),
        path_mappings=mappings, allowed_path_roots=args.allow_path_root, source_hints=args.source_hint,
        max_feedback_rounds=args.max_feedback_rounds,
        max_investigation_rounds=args.max_investigation_rounds,
    )
    print("会话记录：" + str(session.directory / "session.json"))
    if args.task:
        result = submit(session, args.task, mode=args.mode)
        if not interactive:
            if result and result["status"] == "interrupted":
                return 130
            return 0 if result and result["status"] in ("completed", "verified", "completed_unverified") else 2
    print(HELP)
    while True:
        try:
            text = input("ResearchCoder> ").strip()
            if text in ("/quit", "quit", "exit"):
                return 0
            if text:
                submit(session, text, mode=args.mode)
        except EOFError:
            return 0
        except KeyboardInterrupt:
            print("\n输入已取消；/quit 退出。")
        except (ValueError, TypeError, OSError, RuntimeError) as exc:
            print("ResearchCoder: " + str(exc))
