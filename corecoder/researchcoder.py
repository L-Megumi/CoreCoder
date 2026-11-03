"""ResearchCoder: coding, diagnosis, explicit repair, and routed conversations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .workflow import build_coding_llm, load_handoff, project_directory, run_repair


def _code(argv):
    from .cli import main as coding_main

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    args, remaining = parser.parse_known_args(argv)
    # Read launcher configuration before changing cwd; credentials stay in environment.
    load_dotenv(Path.cwd() / ".env", override=False)
    explicit_model = any(value in ("-m", "--model") or value.startswith("--model=") for value in remaining)
    if (not explicit_model and not os.getenv("CORECODER_MODEL") and os.getenv("DEEPSEEK_API_KEY")
            and not (os.getenv("CORECODER_API_KEY") or os.getenv("OPENAI_API_KEY"))):
        remaining = ["--model", "deepseek-flash", *remaining]
    with project_directory(args.project):
        result = coding_main(remaining, llm_factory=build_coding_llm)
    return result or 0


def _repair(argv):
    parser = argparse.ArgumentParser(prog="researchcoder repair", description="Explicitly modify code using a completed diagnosis")
    parser.add_argument("report", type=Path)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-tool-calls", type=int, default=24)
    parser.add_argument("--max-investigation-rounds", type=int, default=8, choices=range(1, 31))
    parser.add_argument("--verification-timeout", type=int, default=30)
    parser.add_argument("--verify", nargs=argparse.REMAINDER, help="Verification command and arguments; place this option last")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new file; preserve existing source and audit records")
    handoff = load_handoff(args.report, args.project)
    config = Config.from_env()
    if args.model:
        config.model = args.model
    elif not os.getenv("CORECODER_MODEL") and os.getenv("DEEPSEEK_API_KEY") and not (
        os.getenv("CORECODER_API_KEY") or os.getenv("OPENAI_API_KEY")
    ):
        config.model = "deepseek-flash"
    if args.base_url:
        config.base_url = args.base_url
    result = run_repair(handoff, args.instruction, build_coding_llm(config), verify_command=args.verify,
                        max_rounds=args.max_rounds, max_tool_calls=args.max_tool_calls,
                        max_investigation_rounds=args.max_investigation_rounds,
                        verification_timeout=args.verification_timeout, mode="live")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Repair status: {result['status']}; audit saved to {args.output}")
    if result["status"] == "interrupted":
        return 130
    return 0 if result["status"] in ("verified", "completed_unverified") else 2


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="researchcoder", description=__doc__)
    commands = ("code", "diagnose", "repair", "demo", "run", "chat")
    parser.add_argument("command", nargs="?", choices=commands)
    if args and args[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    command = args.pop(0) if args and args[0] in commands else "code"
    try:
        if command == "code":
            return _code(args)
        if command == "diagnose":
            from .diagnose import main as diagnosis_main
            return diagnosis_main(args)
        if command == "repair":
            return _repair(args)
        if command in ("run", "chat"):
            from .routed_cli import main as routed_main
            return routed_main(args, interactive=command == "chat")
        from .workflow_demo import main as demo_main
        return demo_main(args)
    except KeyboardInterrupt:
        print("ResearchCoder interrupted.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        print(f"ResearchCoder: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
