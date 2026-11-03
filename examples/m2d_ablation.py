"""Fresh six-case, four-arm, two-repeat read-only DeepSeek experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from corecoder.controlled_experiment import (
    RequestLedger,
    product_hashes,
    schedule_runs,
    score_localization,
    source_unchanged,
    summarize,
)
from corecoder.diagnose import build_deepseek_llm, render_markdown, run_diagnosis
from corecoder.experiments import ARMS, run_offline
from corecoder.fault_injection import digest, prepare_cases, snapshot_sources, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    load_dotenv(args.env_file, override=False)
    if args.direct:
        for key in list(os.environ):
            if key.lower().endswith("_proxy"):
                del os.environ[key]
    cases = prepare_cases(args.source, output / "fixtures", args.experiment_python)
    case_map = {case["id"]: case for case in cases}
    schedule = schedule_runs(list(case_map), args.repeats, args.seed)
    source_hashes = product_hashes(Path.cwd())
    manifest = {"schema_version": 1, "purpose": "Read-only diagnosis optimization factorial ablation",
                "schedule": schedule, "arms": ARMS, "seed_scope": "case and arm order only",
                "seed": args.seed, "repeats": args.repeats,
                "max_requests": 10 * len(schedule), "max_requests_per_run": 10,
                "max_rounds": 8, "max_tool_calls": 16, "max_report_repairs": 1,
                "max_context_tokens": 128000, "completion_mode": "checklist", "findings": False,
                "source_hints": [], "temperature": 0, "model_seed": None,
                "product_source_sha256": source_hashes,
                "log_sha256": {case["id"]: digest(case["log"]) for case in cases},
                "scoring": "Valid schema/references, exact injected-line observation and current hypothesis citation",
                "failure_policy": "No rerun, no report recovery, no human hints; retain all scheduled results",
                "boundaries": "Six known development cases, two stochastic repeats; no held-out benchmark or repair comparison"}
    write_json(output / "manifest.json", manifest)
    offline = output / "offline-mechanisms"
    offline.mkdir()
    write_json(offline / "summary.json", run_offline(offline))
    if not args.live:
        print(json.dumps({"prepared_runs": len(schedule), "api_requests": 0}), flush=True)
        return 0
    ledger = RequestLedger(output / "request-ledger.json", manifest["max_requests"], 10)
    rows = []
    for scheduled in schedule:
        started = time.monotonic()
        case = case_map[scheduled["case_id"]]
        if not source_unchanged(case):
            raise ValueError("Case source changed before a scheduled run")
        run_dir = output / "runs" / scheduled["run_id"]
        run_dir.mkdir(parents=True)
        write_json(run_dir / "started.json", scheduled)
        print(json.dumps({"started": scheduled["run_id"], "completed": len(rows)}), flush=True)
        llm = build_deepseek_llm()
        llm.extra["temperature"] = 0
        ledger.attach(llm, scheduled["run_id"])
        report = run_diagnosis(case["log"], case["project"], llm, mode="live", completion_mode="checklist",
                               max_rounds=8, max_tool_calls=16, max_report_repairs=1, max_context_tokens=128000,
                               **ARMS[scheduled["arm"]])
        write_json(run_dir / "report.json", report)
        (run_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
        events = [event for event in ledger.events if event["run_id"] == scheduled["run_id"]]
        scores = score_localization(report, json.loads(Path(case["annotation"]).read_text(encoding="utf-8")))
        row = {**scheduled, "status": report["status"], "failure_reason": report["failure_reason"], "scores": scores,
               "source_unchanged": source_unchanged(case), "request_attempts": len(events),
               "prompt_tokens": sum(e["prompt_tokens"] for e in events),
               "completion_tokens": sum(e["completion_tokens"] for e in events),
               "tool_schema_chars": sum(e["tool_schema_chars"] for e in events),
               "usage_complete": bool(events) and all(e["status"] == "completed" and e["usage_received"] for e in events),
               "tool_calls": report["metrics"]["tool_calls"], "tool_errors": report["metrics"]["tool_errors"],
               "context_projection_events": len(report["context_events"]),
               "context_group_collapses": sum(e.get("groups_collapsed", False) for e in report["context_events"]),
               "elapsed_seconds": round(time.monotonic() - started, 4),
               "artifact": str((run_dir / "report.json").relative_to(output))}
        rows.append(row)
        write_json(run_dir / "result.json", row)
        write_json(output / "progress.json", {"rows": rows, "summary": summarize(rows)})
        print(json.dumps({"finished": scheduled["run_id"], "status": row["status"],
                          "requests": len(events), "prompt_tokens": row["prompt_tokens"]}), flush=True)
    original = json.loads((output / "fixtures/original-source-hashes.json").read_text())
    unchanged = snapshot_sources(args.source) == original
    summary = {"rows": rows, "summary": summarize(rows), "request_attempts": len(ledger.events),
               "scheduled_runs": len(schedule), "original_source_unchanged": unchanged,
               "product_source_unchanged": all(digest(Path(name)) == value for name, value in source_hashes.items()),
               "all_case_sources_unchanged": all(source_unchanged(case) for case in cases)}
    write_json(output / "summary.json", summary)
    return 0 if unchanged and summary["all_case_sources_unchanged"] and summary["product_source_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
