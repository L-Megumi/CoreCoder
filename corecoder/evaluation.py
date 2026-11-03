"""Score structured model predictions against annotations kept outside the diagnosis project."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from .diagnose import build_deepseek_llm, render_markdown, run_diagnosis
from .diagnostic_cases import GoldReplayLLM, create_cases
from .experiments import summarize_runs, write_json


def score_report(report, case):
    prediction = report.get("diagnosis") or {}
    valid = report.get("status") == "completed"
    observations = {item["id"]: item for item in report["observations"]}
    actual_locations = []
    for location in prediction.get("primary_locations", []):
        item = observations.get(location["observation_id"])
        path = item["data"].get("resolved_path") if item else None
        if path:
            try:
                relative = Path(path).relative_to(Path(case["project"])).as_posix()
            except ValueError:
                continue
            actual_locations.append({"relative_path": relative, "line_no": location["line_no"]})
    checked_paths = {item["data"].get("requested_path") for item in observations.values()
                     if item["kind"] == "path_status" and item["data"]["status"] in ("exists", "missing")}
    path_checks = all(path in checked_paths for path in case["expected_path_checks"])
    return {
        "report_valid": valid,
        "category_correct": valid and prediction.get("failure_category") == case["expected_category"],
        "annotated_cause_code_correct": valid and prediction.get("cause_code") == case["expected_cause_code"],
        "annotated_source_line_correct": valid and any(item in case["expected_locations"] for item in actual_locations),
        "required_literal_path_checks_completed": path_checks,
        "all_checklist_items_attempted": report["investigation_checklist"]["ready_for_report"],
        "predicted_locations": actual_locations,
        "scope": "Exact category/cause-code/source-site matches on hand-authored fixtures; not real-world root-cause accuracy",
    }


def aggregate(rows):
    metrics = ("report_valid", "category_correct", "annotated_cause_code_correct", "annotated_source_line_correct",
               "required_literal_path_checks_completed", "all_checklist_items_attempted")
    result = {}
    for arm in ("baseline", "checklist"):
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        result[arm] = {"runs": len(group), **{metric: sum(bool(row["score"][metric]) for row in group) / len(group)
                                             for metric in metrics}}
        # Reuse cost aggregation without discarding failed reports.
        result[arm]["cost"] = summarize_runs([{**row, "arm": "baseline"} for row in group])["baseline"]
    return result


def run_suite(output_dir, *, live=False, case_ids=None, seed=42, max_rounds=8, model=None, base_url=None):
    cases = create_cases(output_dir / "fixtures")
    selected = [case for case in cases if case_ids is None or case["id"] in case_ids]
    if not selected or (case_ids is not None and set(case_ids) - {case["id"] for case in cases}):
        raise ValueError("Select known case IDs c01 through c07")
    schedule = [(case, arm) for case in selected for arm in ("baseline", "checklist")]
    random.Random(seed).shuffle(schedule)
    write_json(output_dir / "manifest.json", {
        "mode": "live_synthetic" if live else "offline_gold_replay", "seed": seed,
        "schedule": [[case["id"], arm] for case, arm in schedule], "max_rounds": max_rounds,
        "gold_visibility": "Only evaluator/replay can read annotations; live diagnosis receives log, project and entry.py hint",
        "input_sha256": {case["id"]: hashlib.sha256(Path(case["log"]).read_bytes()).hexdigest() for case in selected},
    })
    rows = []
    for case, arm in schedule:
        print(f"Evaluating {case['id']} / {arm}", flush=True)
        llm = build_deepseek_llm(model=model, base_url=base_url) if live else GoldReplayLLM(case)
        report = run_diagnosis(case["log"], case["project"], llm, allowed_path_roots=[case["workspace"]],
                               completion_mode="checklist" if arm == "checklist" else "off", source_hints=case["source_hints"],
                               findings=True, max_rounds=max_rounds, mode="live" if live else "offline_annotated_replay")
        stem = f"{case['id']}-{arm}"
        write_json(output_dir / f"{stem}.json", report)
        (output_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
        rows.append({"case_id": case["id"], "arm": arm, "status": report["status"], "metrics": report["metrics"],
                     "score": score_report(report, case), "artifact": f"{stem}.json"})
        write_json(output_dir / "progress.json", {"runs": rows})
    return {"mode": "live_synthetic" if live else "offline_gold_replay", "runs": rows,
            "summary": aggregate(rows) if live else None,
            "fixture_validation_passed": None if live else all(all(row["score"][key] for key in (
                "report_valid", "category_correct", "annotated_cause_code_correct", "annotated_source_line_correct")) for row in rows),
            "limitations": ["Hand-authored synthetic logs; no training execution and no independent held-out dataset",
                            "Single run per case and arm; model/provider randomness uncontrolled",
                            "Gold replay is only a grader/pipeline round-trip and has no model accuracy interpretation"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=[f"c{index:02d}" for index in range(1, 8)])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    if args.offline and (args.model or args.base_url):
        parser.error("Offline replay does not use model configuration")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        result = run_suite(args.output_dir, live=args.live, case_ids=args.cases, seed=args.seed,
                           max_rounds=args.max_rounds, model=args.model, base_url=args.base_url)
        write_json(args.output_dir / "summary.json", result)
        print(f"Saved evaluation to {args.output_dir}")
        return 0 if all(row["status"] == "completed" for row in result["runs"]) else 2
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
