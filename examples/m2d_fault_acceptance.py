"""Prepare and run bounded live diagnosis/repair on six real-source M2D fault injections."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

from corecoder.diagnose import build_deepseek_llm
from corecoder.fault_injection import child_environment, digest, execute, prepare_cases, snapshot_sources, write_json
from corecoder.repair_efficiency import plan_case_requests
from corecoder.routed_cli import _coding_config
from corecoder.routed_session import RoutedSession
from corecoder.source_visibility import visible_end_line
from corecoder.workflow import build_coding_llm


@contextmanager
def runtime_environment(case):
    keys = ("PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES", "YOLO_CONFIG_DIR", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = case["environment"][key]
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def source_coverage(report, annotation):
    matched = []
    for observation in report.get("observations", []):
        data = observation["data"]
        span = data.get("snippet") or data
        if (observation["kind"] not in ("source_location", "file_excerpt", "config_file")
                or data.get("file_sha256") != annotation["injected_target_sha256"]):
            continue
        if any(span.get("start_line", 0) <= line <= visible_end_line(span) for line in annotation.get("injected_lines", [])):
            matched.append(observation["id"])
    cited = {identifier for item in (report.get("diagnosis") or {}).get("hypotheses", []) for identifier in item["observation_ids"]}
    return {"injected_line_observed": bool(matched), "hypothesis_cites_that_observation": bool(set(matched) & cited),
            "scope": "Source observation/citation coverage, not semantic root-cause accuracy"}


def acceptance_guard(case, rejected):
    project = Path(case["project"]).resolve()
    def guard(name, arguments):
        key = "file_path" if name in ("read_file", "edit_file", "write_file") else "path"
        target = (project / arguments.get(key, ".")).resolve()
        allowed = target.is_relative_to(project)
        if name in ("grep", "glob"):
            pattern = arguments.get("pattern", "") if name == "glob" else arguments.get("include") or "*"
            allowed = allowed and not Path(pattern).is_absolute() and ".." not in Path(pattern).parts
            # Copies contain regular source files only; refuse later links before recursive traversal.
            allowed = allowed and not any(item.is_symlink() for item in project.rglob("*"))
        if name in ("edit_file", "write_file"):
            allowed = (allowed and target.suffix in (".py", ".yaml", ".yml")
                       and target.name not in ("run_experiment.py", "scenario.json")
                       and target.relative_to(project).as_posix() in case["manifest"])
        if not allowed:
            rejected.append({"tool": name, "target": str(target)})
            raise ValueError("Acceptance scope violation: stay in project source; launcher and evaluator are protected")

    return guard


def run_case(case, output, total_requests, max_total=72, max_case=14, previous=None, phase=None, phase_allowance=None):
    project = Path(case["project"])
    case_output = Path(output) / (phase or ("continuations" if previous else "runs")) / case["id"]
    case_output.mkdir(parents=True, exist_ok=False)
    annotation = json.loads(Path(case["annotation"]).read_text())  # Never passed to a model.
    requests = list(previous["requests"]) if previous else []
    phase_start_requests = len(requests)
    responses = list(previous["responses"]) if previous else []
    rejected = []
    started = time.monotonic()

    def factory(diagnosis):
        if phase_allowance == 0:
            raise RuntimeError("No reserved requests for this case; no model was contacted")
        model = build_deepseek_llm() if diagnosis else build_coding_llm(_coding_config())
        original_request, original_chat = model._call_with_retry, model.chat
        def request(params, max_retries=1):
            if len(requests) >= max_case or len(total_requests) >= max_total or time.monotonic() - started > 300:
                raise RuntimeError("Fault acceptance request/time budget exhausted")
            if phase_allowance is not None and len(requests) - phase_start_requests >= phase_allowance:
                raise RuntimeError("Case reserved request budget exhausted; other cases retain their allowances")
            event = {"case": case["id"], "phase": "diagnose" if diagnosis else "repair", "number": len(total_requests) + 1}
            requests.append(event)
            total_requests.append(event)
            write_json(case_output / "request-ledger.json", {"requests": requests, "phase_allowance": phase_allowance,
                                                             "max_total": max_total, "max_case_cumulative": max_case})
            print(json.dumps(event), flush=True)
            return original_request(params, max_retries=1)
        def chat(messages, tools=None, on_token=None):
            response = original_chat(messages=messages, tools=tools, on_token=on_token)
            responses.append({"phase": "diagnose" if diagnosis else "repair", "prompt_tokens": response.prompt_tokens,
                              "completion_tokens": response.completion_tokens})
            return response
        model._call_with_retry, model.chat = request, chat
        return model

    protected = {"launcher": digest(project / "run_experiment.py"), "scenario": digest(project / "scenario.json"),
                 "verifier": digest(case["command"][2]), "expectations": digest(case["command"][4]),
                 "annotation": digest(case["annotation"])}
    previous_artifact = None
    if previous:
        previous_session = json.loads(Path(previous["session"]).read_text())
        previous_artifact = previous.get("diagnosis_artifact") or previous_session["turns"][0]["artifact"]
        if previous.get("reused_diagnosis") and not previous.get("diagnosis_artifact"):
            previous_artifact = previous_session["bindings"][0]["path"]
        if previous["diagnosis_status"] != "completed" or previous["file_changes"]:
            raise ValueError("Continuation requires a completed diagnosis and an unchanged injection")
    session = RoutedSession(project, log_file=None if previous else case["log"], report_file=previous_artifact,
                            session_dir=case_output / "session",
                            coding_factory=lambda: factory(False), diagnosis_factory=lambda: factory(True),
                            verify_command=case["command"], execution_mode="live", max_feedback_rounds=0,
                            source_hints=case.get("source_hints", ()),
                            repair_tool_names=("read_file", "edit_file", "write_file", "grep", "glob"),
                            repair_tool_guard=acceptance_guard(case, rejected))
    with runtime_environment(case):
        diagnosis = ({"status": "completed", "artifact": previous_artifact} if previous else session.handle(
            case.get("diagnosis_instruction", "只读调查这次实验输出异常，定位相关项目代码或配置，核对当前事实并说明原因假设和验证方法。"), mode="diagnose"))
        report = json.loads(Path(diagnosis["artifact"]).read_text()) if diagnosis["artifact"] else {}
        repair = None
        if diagnosis["status"] == "completed":
            repair = session.handle(
                case.get("repair_instruction", "修复诊断指出的项目源码或配置问题，保持接口与原操作语义。保留 run_experiment.py 和 scenario.json，"
                         "不改输入或校验、不创建替代依赖、不下载资源。使用项目内读取、搜索和编辑工具；用 grep 定位函数，避免逐页读取长文件；验证由控制器执行。"), mode="repair")
    final = execute(case["command"], project, case["environment"])
    write_json(case_output / "independent-final-check.json", final)
    intact = protected == {"launcher": digest(project / "run_experiment.py"), "scenario": digest(project / "scenario.json"),
                           "verifier": digest(case["command"][2]), "expectations": digest(case["command"][4]),
                           "annotation": digest(case["annotation"])}
    changes = [{"path": path, "before_sha256": before, "after_sha256": digest(project / path) if (project / path).is_file() else None}
               for path, before in case["manifest"].items() if not (project / path).is_file() or digest(project / path) != before]
    row = {"case_id": case["id"], "fault_kind": annotation["fault_kind"], "diagnosis_status": diagnosis["status"],
           "repair_status": repair["status"] if repair else "not_started", "protected_files_unchanged": intact,
           "independent_verification_passed": final["exit_code"] == 0 and intact,
           "independent_final_exit_code": final["exit_code"],
           "repair_failure_reason": (session.last_result or {}).get("failure_reason"),
           "phase_request_attempts": len(requests) - phase_start_requests,
           "phase_request_allowance": phase_allowance,
           "prior_scope_rejections": previous.get("scope_rejections", []) if previous else [],
           "healthy_control_passed": annotation["healthy_control"]["exit_code"] == 0,
           "injected_control_failed": annotation["injected_control"]["exit_code"] != 0,
           "coverage": source_coverage(report, annotation), "file_changes": changes,
           "requests": requests, "responses": responses, "scope_rejections": rejected,
           "prompt_tokens": sum(item["prompt_tokens"] for item in responses),
           "completion_tokens": sum(item["completion_tokens"] for item in responses),
           "usage_complete": len(requests) == len(responses) and bool(responses)
                             and all(item["prompt_tokens"] or item["completion_tokens"] for item in responses),
           "feedback_records": session.feedback_records,
           "elapsed_seconds": round(time.monotonic() - started + (previous["elapsed_seconds"] if previous else 0), 3),
           "previous_repair_status": previous["repair_status"] if previous else None,
           "reused_diagnosis": bool(previous), "diagnosis_artifact": diagnosis["artifact"],
           "session": str(session.directory / "session.json"), "mode": "live_project_source_injection"}
    write_json(case_output / "result.json", row)
    print(json.dumps({"finished": case["id"], "diagnosis": row["diagnosis_status"],
                      "repair": row["repair_status"], "verified": row["independent_verification_passed"]}), flush=True)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared", action="store_true", help="Use previously prepared, untouched injections")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--continue-repairs", action="store_true", help="Preserve initial results; reuse validated reports for untouched cases")
    parser.add_argument("--recover-repairs", action="store_true", help="Preserve both earlier phases; recover unchanged failed cases")
    parser.add_argument("--env-file", type=Path, default=Path.cwd() / ".env")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--max-requests", type=int, default=72, choices=range(1, 101))
    parser.add_argument("--max-case-requests", type=int, default=14, choices=range(1, 21))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if args.continue_repairs and args.recover_repairs:
        parser.error("Select one continuation phase")
    if (args.continue_repairs or args.recover_repairs) and not (args.prepared and args.live):
        parser.error("Continuation requires --prepared --live")
    result_name = "recovery-summary.json" if args.recover_repairs else "continuation-summary.json" if args.continue_repairs else "summary.json"
    phase_name = "recovery" if args.recover_repairs else "continuation" if args.continue_repairs else "initial"
    phase_directory = "recoveries" if args.recover_repairs else "continuations" if args.continue_repairs else "runs"
    if args.live and any((output / name).exists() for name in (result_name, phase_directory, "budget-plan-" + phase_name + ".json")):
        raise ValueError("Acceptance phase already recorded; never reset its request ledger or overwrite results")
    recovery_rows = {}
    if args.recover_repairs:
        recovery_rows = {row["case_id"]: row for row in json.loads((output / "continuation-summary.json").read_text())["rows"]}
        if (output / "recovery-summary.json").exists():
            raise ValueError("Recovery already recorded; never overwrite prior acceptance results")
    if args.prepared:
        cases = json.loads((output / "evaluator/cases.json").read_text())
        for case in cases:
            case["environment"] = child_environment(case["project"], output / "runtime_settings" / case["id"])
            expected = dict(case["manifest"])
            expected.update({item["path"]: item["after_sha256"] for item in recovery_rows.get(case["id"], {}).get("file_changes", [])})
            if any(not (Path(case["project"]) / name).is_file() or digest(Path(case["project"]) / name) != value for name, value in expected.items()):
                raise ValueError("Prepared injection changed; do not silently reuse a previously repaired case")
    else:
        cases = prepare_cases(args.source, output, args.experiment_python)
    if not args.live:
        print(json.dumps({"status": "prepared", "cases": len(cases), "provider_requests": 0}))
        return 0
    load_dotenv(args.env_file, override=False)
    if args.direct:
        for key in list(os.environ):
            if key.lower().endswith("_proxy"):
                del os.environ[key]
    previous_rows = recovery_rows
    if args.continue_repairs and args.recover_repairs:
        raise ValueError("Select one continuation phase")
    if args.continue_repairs:
        previous_summary = json.loads((output / "summary.json").read_text())
        previous_rows = {row["case_id"]: row for row in previous_summary["rows"]}
        if set(previous_rows) != {case["id"] for case in cases}:
            raise ValueError("Continuation requires the same complete initial case set")
    if previous_rows and set(previous_rows) != {case["id"] for case in cases}:
        raise ValueError("Continuation requires the same complete case set")
    requests = [event for case in cases for event in previous_rows.get(case["id"], {}).get("requests", [])]
    if len(requests) > args.max_requests:
        raise ValueError("Previous requests already exceed this acceptance budget")
    previous_counts = {cid: len(row["requests"]) for cid, row in previous_rows.items()}
    active_ids = [case["id"] for case in cases
                  if not (args.recover_repairs and previous_rows.get(case["id"], {}).get("file_changes"))]
    allowances = plan_case_requests(active_ids, previous_counts, args.max_requests, args.max_case_requests)
    budget_plan = {"previous_request_counts": previous_counts, "max_total": args.max_requests,
                   "max_case_cumulative": args.max_case_requests, "phase_allowances": allowances,
                   "policy": "Reserved before execution; unused allowances are not automatically borrowed"}
    write_json(output / ("budget-plan-" + phase_name + ".json"), budget_plan)
    rows = []
    for case in cases:
        previous = previous_rows.get(case["id"])
        if args.recover_repairs and previous and previous["file_changes"]:
            # Preserve the incomplete workflow; independently recheck its existing patch without stale handoff reuse.
            row = dict(previous)
            final = execute(case["command"], case["project"], case["environment"])
            check_path = output / "recoveries" / case["id"] / "independent-final-check.json"
            if check_path.exists():
                raise ValueError("Recovery check already exists")
            write_json(check_path, final)
            row["independent_verification_passed"] = final["exit_code"] == 0 and previous["protected_files_unchanged"]
            row["independent_final_exit_code"] = final["exit_code"]
            row["phase_request_attempts"] = 0
            row["phase_request_allowance"] = 0
            row["recovery_action"] = "Existing patch independently rechecked; incomplete workflow retained"
        else:
            row = run_case(case, output, requests, args.max_requests, args.max_case_requests, previous,
                           phase="recoveries" if args.recover_repairs else None, phase_allowance=allowances[case["id"]])
        rows.append(row)
        write_json(output / ("progress-recovery.json" if args.recover_repairs else "progress-continuation.json" if args.continue_repairs else "progress.json"),
                   {"rows": rows, "request_attempts": len(requests)})
    original = json.loads((output / "original-source-hashes.json").read_text())
    original_unchanged = snapshot_sources(args.source) == original
    summary = {"mode": "live_project_source_injection", "cases": len(rows),
               "diagnoses_completed": sum(row["diagnosis_status"] == "completed" for row in rows),
               "repair_workflows_verified": sum(row["repair_status"] == "verified" for row in rows),
               "independent_verifications_passed": sum(row["independent_verification_passed"] for row in rows),
               "source_lines_observed": sum(row["coverage"]["injected_line_observed"] for row in rows),
               "request_attempts": len(requests), "max_requests": args.max_requests, "original_sources_unchanged": original_unchanged,
               "prompt_tokens": sum(row["prompt_tokens"] for row in rows), "completion_tokens": sum(row["completion_tokens"] for row in rows),
               "usage_complete": all(row["usage_complete"] for row in rows), "rows": rows,
               "initial_results_preserved": bool(args.continue_repairs or args.recover_repairs),
               "budget_plan": budget_plan,
               "scope": "Six hand-selected injected faults in copied M2D-LIF source, recorded initial and recovery attempts, CPU-sized semantic checks. "
                        "No training, no held-out real-world dataset, no baseline comparison, no universal accuracy or cost claim. "
                        "Repair tools are restricted by this acceptance harness; product coding tools are unchanged."}
    write_json(output / ("recovery-summary.json" if args.recover_repairs else "continuation-summary.json" if args.continue_repairs else "summary.json"), summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0 if original_unchanged and summary["independent_verifications_passed"] == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
