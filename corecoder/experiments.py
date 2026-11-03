"""Reproducible mechanism checks and explicitly requested live diagnosis ablations."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .context import ContextManager, estimate_tokens
from .diagnose import build_deepseek_llm, render_markdown, run_diagnosis
from .diagnosis_demo import DemoLLM, create_demo_case
from .diagnostic_tools import Investigation
from .evidence import NUMERIC_ANOMALY_PATTERN
from .evidence_context import EvidenceContextManager, encode

ARMS = {
    "baseline": {"tool_selection": "all", "context_mode": "baseline"},
    "tools": {"tool_selection": "guided", "context_mode": "baseline"},
    "context": {"tool_selection": "all", "context_mode": "evidence"},
    "both": {"tool_selection": "guided", "context_mode": "evidence"},
}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_runs(rows):
    result = {}
    for arm in ARMS:
        group = [row for row in rows if row["arm"] == arm]
        if not group:
            continue
        result[arm] = {
            "runs": len(group), "completed": sum(row["status"] == "completed" for row in group),
            "mean_llm_calls": statistics.mean(row["metrics"]["llm_calls"] for row in group),
            "mean_tool_calls": statistics.mean(row["metrics"]["tool_calls"] for row in group),
            "mean_tool_errors": statistics.mean(row["metrics"]["tool_errors"] for row in group),
            "mean_elapsed_seconds": statistics.mean(row["metrics"]["elapsed_seconds"] for row in group),
            "mean_request_tokens_estimate": statistics.mean(row["metrics"]["request_tokens_estimate_total"] for row in group),
            "all_provider_usage_complete": all(row["metrics"]["provider_usage_complete"] for row in group),
            "mean_prompt_tokens_reported": statistics.mean(row["metrics"]["prompt_tokens"] for row in group),
            "mean_completion_tokens_reported": statistics.mean(row["metrics"]["completion_tokens"] for row in group),
        }
    return result


def _facts_in(messages):
    facts = set()

    def visit(value):
        if isinstance(value, dict):
            if "id" in value and str(value["id"]).startswith("ev_"):
                facts.add(("evidence", value["id"]))
                for link in ("cause_id", "context_id"):
                    if value.get(link):
                        facts.add((link, value["id"], value[link]))
                text = value.get("message") or value.get("content") or ""
                if isinstance(text, str):
                    facts.update(("numeric_signal", match.group().lower()) for match in NUMERIC_ANOMALY_PATTERN.finditer(text))
            if "id" in value and "data" in value:
                data = value["data"]
                path = data.get("resolved_path") or data.get("requested_path")
                if path and data.get("status"):
                    facts.add(("observation", value["id"], path, data["status"]))
            for key, item in value.items():
                if key == "numeric_signals":
                    facts.update(("numeric_signal", signal) for signal in item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for message in messages:
        content = message.get("content") or ""
        if content.startswith("[Evidence memory]"):
            content = content.split("\n", 1)[1]
        try:
            visit(json.loads(content))
        except ValueError:
            pass
    return facts


def context_stress_case(directory, variant):
    """A constructed transcript for exact structured-fact retention, not model accuracy."""
    workspace, root, log, current, previous = create_demo_case(directory)
    if variant == "numeric":
        log.write_text("INFO warmup\nstep=10 loss=nan gradient=inf\n", encoding="utf-8")
    elif variant == "long_message":
        text = log.read_text(encoding="utf-8")
        text = text.replace("FileNotFoundError:\n", "FileNotFoundError:\n" + "diagnostic-padding " * 180 + "\n")
        log.write_text(text, encoding="utf-8")
    state = Investigation(log, root, allowed_path_roots=[workspace])
    requests = [("extract_log_evidence", {})]
    first = state.execute(*requests[0])
    outputs = [first]
    ids = [item["id"] for item in state.evidence]
    if variant != "numeric":
        requests.append(("locate_source", {"evidence_ids": ids}))
        outputs.append(state.execute(*requests[-1]))
    requests.append(("inspect_paths", {"paths": [str(current), str(previous)]}))
    outputs.append(state.execute(*requests[-1]))
    path_states = [item["data"]["status"] for item in json.loads(outputs[-1])["observations"]]
    if path_states != ["exists", "missing"]:
        raise ValueError("Stress fixture paths did not match independently constructed expected states")
    messages = [{"role": "user", "content": "Diagnose this historical log using current observations."}]
    for index, ((name, arguments), output) in enumerate(zip(requests, outputs)):
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": str(index), "type": "function", "function": {"name": name, "arguments": encode(arguments)},
            }]},
            {"role": "tool", "tool_call_id": str(index), "content": output},
        ])
    # Repeat irrelevant complete tool groups to trigger both compression implementations.
    for index in range(8):
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"noise_{index}"}]},
            {"role": "tool", "tool_call_id": f"noise_{index}",
             "content": json.dumps({"diagnostic_noise": ["irrelevant progress output " * 20] * 16}, indent=2)},
        ])
    golden = {("evidence", item["id"]) for item in state.evidence}
    golden.update(("numeric_signal", match.group().lower()) for match in NUMERIC_ANOMALY_PATTERN.finditer(state.log_text))
    golden.update(("cause_id", item["id"], item["cause_id"]) for item in state.evidence if item.get("cause_id"))
    golden.update(("observation", item["id"], item["data"]["resolved_path"], item["data"]["status"])
                  for item in state.observations.values() if item["kind"] == "path_status")
    variants = {}
    for name in ("baseline", "evidence"):
        history = copy.deepcopy(messages)
        manager = ContextManager(5000) if name == "baseline" else EvidenceContextManager(state, 5000)
        manager.maybe_compress(history, llm=None)
        if name == "evidence" and manager.memory:
            history.insert(0, {"role": "user", "content": "[Evidence memory]\n" + manager.memory})
        retained = _facts_in(history)
        variants[name] = {
            "facts_total": len(golden), "facts_retained": len(golden & retained),
            "structured_fact_retention": len(golden & retained) / len(golden),
            "missing_facts": [list(item) for item in sorted(golden - retained)],
            "context_tokens_estimate": estimate_tokens(history),
            "input_tokens_estimate": estimate_tokens(messages),
        }
    return {"variant": variant, "constructed_transcript": True, "variants": variants}


def run_offline(output_dir):
    rows = []
    workspace, root, log, current, previous = create_demo_case(output_dir / "shared-fixture")
    for arm, configuration in ARMS.items():
        report = run_diagnosis(log, root, DemoLLM(root, current, previous), allowed_path_roots=[workspace],
                               mode="offline_scripted", **configuration)
        write_json(output_dir / f"{arm}.json", report)
        rows.append({"arm": arm, "status": report["status"], "metrics": report["metrics"]})
    stress = [context_stress_case(output_dir / f"stress-{variant}", variant)
              for variant in ("chain", "numeric", "long_message")]
    return {
        "mode": "offline_mechanism_checks", "runs": rows, "summary": summarize_runs(rows), "context_stress": stress,
        "limitations": ["Scripted model, not diagnosis accuracy or provider token usage",
                        "Constructed transcript; exact structured facts after compression, not semantic recall",
                        "Character-based token estimates are not provider billing tokens"],
    }


def run_live(output_dir, log, project, *, allowed_path_roots=(), repeats=1, seed=42,
             max_context_tokens=128_000, model=None, base_url=None):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 3:
        raise ValueError("repeats must be 1 to 3")
    rng = random.Random(seed)
    rows = []
    schedule = []
    for repeat in range(repeats):
        arms = list(ARMS)
        rng.shuffle(arms)
        schedule.extend((repeat, arm) for arm in arms)
    write_json(output_dir / "manifest.json", {
        "mode": "live", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(Path(log).resolve()), "project": str(Path(project).resolve()),
        "seed": seed, "seed_scope": "arm ordering only; does not seed the model", "schedule": schedule,
        "max_context_tokens": max_context_tokens, "max_rounds": 8, "max_tool_calls": 16, "max_report_repairs": 1,
    })
    for repeat, arm in schedule:
        print(f"Running {arm}, repetition {repeat + 1}/{repeats}", flush=True)
        llm = build_deepseek_llm(model=model, base_url=base_url)
        report = run_diagnosis(log, project, llm, allowed_path_roots=allowed_path_roots,
                               max_context_tokens=max_context_tokens, mode="live", **ARMS[arm])
        stem = f"{repeat + 1:02d}-{arm}"
        write_json(output_dir / f"{stem}.json", report)
        (output_dir / f"{stem}.md").write_text(render_markdown(report), encoding="utf-8")
        rows.append({"arm": arm, "repeat": repeat, "model": report["model"], "status": report["status"],
                     "failure_reason": report["failure_reason"], "metrics": report["metrics"], "artifact": f"{stem}.json"})
        write_json(output_dir / "progress.json", {"runs": rows, "summary": summarize_runs(rows)})
    return {"mode": "live", "runs": rows, "summary": summarize_runs(rows), "limitations": [
        "One supplied historical case; completion is structural/reference validation, not diagnosis accuracy",
        "Arm ordering is reproducible; model generation and provider cache state are not controlled",
        "Small-sample descriptive results; no statistical significance or causal performance guarantee",
    ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--offline", action="store_true")
    modes.add_argument("--live", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--allow-path-root", action="append", default=[], type=Path)
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-context-tokens", type=int, default=128_000)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    args = parser.parse_args(argv)
    if args.live and (args.log is None or args.project is None):
        parser.error("--live requires --log and --project")
    if args.offline and any((args.log, args.project, args.allow_path_root, args.model, args.base_url)):
        parser.error("--offline generates its own fixtures; omit live inputs")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        if args.offline:
            result = run_offline(args.output_dir)
        else:
            result = run_live(args.output_dir, args.log, args.project, allowed_path_roots=args.allow_path_root,
                              repeats=args.repeats, seed=args.seed, max_context_tokens=args.max_context_tokens,
                              model=args.model, base_url=args.base_url)
        write_json(args.output_dir / "summary.json", result)
        print(f"Saved results to {args.output_dir}")
        return 0 if all(row["status"] == "completed" for row in result["runs"]) else 2
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
