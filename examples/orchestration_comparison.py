"""Reproducible engineering comparison; ScriptedLLM is not a quality benchmark."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from examples.multi_agent_autograd_demo import prepare_demo, call
from researchcoder.runtime import ResearchHarness
from researchcoder.runtime.protocol import CodingTaskSpec, ModelSpec, ScriptedTurn
from researchcoder.contracts import ExecutionPolicy
from researchcoder.orchestration import DAGScheduler
from researchcoder.orchestration.budget import JobBudget
from researchcoder.orchestration.planning import Planner, PlanningJob, PlanningConstraints


def metrics(mode, status, verification, records, elapsed, budget, *, planning=(), diagnosis="validated_structured_report"):
    rows = list(records) + list(planning)
    settlements = budget.snapshot()["settlements"]
    known = all(not r["usage_unknown"] for r in settlements.values())
    return dict(
        mode=mode,
        status=status,
        diagnosis_result=diagnosis,
        verification_result=verification,
        wall_seconds=elapsed,
        worker_cumulative_seconds=sum(
            max(0, (r.get("ended_ns", 0) - r.get("started_ns", 0)) / 1e9)
            if "started_ns" in r
            else max(0, (r.get("budget") or {}).get("elapsed_seconds", 0))
            for r in rows
        ),
        model_requests=budget.charged["model_calls"],
        tool_calls=budget.charged["tool_calls"],
        actual_prompt_tokens=sum(r["known_prompt_tokens"] or 0 for r in settlements.values()) if known else None,
        actual_completion_tokens=sum(r["known_completion_tokens"] or 0 for r in settlements.values()) if known else None,
        known_partial_prompt_tokens=sum(r["known_prompt_tokens"] or 0 for r in settlements.values()),
        known_partial_completion_tokens=sum(r["known_completion_tokens"] or 0 for r in settlements.values()),
        estimated_input_tokens=sum(r["estimated_input_tokens"] or 0 for r in settlements.values()),
        token_usage_unknown=not known,
        charged_token_liability=budget.charged["total_tokens"],
        failed_tasks=0 if status == "succeeded" else 1,
        replans=0,
        worker_count=len(rows),
    )


def run_comparison(directory=None, *, torch_python=None, repetitions=2):
    if not 1 <= repetitions <= 5:
        raise ValueError("Bounded comparison repetitions required")
    directory = Path(tempfile.mkdtemp(prefix="researchcoder-comparison-")) if directory is None else Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    all_rows = []
    raw = []
    modes = ("single_agent_sequential", "multi_agent_sequential", "multi_agent_dag_parallel")
    for repetition in range(repetitions):
        # Rotate order to expose rather than hide warm-start/order effects.
        order = modes[repetition % len(modes) :] + modes[: repetition % len(modes)]
        for mode in order:
            target = directory / (str(repetition) + "-" + mode)
            plan, cfg, io, initial = prepare_demo(
                target, torch_python=torch_python, max_workers=2 if mode.endswith("parallel") else 1
            )
            budget = JobBudget(cfg.job_limits)
            started = time.monotonic()
            if mode == "single_agent_sequential":
                repair = cfg.templates["repair"]
                verify = cfg.templates["verify"]
                policy = ExecutionPolicy(
                    allowed_tool_names=("read_file", "edit_file"), writable_paths=("training.py",), allow_verification=True
                )
                spec = CodingTaskSpec(
                    job_id="autograd",
                    task_id="single",
                    objective="Find the unintended detach and repair joint gradient flow; preserve all contracts.",
                    workspace=repair.spec.workspace,
                    inputs=cfg.templates["log"].spec.inputs,
                    policy=policy,
                    verification=verify.spec.verification,
                )
                model = ModelSpec(
                    turns=(
                        call("r", "read_file", {"file_path": "training.py"}),
                        call(
                            "w",
                            "edit_file",
                            {
                                "file_path": "training.py",
                                "old_string": "features = model.backbone(x).detach()",
                                "new_string": "features = model.backbone(x)",
                            },
                        ),
                        ScriptedTurn(
                            content="Hypothesis: detach interrupts backbone gradients. Applied the small code change; independent oracle decides correctness."
                        ),
                    )
                )
                config = replace(repair.config, max_policy=policy, model=model, permitted_verifications=(spec.verification,))
                budget.reserve("single", config.limits)
                harness = ResearchHarness()
                s = time.monotonic_ns()
                result = harness.execute(spec, config)
                e = time.monotonic_ns()
                record = dict(harness.records[result.attempt_id], started_ns=s, ended_ns=e)
                budget.settle("single", record)
                row = metrics(
                    mode,
                    result.status,
                    result.verification.status if result.verification else "not_run",
                    [record],
                    time.monotonic() - started,
                    budget,
                    diagnosis="unverified_model_hypothesis; no specialist structured diagnosis",
                )
                original = dict(result=result.to_dict(), record=record, budget=budget.snapshot())
            else:
                # Include one Lead request in both multi-role arms, under the same Job budget.
                planner = Planner()
                planned = planner.plan(
                    PlanningJob("autograd", "Diagnose and repair joint-training gradient interruption"),
                    [],
                    PlanningConstraints(
                        config=cfg, model=ModelSpec(turns=(ScriptedTurn(content=plan.to_json()),)), budget=budget
                    ),
                )
                scheduler = DAGScheduler()
                result = scheduler.execute(planned, cfg, budget=budget)
                verification = next((r.verification.status for r in reversed(result.results) if r.verification), "not_run")
                planning_records = [r["record"] for r in planner.records]
                row = metrics(
                    mode,
                    result.status,
                    verification,
                    list(scheduler.records.values()),
                    time.monotonic() - started,
                    budget,
                    planning=planning_records,
                )
                original = dict(
                    result=result.to_dict(),
                    execution=io.read_json(result.summary_ref),
                    planning=planner.records,
                    budget=budget.snapshot(),
                )
            row["repetition"] = repetition
            row["model_profile"] = "scripted-runtime"
            row["provider_requests"] = 0
            row["job_limits"] = cfg.job_limits.to_dict()
            (target / "comparison-run.json").write_text(json.dumps(original, indent=2), encoding="utf-8")
            all_rows.append(row)
            raw.append(str(target / "comparison-run.json"))
    result = dict(
        schema_version=1,
        mode="offline_engineering",
        provider_requests=0,
        repetitions=repetitions,
        rows=all_rows,
        raw_records=raw,
        scope="Same injected fault, CPU oracle, model/provider label and Job caps; fixed per-role scripts. Measures engineering behavior, not model accuracy.",
        timing_scope="Excludes fixture creation and initial fault probe; includes Worker startup, Lead planning, tools, verification and archival.",
        limitations=[
            "No controlled simulated LLM latency, no guaranteed parallel speedup on this tiny CPU case.",
            "Single arm uses one existing guarded Coding Agent in one Worker; multi arms use five domain tasks plus Lead.",
            "Single arm outputs a hypothesis; multi arms additionally validate structured specialist diagnosis.",
        ],
    )
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_comparison(args.output_dir, repetitions=args.repetitions), indent=2))
