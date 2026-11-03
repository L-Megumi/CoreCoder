"""Offline by default. Live model and observability require separate explicit flags."""

import argparse
import json
import os
from pathlib import Path
from researchcoder.contracts import ArtifactRef
from researchcoder.adapters.artifacts import ArtifactIO
from researchcoder.runtime.protocol import ModelSpec, ModelRequestOptions
from .contracts import EvaluationConfig, EvaluationResult
from .dataset import load_manifest
from .runner import EvaluationRunner, recompute, write_json
from .observability import LangfuseExporter


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ResearchCoder reproducible offline evaluation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--subset", choices=("pytorch", "complexity_benchmark"), default="pytorch")
    run = sub.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--torch-python", default=os.environ.get("RESEARCHCODER_EXPERIMENT_PYTHON")
    )
    run.add_argument("--config", type=Path)
    run.add_argument("--subset", choices=("pytorch", "complexity_benchmark"))
    run.add_argument("--feedback-rounds", type=int, default=0)
    run.add_argument('--harness-strategy',choices=('fixed-v1','budget-v1','compact-v1','constraints-v1','combined-v1'),default='fixed-v1')
    run.add_argument("--cases", nargs="+")
    run.add_argument(
        "--modes", nargs="+", choices=("single", "sequential", "parallel", "evidence")
    )
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument(
        "--behavior",
        choices=("correct", "no_patch", "wrong_patch", "refuse", "diagnosis_fail", "staged_correct", "repeat_patch"),
        default="correct",
    )
    run.add_argument("--provider", choices=("scripted", "openai"), default="scripted")
    run.add_argument("--model", default="scripted-runtime")
    run.add_argument("--base-url")
    run.add_argument(
        "--credential-env",
        choices=("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "CORECODER_API_KEY"),
        default="OPENAI_API_KEY",
    )
    run.add_argument("--allow-live", action="store_true")
    run.add_argument("--disable-thinking", action="store_true")
    run.add_argument("--offline-observability", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("directory", type=Path)
    report.add_argument("--output", type=Path)
    export = sub.add_parser("export")
    export.add_argument("directory", type=Path)
    export.add_argument("--enable-network", action="store_true")
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "list":
        manifest = load_manifest(args.subset)
        print(
            json.dumps(
                dict(
                    benchmark=manifest.benchmark_id,
                    version=manifest.version,
                    cases=[
                        dict(case_id=c.case_id, category=c.category, split=c.split)
                        for c in manifest.cases
                    ],
                ),
                indent=2,
            )
        )
    elif args.command == "run":
        if not args.torch_python:
            parser.error(
                "An existing CPU Torch interpreter is required; no automatic installation"
            )
        if args.config:
            config = EvaluationConfig.from_json(args.config.read_text())
            if config.model.provider != "scripted" and not args.allow_live:
                parser.error(
                    "Live configuration additionally requires --allow-live on the command line"
                )
        else:
            config = EvaluationConfig(
                harness_strategy=args.harness_strategy,
                benchmark_subset=args.subset or "pytorch",
                max_feedback_rounds=args.feedback_rounds,
                model_options=ModelRequestOptions() if args.disable_thinking else None,
                model=ModelSpec(
                    provider=args.provider,
                    model=args.model,
                    base_url=args.base_url,
                    credential_env=args.credential_env,
                ),
                allow_live=args.allow_live,
                modes=tuple(
                    args.modes or ("single", "sequential", "parallel", "evidence")
                ),
                case_ids=tuple(args.cases or ()),
                repeats=args.repeats,
                scripted_behavior=args.behavior,
            )
            if args.subset == "complexity_benchmark" and args.provider == "scripted":
                from .complexity.protocol import offline_config
                config = offline_config(model=config.model, modes=config.modes, case_ids=config.case_ids,
                    harness_strategy=config.harness_strategy,
                    repeats=config.repeats, scripted_behavior=config.scripted_behavior,
                    max_feedback_rounds=config.max_feedback_rounds)
        summary = EvaluationRunner().run(
            args.output_dir,
            config,
            torch_python=args.torch_python,
            exporter=LangfuseExporter(
                mode="offline" if args.offline_observability else "disabled"
            ),
        )
        write_json(args.output_dir / "report.json", summary)
        print(json.dumps(summary, indent=2))
    elif args.command == "report":
        summary = recompute(args.directory)
        if args.output:
            write_json(args.output, summary)
        print(json.dumps(summary, indent=2))
    else:
        exporter = LangfuseExporter(
            mode="langfuse" if args.enable_network else "offline",
            allow_network=args.enable_network,
        )
        rows = []
        for path in sorted((args.directory / "records").glob("*.json")):
            io = ArtifactIO(args.directory / "trials" / path.stem / "artifacts")
            result = EvaluationResult.from_dict(
                io.read_json(ArtifactRef.from_json(path.read_text()))
            )
            rows.append(exporter.export_result(result, args.directory))
        write_json(args.output, dict(schema_version=1, records=rows))
        print(json.dumps(dict(records=len(rows), statuses=[r["status"] for r in rows])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
