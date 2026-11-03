"""Reproduce the fixed-action, common-budget independent-input protocol.

No live-provider argument; all runs use existing ScriptedLLM/Worker/Runner paths.
Run this standalone controller in its own process, not as a concurrent service.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

from researchcoder.evaluation import runner
from researchcoder.evaluation.contracts import EvaluationConfig
from researchcoder.evaluation.complexity.protocol import offline_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-python", default=os.environ.get("RESEARCHCODER_EXPERIMENT_PYTHON"))
    args = parser.parse_args(argv)
    if not args.torch_python:
        parser.error("Provide an existing PyTorch interpreter; no automatic installation")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = offline_config(splits=("development", "validation", "holdout"))
    original = runner.build_workflow

    def uniform_builder(case, trial, directory, config, torch_python, **kwargs):
        built = original(case, trial, directory, config, torch_python, **kwargs)
        command = runner.oracle_command(case, built["project"], torch_python, "discovery")
        cfg = built["config"]
        if "spec" in built:
            built["spec"] = replace(built["spec"], verification=command)
            built["config"] = replace(cfg, record_requests=True, permitted_verifications=(command,))
        else:
            templates = {}
            for key, template in cfg.templates.items():
                spec = replace(template.spec, verification=command) if template.role == "verification" else template.spec
                worker = replace(template.config, record_requests=True,
                                 permitted_verifications=(command,)) if template.role == "verification" else replace(template.config, record_requests=True)
                templates[key] = replace(template, spec=spec, config=worker)
            built["config"] = replace(cfg, templates=templates)
        return built

    protocol = dict(id="6d-independent-inputs-v2", controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    job_limits=config.job_limits.to_dict(), task_limits=config.task_limits.to_dict(),
                    record_requests=True, process_profile="discovery", final_f2p="holdout", final_p2p="regression",
                    model="scripted-runtime", paid_requests=0, case_labels_changed=False)
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    # A scoped controller-only recording/verification selection. Tools and model
    # objects remain per Worker; the original builder is restored on every exit.
    runner.build_workflow = uniform_builder
    try:
        anchors = EvaluationConfig(splits=("development",),
            case_ids=("autograd_detach", "shape_transpose", "dtype_features"),
            task_limits=config.task_limits, job_limits=config.job_limits)
        for name, profile in (("complexity", config), ("level1-anchors", anchors)):
            report = runner.EvaluationRunner().run(root / name, profile, torch_python=args.torch_python)
            assert runner.recompute(root / name) == report
            (root / (name + "-summary.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(dict(subset=name, planned=report["planned"], recorded=report["recorded"])), flush=True)
    finally:
        runner.build_workflow = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
