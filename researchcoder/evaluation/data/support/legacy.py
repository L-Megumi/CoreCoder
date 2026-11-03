"""External CPU oracle. Keep this file and its manifest outside the editable project."""

import argparse
import importlib.util
import json
import math
from pathlib import Path


def evaluate(project, *, profile="joint", seed_group="discovery"):
    import torch

    torch.set_num_threads(1)
    manifest = json.loads(Path(__file__).with_name("case.json").read_text(encoding="utf-8"))
    expected = set(manifest["joint_trainable" if profile == "joint" else "frozen_trainable"])
    source = Path(project).resolve() / "training.py"
    spec = importlib.util.spec_from_file_location("candidate_training", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    failures = []
    for seed in manifest[seed_group + "_seeds"]:
        torch.manual_seed(seed)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh())
                self.head = torch.nn.Linear(8, 2)

            def forward(self, x):
                return self.head(self.backbone(x))

        model = Model().double().cpu()
        if profile == "frozen":
            model.backbone.requires_grad_(False)
        parameters = dict(model.named_parameters())
        optimizer = torch.optim.SGD([p for name, p in parameters.items() if name in expected], lr=manifest["learning_rate"])
        x = torch.randn(6, 4, dtype=torch.float64)
        y = torch.randn(6, 2, dtype=torch.float64)
        for step in range(manifest["steps"]):
            before = {name: p.detach().clone() for name, p in parameters.items()}
            loss = module.training_step(model, x, y, optimizer)
            actual_parameters = dict(model.named_parameters())
            if set(actual_parameters) != set(parameters) or any(actual_parameters[n] is not p for n, p in parameters.items()):
                failures.append("Model parameters were replaced")
            actual_trainable = {name for name, p in parameters.items() if p.requires_grad}
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if actual_trainable != expected or optimizer_ids != {id(parameters[n]) for n in expected}:
                failures.append("Expected trainable set or optimizer parameter groups changed")
            if model(x).shape != (6, 2) or not torch.isfinite(model(x)).all() or not math.isfinite(loss):
                failures.append("Invalid forward shape or nonfinite output/loss")
            measurements = {}
            for name, p in parameters.items():
                grad = p.grad
                gradient_ok = grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 0
                update = float((p.detach() - before[name]).norm())
                measurements[name] = {"grad_present": grad is not None, "gradient_ok": gradient_ok, "update_norm": update}
                if name in expected and (not gradient_ok or not math.isfinite(update) or update <= 0):
                    failures.append("Expected gradient and parameter update: " + name)
                if name not in expected and (grad is not None or update != 0):
                    failures.append("Intentionally frozen parameter changed: " + name)
            rows.append({"seed": seed, "step": step, "loss": loss, "parameters": measurements})
    return {
        "profile": profile,
        "seed_group": seed_group,
        "passed": not failures,
        "failures": sorted(set(failures)),
        "rows": rows,
        "scope": "Independent gradient/update checks, not model diagnosis accuracy or an OS sandbox.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--profile", choices=("joint", "frozen"), default="joint")
    parser.add_argument("--seed-group", choices=("discovery", "holdout"), default="holdout")
    args = parser.parse_args(argv)
    result = evaluate(args.project, profile=args.profile, seed_group=args.seed_group)
    if result["failures"]:
        print(
            "WARNING: training.py: expected joint-training gradients/parameter updates missing; inspect the training_step graph.",
            flush=True,
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
