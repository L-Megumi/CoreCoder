"""Trusted external CPU checks. No ResearchCoder/runtime dependency in Torch env.

The candidate is restricted training code, not an adversarial execution sandbox.
Answers, seeds and reference calculations remain outside the editable workspace.
"""

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys


def audit(project):
    allowed = {"torch", "math", "ops"}
    forbidden = {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "input",
        "breakpoint",
    }
    paths = list(project.glob("*.py"))
    if not paths or len(paths) > 8 or any(p.is_symlink() for p in paths):
        raise ValueError("Invalid candidate files")
    for path in paths:
        if path.stat().st_size > 100000:
            raise ValueError("Candidate too large")
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                n.name.split(".")[0] not in allowed for n in node.names
            ):
                raise ValueError("Unsupported benchmark import")
            if isinstance(node, ast.ImportFrom) and (
                node.level or (node.module or "").split(".")[0] not in allowed
            ):
                raise ValueError("Unsupported benchmark import")
            if isinstance(node, ast.Name) and node.id in forbidden:
                raise ValueError(
                    "Reflective/file execution not allowed in benchmark candidate"
                )
            if isinstance(node, ast.Attribute) and (
                node.attr.startswith("__")
                or node.attr
                in {
                    "save",
                    "load",
                    "set_default_dtype",
                    "set_default_device",
                    "set_num_threads",
                    "manual_seed",
                }
            ):
                raise ValueError("Global runtime mutation is outside the task")
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    while isinstance(target, ast.Attribute):
                        target = target.value
                    if isinstance(target, ast.Name) and target.id == "torch":
                        raise ValueError("Cannot replace trusted Torch implementation")


def load(project):
    sys.path.insert(0, str(project))
    sys.modules.pop("ops", None)
    spec = importlib.util.spec_from_file_location(
        "candidate_training", project / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(project, case_id, profile):
    import torch
    import torch.nn.functional as F

    project = Path(project).resolve()
    audit(project)
    torch.set_num_threads(1)
    if case_id == "autograd_detach":
        spec = importlib.util.spec_from_file_location(
            "legacy_oracle", Path(__file__).with_name("data") / "support/legacy.py"
        )
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        return legacy.evaluate(
            project,
            profile="frozen" if profile == "regression" else "joint",
            seed_group="discovery"
            if profile in ("discovery", "regression")
            else "holdout",
        )
    module = load(project)
    seeds = (
        (7, 19)
        if profile == "discovery"
        else (3, 11)
        if profile == "regression"
        else (142, 314)
    )
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        regression = profile == "regression"
        if case_id == "shape_transpose":
            n = 3 if regression else 5
            model = torch.nn.Linear(4, 3).double()
            x = torch.randn(n, 4, dtype=torch.float64)
            before = x.clone()
            state = {name: p.detach().clone() for name, p in model.named_parameters()}
            reference = model(x)
            actual = module.predict(model, x)
            assert actual.shape == (n, 3), "Batch/class axes changed"
            if not regression:
                torch.testing.assert_close(actual, reference)
            assert all(
                torch.equal(p, state[name]) for name, p in model.named_parameters()
            ), "Forward mutated model state"
            actual.square().mean().backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            assert torch.equal(x, before)
        elif case_id == "dtype_features":
            dtype = torch.float32 if seed in (7, 142) else torch.float64
            model = torch.nn.Linear(4, 2).to(dtype=dtype)
            state = {name: p.detach().clone() for name, p in model.named_parameters()}
            x = torch.arange(20).reshape(5, 4) if regression else torch.randn(5, 4)
            before = x.clone()
            actual = module.prepare_features(model, x)
            assert actual.shape == x.shape and torch.equal(x, before)
            assert all(
                torch.equal(p, state[name]) for name, p in model.named_parameters()
            )
            if not regression:
                assert actual.dtype == dtype and next(model.parameters()).dtype == dtype
                torch.testing.assert_close(actual, x.to(dtype=dtype))
                model(actual).sum().backward()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
        elif case_id == "nonfinite_logsigmoid":
            x = torch.tensor(
                [-1.0, 0.0, 1.0] if regression else [-1000.0, -5.0, 0.0, 5.0, 1000.0],
                dtype=torch.float64,
                requires_grad=True,
            )
            actual = module.positive_loss(x)
            reference = F.softplus(-x)
            assert actual.shape == x.shape and torch.isfinite(actual).all(), (
                "Nonfinite training loss"
            )
            torch.testing.assert_close(actual, reference)
            grad = torch.autograd.grad(actual.sum(), x)[0]
            expected = -torch.sigmoid(-x)
            torch.testing.assert_close(grad, expected)
            assert torch.isfinite(grad).all()
        elif case_id == "label_indices":
            logits = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
            labels = (
                torch.softmax(torch.randn(5, 3, dtype=torch.float64), -1)
                if regression
                else torch.tensor([0, 2, 1, 0, 2])
            )
            before = labels.clone()
            before_logits = logits.detach().clone()
            reference = F.cross_entropy(logits, labels)
            actual = module.classification_loss(logits, labels)
            assert torch.equal(logits, before_logits), "Loss mutated logits"
            torch.testing.assert_close(actual, reference)
            torch.testing.assert_close(
                torch.autograd.grad(actual, logits)[0],
                torch.autograd.grad(reference, logits)[0],
            )
            assert torch.equal(labels, before)
        elif case_id == "optimizer_omission":

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = torch.nn.Linear(4, 6)
                    self.head = torch.nn.Linear(6, 2)

                def forward(self, x):
                    return self.head(torch.tanh(self.backbone(x)))

            model = Model().double()
            parameters = dict(model.named_parameters())
            before = {n: p.detach().clone() for n, p in parameters.items()}
            optimizer = module.build_optimizer(model)
            assert isinstance(optimizer, torch.optim.SGD)
            assert all(g["lr"] == 0.05 for g in optimizer.param_groups), (
                "Changed prescribed learning rate"
            )
            group = [p for g in optimizer.param_groups for p in g["params"]]
            assert len(group) == len({id(p) for p in group})
            if not regression:
                assert {id(p) for p in group} == {id(p) for p in parameters.values()}, (
                    "Trainable parameters omitted"
                )
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(
                model(torch.randn(5, 4, dtype=torch.float64)),
                torch.randn(5, 2, dtype=torch.float64),
            )
            loss.backward()
            optimizer.step()
            assert all(
                p.requires_grad and dict(model.named_parameters())[n] is p
                for n, p in parameters.items()
            )
            for n, p in parameters.items():
                if not regression or n.startswith("head."):
                    assert (
                        p.grad is not None
                        and torch.isfinite(p.grad).all()
                        and float((p - before[n]).norm()) > 0
                    )
                    torch.testing.assert_close(p.detach() - before[n], -0.05 * p.grad)
        elif case_id == "normalization_scale":
            x = torch.randn(5, 3, dtype=torch.float64)
            before = x.clone()
            mean = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
            std = (
                torch.ones(3, dtype=torch.float64)
                if regression
                else torch.tensor([0.2, 2.0, 3.0], dtype=torch.float64)
            )
            before_mean, before_std = mean.clone(), std.clone()
            reference = (x - mean) / std
            actual = module.normalize(x, mean, std)
            torch.testing.assert_close(actual, reference)
            assert (
                torch.equal(x, before)
                and torch.equal(mean, before_mean)
                and torch.equal(std, before_std)
            )
            torch.testing.assert_close(module.normalize(x + mean, mean, std), x / std)
        elif case_id == "crossfile_batch":
            model = torch.nn.Linear(3, 1).double()
            n = 3 if regression else 1
            x = torch.randn(n, 3, dtype=torch.float64)
            before = x.clone()
            state = {name: p.detach().clone() for name, p in model.named_parameters()}
            reference = model(x).reshape(n)
            actual = module.predict(model, x)
            assert actual.shape == (n,), "Batch axis disappeared through helper"
            torch.testing.assert_close(actual, reference)
            assert torch.equal(x, before) and all(
                torch.equal(p, state[name]) for name, p in model.named_parameters()
            )
            actual.square().mean().backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        else:
            raise ValueError("Unknown fixed oracle")
        rows.append(dict(seed=seed, passed=True))
    return dict(passed=True, rows=rows, profile=profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("profile", choices=("discovery", "holdout", "regression"))
    args = parser.parse_args()
    try:
        result = evaluate(args.project, args.case_id, args.profile)
    except Exception as exc:
        # Scores are private evaluator outputs, never automatically agent context.
        result = dict(passed=False, failure_kind=type(exc).__name__)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
