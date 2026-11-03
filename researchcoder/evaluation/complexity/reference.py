"""Evaluator-only mathematical reference, independent of healthy fixture imports.

No Agent/runtime/provider imports. Runs in the existing CPU PyTorch interpreter.
Candidate architecture files are immutable and hash-checked by oracle.py.
"""

import importlib.util
from pathlib import Path
import sys


MODULES = ("network", "pipeline", "objective", "settings", "fusion", "pooling", "evaluation")


def load_candidate(project):
    project = Path(project).resolve()
    for name in MODULES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location("candidate_entry", project / "entry.py")
    entry = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(entry)
    finally:
        sys.path.pop(0)
    return entry


def close(a, b):
    import torch
    try:
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
        return True
    except (AssertionError, TypeError, RuntimeError):
        return False


def cloned(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, list):
        return [cloned(v) for v in value]
    return value


def inputs_equal(a, b):
    import torch
    return set(a) == set(b) and all(
        len(value) == len(b[key]) and all(torch.equal(x, y) for x, y in zip(value, b[key]))
        if isinstance(value, list) else torch.equal(value, b[key])
        for key, value in a.items()
    )


def architecture(identity, options):
    """Independent architecture/initialization: never clone a candidate's freeze hack."""
    import torch

    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.head = torch.nn.Linear(4, 3 if identity == "cpx_02" else 2)

        def forward(self, x):
            return self.head(torch.tanh(self.encoder(x)))

    class Dual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image = torch.nn.Linear(3, 3)
            self.text = torch.nn.Linear(2, 3)
            self.head = torch.nn.Linear(6, 2)

        def forward(self, image, text):
            return self.head(torch.cat((torch.tanh(self.image(image)), torch.tanh(self.text(text))), dim=-1))

    class Normalized(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm1d(3)
            self.head = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.head(self.norm(x))

    model = (Dual() if identity == "cpx_04" else Normalized() if identity == "cpx_06" else Plain()).double()
    freeze = "text" if options.get("freeze_text") else "encoder" if options.get("freeze_encoder") else None
    if freeze is not None:
        for name, parameter in model.named_parameters():
            if name.startswith(freeze + "."):
                parameter.requires_grad_(False)
    return model


def prepare(identity, seed, regression=False):
    import torch
    torch.manual_seed(seed)
    n = 4 + seed % 3
    if identity == "cpx_04":
        batch = dict(image=torch.randn(n, 3, dtype=torch.float64),
                     text=torch.randn(n, 2, dtype=torch.float64), y=torch.randn(n, 2, dtype=torch.float64))
        options = dict(freeze_text=regression)
    elif identity == "cpx_05":
        t = 5 + seed % 2
        lengths = torch.full((n,), t, dtype=torch.long) if regression else torch.arange(n) % (t - 1) + 1
        batch = dict(tokens=torch.randn(n, t, 3, dtype=torch.float64), lengths=lengths,
                     weights=torch.ones(n, dtype=torch.float64) if regression else torch.arange(n, dtype=torch.float64) + 1,
                     y=torch.randn(n, 2, dtype=torch.float64))
        options = {}
    elif identity == "cpx_06":
        count = 1 if regression else 2
        batch = dict(micro_x=[torch.randn(n, 3, dtype=torch.float64) + i for i in range(count)],
                     micro_y=[torch.randn(n, 2, dtype=torch.float64) for _ in range(count)],
                     validation=torch.randn(n + 1, 3, dtype=torch.float64))
        options = {}
    else:
        x = torch.randn(n, 3, dtype=torch.float64)
        if identity == "cpx_03":
            if not regression:
                x[0] = 0
            x.requires_grad_(True)
        batch = dict(x=x, y=torch.randn(n, 2, dtype=torch.float64))
        options = dict(freeze_encoder=regression and identity == "cpx_01")
        if identity == "cpx_02":
            batch["y"] = torch.arange(n) % 3
            batch["lengths"] = torch.arange(n, 0, -1) if regression else torch.arange(n).roll(1) + 1
    return batch, options


def expected_step(model, batch, identity, lr):
    import torch
    import torch.nn.functional as F
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    optimizer.zero_grad(set_to_none=True)
    if identity == "cpx_06":
        previous = model.training
        model.eval()
        with torch.no_grad():
            score = float(model(batch["validation"]).square().mean())
        model.train(previous)
        value = 0.0
        for x, target in zip(batch["micro_x"], batch["micro_y"]):
            loss = F.mse_loss(model(x), target) / len(batch["micro_x"])
            loss.backward()
            value += float(loss.detach())
        optimizer.step()
        return dict(loss=value, score=score)
    if identity == "cpx_04":
        output = model(batch["image"], batch["text"])
        loss = F.mse_loss(output, batch["y"])
    elif identity == "cpx_05":
        features = torch.tanh(model.encoder(batch["tokens"]))
        mask = torch.arange(features.shape[1]).unsqueeze(0) < batch["lengths"].unsqueeze(1)
        masked = features * mask.unsqueeze(-1)
        output = model.head(masked.sum(dim=1) / batch["lengths"].unsqueeze(-1))
        loss = ((output - batch["y"]).square().mean(dim=-1) * batch["weights"]).sum() / batch["weights"].sum()
    else:
        features = F.normalize(batch["x"], dim=-1, eps=torch.finfo(batch["x"].dtype).eps) if identity == "cpx_03" else batch["x"]
        output = model(features)
        loss = F.cross_entropy(output, batch["y"]) if identity == "cpx_02" else F.mse_loss(output, batch["y"])
    loss.backward()
    optimizer.step()
    result = dict(loss=float(loss.detach()), output=output.detach())
    if identity == "cpx_02":
        result["order"] = torch.argsort(batch["lengths"], descending=True)
    return result


def run_once(entry, identity, seed, regression=False):
    import torch
    batch, options = prepare(identity, seed, regression)
    random_state = torch.random.get_rng_state()
    reference = architecture(identity, options)
    torch.random.set_rng_state(random_state)
    candidate = entry.create_model(**options)
    if identity == "cpx_06" and regression:
        candidate.eval()
        reference.eval()
    initial_valid = set(candidate.state_dict()) == set(reference.state_dict()) and all(
        close(candidate.state_dict()[key], value) for key, value in reference.state_dict().items()
    ) and all(p.requires_grad == dict(reference.named_parameters())[name].requires_grad for name, p in candidate.named_parameters())
    before = {k: p.detach().clone() for k, p in candidate.named_parameters()}
    original = {k: cloned(v) for k, v in batch.items()}
    other = {k: cloned(v) for k, v in batch.items()}
    lr = .02 + (seed % 3) * .01
    expected = expected_step(reference, other, identity, lr)
    actual = entry.step(candidate, batch, lr=lr)
    checks = dict(architecture_and_freeze_contract=initial_valid, output_contract=isinstance(actual, dict) and set(actual) == set(expected))
    checks["loss_value"] = checks["output_contract"] and abs(actual["loss"] - expected["loss"]) <= 1e-7
    if "output" in expected:
        checks["predictions"] = checks["output_contract"] and close(actual["output"], expected["output"])
    if "score" in expected:
        checks["validation_score"] = checks["output_contract"] and abs(actual["score"] - expected["score"]) <= 1e-7
    checks["input_preserved"] = inputs_equal(batch, original)
    checks["training_mode"] = candidate.training == reference.training
    checks["trainability_preserved"] = all(p.requires_grad == dict(reference.named_parameters())[name].requires_grad for name, p in candidate.named_parameters())
    states = candidate.state_dict()
    checks["parameter_and_buffer_values"] = all(close(states[k], value) for k, value in reference.state_dict().items())
    rows = {}
    for name, parameter in candidate.named_parameters():
        target = dict(reference.named_parameters())[name]
        grad_ok = parameter.grad is None if target.grad is None else parameter.grad is not None and close(parameter.grad, target.grad)
        changed = float((parameter.detach() - before[name]).norm())
        finite = bool(torch.isfinite(parameter).all()) and (parameter.grad is None or bool(torch.isfinite(parameter.grad).all()))
        expected_changed = float((target.detach() - before[name]).norm())
        rows[name] = dict(grad_present=parameter.grad is not None,
                          grad_norm=None if parameter.grad is None else float(parameter.grad.norm()),
                          update_norm=changed, finite=finite)
        checks["gradient:" + name] = grad_ok and finite
        checks["update:" + name] = close(parameter.detach(), target.detach()) and (changed > 0 if expected_changed > 1e-12 else changed <= 1e-12)
    if identity in ("cpx_01", "cpx_04"):
        optimizer = entry.settings.optimizer(candidate, lr)
        supplied = [id(p) for group in optimizer.param_groups for p in group["params"]]
        required = {id(p) for p in candidate.parameters() if p.requires_grad}
        checks["optimizer_parameter_ownership"] = set(supplied) == required and len(supplied) == len(required)
    if identity == "cpx_04":
        # Public component API must work independently, not only via an inline entry workaround.
        probe = architecture(identity, options)
        combined = entry.fusion.combine(probe, batch["image"], batch["text"])
        probe.head(combined).square().mean().backward()
        checks["fusion_gradient_interface"] = all(
            p.grad is not None if p.requires_grad else p.grad is None for p in probe.parameters())
    if identity == "cpx_02":
        packed, target, order = entry.pipeline.pack(batch["x"], batch["y"], batch["lengths"])
        expected_order = torch.argsort(batch["lengths"], descending=True)
        checks["sample_pairing"] = torch.equal(order, expected_order) and close(packed, batch["x"][expected_order]) and torch.equal(target, batch["y"][expected_order])
        checks["returned_order"] = torch.equal(actual["order"], expected_order)
    if identity == "cpx_03":
        values = entry.pipeline.prepare(batch["x"])
        normalized = torch.nn.functional.normalize(batch["x"], dim=-1, eps=torch.finfo(batch["x"].dtype).eps)
        checks["feature_normalization"] = close(values, normalized)
        checks["finite_input_gradient"] = batch["x"].grad is not None and bool(torch.isfinite(batch["x"].grad).all())
    if identity == "cpx_05" and not regression:
        _, mask = entry.pipeline.prepare(batch["tokens"], batch["lengths"])
        features = torch.tanh(candidate.encoder(batch["tokens"]))
        value = entry.pooling.reduce(features, mask)
        extended = torch.cat((features, torch.randn(features.shape[0], 2, features.shape[2], dtype=features.dtype)), dim=1)
        extended_mask = torch.cat((mask, torch.zeros(mask.shape[0], 2, dtype=torch.bool)), dim=1)
        checks["padding_invariance"] = close(value, entry.pooling.reduce(extended, extended_mask))
        output = candidate.head(value).detach()
        checks["weight_scale_invariance"] = close(entry.objective.compute(output, batch["y"], batch["weights"]), entry.objective.compute(output, batch["y"], batch["weights"] * 7))
    if identity == "cpx_06":
        # Mode restoration must also hold when validation itself raises.
        prior = candidate.training
        try:
            entry.evaluation.probe(candidate, torch.zeros(3, 99, dtype=torch.float64))
        except (RuntimeError, ValueError):
            pass
        checks["mode_on_validation_error"] = candidate.training == prior
    telemetry = dict(loss=float(actual.get("loss", float("nan"))), mode_after=bool(candidate.training), parameters=rows,
                     passed=sum(bool(v) for v in checks.values()), total=len(checks))
    return checks, telemetry


def evaluate(project, identity, profile):
    import torch
    torch.set_num_threads(1)
    entry = load_candidate(project)
    seeds = (5, 17) if profile == "discovery" else (11, 23) if profile == "regression" else (142, 317, 509)
    rows = []
    for seed in seeds:
        checks, _ = run_once(entry, identity, seed, profile == "regression")
        # Never publish hidden random inputs, expected tensors or reference code.
        rows.append(dict(checks=checks, passed=all(checks.values())))
    return dict(passed=all(r["passed"] for r in rows), rows=rows)
