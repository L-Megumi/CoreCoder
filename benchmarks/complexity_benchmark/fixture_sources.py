"""Reproducible CPU fixtures. Opaque IDs do not encode the repair location.

Build-time only: reference repairs below are never materialized into Agent input.
The dataset builder freezes these bytes before any engineering Agent trial.
"""

import textwrap


def code(value):
    return textwrap.dedent(value).lstrip()


NETWORK = code('''
    import torch

    class Model(torch.nn.Module):
        def __init__(self, classes=2):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 4)
            self.head = torch.nn.Linear(4, classes)

        def forward(self, x):
            return self.head(torch.tanh(self.encoder(x)))

    def make_model(freeze_encoder=False, classes=2):
        model = Model(classes).double()
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not freeze_encoder)
        return model
''')
SETTINGS = code('''
    import torch

    def optimizer(model, lr):
        return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
''')
OBJECTIVE = code('''
    import torch

    def compute(output, target):
        return torch.nn.functional.mse_loss(output, target)
''')
PIPELINE = code('''
    def prepare(x):
        return x
''')
ENTRY = code('''
    import network
    import pipeline
    import objective
    import settings

    def create_model(freeze_encoder=False):
        return network.make_model(freeze_encoder=freeze_encoder)

    def step(model, batch, lr=0.03):
        optimizer = settings.optimizer(model, lr)
        optimizer.zero_grad(set_to_none=True)
        features = pipeline.prepare(batch["x"])
        output = model(features)
        loss = objective.compute(output, batch["y"])
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.detach()), "output": output.detach()}
''')


def fixtures():
    """Six distinct programs, with executable partial states for each Level 3."""
    records = {}
    def add(identity, level, healthy, changes, goal, cause, concepts, *, partial=None, split="development"):
        faulty = dict(healthy, **changes)
        assert changes and all(faulty[p] != healthy[p] for p in changes)
        records[identity] = dict(level=level, healthy=healthy, faulty=faulty,
            writable=tuple(changes), partial=partial, goal=goal, cause=cause,
            concepts=concepts, split=split)

    healthy = dict(entry=ENTRY, network=NETWORK, pipeline=PIPELINE, objective=OBJECTIVE, settings=SETTINGS)
    healthy = {name + ".py": value for name, value in healthy.items()}
    add("cpx_01", 2, healthy,
        {"settings.py": SETTINGS.replace('[p for p in model.parameters() if p.requires_grad]', 'model.head.parameters()')},
        "Repair the documented CPU training step. Update every trainable model component; preserve the explicit encoder freeze option, caller inputs, and the loss/output API.",
        "The optimizer owns only head parameters; encoder gradients exist but no encoder update occurs. Correct the parameter collection while retaining intentional freezing.",
        (("optimizer", "parameter group"), ("encoder", "backbone")))

    paired_pipeline = code('''
        import torch

        def pack(x, target, lengths):
            order = torch.argsort(lengths, descending=True)
            return x[order], target[order], order
    ''')
    paired_entry = code('''
        import torch
        import network
        import pipeline
        import objective
        import settings

        def create_model(freeze_encoder=False):
            return network.make_model(freeze_encoder=freeze_encoder, classes=3)

        def step(model, batch, lr=0.03):
            optimizer = settings.optimizer(model, lr)
            optimizer.zero_grad(set_to_none=True)
            x, target, order = pipeline.pack(batch["x"], batch["y"], batch["lengths"])
            output = model(x)
            loss = objective.compute(output, target)
            loss.backward()
            optimizer.step()
            inverse = torch.argsort(order)
            return {"loss": float(loss.detach()), "output": output[inverse].detach(), "order": order}
    ''')
    paired = dict(healthy, **{"entry.py": paired_entry, "pipeline.py": paired_pipeline,
        "objective.py": OBJECTIVE.replace("mse_loss", "cross_entropy")})
    add("cpx_02", 2, paired,
        {"pipeline.py": paired_pipeline.replace("target[order]", "target")},
        "Repair paired-batch training. Packing must sort descending lengths while preserving sample/label pairing; return predictions in original sample order and keep caller inputs unchanged.",
        "Packing permutes feature rows but leaves targets in their original order; loss gradients then associate predictions with different samples. Apply the same ordering to paired data.",
        (("target", "label"), ("permut", "sort", "order")), split="validation")

    normalize = code('''
        import torch

        def prepare(x):
            scale = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
            return x / scale.clamp_min(torch.finfo(x.dtype).eps)
    ''')
    add("cpx_03", 2, dict(healthy, **{"pipeline.py": normalize}),
        {"pipeline.py": normalize.replace("scale.clamp_min(torch.finfo(x.dtype).eps)", "scale")},
        "Repair stable per-row feature normalization and training. Nonzero rows retain L2 normalization, zero feature rows remain zero, all gradients/updates are finite, and caller inputs stay unchanged.",
        "Zero feature vectors are divided by their zero norm in preprocessing; nonfinite values propagate through the network and loss. Guard the normalization denominator, not merely the final loss.",
        (("zero", "norm"), ("normaliz", "denominator")))

    dual_network = code('''
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.image = torch.nn.Linear(3, 3)
                self.text = torch.nn.Linear(2, 3)
                self.head = torch.nn.Linear(6, 2)

            def forward(self, image, text):
                return self.head(torch.cat((torch.tanh(self.image(image)), torch.tanh(self.text(text))), dim=-1))

        def make_model(freeze_text=False):
            model = Model().double()
            for parameter in model.text.parameters():
                parameter.requires_grad_(not freeze_text)
            return model
    ''')
    fusion = code('''
        import torch

        def combine(model, image, text):
            image_features = torch.tanh(model.image(image))
            text_features = torch.tanh(model.text(text))
            return torch.cat((image_features, text_features), dim=-1)
    ''')
    dual_entry = code('''
        import network
        import pipeline
        import fusion
        import objective
        import settings

        def create_model(freeze_text=False):
            return network.make_model(freeze_text=freeze_text)

        def step(model, batch, lr=0.03):
            optimizer = settings.optimizer(model, lr)
            optimizer.zero_grad(set_to_none=True)
            image = pipeline.prepare(batch["image"])
            text = pipeline.prepare(batch["text"])
            output = model.head(fusion.combine(model, image, text))
            loss = objective.compute(output, batch["y"])
            loss.backward()
            optimizer.step()
            return {"loss": float(loss.detach()), "output": output.detach()}
    ''')
    dual = dict(healthy, **{"network.py": dual_network, "entry.py": dual_entry, "fusion.py": fusion})
    dual_fault = {"fusion.py": fusion.replace("text_features = torch.tanh(model.text(text))", "text_features = torch.tanh(model.text(text)).detach()"),
        "settings.py": SETTINGS.replace('[p for p in model.parameters() if p.requires_grad]', 'list(model.image.parameters()) + list(model.head.parameters())')}
    add("cpx_04", 3, dual, dual_fault,
        "Repair joint multimodal training of image, text and head modules. Every explicitly trainable component must receive gradients and optimizer updates; preserve the intentional text-freeze option and input/output API.",
        "Text features are detached in fusion and text parameters are also absent from optimizer groups. Restoring gradient flow alone leaves updates missing; both independent obligations must be repaired.",
        (("detach", "gradient"), ("optimizer", "parameter")), partial={"fusion.py": fusion})

    pooling = code('''
        def reduce(features, mask):
            weight = mask.to(features.dtype).unsqueeze(-1)
            return (features * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
    ''')
    weighted = code('''
        def compute(output, target, weights):
            per_sample = (output - target).square().mean(dim=-1)
            return (per_sample * weights).sum() / weights.sum()
    ''')
    masked_pipeline = code('''
        import torch

        def prepare(tokens, lengths):
            mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
            return tokens, mask
    ''')
    masked_entry = code('''
        import torch
        import network
        import pipeline
        import pooling
        import objective
        import settings

        def create_model(freeze_encoder=False):
            return network.make_model(freeze_encoder=freeze_encoder)

        def step(model, batch, lr=0.03):
            optimizer = settings.optimizer(model, lr)
            optimizer.zero_grad(set_to_none=True)
            tokens, mask = pipeline.prepare(batch["tokens"], batch["lengths"])
            features = torch.tanh(model.encoder(tokens))
            output = model.head(pooling.reduce(features, mask))
            loss = objective.compute(output, batch["y"], batch["weights"])
            loss.backward()
            optimizer.step()
            return {"loss": float(loss.detach()), "output": output.detach()}
    ''')
    masked = dict(healthy, **{"entry.py": masked_entry, "pipeline.py": masked_pipeline,
        "pooling.py": pooling, "objective.py": weighted})
    add("cpx_05", 3, masked,
        {"pooling.py": code('''
            def reduce(features, mask):
                return features.mean(dim=1)
        '''),
         "objective.py": weighted.replace('(per_sample * weights).sum() / weights.sum()', '(per_sample * weights).mean()')},
        "Repair masked, weighted sequence regression. Padding must not affect predictions; sample weights are normalized by total weight so uniform weight rescaling cannot change gradients. Preserve input tensors and all trainable updates.",
        "Pooling includes padding in its denominator and the weighted objective divides by batch size rather than total sample weight. Fixing pooling alone does not restore weight-scale invariant gradients.",
        (("mask", "padding"), ("weight", "normaliz")), partial={"pooling.py": pooling})

    norm_network = code('''
        import torch

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.BatchNorm1d(3)
                self.head = torch.nn.Linear(3, 2)

            def forward(self, x):
                return self.head(self.norm(x))

        def make_model():
            return Model().double()
    ''')
    evaluation = code('''
        import torch

        def probe(model, x):
            previous = model.training
            model.eval()
            try:
                with torch.no_grad():
                    score = float(model(x).square().mean())
                return score
            finally:
                model.train(previous)
    ''')
    accumulation = code('''
        import network
        import evaluation
        import objective
        import settings

        def create_model():
            return network.make_model()

        def step(model, batch, lr=0.03):
            optimizer = settings.optimizer(model, lr)
            score = evaluation.probe(model, batch["validation"])
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            count = len(batch["micro_x"])
            for x, y in zip(batch["micro_x"], batch["micro_y"]):
                loss = objective.compute(model(x), y) / count
                loss.backward()
                loss_value += float(loss.detach())
            optimizer.step()
            return {"loss": loss_value, "score": score}
    ''')
    staged = {"entry.py": accumulation, "network.py": norm_network, "evaluation.py": evaluation,
              "objective.py": OBJECTIVE, "settings.py": SETTINGS}
    bad_eval = evaluation.replace('    previous = model.training\n', '').replace('    try:\n', '').replace('    finally:\n        model.train(previous)\n', '')
    # Remove exactly the try suite indentation, retaining the no_grad suite.
    bad_eval = bad_eval.replace('        with torch.no_grad():', '    with torch.no_grad():').replace('            score =', '        score =').replace('        return score', '    return score')
    bad_accum = accumulation.replace('    optimizer.zero_grad(set_to_none=True)\n', '').replace('    for x, y in zip(batch["micro_x"], batch["micro_y"]):\n', '    for x, y in zip(batch["micro_x"], batch["micro_y"]):\n        optimizer.zero_grad(set_to_none=True)\n')
    add("cpx_06", 3, staged, {"evaluation.py": bad_eval, "entry.py": bad_accum},
        "Repair validation followed by microbatch training. Validation must preserve caller model mode and buffers; accumulated gradients must equal the mean of all microbatch losses, with one optimizer update. Preserve inputs and eval-mode behavior.",
        "Validation leaves the model in eval mode and gradient buffers are cleared inside each microbatch. Restore mode reliably and move zero_grad outside accumulation; fixing only mode leaves earlier microbatch gradients discarded.",
        (("mode", "eval"), ("accumul", "zero_grad")), partial={"evaluation.py": evaluation}, split="holdout")
    return records
