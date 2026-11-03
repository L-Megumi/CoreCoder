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
