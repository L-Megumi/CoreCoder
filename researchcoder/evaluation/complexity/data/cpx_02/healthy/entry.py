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
