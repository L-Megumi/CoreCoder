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
