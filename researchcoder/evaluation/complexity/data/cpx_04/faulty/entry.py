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
