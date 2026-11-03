import network
import evaluation
import objective
import settings

def create_model():
    return network.make_model()

def step(model, batch, lr=0.03):
    optimizer = settings.optimizer(model, lr)
    score = evaluation.probe(model, batch["validation"])
    loss_value = 0.0
    count = len(batch["micro_x"])
    for x, y in zip(batch["micro_x"], batch["micro_y"]):
        optimizer.zero_grad(set_to_none=True)
        loss = objective.compute(model(x), y) / count
        loss.backward()
        loss_value += float(loss.detach())
    optimizer.step()
    return {"loss": loss_value, "score": score}
