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
