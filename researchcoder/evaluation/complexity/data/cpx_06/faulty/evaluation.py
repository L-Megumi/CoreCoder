import torch

def probe(model, x):
    model.eval()
    with torch.no_grad():
        score = float(model(x).square().mean())
    return score
