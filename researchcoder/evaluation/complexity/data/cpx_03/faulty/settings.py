import torch

def optimizer(model, lr):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
