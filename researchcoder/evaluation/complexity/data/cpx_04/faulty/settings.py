import torch

def optimizer(model, lr):
    return torch.optim.SGD(list(model.image.parameters()) + list(model.head.parameters()), lr=lr)
