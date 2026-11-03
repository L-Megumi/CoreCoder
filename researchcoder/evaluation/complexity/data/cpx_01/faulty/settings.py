import torch

def optimizer(model, lr):
    return torch.optim.SGD(model.head.parameters(), lr=lr)
