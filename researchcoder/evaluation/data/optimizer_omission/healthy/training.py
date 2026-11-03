import torch

def build_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.05)
