import torch

def compute(output, target):
    return torch.nn.functional.mse_loss(output, target)
