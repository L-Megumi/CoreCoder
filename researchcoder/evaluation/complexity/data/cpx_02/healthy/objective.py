import torch

def compute(output, target):
    return torch.nn.functional.cross_entropy(output, target)
