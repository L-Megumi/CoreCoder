import torch

def positive_loss(logits):
    return torch.nn.functional.softplus(-logits)
