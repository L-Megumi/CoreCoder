import torch

def pack(x, target, lengths):
    order = torch.argsort(lengths, descending=True)
    return x[order], target, order
