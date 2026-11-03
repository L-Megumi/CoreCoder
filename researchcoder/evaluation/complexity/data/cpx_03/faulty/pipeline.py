import torch

def prepare(x):
    scale = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / scale
