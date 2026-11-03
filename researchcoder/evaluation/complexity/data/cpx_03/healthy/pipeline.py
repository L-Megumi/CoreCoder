import torch

def prepare(x):
    scale = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / scale.clamp_min(torch.finfo(x.dtype).eps)
