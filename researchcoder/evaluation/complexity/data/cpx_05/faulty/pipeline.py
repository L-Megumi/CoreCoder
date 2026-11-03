import torch

def prepare(tokens, lengths):
    mask = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
    return tokens, mask
