import torch

def prepare_features(model, x):
    return x.to(torch.int64)
