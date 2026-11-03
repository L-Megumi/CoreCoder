import torch

def positive_loss(logits):
    return -torch.log(torch.sigmoid(logits))
