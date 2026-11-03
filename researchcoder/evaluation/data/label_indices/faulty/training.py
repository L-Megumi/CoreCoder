import torch

def classification_loss(logits, labels):
    target = labels.float()
    return torch.nn.functional.cross_entropy(logits, target)
