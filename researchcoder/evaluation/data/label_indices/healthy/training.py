import torch

def classification_loss(logits, labels):
    target = labels.long() if labels.ndim == 1 else labels
    return torch.nn.functional.cross_entropy(logits, target)
