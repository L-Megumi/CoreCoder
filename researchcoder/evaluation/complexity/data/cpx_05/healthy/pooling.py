def reduce(features, mask):
    weight = mask.to(features.dtype).unsqueeze(-1)
    return (features * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
