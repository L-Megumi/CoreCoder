def compute(output, target, weights):
    per_sample = (output - target).square().mean(dim=-1)
    return (per_sample * weights).mean()
