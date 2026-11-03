def prepare_features(model, x):
    parameter = next(model.parameters())
    return x.to(dtype=parameter.dtype, device=parameter.device)
