"""Joint training: both model components must learn."""


def training_step(model, x, y, optimizer):
    import torch

    optimizer.zero_grad(set_to_none=True)
    features = model.backbone(x).detach()
    predictions = model.head(features)
    loss = torch.nn.functional.mse_loss(predictions, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
