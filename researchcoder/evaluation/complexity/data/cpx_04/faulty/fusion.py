import torch

def combine(model, image, text):
    image_features = torch.tanh(model.image(image))
    text_features = torch.tanh(model.text(text)).detach()
    return torch.cat((image_features, text_features), dim=-1)
