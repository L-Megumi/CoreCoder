import torch

class Model(torch.nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.head = torch.nn.Linear(4, classes)

    def forward(self, x):
        return self.head(torch.tanh(self.encoder(x)))

def make_model(freeze_encoder=False, classes=2):
    model = Model(classes).double()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not freeze_encoder)
    return model
