import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image = torch.nn.Linear(3, 3)
        self.text = torch.nn.Linear(2, 3)
        self.head = torch.nn.Linear(6, 2)

    def forward(self, image, text):
        return self.head(torch.cat((torch.tanh(self.image(image)), torch.tanh(self.text(text))), dim=-1))

def make_model(freeze_text=False):
    model = Model().double()
    for parameter in model.text.parameters():
        parameter.requires_grad_(not freeze_text)
    return model
