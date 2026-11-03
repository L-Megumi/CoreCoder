import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(3)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.head(self.norm(x))

def make_model():
    return Model().double()
