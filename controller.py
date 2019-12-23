import torch.nn as nn

class Controller(nn.Module):

    def __init__(self):
        super(Controller, self).__init__()

    def forward(self, x):
        return x



