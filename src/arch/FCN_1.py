import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 1)

class FCN_1(torch.nn.Module):
    def __init__(self, width, seed=888):
        super(FCN_1, self).__init__()

        torch.manual_seed(seed)

        self.width = width

        self.fc1 = nn.Linear(784, width, bias=False)
        self.fc5 = nn.Linear(width, 1, bias=False)

        self.apply(_weights_init)

    def forward(self, x):
        x = x.view(-1, 784)

        x = 1 / math.sqrt(784) * self.fc1(x)
        x = F.relu(x)

        x = 1 / math.sqrt(self.width) * self.fc5(x)

        return x

  #试一层两千width