import numpy as np
import torch.nn as nn

class SingleLinear(nn.Module):
    def __init__(self):
        super(SingleLinear, self).__init__()
        self.linear = nn.Linear(1000, 4)

    def forward(self, x):
        output = self.linear(x)
        return output