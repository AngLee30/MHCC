import torch
import torch.nn as nn
from collections import OrderedDict

class WeightedFeatureCombiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.tensor([0, 0, 0], dtype=torch.float32))

    def forward(self, f1, f2, f3):
        weights = torch.softmax(self.raw_weights, dim=0)
        return weights[0] * f1 + weights[1] * f2 + weights[2] * f3


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, d1, d2, d3):
        super().__init__()
        self.MLP = nn.Sequential(
            OrderedDict([
                ("c_fc", nn.Linear(d1, d2)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d2, d3))
            ])
        )
    
    def forward(self, x):
        return(self.MLP(x))