import torch
import torch.nn as nn
from cs336_basics.model.linear import Linear

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int, device:torch.device | None = None , dtype:torch.dtype | None = None ):
        super().__init__()
        self.W1 =Linear(d_model, d_ff, device, dtype)
        self.W2 =Linear(d_ff, d_model,device, dtype)
        self.W3 = Linear(d_model, d_ff,device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1 = self.W1(x)
        layer2 = self.silu(layer1)
        layer3 = self.W3(x)
        layer4 = layer2*layer3
        return self.W2(layer4)


    def silu(self, x:torch.Tensor):
        return x*torch.sigmoid(x)