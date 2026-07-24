import torch
import torch.nn as nn
from einops import rearrange, einsum
import math

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device | None = None , dtype:torch.dtype | None = None ):
        super().__init__()
        empty_weights = torch.empty(out_features,in_features, device=device, dtype=dtype)
        sigma = math.sqrt(2/(out_features+in_features))
        self.W = nn.Parameter(nn.init.trunc_normal_(empty_weights, mean=0, std = sigma , a =-3*sigma, b=3*sigma))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wt = rearrange(self.W, "out_features in_features -> in_features out_features")
        return einsum(Wt, x, "in_features out_features, ... in_features -> ... out_features")