import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device | None = None , dtype:torch.dtype | None = None ):
        super().__init__()
        empty_weights = torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype)
        self.W = nn.Parameter(nn.init.trunc_normal_(empty_weights, mean=0, std = 1 , a =-3, b=3))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]