import torch
import torch.nn as nn
from einops import rearrange

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()

        angle: torch.Tensor = torch.zeros(max_seq_len, d_k//2, device=device)
        frequency: torch.Tensor = torch.zeros(d_k//2,device=device)

        for pair_index in range(d_k//2):
            frequency[pair_index] = theta ** (-2 * pair_index/d_k)
            
            for position in range(max_seq_len):
                 angle[position, pair_index] = position * frequency[pair_index]

        sin_table = torch.sin(angle)
        cos_table = torch.cos(angle)
        self.register_buffer("sin",sin_table,persistent=False)
        self.register_buffer("cos",cos_table,persistent=False)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
            x_even, x_odd = x[...,0::2], x[...,1::2]
            new_odd = x_even * self.sin[token_positions] + x_odd * self.cos[token_positions]
            new_even = x_even *  self.cos[token_positions] - x_odd * self.sin[token_positions] 

            stacked = torch.stack(( new_even, new_odd ), dim=-1)
            return rearrange(stacked, "... sequence_length pairs component -> ... sequence_length (pairs component)")