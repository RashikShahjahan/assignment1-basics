import torch
import torch.nn as nn
from cs336_basics.model.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.model.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.model.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, theta:float |None=None, max_seq_len:int|None=None ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.ffn = PositionwiseFeedForward(d_model,d_ff)

    def forward(self, x: torch.Tensor,  token_positions: torch.Tensor | None=None) -> torch.Tensor:
        h = x+ self.attn(self.ln1(x), token_positions)
        return h+self.ffn(self.ln2(h))