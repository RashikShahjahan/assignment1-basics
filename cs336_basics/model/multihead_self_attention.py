import torch
import torch.nn as nn
from cs336_basics.model.linear import Linear
from cs336_basics.model.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.model.rope import RotaryPositionalEmbedding


from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta:float |None=None, max_seq_len:int|None=None ):
        super().__init__()
        self.WQ = Linear(d_model, d_model)
        self.WK =Linear(d_model, d_model)
        self.WV = Linear(d_model, d_model)
        self.WO = Linear(d_model, d_model)
        self.num_heads = num_heads
        if theta is not None and max_seq_len is not None:
            head_dim = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(theta, head_dim, max_seq_len)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        Q = rearrange(self.WQ(x), "... q (num_heads head_dim) -> ... num_heads q head_dim", num_heads=self.num_heads)
        K = rearrange(self.WK(x), "... k (num_heads head_dim) -> ... num_heads k head_dim", num_heads=self.num_heads)
        V =  rearrange(self.WV(x), "... v (num_heads head_dim) -> ... num_heads v head_dim", num_heads=self.num_heads)
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        sequence_length = Q.shape[-2]
        bool_tensor = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=Q.device)
        mask = torch.tril(bool_tensor)
        attn = rearrange(scaled_dot_product_attention(Q, K, V, mask), "... num_heads sequence_length head_dim -> ... sequence_length (num_heads head_dim)")

        return self.WO(attn)