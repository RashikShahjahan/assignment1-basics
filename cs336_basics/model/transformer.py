import torch
import torch.nn as nn
from cs336_basics.model.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.model.positionwise_feedforward import PositionwiseFeedForward
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear




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

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length:int, num_layers:int, d_model: int, num_heads: int, d_ff:int, theta:float |None=None, max_seq_len:int|None=None ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len) for i in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.context_length = context_length

    def forward(self, x: torch.Tensor,  token_positions: torch.Tensor | None=None) -> torch.Tensor:
        h = self.embedding(x)
        for block in self.transformer_blocks:
            h = block(h, token_positions)

        return self.lm_head(self.norm(h))


