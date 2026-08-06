import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.tokenizer import Tokenizer
import math

def softmax(x:torch.Tensor, i:int)->torch.Tensor:
    x = x-torch.max(x, dim=i, keepdim=True).values
    return torch.exp(x)/torch.sum(torch.exp(x), dim=i, keepdim=True)

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor,V:torch.Tensor, mask:torch.Tensor):
    d_k = Q.shape[-1] 
    QKt = einsum(Q,K," ... q d_k ,  ...  k d_k  ->   ... q k")
    norm = QKt/math.sqrt(d_k)
    if mask is not None:
        norm = norm.masked_fill(torch.Tensor.logical_not(mask),-torch.inf)

    scores = softmax(norm, i =-1)

    return einsum(scores, V, " ... q k ,  ... k d_v ->  ... q d_v")

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device | None = None , dtype:torch.dtype | None = None ):
        super().__init__()
        empty_weights = torch.empty(num_embeddings,embedding_dim, device=device, dtype=dtype)
        self.W = nn.Parameter(nn.init.trunc_normal_(empty_weights, mean=0, std = 1 , a =-3, b=3))
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]

class Linear(nn.Module):
    def __init__(self, in_features:int, out_features:int, device:torch.device | None = None , dtype:torch.dtype | None = None ):
        super().__init__()
        empty_weights = torch.empty(out_features,in_features, device=device, dtype=dtype)
        sigma = math.sqrt(2/(out_features+in_features))
        self.W = nn.Parameter(nn.init.trunc_normal_(empty_weights, mean=0, std = sigma , a =-3*sigma, b=3*sigma))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wt = rearrange(self.W, "out_features in_features -> in_features out_features")
        return einsum(Wt, x, "in_features out_features, ... in_features -> ... out_features")

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device:torch.device | None = None , dtype:torch.dtype | None = None ):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x*x, dim=-1,keepdim=True)+self.eps)
        result = (self.gain*x)/rms

        return result.to(in_dtype)

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

@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer:Tokenizer,
    input_tokens: torch.Tensor,
    max_tokens: int = 64,
    temperature: float = 0.7,
    p: float = 1.0,
) -> torch.Tensor:

    model.eval()
    eot_token = tokenizer.encode("<|endoftext|>")

    for _ in range(max_tokens):
        logits = model(input_tokens)
        next_token_logits = logits[:, -1, :] / temperature
        probabilities = softmax(next_token_logits, i=-1)


        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            dim=-1,
            descending=True,
        )

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs - sorted_probs >= p
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        sampled_position = torch.multinomial(
            sorted_probs,
            num_samples=1,
        )

        next_token = torch.gather(
            sorted_indices,
            dim=-1,
            index=sampled_position,
        )

        input_tokens = torch.cat(
            [input_tokens, next_token],
            dim=-1,
        )
        if next_token == eot_token:
            break

    return input_tokens

