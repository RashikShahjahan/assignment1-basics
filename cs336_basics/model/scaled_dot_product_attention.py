import torch
from einops import einsum
from cs336_basics.model.softmax import softmax
import math

def scaled_dot_product_attention(Q:torch.Tensor, K:torch.Tensor,V:torch.Tensor, mask:torch.Tensor):
    d_k = Q.shape[-1] 
    QKt = einsum(Q,K," ... q d_k ,  ...  k d_k  ->   ... q k")
    norm = QKt/math.sqrt(d_k)
    if mask is not None:
        norm = norm.masked_fill(torch.Tensor.logical_not(mask),-torch.inf)

    scores = softmax(norm, i =-1)

    return einsum(scores, V, " ... q k ,  ... k d_v ->  ... q d_v")