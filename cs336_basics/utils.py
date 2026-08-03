import torch
import math

def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor)->torch.Tensor:
    log_probs = torch.log_softmax(inputs, dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()

def lr_cosine_schedule( it: int,max_learning_rate: float, min_learning_rate: float,warmup_iters: int,cosine_cycle_iters: int,): 
    if it < warmup_iters:
        return it/warmup_iters *max_learning_rate
    if warmup_iters <= it and cosine_cycle_iters >= it:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    if it > cosine_cycle_iters:
        return min_learning_rate
