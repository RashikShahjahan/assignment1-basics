import torch

def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor)->torch.Tensor:
    log_probs = torch.log_softmax(inputs, dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()