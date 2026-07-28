import torch

def softmax(x:torch.Tensor, i:int)->torch.Tensor:
    x = x-torch.max(x, dim=i, keepdim=True).values
    return torch.exp(x)/torch.sum(torch.exp(x), dim=i, keepdim=True)
