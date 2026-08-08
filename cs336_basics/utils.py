import torch
import math
from collections.abc import Iterable
import typing
import os
import numpy.typing as npt
import numpy as np

def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor)->torch.Tensor:
    log_probs = torch.log_softmax(inputs, dim=-1)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()

def lr_cosine_schedule( it: int,max_learning_rate: float, min_learning_rate: float,warmup_iters: int,cosine_cycle_iters: int,)->float: 
    if it < warmup_iters:
        return it/warmup_iters *max_learning_rate
    if warmup_iters <= it and cosine_cycle_iters >= it:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
    if it > cosine_cycle_iters:
        return min_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    total = 0
    for param in list(parameters):
        if param.grad is not None:
            total+=torch.sum(torch.square(param.grad))
    norm = math.sqrt(total)

    if norm >= max_l2_norm:
        for param in list(parameters):
            if param.grad is not None:
                param.grad = (max_l2_norm/(norm+eps))* param.grad

rng = np.random.default_rng(42)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(dataset)-context_length, batch_size)[:,None]
    offsets = np.arange(0,context_length+1)
    indices = starts+offsets
    

    sample = dataset[indices]
    return torch.tensor(sample[:,:context_length],device=device, dtype=torch.long), torch.tensor(sample[:,1:context_length+1],device=device, dtype=torch.long)

def save_checkpoint(model: torch.nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    torch.save({"model_state":model_state,"optimizer_state":optimizer_state,"it":iteration},out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] , model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model_state"])
    optimizer.load_state_dict(state_dict["optimizer_state"])

    return state_dict["it"]