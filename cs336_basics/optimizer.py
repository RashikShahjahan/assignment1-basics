import torch
from collections.abc import Callable, Iterable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params:Iterable[torch.nn.Parameter], lr:float, betas:tuple[float, float], eps:float, weight_decay:float):

        defaults = {"lr": lr, "beta1":betas[0], "beta2":betas[1], "eps":eps, "weight_decay":weight_decay}

        super().__init__(params,defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                if  "t" not in state:
                    state["t"] = state.get("t", 0) 
                if  "m" not in state:
                    state["m"] = state.get("m", torch.zeros_like(p))
                if  "v" not in state:
                    state["v"] = state.get("v", torch.zeros_like(p))
                grad = p.grad.data  
                
                state["t"] = state["t"] + 1  # Increment iteration number.

                alpha_t = lr*math.sqrt(1-group["beta2"]**state["t"])/(1-group["beta1"]**state["t"])
                p.data -= lr * group["weight_decay"] *p.data  
                state["m"] = group["beta1"]* state["m"]  + (1-group["beta1"])*grad
                state["v"] = group["beta2"]* state["v"]  + (1-group["beta2"])*grad**2

                p.data -= alpha_t * state["m"]/(torch.sqrt(state["v"])+group["eps"])
        return loss
