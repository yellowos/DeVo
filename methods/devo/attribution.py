"""Input attribution utilities for DeVo."""

from __future__ import annotations

from typing import Optional

import torch


def prediction_error_gradient_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    *,
    batch_size: int = 128,
    mode: str = "gradient",
) -> torch.Tensor:
    """Return per-sample attribution maps over `[time, variable]`.

    The default objective is the prediction-error MSE. When `targets` is not
    provided, the objective falls back to prediction energy for quick probing.
    """

    if mode not in {"gradient", "grad_x_input"}:
        raise ValueError("mode must be 'gradient' or 'grad_x_input'.")

    model.eval()
    device = next(model.parameters()).device
    outputs = []

    for start in range(0, inputs.shape[0], batch_size):
        stop = min(start + batch_size, inputs.shape[0])
        batch = inputs[start:stop].to(device).clone().detach().requires_grad_(True)
        pred = model(batch)
        if targets is None:
            objective = pred.pow(2).mean(dim=(1, 2))
        else:
            batch_target = targets[start:stop].to(device)
            objective = (pred - batch_target).pow(2).mean(dim=(1, 2))
        grads = torch.autograd.grad(objective.sum(), batch, retain_graph=False, create_graph=False)[0]
        if mode == "grad_x_input":
            grads = grads * batch
        outputs.append(grads.detach().cpu())

    return torch.cat(outputs, dim=0)
