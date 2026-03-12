"""Multi-branch parameterization for DeVo."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
from torch import nn


class MultiBranchParameterization(nn.Module):
    """Over-parameterized linear coefficients for canonical Volterra features.

    DeVo does not train several independent models and average their outputs.
    Instead, each Volterra order owns multiple branch tensors that are collapsed
    into one effective kernel parameter set used by the forward pass. This keeps
    one shared structural model while injecting optimization redundancy.
    """

    def __init__(
        self,
        *,
        feature_counts: Mapping[int, int],
        response_dim: int,
        num_branches: int = 1,
        include_bias: bool = True,
        init_scale: float = 1e-2,
    ) -> None:
        super().__init__()
        if num_branches <= 0:
            raise ValueError("num_branches must be positive.")

        self.feature_counts = {int(order): int(count) for order, count in feature_counts.items()}
        self.response_dim = int(response_dim)
        self.num_branches = int(num_branches)
        self.branch_parameters = nn.ParameterDict()
        self.branch_logits = nn.ParameterDict()

        for order, feature_count in sorted(self.feature_counts.items()):
            parameter = nn.Parameter(
                torch.empty(self.num_branches, self.response_dim, feature_count)
            )
            nn.init.normal_(parameter, mean=0.0, std=init_scale / math.sqrt(max(order, 1)))
            self.branch_parameters[str(order)] = parameter
            if self.num_branches > 1:
                self.branch_logits[str(order)] = nn.Parameter(torch.zeros(self.num_branches))

        self.bias = nn.Parameter(torch.zeros(self.response_dim)) if include_bias else None

    def branch_weights(self, order: int) -> torch.Tensor:
        """Return branch mixing weights for one Volterra order."""

        if self.num_branches == 1:
            return torch.ones(1, device=self.branch_parameters[str(order)].device)
        return torch.softmax(self.branch_logits[str(order)], dim=0)

    def effective_order_parameters(self, order: int) -> torch.Tensor:
        """Collapse branches into the effective kernel coefficients."""

        raw = self.branch_parameters[str(order)]
        weights = self.branch_weights(order).to(dtype=raw.dtype)
        return torch.einsum("k,kof->of", weights, raw)

    def export(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Detach branch and effective parameters for downstream analysis."""

        exported: Dict[int, Dict[str, torch.Tensor]] = {}
        for order in sorted(self.feature_counts):
            weights = self.branch_weights(order).detach().cpu()
            branches = self.branch_parameters[str(order)].detach().cpu()
            effective = self.effective_order_parameters(order).detach().cpu()
            exported[order] = {
                "branch_weights": weights,
                "branch_parameters": branches,
                "effective_parameters": effective,
            }
        return exported
