"""Core DeVo model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

from .canonical_features import CanonicalOrderSpec
from .parameterization import MultiBranchParameterization


@dataclass
class DeVoConfig:
    """Configuration for DeVo.

    Important knobs are:

    - `orders`: Volterra orders represented by canonical features.
    - `num_branches`: multi-branch over-parameterization count. `1` is the
      single-branch ablation / degenerate mode.
    - `feature_chunk_size`: chunk size for on-the-fly canonical feature
      evaluation. This is the main memory control for MPS / CPU execution.
    """

    orders: Tuple[int, ...] = (1, 2, 3)
    num_branches: int = 4
    feature_mode: str = "canonical"
    apply_multiplicity_correction: bool = True
    epochs: int = 25
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_norm: Optional[float] = 1.0
    early_stop_patience: Optional[int] = None
    early_stop_min_delta: float = 0.0
    feature_chunk_size: int = 4_096
    index_cache_limit: int = 500_000
    max_canonical_terms_per_order: int = 1_000_000
    max_full_recovery_elements: int = 4_000_000
    seed: int = 7
    include_bias: bool = True
    init_scale: float = 1e-2
    verbose: bool = True
    log_every: int = 1
    device: Optional[str] = None
    dtype: torch.dtype = torch.float32

    def normalized_orders(self) -> Tuple[int, ...]:
        """Return validated and sorted Volterra orders."""

        unique = sorted({int(order) for order in self.orders})
        if not unique or unique[0] <= 0:
            raise ValueError("orders must contain positive integers.")
        return tuple(unique)

    def normalized_feature_mode(self) -> str:
        mode = str(self.feature_mode).strip().lower() or "canonical"
        if mode not in {"canonical", "full"}:
            raise ValueError("feature_mode must be 'canonical' or 'full'.")
        return mode


class DeVoModel(nn.Module):
    """Structured nonlinear model built around canonical Volterra features."""

    def __init__(
        self,
        *,
        window_length: int,
        input_dim: int,
        output_dim: int,
        horizon: int,
        config: Optional[DeVoConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or DeVoConfig()
        self.window_length = int(window_length)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.horizon = int(horizon)
        self.response_dim = self.horizon * self.output_dim
        self.orders = self.config.normalized_orders()
        self.feature_mode = self.config.normalized_feature_mode()

        self.order_specs: Dict[int, CanonicalOrderSpec] = {}
        feature_counts: Dict[int, int] = {}
        for order in self.orders:
            spec = CanonicalOrderSpec(
                order=order,
                window_length=self.window_length,
                input_dim=self.input_dim,
                index_cache_limit=self.config.index_cache_limit,
                index_mode=self.feature_mode,
            )
            if spec.feature_count > self.config.max_canonical_terms_per_order:
                raise ValueError(
                    "Feature count exceeds the configured memory budget: "
                    f"order={order}, feature_count={spec.feature_count}, "
                    f"max={self.config.max_canonical_terms_per_order}."
                )
            self.order_specs[order] = spec
            feature_counts[order] = spec.feature_count

        self.parameterization = MultiBranchParameterization(
            feature_counts=feature_counts,
            response_dim=self.response_dim,
            num_branches=self.config.num_branches,
            include_bias=self.config.include_bias,
            init_scale=self.config.init_scale,
        )

    def _validate_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        if x.ndim != 3:
            raise ValueError("DeVo expects X with shape [N, M, D] or [N, M].")
        if x.shape[1] != self.window_length or x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected X shape [N, {self.window_length}, {self.input_dim}], got {tuple(x.shape)}."
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_inputs(x)
        x_flat = x.reshape(x.shape[0], -1)

        if self.parameterization.bias is None:
            output = x_flat.new_zeros((x_flat.shape[0], self.response_dim))
        else:
            output = self.parameterization.bias.unsqueeze(0).expand(x_flat.shape[0], -1).clone()

        for order in self.orders:
            spec = self.order_specs[order]
            effective = self.parameterization.effective_order_parameters(order)
            for chunk in spec.iter_chunks(self.config.feature_chunk_size):
                features = spec.build_features(x_flat, chunk.indices)
                weights = effective[:, chunk.start:chunk.stop].transpose(0, 1)
                output = output + features.matmul(weights)

        return output.reshape(x.shape[0], self.horizon, self.output_dim)

    def export_parameters(self) -> Dict[str, object]:
        """Export effective and branch parameters in model-aware shapes."""

        exported_orders = {}
        raw = self.parameterization.export()
        for order, payload in raw.items():
            exported_orders[order] = {
                "branch_weights": payload["branch_weights"],
                "branch_parameters": payload["branch_parameters"].reshape(
                    self.parameterization.num_branches,
                    self.horizon,
                    self.output_dim,
                    -1,
                ),
                "effective_parameters": payload["effective_parameters"].reshape(
                    self.horizon,
                    self.output_dim,
                    -1,
                ),
                "parameterization": self.order_specs[order].parameterization_summary(),
            }

        bias = None
        if self.parameterization.bias is not None:
            bias = self.parameterization.bias.detach().cpu().reshape(self.horizon, self.output_dim)

        return {
            "window_length": self.window_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "horizon": self.horizon,
            "num_branches": self.parameterization.num_branches,
            "bias": bias,
            "orders": exported_orders,
        }
