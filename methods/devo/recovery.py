"""Kernel recovery for DeVo."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .multiplicity import effective_to_symmetric_coefficients


def _unique_permutations(index_tuple: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    return list({tuple(value) for value in itertools.permutations(index_tuple)})


@dataclass
class RecoveredKernelOrder:
    """Recovered kernel for one Volterra order.

    The canonical representation stores only ordered monomials plus exact
    multiplicity factors. `symmetric_coefficients` is already converted to the
    complete symmetric kernel semantics, so expanding to a full tensor simply
    copies each canonical value to all unique permutations.
    """

    order: int
    window_length: int
    input_dim: int
    output_dim: int
    horizon: int
    canonical_indices: torch.Tensor
    lag_input_indices: torch.Tensor
    multiplicity: torch.Tensor
    effective_coefficients: torch.Tensor
    symmetric_coefficients: torch.Tensor
    full_tensor_shape: Tuple[int, ...]
    full_tensor: Optional[torch.Tensor] = None

    @property
    def flattened_dim(self) -> int:
        return self.window_length * self.input_dim

    @property
    def feature_count(self) -> int:
        return int(self.canonical_indices.shape[0])

    @property
    def flat_full_tensor_shape(self) -> Tuple[int, ...]:
        return (self.horizon * self.output_dim,) + (self.flattened_dim,) * self.order

    def full_tensor_element_count(self) -> int:
        return math.prod(self.flat_full_tensor_shape)

    def can_materialize_full_tensor(self, max_elements: int) -> bool:
        return self.full_tensor_element_count() <= int(max_elements)

    def materialize_full_tensor(self, max_elements: int = 4_000_000) -> torch.Tensor:
        """Expand the compact symmetric representation into a full kernel tensor."""

        if self.full_tensor is not None:
            return self.full_tensor
        if not self.can_materialize_full_tensor(max_elements):
            raise MemoryError(
                f"Full tensor for order {self.order} needs {self.full_tensor_element_count()} "
                f"elements, above max_elements={max_elements}."
            )

        flat = torch.zeros(self.flat_full_tensor_shape, dtype=self.symmetric_coefficients.dtype)
        coeff = self.symmetric_coefficients.reshape(self.horizon * self.output_dim, self.feature_count)

        for feature_idx, tuple_tensor in enumerate(self.canonical_indices):
            tuple_key = tuple(int(value) for value in tuple_tensor.tolist())
            value = coeff[:, feature_idx]
            for perm in _unique_permutations(tuple_key):
                flat[(slice(None),) + perm] = value

        structured = flat.view(self.full_tensor_shape)
        self.full_tensor = structured
        return structured

    def to_dict(self, *, include_full_tensor: bool = False, max_full_elements: int = 4_000_000) -> Dict[str, Any]:
        payload = {
            "order": self.order,
            "feature_count": self.feature_count,
            "canonical_indices": self.canonical_indices.numpy(),
            "lag_input_indices": self.lag_input_indices.numpy(),
            "multiplicity": self.multiplicity.numpy(),
            "effective_coefficients": self.effective_coefficients.numpy(),
            "symmetric_coefficients": self.symmetric_coefficients.numpy(),
            "full_tensor_shape": self.full_tensor_shape,
        }
        if include_full_tensor and self.can_materialize_full_tensor(max_full_elements):
            payload["full_tensor"] = self.materialize_full_tensor(max_full_elements).numpy()
        else:
            payload["full_tensor"] = None
        return payload


@dataclass
class RecoveredKernelBundle:
    """Structured kernel recovery output for DeVo."""

    method_name: str
    window_length: int
    input_dim: int
    output_dim: int
    horizon: int
    bias: Optional[torch.Tensor]
    orders: List[RecoveredKernelOrder] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_full_tensor: bool = False, max_full_elements: int = 4_000_000) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "window_length": self.window_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "horizon": self.horizon,
            "bias": None if self.bias is None else self.bias.numpy(),
            "orders": [
                order.to_dict(
                    include_full_tensor=include_full_tensor,
                    max_full_elements=max_full_elements,
                )
                for order in self.orders
            ],
            "metadata": dict(self.metadata),
        }


def recover_devo_kernels(
    model: Any,
    *,
    materialize_full: bool = False,
    max_full_elements: Optional[int] = None,
) -> RecoveredKernelBundle:
    """Recover DeVo kernels in canonical and symmetric forms."""

    exported = model.export_parameters()
    limit = max_full_elements or model.config.max_full_recovery_elements
    recovered_orders: List[RecoveredKernelOrder] = []

    for order in model.orders:
        spec = model.order_specs[order]
        effective = exported["orders"][order]["effective_parameters"].clone()
        indices = spec.materialize_indices()
        multiplicity = spec.materialize_multiplicity()
        lag_input = spec.decode_indices(indices)
        symmetric = effective_to_symmetric_coefficients(effective, multiplicity)

        full_tensor_shape = (model.horizon, model.output_dim) + tuple(
            axis for _ in range(order) for axis in (model.window_length, model.input_dim)
        )
        recovered = RecoveredKernelOrder(
            order=order,
            window_length=model.window_length,
            input_dim=model.input_dim,
            output_dim=model.output_dim,
            horizon=model.horizon,
            canonical_indices=indices.cpu(),
            lag_input_indices=lag_input.cpu(),
            multiplicity=multiplicity.cpu(),
            effective_coefficients=effective.cpu(),
            symmetric_coefficients=symmetric.cpu(),
            full_tensor_shape=full_tensor_shape,
        )
        if materialize_full and recovered.can_materialize_full_tensor(limit):
            recovered.materialize_full_tensor(limit)
        recovered_orders.append(recovered)

    return RecoveredKernelBundle(
        method_name="DeVo",
        window_length=model.window_length,
        input_dim=model.input_dim,
        output_dim=model.output_dim,
        horizon=model.horizon,
        bias=exported["bias"],
        orders=recovered_orders,
        metadata={
            "num_branches": model.parameterization.num_branches,
            "feature_chunk_size": model.config.feature_chunk_size,
            "orders": list(model.orders),
        },
    )
