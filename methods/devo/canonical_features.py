"""Canonical feature construction for structured Volterra models."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

from .multiplicity import multiplicity_from_canonical_indices


def flat_slot_count(window_length: int, input_dim: int) -> int:
    """Return the flattened `(lag, input_channel)` slot count."""

    return int(window_length) * int(input_dim)


def naive_full_tensor_parameter_count(flattened_dim: int, order: int) -> int:
    """Parameter count for a naive full Volterra tensor of a single response."""

    return int(flattened_dim) ** int(order)


def canonical_parameter_count(flattened_dim: int, order: int) -> int:
    """Parameter count for ordered canonical monomials.

    This uses combinations with replacement. It is the symmetry-aware compact
    alternative to the naive `flattened_dim ** order` parameterization.
    """

    return math.comb(int(flattened_dim) + int(order) - 1, int(order))


def flat_to_lag_input(indices: torch.Tensor, input_dim: int) -> torch.Tensor:
    """Decode flattened slots back to `(lag, input_channel)` pairs."""

    base = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    lag = torch.div(base, input_dim, rounding_mode="floor")
    channel = torch.remainder(base, input_dim)
    return torch.stack((lag, channel), dim=-1)


@dataclass(frozen=True)
class CanonicalFeatureChunk:
    """Chunk of canonical feature definitions.

    `indices` stores ordered flattened slots for each canonical monomial and
    `multiplicity` stores the exact symmetry correction needed to recover the
    corresponding complete symmetric Volterra kernel.
    """

    start: int
    stop: int
    indices: torch.Tensor
    multiplicity: torch.Tensor


class CanonicalOrderSpec:
    """Definition of one canonical Volterra order.

    Canonical features are ordered monomials over flattened `(lag, channel)`
    slots. They avoid the memory blow-up of a naive full tensor parameterization
    while remaining exactly recoverable through multiplicity correction.
    """

    def __init__(
        self,
        *,
        order: int,
        window_length: int,
        input_dim: int,
        index_cache_limit: int = 500_000,
    ) -> None:
        if order <= 0:
            raise ValueError("order must be positive.")
        self.order = int(order)
        self.window_length = int(window_length)
        self.input_dim = int(input_dim)
        self.flattened_dim = flat_slot_count(self.window_length, self.input_dim)
        self.feature_count = canonical_parameter_count(self.flattened_dim, self.order)
        self.index_cache_limit = int(index_cache_limit)
        self._cached_indices: Optional[torch.Tensor] = None
        self._cached_multiplicity: Optional[torch.Tensor] = None
        if self.feature_count <= self.index_cache_limit:
            self._materialize_cache()

    @property
    def is_cached(self) -> bool:
        return self._cached_indices is not None and self._cached_multiplicity is not None

    def _materialize_cache(self) -> None:
        tuples = list(itertools.combinations_with_replacement(range(self.flattened_dim), self.order))
        self._cached_indices = torch.tensor(tuples, dtype=torch.long)
        self._cached_multiplicity = multiplicity_from_canonical_indices(self._cached_indices)

    def parameterization_summary(self) -> Dict[str, int]:
        """Return a small summary contrasting naive and canonical sizes."""

        return {
            "order": self.order,
            "flattened_dim": self.flattened_dim,
            "canonical_feature_count": self.feature_count,
            "naive_full_tensor_parameter_count": naive_full_tensor_parameter_count(
                self.flattened_dim, self.order
            ),
        }

    def iter_chunks(self, chunk_size: int) -> Iterator[CanonicalFeatureChunk]:
        """Yield canonical feature definitions without building a full design matrix."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if self.is_cached:
            assert self._cached_indices is not None
            assert self._cached_multiplicity is not None
            for start in range(0, self.feature_count, chunk_size):
                stop = min(start + chunk_size, self.feature_count)
                yield CanonicalFeatureChunk(
                    start=start,
                    stop=stop,
                    indices=self._cached_indices[start:stop],
                    multiplicity=self._cached_multiplicity[start:stop],
                )
            return

        start = 0
        generator = itertools.combinations_with_replacement(range(self.flattened_dim), self.order)
        while True:
            batch = list(itertools.islice(generator, chunk_size))
            if not batch:
                break
            indices = torch.tensor(batch, dtype=torch.long)
            stop = start + indices.shape[0]
            yield CanonicalFeatureChunk(
                start=start,
                stop=stop,
                indices=indices,
                multiplicity=multiplicity_from_canonical_indices(indices),
            )
            start = stop

    def materialize_indices(self, chunk_size: int = 16_384) -> torch.Tensor:
        """Materialize the canonical index table on CPU when recovery needs it."""

        if self.is_cached:
            assert self._cached_indices is not None
            return self._cached_indices.clone()
        return torch.cat([chunk.indices for chunk in self.iter_chunks(chunk_size)], dim=0)

    def materialize_multiplicity(self, chunk_size: int = 16_384) -> torch.Tensor:
        """Materialize multiplicities aligned with `materialize_indices`."""

        if self.is_cached:
            assert self._cached_multiplicity is not None
            return self._cached_multiplicity.clone()
        return torch.cat([chunk.multiplicity for chunk in self.iter_chunks(chunk_size)], dim=0)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode flattened canonical indices to `(lag, input_channel)` pairs."""

        return flat_to_lag_input(indices, self.input_dim)

    def build_features(self, x_flat: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Evaluate a canonical feature block for a mini-batch.

        The model never constructs the full design matrix. Instead, it gathers a
        small chunk of canonical monomials, evaluates them, and immediately
        accumulates the corresponding linear projection.
        """

        if x_flat.ndim != 2:
            raise ValueError("x_flat must have shape [batch, flattened_dim].")
        if x_flat.shape[1] != self.flattened_dim:
            raise ValueError(
                f"Expected flattened_dim={self.flattened_dim}, got {x_flat.shape[1]}."
            )
        device_indices = indices.to(x_flat.device)
        gathered = x_flat[:, device_indices.reshape(-1)]
        gathered = gathered.reshape(x_flat.shape[0], device_indices.shape[0], self.order)
        return gathered.prod(dim=-1)
