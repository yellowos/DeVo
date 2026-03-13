"""Feature construction for structured Volterra models."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import torch

from .multiplicity import multiplicity_from_canonical_indices


@dataclass(frozen=True)
class AlignmentRecord:
    window_start: int
    window_end: int
    target_index: int
    horizon: int
    run_id: Any = None


def build_targets_with_horizon(
    sequence: Any,
    *,
    window_length: int,
    horizon: int,
) -> tuple[np.ndarray, list[AlignmentRecord]]:
    values = np.asarray(sequence)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("sequence must be 1D or 2D when building aligned targets.")

    sample_count = values.shape[0] - int(window_length) - int(horizon) + 1
    if sample_count < 0:
        raise ValueError(
            f"Need at least window_length+horizon samples, got n={values.shape[0]}, "
            f"window_length={window_length}, horizon={horizon}."
        )

    targets = []
    alignment: list[AlignmentRecord] = []
    for window_start in range(sample_count):
        window_end = window_start + int(window_length) - 1
        target_index = window_end + int(horizon)
        targets.append(values[window_end + 1 : target_index + 1])
        alignment.append(
            AlignmentRecord(
                window_start=window_start,
                window_end=window_end,
                target_index=target_index,
                horizon=int(horizon),
            )
        )
    return np.asarray(targets), alignment


def build_aligned_windows(
    sequence: Any,
    *,
    window_length: int,
    horizon: int,
    run_id: Any = None,
) -> dict[str, Any]:
    values = np.asarray(sequence)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("sequence must be 1D or 2D when building aligned windows.")

    targets, alignment = build_targets_with_horizon(
        values,
        window_length=window_length,
        horizon=horizon,
    )
    run_values = None if run_id is None else np.asarray(run_id, dtype=object).reshape(-1)
    windows = []
    window_run_ids = []
    for record in alignment:
        windows.append(values[record.window_start : record.window_end + 1])
        if run_values is None:
            window_run_ids.append(None)
            continue
        if len(run_values) != len(values):
            raise ValueError("run_id must match the raw sequence length.")
        covered = run_values[record.window_start : record.target_index + 1]
        if any(item != covered[0] for item in covered[1:]):
            raise ValueError("Window/target alignment crosses run boundaries.")
        window_run_ids.append(covered[0])

    return {
        "X": np.asarray(windows),
        "Y": targets,
        "window_start": np.asarray([item.window_start for item in alignment], dtype=np.int64),
        "window_end": np.asarray([item.window_end for item in alignment], dtype=np.int64),
        "target_index": np.asarray([item.target_index for item in alignment], dtype=np.int64),
        "horizon": np.asarray([item.horizon for item in alignment], dtype=np.int64),
        "run_id": np.asarray(window_run_ids, dtype=object),
    }


def infer_alignment_from_windowed_batch(
    *,
    num_samples: int,
    window_length: int,
    horizon: int,
    alignment: Optional[Mapping[str, Any]] = None,
) -> dict[str, np.ndarray]:
    sample_count = int(num_samples)
    base = np.arange(sample_count, dtype=np.int64)
    if alignment:
        payload = {
            key: np.asarray(value, dtype=np.int64).reshape(-1)
            for key, value in alignment.items()
            if value is not None and key in {"window_start", "window_end", "target_index", "horizon"}
        }
        if payload:
            if "window_start" not in payload:
                if "window_end" in payload:
                    payload["window_start"] = payload["window_end"] - int(window_length) + 1
                else:
                    payload["window_start"] = base
            if "window_end" not in payload:
                payload["window_end"] = payload["window_start"] + int(window_length) - 1
            if "target_index" not in payload:
                payload["target_index"] = payload["window_end"] + int(horizon)
            if "horizon" not in payload:
                payload["horizon"] = np.full(sample_count, int(horizon), dtype=np.int64)
            return {
                key: np.asarray(value, dtype=np.int64).reshape(-1)
                for key, value in payload.items()
            }
    return {
        "window_start": base,
        "window_end": base + int(window_length) - 1,
        "target_index": base + int(window_length) - 1 + int(horizon),
        "horizon": np.full(sample_count, int(horizon), dtype=np.int64),
    }


def validate_alignment(
    x: Any,
    y: Any | None,
    *,
    window_length: int,
    input_dim: int,
    horizon: int,
    output_dim: Optional[int] = None,
    alignment: Optional[Mapping[str, Any]] = None,
) -> None:
    x_array = np.asarray(x)
    if x_array.ndim == 2:
        x_array = x_array[..., None]
    if x_array.ndim != 3:
        raise ValueError("Aligned inputs must have shape [N, M, D] or [N, M].")
    if x_array.shape[1] != int(window_length) or x_array.shape[2] != int(input_dim):
        raise ValueError(
            f"Aligned inputs expect trailing shape ({window_length}, {input_dim}), got {tuple(x_array.shape[1:])}."
        )

    if y is not None:
        y_array = np.asarray(y)
        if y_array.ndim != 3:
            raise ValueError("Aligned targets must have shape [N, H, O].")
        if y_array.shape[0] != x_array.shape[0]:
            raise ValueError("Aligned inputs/targets must have the same batch size.")
        if y_array.shape[1] != int(horizon):
            raise ValueError(f"Aligned targets expect horizon={horizon}, got {y_array.shape[1]}.")
        if output_dim is not None and y_array.shape[2] != int(output_dim):
            raise ValueError(f"Aligned targets expect output_dim={output_dim}, got {y_array.shape[2]}.")

    payload = infer_alignment_from_windowed_batch(
        num_samples=x_array.shape[0],
        window_length=window_length,
        horizon=horizon,
        alignment=alignment,
    )
    if any(len(value) != x_array.shape[0] for value in payload.values()):
        raise ValueError("Alignment metadata length must match the batch size.")
    if np.any(payload["window_end"] < payload["window_start"]):
        raise ValueError("window_end must be greater than or equal to window_start.")
    if np.any((payload["window_end"] - payload["window_start"] + 1) != int(window_length)):
        raise ValueError("Alignment window span does not match window_length.")
    if np.any((payload["target_index"] - payload["window_end"]) != int(horizon)):
        raise ValueError("Alignment target_index does not match the configured horizon.")


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
        index_mode: str = "canonical",
    ) -> None:
        if order <= 0:
            raise ValueError("order must be positive.")
        self.order = int(order)
        self.window_length = int(window_length)
        self.input_dim = int(input_dim)
        self.index_mode = str(index_mode).strip().lower() or "canonical"
        if self.index_mode not in {"canonical", "full"}:
            raise ValueError("index_mode must be 'canonical' or 'full'.")
        self.flattened_dim = flat_slot_count(self.window_length, self.input_dim)
        if self.index_mode == "full":
            self.feature_count = naive_full_tensor_parameter_count(self.flattened_dim, self.order)
        else:
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
        if self.index_mode == "full":
            tuples = list(itertools.product(range(self.flattened_dim), repeat=self.order))
        else:
            tuples = list(itertools.combinations_with_replacement(range(self.flattened_dim), self.order))
        self._cached_indices = torch.tensor(tuples, dtype=torch.long)
        if self.index_mode == "full":
            self._cached_multiplicity = torch.ones(self.feature_count, dtype=torch.float32)
        else:
            self._cached_multiplicity = multiplicity_from_canonical_indices(self._cached_indices)

    def parameterization_summary(self) -> Dict[str, int | str]:
        """Return a small summary contrasting naive and canonical sizes."""

        return {
            "order": self.order,
            "index_mode": self.index_mode,
            "flattened_dim": self.flattened_dim,
            "feature_count": self.feature_count,
            "canonical_feature_count": canonical_parameter_count(self.flattened_dim, self.order),
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
        if self.index_mode == "full":
            generator = itertools.product(range(self.flattened_dim), repeat=self.order)
        else:
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
                multiplicity=(
                    torch.ones(indices.shape[0], dtype=torch.float32)
                    if self.index_mode == "full"
                    else multiplicity_from_canonical_indices(indices)
                ),
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
