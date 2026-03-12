"""Multiplicity correction for canonical Volterra features."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def multiplicity_from_canonical_indices(indices: torch.Tensor | np.ndarray | Iterable[Iterable[int]]) -> torch.Tensor:
    """Return permutation multiplicities for ordered canonical monomials.

    A canonical feature keeps only one ordered tuple `i1 <= i2 <= ... <= iq`
    instead of all permutations of a full Volterra tensor entry. Its
    multiplicity is:

    `q! / prod_r count_r!`

    where `count_r` are the repetition counts of identical indices. This is the
    exact factor required to align canonical coefficients with the semantics of a
    complete symmetric Volterra kernel.
    """

    tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    if tensor.ndim != 2:
        raise ValueError("canonical indices must be a rank-2 array: [feature_count, order].")

    feature_count, order = tensor.shape
    if order == 0:
        raise ValueError("order 0 is handled as bias outside multiplicity correction.")
    if order == 1:
        return torch.ones(feature_count, dtype=torch.float32)

    raw = tensor.numpy()
    factorial_order = math.factorial(order)
    multiplicity = np.empty(feature_count, dtype=np.float32)
    for row_idx, row in enumerate(raw):
        run_length = 1
        denominator = 1
        for col_idx in range(1, order):
            if row[col_idx] == row[col_idx - 1]:
                run_length += 1
            else:
                denominator *= math.factorial(run_length)
                run_length = 1
        denominator *= math.factorial(run_length)
        multiplicity[row_idx] = float(factorial_order // denominator)
    return torch.from_numpy(multiplicity)


def effective_to_symmetric_coefficients(
    effective_coefficients: torch.Tensor,
    multiplicity: torch.Tensor,
) -> torch.Tensor:
    """Convert canonical effective coefficients to symmetric kernel values."""

    scale = multiplicity.to(device=effective_coefficients.device, dtype=effective_coefficients.dtype)
    expand_shape = (1,) * (effective_coefficients.ndim - 1) + (scale.numel(),)
    return effective_coefficients / scale.view(expand_shape)


def symmetric_to_effective_coefficients(
    symmetric_coefficients: torch.Tensor,
    multiplicity: torch.Tensor,
) -> torch.Tensor:
    """Convert symmetric kernel values back to canonical effective coefficients."""

    scale = multiplicity.to(device=symmetric_coefficients.device, dtype=symmetric_coefficients.dtype)
    expand_shape = (1,) * (symmetric_coefficients.ndim - 1) + (scale.numel(),)
    return symmetric_coefficients * scale.view(expand_shape)
