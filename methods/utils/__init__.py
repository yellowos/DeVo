"""Utility exports for methods layer."""

from __future__ import annotations

from typing import Any, Optional

from .device import (
    DeviceContext,
    is_mps_available,
    is_torch_available,
    resolve_dtype,
    select_default_device,
    select_device,
)
from .runtime import set_random_seed


def coerce_dataset_bundle(bundle: Any) -> Any:
    from .bundle import coerce_dataset_bundle as _coerce_dataset_bundle

    return _coerce_dataset_bundle(bundle)


def load_processed_dataset_bundle(path: str) -> Any:
    from .bundle import load_processed_dataset_bundle as _load_processed_dataset_bundle

    return _load_processed_dataset_bundle(path)


def slice_dataset_bundle(
    bundle: Any,
    *,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> Any:
    from .bundle import slice_dataset_bundle as _slice_dataset_bundle

    return _slice_dataset_bundle(
        bundle,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )


__all__ = [
    "DeviceContext",
    "coerce_dataset_bundle",
    "is_mps_available",
    "is_torch_available",
    "load_processed_dataset_bundle",
    "resolve_dtype",
    "select_default_device",
    "select_device",
    "set_random_seed",
    "slice_dataset_bundle",
]
