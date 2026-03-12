"""Compatibility bundle helpers layered on top of the methods IO schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from methods.base import MethodDatasetBundle, MethodDatasetSplit, load_dataset_bundle
from methods.base.io_schema import coerce_dataset_bundle as _coerce_dataset_bundle


def coerce_dataset_bundle(bundle: Any) -> MethodDatasetBundle:
    """Normalize supported bundle inputs into `MethodDatasetBundle`."""

    return _coerce_dataset_bundle(bundle)


def load_processed_dataset_bundle(path: str | Path) -> MethodDatasetBundle:
    """Load a processed manifest or directory into `MethodDatasetBundle`."""

    return load_dataset_bundle(path)


def _slice_value(value: Any, size: Optional[int]) -> Any:
    if size is None or value is None:
        return value
    return value[:size]


def _slice_split(split: MethodDatasetSplit, size: Optional[int]) -> MethodDatasetSplit:
    return MethodDatasetSplit(
        X=_slice_value(split.X, size),
        Y=_slice_value(split.Y, size),
        sample_id=_slice_value(split.sample_id, size),
        run_id=_slice_value(split.run_id, size),
        timestamp=_slice_value(split.timestamp, size),
        meta=dict(split.meta),
        extra_fields=dict(split.extra_fields),
    )


def slice_dataset_bundle(
    bundle: Any,
    *,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> MethodDatasetBundle:
    """Create a smaller methods-layer bundle view for smoke tests or debugging."""

    normalized = _coerce_dataset_bundle(bundle)
    return MethodDatasetBundle(
        train=_slice_split(normalized.train, train_size),
        val=_slice_split(normalized.val, val_size),
        test=_slice_split(normalized.test, test_size),
        meta=normalized.meta,
        artifacts=normalized.artifacts,
        source_manifest=normalized.source_manifest,
        source_root=normalized.source_root,
    )
