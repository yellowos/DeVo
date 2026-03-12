"""Minimal validation utilities for DatasetBundle and shared schema."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from data.adapters.base import DataProtocolError, DatasetArtifacts, DatasetBundle, DatasetMeta, DatasetSplit


def _as_len(value: Any) -> Optional[int]:
    try:
        return len(value)
    except Exception:
        return None


def check_split(split: DatasetSplit, split_name: str = "split") -> None:
    if not isinstance(split, DatasetSplit):
        raise DataProtocolError(f"{split_name}: must be a DatasetSplit.")

    x_len = _as_len(split.X)
    y_len = _as_len(split.Y)
    if x_len is None or y_len is None:
        raise DataProtocolError(f"{split_name}: X and Y must be sized objects.")
    if x_len != y_len:
        raise DataProtocolError(
            f"{split_name}: X and Y length mismatch (X={x_len}, Y={y_len})."
        )
    if x_len == 0:
        raise DataProtocolError(f"{split_name}: empty samples are not allowed.")

    for field_name in ("sample_id", "run_id", "timestamp"):
        value = getattr(split, field_name)
        if value is None:
            continue
        v_len = _as_len(value)
        if v_len is not None and v_len != x_len:
            raise DataProtocolError(
                f"{split_name}: {field_name} length ({v_len}) must match X/Y length ({x_len})."
            )


def check_meta(meta: DatasetMeta) -> None:
    if not isinstance(meta, DatasetMeta):
        raise DataProtocolError("meta must be a DatasetMeta.")
    required = set(DatasetMeta.REQUIRED_FIELDS)
    missing = [f for f in required if getattr(meta, f, None) is None]
    if missing:
        raise DataProtocolError(f"meta missing required fields: {', '.join(missing)}")
    if not isinstance(meta.input_dim, int) or meta.input_dim <= 0:
        raise DataProtocolError("meta.input_dim must be a positive integer.")
    if not isinstance(meta.output_dim, int) or meta.output_dim <= 0:
        raise DataProtocolError("meta.output_dim must be a positive integer.")
    if not isinstance(meta.window_length, int) or meta.window_length <= 0:
        raise DataProtocolError("meta.window_length must be a positive integer.")
    if not isinstance(meta.horizon, int) or meta.horizon <= 0:
        raise DataProtocolError("meta.horizon must be a positive integer.")
    if not isinstance(meta.has_ground_truth_kernel, bool):
        raise DataProtocolError("meta.has_ground_truth_kernel must be bool.")
    if not isinstance(meta.has_ground_truth_gfrf, bool):
        raise DataProtocolError("meta.has_ground_truth_gfrf must be bool.")
    if not meta.split_protocol:
        raise DataProtocolError("meta.split_protocol must be non-empty.")


def check_artifacts(artifacts: DatasetArtifacts) -> None:
    if not isinstance(artifacts, DatasetArtifacts):
        raise DataProtocolError("artifacts must be a DatasetArtifacts.")

    for key in ("truth_file", "grouping_file", "protocol_file"):
        value = getattr(artifacts, key)
        if value is not None and not isinstance(value, str):
            raise DataProtocolError(f"artifacts.{key} must be a string path/reference or None.")


def check_dataset_bundle(
    bundle: DatasetBundle,
    *,
    strict: bool = True,
    required_extra_fields: Optional[Iterable[str]] = None,
) -> None:
    """Validate DatasetBundle contract fields and basic shape compatibility."""

    if not isinstance(bundle, DatasetBundle):
        raise DataProtocolError("bundle must be a DatasetBundle.")
    if not isinstance(bundle.train, DatasetSplit):
        raise DataProtocolError("bundle.train must be DatasetSplit.")
    if not isinstance(bundle.val, DatasetSplit):
        raise DataProtocolError("bundle.val must be DatasetSplit.")
    if not isinstance(bundle.test, DatasetSplit):
        raise DataProtocolError("bundle.test must be DatasetSplit.")
    if not isinstance(bundle.meta, DatasetMeta):
        raise DataProtocolError("bundle.meta must be DatasetMeta.")
    if not isinstance(bundle.artifacts, DatasetArtifacts):
        raise DataProtocolError("bundle.artifacts must be DatasetArtifacts.")

    check_split(bundle.train, "train")
    check_split(bundle.val, "val")
    check_split(bundle.test, "test")
    check_meta(bundle.meta)
    check_artifacts(bundle.artifacts)

    if required_extra_fields:
        missing = [f for f in required_extra_fields if f not in bundle.meta.extras]
        if strict and missing:
            raise DataProtocolError(f"meta.extras missing fields: {', '.join(missing)}")
