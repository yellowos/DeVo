"""Dataset bundle helpers for methods."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from data.adapters.base import DatasetArtifacts, DatasetBundle, DatasetMeta, DatasetSplit


def coerce_dataset_bundle(bundle: Any) -> DatasetBundle:
    """Normalize a mapping into the shared `DatasetBundle` dataclass."""

    if isinstance(bundle, DatasetBundle):
        return bundle
    if not isinstance(bundle, Mapping):
        raise TypeError("dataset_bundle must be a DatasetBundle or a mapping with train/val/test/meta.")
    return DatasetBundle(
        train=DatasetSplit.from_mapping(bundle["train"], split_name="train"),
        val=DatasetSplit.from_mapping(bundle["val"], split_name="val"),
        test=DatasetSplit.from_mapping(bundle["test"], split_name="test"),
        meta=DatasetMeta.from_mapping(bundle["meta"]),
        artifacts=DatasetArtifacts.from_mapping(bundle.get("artifacts")),
    )


def _load_array(root: Path, filename: Optional[str]) -> Any:
    if not filename:
        return None
    path = root / filename
    if not path.exists():
        raise FileNotFoundError(f"Processed artifact missing: {path}")
    return np.load(path, allow_pickle=True)


def load_processed_dataset_bundle(path: str | Path) -> DatasetBundle:
    """Load a dataset bundle from a processed manifest directory or file.

    The builders in this repository persist arrays plus a JSON manifest rather
    than a pickled `DatasetBundle`. This loader reconstructs the bundle without
    changing any data-layer protocol.
    """

    manifest_path = Path(path)
    if manifest_path.is_dir():
        candidates = sorted(manifest_path.glob("*_processed_manifest.json"))
        if not candidates:
            raise FileNotFoundError(f"No processed manifest found under {manifest_path}")
        manifest_path = candidates[0]

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    processed_root = manifest_path.parent
    processed_files = payload.get("processed_files")
    if not isinstance(processed_files, Mapping):
        raise ValueError(f"{manifest_path} missing processed_files.")

    meta_payload = payload.get("bundle_meta") or payload.get("meta")
    artifact_payload = payload.get("bundle_artifacts") or payload.get("artifacts")
    if meta_payload is None:
        raise ValueError(f"{manifest_path} missing bundle meta payload.")

    split_payloads = {}
    for split_name in ("train", "val", "test"):
        split_payloads[split_name] = {
            "X": _load_array(processed_root, processed_files.get(f"{split_name}_X")),
            "Y": _load_array(processed_root, processed_files.get(f"{split_name}_Y")),
            "sample_id": _load_array(processed_root, processed_files.get(f"{split_name}_sample_id")),
            "run_id": _load_array(processed_root, processed_files.get(f"{split_name}_run_id")),
            "timestamp": _load_array(processed_root, processed_files.get(f"{split_name}_timestamp")),
            "meta": {},
        }

    return DatasetBundle(
        train=DatasetSplit.from_mapping(split_payloads["train"], split_name="train"),
        val=DatasetSplit.from_mapping(split_payloads["val"], split_name="val"),
        test=DatasetSplit.from_mapping(split_payloads["test"], split_name="test"),
        meta=DatasetMeta.from_mapping(meta_payload),
        artifacts=DatasetArtifacts.from_mapping(artifact_payload),
    )


def _slice_value(value: Any, size: Optional[int]) -> Any:
    if size is None or value is None:
        return value
    return value[:size]


def _slice_split(split: DatasetSplit, size: Optional[int]) -> DatasetSplit:
    return DatasetSplit(
        X=_slice_value(split.X, size),
        Y=_slice_value(split.Y, size),
        sample_id=_slice_value(split.sample_id, size),
        run_id=_slice_value(split.run_id, size),
        timestamp=_slice_value(split.timestamp, size),
        meta=dict(split.meta),
    )


def slice_dataset_bundle(
    bundle: DatasetBundle,
    *,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
) -> DatasetBundle:
    """Create a smaller bundle view for smoke tests or interactive debugging."""

    bundle = coerce_dataset_bundle(bundle)
    return DatasetBundle(
        train=_slice_split(bundle.train, train_size),
        val=_slice_split(bundle.val, val_size),
        test=_slice_split(bundle.test, test_size),
        meta=bundle.meta,
        artifacts=bundle.artifacts,
    )
