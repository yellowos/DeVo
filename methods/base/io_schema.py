"""Compatibility IO schema for reading data-layer dataset bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TypeAlias

import numpy as np

from data.adapters.base import DataProtocolError, DatasetArtifacts, DatasetBundle, DatasetMeta, DatasetSplit
from data.checks.bundle_checks import check_dataset_bundle

PathLike: TypeAlias = str | Path

_STANDARD_SPLIT_KEYS = {"X", "Y", "sample_id", "run_id", "timestamp", "meta"}

# Fields that may contain object arrays (e.g. string identifiers) and therefore
# require ``allow_pickle=True`` when loaded via ``np.load``.
_PICKLE_ALLOWED_FIELDS = {"sample_id", "run_id", "timestamp", "meta"}


def _load_npy_with_pickle_fallback(path: Path, *, allow_pickle: bool) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=allow_pickle)
    except ValueError as exc:
        message = str(exc)
        if allow_pickle or "Object arrays cannot be loaded when allow_pickle=False" not in message:
            raise
        return np.load(path, allow_pickle=True)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise DataProtocolError(f"Expected JSON object at {path}.")
    return payload


def _manifest_candidates(processed_root: Path, dataset_name: Optional[str]) -> list[Path]:
    candidates: list[Path] = []
    if dataset_name:
        candidates.append(processed_root / f"{dataset_name}_processed_manifest.json")
    candidates.extend(
        [
            processed_root / "processed_manifest.json",
            processed_root / "manifest.json",
        ]
    )
    return candidates


def _resolve_manifest_path(source: Path, dataset_name: Optional[str]) -> Path:
    if source.is_file():
        return source
    for candidate in _manifest_candidates(source, dataset_name):
        if candidate.exists():
            return candidate
    by_glob = sorted(source.glob("*processed_manifest.json"))
    if by_glob:
        return by_glob[0]
    any_json = sorted(source.glob("*.json"))
    if any_json:
        return any_json[0]
    raise FileNotFoundError(f"Cannot locate processed manifest under {source}.")


def _bundle_meta_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("bundle_meta")
    if payload is None:
        payload = manifest.get("meta")
    if not isinstance(payload, Mapping):
        raise DataProtocolError("Processed manifest missing bundle_meta/meta mapping.")
    return payload


def _bundle_artifacts_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("bundle_artifacts")
    if payload is None:
        payload = manifest.get("artifacts", {})
    if not isinstance(payload, Mapping):
        raise DataProtocolError("Processed manifest has invalid bundle_artifacts/artifacts payload.")
    return payload


def _processed_files_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = manifest.get("processed_files")
    if isinstance(payload, Mapping):
        return payload
    payload = manifest.get("files")
    if isinstance(payload, Mapping):
        return payload
    return {}


def _normalize_split_field(raw_name: str) -> str:
    key = raw_name.strip()
    lowered = key.lower()
    if lowered == "x":
        return "X"
    if lowered == "y":
        return "Y"
    return lowered


def _default_field_candidates(split_name: str, field_name: str) -> list[str]:
    token = field_name if field_name in {"X", "Y"} else field_name.lower()
    candidates = [
        f"{split_name}_{token}.npy",
        f"{split_name}_{token.lower()}.npy",
        f"{split_name}_{token.upper()}.npy",
    ]
    return list(dict.fromkeys(candidates))


def _load_split_payload(
    processed_root: Path,
    processed_files: Mapping[str, Any],
    split_name: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    prefix = f"{split_name}_"
    for key, value in processed_files.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not key.lower().startswith(prefix):
            continue
        field_name = _normalize_split_field(key[len(prefix) :])
        array_path = Path(value)
        if not array_path.is_absolute():
            array_path = processed_root / value
        if array_path.suffix.lower() != ".npy":
            continue
        if not array_path.exists():
            raise FileNotFoundError(f"Missing processed file: {array_path}")
        payload[field_name] = _load_npy_with_pickle_fallback(
            array_path,
            allow_pickle=(field_name in _PICKLE_ALLOWED_FIELDS),
        )

    for field_name in ("X", "Y", "sample_id", "run_id", "timestamp"):
        if field_name in payload:
            continue
        for candidate in _default_field_candidates(split_name, field_name):
            array_path = processed_root / candidate
            if array_path.exists():
                payload[field_name] = _load_npy_with_pickle_fallback(
                    array_path,
                    allow_pickle=(field_name in _PICKLE_ALLOWED_FIELDS),
                )
                break

    if "X" not in payload or "Y" not in payload:
        raise DataProtocolError(f"{split_name} split requires X and Y arrays.")
    return payload


def _coerce_meta(value: DatasetMeta | Mapping[str, Any]) -> DatasetMeta:
    if isinstance(value, DatasetMeta):
        return value
    if isinstance(value, Mapping):
        return DatasetMeta.from_mapping(value)
    raise DataProtocolError("meta must be DatasetMeta or mapping.")


def _coerce_artifacts(value: DatasetArtifacts | Mapping[str, Any] | None) -> DatasetArtifacts:
    if isinstance(value, DatasetArtifacts):
        return value
    return DatasetArtifacts.from_mapping(value)


def _flatten_optional(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=object).reshape(-1)


def _numeric_if_possible(value: np.ndarray) -> Optional[np.ndarray]:
    try:
        numeric = value.astype(np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric).all():
        return None
    return numeric


def validate_split_disjointness(bundle: "MethodDatasetBundle") -> None:
    split_names = ("train", "val", "test")
    alignment_available = any(
        any(key in (bundle.get_split(name).extra_fields or {}) for key in ("window_start", "window_end"))
        for name in split_names
    )
    fingerprints: dict[str, Optional[set[tuple[Any, ...] | Any]]] = {}
    for split_name in split_names:
        split = bundle.get_split(split_name)
        sample_id = _flatten_optional(split.sample_id)
        run_id = _flatten_optional(split.run_id)
        window_start = _flatten_optional(split.extra_fields.get("window_start"))
        window_end = _flatten_optional(split.extra_fields.get("window_end"))
        if run_id is not None and window_start is not None and window_end is not None:
            fingerprints[split_name] = set(zip(run_id.tolist(), window_start.tolist(), window_end.tolist()))
            continue
        if sample_id is not None and run_id is not None and (alignment_available or bundle.source_manifest is not None):
            fingerprints[split_name] = set(zip(run_id.tolist(), sample_id.tolist()))
            continue
        fingerprints[split_name] = None

    for index, left_name in enumerate(split_names):
        left = fingerprints[left_name]
        if left is None:
            continue
        for right_name in split_names[index + 1 :]:
            right = fingerprints[right_name]
            if right is None:
                continue
            overlap = left & right
            if overlap:
                raise DataProtocolError(
                    f"Dataset bundle has overlapping {left_name}/{right_name} splits; "
                    f"overlap size={len(overlap)}."
                )


def validate_temporal_order(split: "MethodDatasetSplit", *, split_name: str) -> None:
    timestamp = _flatten_optional(split.timestamp)
    run_id = _flatten_optional(split.run_id)
    if timestamp is not None:
        numeric_timestamp = _numeric_if_possible(timestamp)
        if numeric_timestamp is None:
            return
        if run_id is None:
            if np.any(np.diff(numeric_timestamp) < 0):
                raise DataProtocolError(f"{split_name}: timestamp must be monotonic.")
        else:
            unique_runs = []
            for item in run_id.tolist():
                if item not in unique_runs:
                    unique_runs.append(item)
            for run_value in unique_runs:
                mask = run_id == run_value
                run_ts = numeric_timestamp[mask]
                if run_ts.size > 1 and np.any(np.diff(run_ts) < 0):
                    raise DataProtocolError(
                        f"{split_name}: timestamp must be monotonic within run_id={run_value!r}."
                    )


def validate_run_window_boundaries(split: "MethodDatasetSplit", meta: DatasetMeta, *, split_name: str) -> None:
    window_start = _flatten_optional(split.extra_fields.get("window_start"))
    window_end = _flatten_optional(split.extra_fields.get("window_end"))
    target_index = _flatten_optional(split.extra_fields.get("target_index"))
    window_run_id = _flatten_optional(split.extra_fields.get("window_run_id"))
    target_run_id = _flatten_optional(split.extra_fields.get("target_run_id"))
    run_length = _flatten_optional(split.extra_fields.get("run_length"))

    if window_start is None and window_end is None and target_index is None:
        return
    if window_start is None or window_end is None:
        raise DataProtocolError(f"{split_name}: window_start/window_end must be provided together.")

    sample_count = split.num_samples
    for field_name, value in (
        ("window_start", window_start),
        ("window_end", window_end),
        ("target_index", target_index),
        ("window_run_id", window_run_id),
        ("target_run_id", target_run_id),
        ("run_length", run_length),
    ):
        if value is not None and len(value) != sample_count:
            raise DataProtocolError(
                f"{split_name}: {field_name} length {len(value)} must match sample count {sample_count}."
            )

    start_int = window_start.astype(np.int64)
    end_int = window_end.astype(np.int64)
    if np.any(start_int < 0) or np.any(end_int < start_int):
        raise DataProtocolError(f"{split_name}: invalid window boundaries.")
    expected_window = int(meta.window_length)
    if np.any((end_int - start_int + 1) != expected_window):
        raise DataProtocolError(
            f"{split_name}: window boundaries must span exactly window_length={expected_window}."
        )

    if target_index is not None:
        target_int = target_index.astype(np.int64)
        if np.any(target_int <= end_int):
            raise DataProtocolError(f"{split_name}: target_index must be strictly after window_end.")
        if np.any((target_int - end_int) != int(meta.horizon)):
            raise DataProtocolError(
                f"{split_name}: target_index must satisfy target_index-window_end == horizon={meta.horizon}."
            )

    run_id = _flatten_optional(split.run_id)
    if run_id is not None and window_run_id is not None and np.any(window_run_id != run_id):
        raise DataProtocolError(f"{split_name}: window_run_id must match run_id for every sample.")
    if run_id is not None and target_run_id is not None and np.any(target_run_id != run_id):
        raise DataProtocolError(f"{split_name}: target_run_id must match run_id for every sample.")
    if run_length is not None and target_index is not None:
        if np.any(target_index.astype(np.int64) >= run_length.astype(np.int64)):
            raise DataProtocolError(f"{split_name}: target_index exceeds run_length boundary.")


@dataclass(frozen=True)
class MethodDatasetSplit:
    """Methods-layer view of one dataset split."""

    X: Any
    Y: Any
    sample_id: Optional[Sequence[Any]] = None
    run_id: Optional[Sequence[Any]] = None
    timestamp: Optional[Sequence[Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, split_name: str) -> "MethodDatasetSplit":
        data_split = DatasetSplit.from_mapping(value, split_name=split_name)
        extra_fields = {key: item for key, item in value.items() if key not in _STANDARD_SPLIT_KEYS}
        return cls(
            X=data_split.X,
            Y=data_split.Y,
            sample_id=data_split.sample_id,
            run_id=data_split.run_id,
            timestamp=data_split.timestamp,
            meta=dict(data_split.meta),
            extra_fields=extra_fields,
        )

    @classmethod
    def from_data_split(cls, split: DatasetSplit) -> "MethodDatasetSplit":
        return cls(
            X=split.X,
            Y=split.Y,
            sample_id=split.sample_id,
            run_id=split.run_id,
            timestamp=split.timestamp,
            meta=dict(split.meta),
        )

    @property
    def num_samples(self) -> int:
        return len(self.X)

    def to_data_split(self) -> DatasetSplit:
        return DatasetSplit(
            X=self.X,
            Y=self.Y,
            sample_id=self.sample_id,
            run_id=self.run_id,
            timestamp=self.timestamp,
            meta=dict(self.meta),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "X": self.X,
            "Y": self.Y,
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "meta": dict(self.meta),
        }
        payload.update(self.extra_fields)
        return payload


@dataclass(frozen=True)
class MethodDatasetBundle:
    """Methods-layer compatible dataset bundle."""

    train: MethodDatasetSplit
    val: MethodDatasetSplit
    test: MethodDatasetSplit
    meta: DatasetMeta
    artifacts: DatasetArtifacts
    source_manifest: Optional[str] = None
    source_root: Optional[str] = None

    @classmethod
    def from_data_bundle(cls, bundle: DatasetBundle) -> "MethodDatasetBundle":
        check_dataset_bundle(bundle)
        normalized = cls(
            train=MethodDatasetSplit.from_data_split(bundle.train),
            val=MethodDatasetSplit.from_data_split(bundle.val),
            test=MethodDatasetSplit.from_data_split(bundle.test),
            meta=bundle.meta,
            artifacts=bundle.artifacts,
        )
        validate_split_disjointness(normalized)
        for split_name in ("train", "val", "test"):
            validate_temporal_order(normalized.get_split(split_name), split_name=split_name)
        return normalized

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_manifest: Optional[str] = None,
        source_root: Optional[str] = None,
    ) -> "MethodDatasetBundle":
        required = ("train", "val", "test")
        missing = [name for name in required if name not in value]
        if missing:
            raise DataProtocolError(f"Dataset bundle missing splits: {', '.join(missing)}.")

        meta_key = "meta" if "meta" in value else "bundle_meta"
        artifacts_key = "artifacts" if "artifacts" in value else "bundle_artifacts"
        if meta_key not in value:
            raise DataProtocolError("Dataset bundle mapping requires meta or bundle_meta.")

        bundle = cls(
            train=MethodDatasetSplit.from_mapping(value["train"], split_name="train"),
            val=MethodDatasetSplit.from_mapping(value["val"], split_name="val"),
            test=MethodDatasetSplit.from_mapping(value["test"], split_name="test"),
            meta=_coerce_meta(value[meta_key]),
            artifacts=_coerce_artifacts(value.get(artifacts_key)),
            source_manifest=source_manifest,
            source_root=source_root,
        )
        check_dataset_bundle(bundle.to_data_bundle())
        validate_split_disjointness(bundle)
        for split_name in ("train", "val", "test"):
            split = bundle.get_split(split_name)
            validate_temporal_order(split, split_name=split_name)
            validate_run_window_boundaries(split, bundle.meta, split_name=split_name)
        return bundle

    def get_split(self, split_name: str) -> MethodDatasetSplit:
        normalized = split_name.strip().lower()
        if normalized not in {"train", "val", "test"}:
            raise KeyError(f"Unknown split: {split_name}")
        return getattr(self, normalized)

    def to_data_bundle(self) -> DatasetBundle:
        return DatasetBundle(
            train=self.train.to_data_split(),
            val=self.val.to_data_split(),
            test=self.test.to_data_split(),
            meta=self.meta,
            artifacts=self.artifacts,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "meta": self.meta.to_dict(),
            "artifacts": self.artifacts.to_dict(),
            "source_manifest": self.source_manifest,
            "source_root": self.source_root,
        }


DatasetBundleSource: TypeAlias = MethodDatasetBundle | DatasetBundle | Mapping[str, Any] | PathLike


def load_dataset_bundle(
    source: DatasetBundleSource,
    *,
    dataset_name: Optional[str] = None,
) -> MethodDatasetBundle:
    """Load a methods-layer bundle from a data-layer bundle, mapping, or processed manifest."""

    if isinstance(source, MethodDatasetBundle):
        return source
    if isinstance(source, DatasetBundle):
        return MethodDatasetBundle.from_data_bundle(source)
    if isinstance(source, Mapping):
        return MethodDatasetBundle.from_mapping(source)

    source_path = Path(source).expanduser().resolve()
    manifest_path = _resolve_manifest_path(source_path, dataset_name)
    manifest_payload = _read_json(manifest_path)
    processed_root = manifest_path.parent
    processed_files = _processed_files_payload(manifest_payload)
    bundle_payload = {
        "train": _load_split_payload(processed_root, processed_files, "train"),
        "val": _load_split_payload(processed_root, processed_files, "val"),
        "test": _load_split_payload(processed_root, processed_files, "test"),
        "meta": _bundle_meta_payload(manifest_payload),
        "artifacts": _bundle_artifacts_payload(manifest_payload),
    }
    return MethodDatasetBundle.from_mapping(
        bundle_payload,
        source_manifest=str(manifest_path),
        source_root=str(processed_root),
    )


def coerce_dataset_bundle(
    source: DatasetBundleSource,
    *,
    dataset_name: Optional[str] = None,
) -> MethodDatasetBundle:
    """Normalize any supported data-layer input into MethodDatasetBundle."""

    return load_dataset_bundle(source, dataset_name=dataset_name)
