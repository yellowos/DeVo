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
        payload[field_name] = np.load(array_path, allow_pickle=(field_name in _PICKLE_ALLOWED_FIELDS))

    for field_name in ("X", "Y", "sample_id", "run_id", "timestamp"):
        if field_name in payload:
            continue
        for candidate in _default_field_candidates(split_name, field_name):
            array_path = processed_root / candidate
            if array_path.exists():
                payload[field_name] = np.load(array_path, allow_pickle=(field_name in _PICKLE_ALLOWED_FIELDS))
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
        return cls(
            train=MethodDatasetSplit.from_data_split(bundle.train),
            val=MethodDatasetSplit.from_data_split(bundle.val),
            test=MethodDatasetSplit.from_data_split(bundle.test),
            meta=bundle.meta,
            artifacts=bundle.artifacts,
        )

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
