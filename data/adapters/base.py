"""Data layer protocol and dataset bundle schema for experiment project.

This module defines a stable, task-agnostic schema used by all dataset
constructors. It intentionally contains no dataset-specific loading logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


class DataProtocolError(ValueError):
    """Raised when dataset bundle schema or metadata does not meet contract."""


class TaskFamily(str, Enum):
    """Supported data task families."""

    NONLINEAR = "nonlinear"
    HYDRAULIC = "hydraulic"
    TEP = "tep"


@dataclass(frozen=True)
class DatasetSplit:
    """Single split content.

    X / Y are intentionally typed as generic arrays; loaders may return np.ndarray,
    list-of-arrays, or any sequence-like object with len/shape.
    """

    X: Any
    Y: Any
    sample_id: Optional[Sequence[Any]] = None
    run_id: Optional[Sequence[Any]] = None
    timestamp: Optional[Sequence[Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, split_name: str) -> "DatasetSplit":
        if not isinstance(value, Mapping):
            raise DataProtocolError(f"[{split_name}] expected mapping for split payload.")
        if "X" not in value or "Y" not in value:
            raise DataProtocolError(f"[{split_name}] requires keys: X, Y.")
        return cls(
            X=value["X"],
            Y=value["Y"],
            sample_id=value.get("sample_id"),
            run_id=value.get("run_id"),
            timestamp=value.get("timestamp"),
            meta=dict(value.get("meta", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetMeta:
    """Metadata fields required by unified data interface."""

    dataset_name: str
    task_family: TaskFamily
    input_dim: int
    output_dim: int
    window_length: int
    horizon: int
    split_protocol: str
    has_ground_truth_kernel: bool
    has_ground_truth_gfrf: bool
    extras: Dict[str, Any] = field(default_factory=dict)

    REQUIRED_FIELDS = (
        "dataset_name",
        "task_family",
        "input_dim",
        "output_dim",
        "window_length",
        "horizon",
        "split_protocol",
        "has_ground_truth_kernel",
        "has_ground_truth_gfrf",
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetMeta":
        missing = [f for f in cls.REQUIRED_FIELDS if f not in value]
        if missing:
            raise DataProtocolError(
                f"Dataset meta missing required fields: {', '.join(missing)}."
            )

        task_family = value["task_family"]
        if isinstance(task_family, str):
            try:
                task_family = TaskFamily(task_family)
            except ValueError as exc:
                raise DataProtocolError(f"Unknown task_family: {task_family}") from exc

        return cls(
            dataset_name=str(value["dataset_name"]),
            task_family=task_family,
            input_dim=int(value["input_dim"]),
            output_dim=int(value["output_dim"]),
            window_length=int(value["window_length"]),
            horizon=int(value["horizon"]),
            split_protocol=str(value["split_protocol"]),
            has_ground_truth_kernel=bool(value["has_ground_truth_kernel"]),
            has_ground_truth_gfrf=bool(value["has_ground_truth_gfrf"]),
            extras=dict(value.get("extras", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["task_family"] = self.task_family.value
        return raw


@dataclass(frozen=True)
class DatasetArtifacts:
    """Artifact references for reproducibility and audit.

    These fields describe protocol/groupping/truth materials and can be file paths
    or object references supported by the loader.
    """

    truth_file: Optional[str] = None
    grouping_file: Optional[str] = None
    protocol_file: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Optional[Mapping[str, Any]]) -> "DatasetArtifacts":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise DataProtocolError("artifacts must be a mapping.")
        return cls(
            truth_file=value.get("truth_file"),
            grouping_file=value.get("grouping_file"),
            protocol_file=value.get("protocol_file"),
            extra=dict(value.get("extra", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetBundle:
    """统一输出格式：train / val / test / meta / artifacts."""

    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    meta: DatasetMeta
    artifacts: DatasetArtifacts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "meta": self.meta.to_dict(),
            "artifacts": self.artifacts.to_dict(),
        }


class BaseDatasetAdapter:
    """Base adapter with no IO implementation; only schema binding and validation."""

    task_family: TaskFamily = TaskFamily.NONLINEAR
    default_split_protocol: str = "time_series_temporal_holdout_v1"
    # Keep as optional to avoid fabricating nonexistent dataset names.
    supported_dataset_names = None

    @classmethod
    def build_bundle(
        cls,
        dataset_name: str,
        train: Mapping[str, Any],
        val: Mapping[str, Any],
        test: Mapping[str, Any],
        meta: Mapping[str, Any],
        artifacts: Optional[Mapping[str, Any]] = None,
    ) -> DatasetBundle:
        """Assemble a standardized DatasetBundle.

        This method only validates and normalizes schema; concrete parsing/loading
        must be implemented in dedicated builder scripts under data/builders/.
        """

        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise DataProtocolError("dataset_name must be a non-empty string.")
        if cls.supported_dataset_names and dataset_name not in cls.supported_dataset_names:
            raise DataProtocolError(
                f"{cls.__name__} only supports: {', '.join(sorted(cls.supported_dataset_names))}"
            )

        meta_dict = dict(meta)
        meta_dict.setdefault("dataset_name", dataset_name)
        meta_dict.setdefault("task_family", cls.task_family.value)
        meta_dict.setdefault("split_protocol", cls.default_split_protocol)

        return DatasetBundle(
            train=DatasetSplit.from_mapping(train, split_name="train"),
            val=DatasetSplit.from_mapping(val, split_name="val"),
            test=DatasetSplit.from_mapping(test, split_name="test"),
            meta=DatasetMeta.from_mapping(meta_dict),
            artifacts=DatasetArtifacts.from_mapping(artifacts),
        )

