"""Shared builders for nonlinear benchmarks.

This module only defines reusable interfaces; concrete dataset builders must
implement raw loading and preprocessing in their own scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from data.adapters.nonlinear_adapter import NonlinearAdapter


@dataclass(frozen=True)
class NonlinearBuilderContext:
    """Canonical paths used across nonlinear builders."""

    dataset_name: str
    raw_root: Path
    interim_root: Path
    processed_root: Path
    splits_root: Path


@dataclass(frozen=True)
class SplitTransformStats:
    """Train-only standardization statistics for nonlinear splits."""

    mean: np.ndarray
    std: np.ndarray
    source_split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_split": self.source_split,
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "std": np.asarray(self.std, dtype=float).tolist(),
            "policy": "split_before_fit_transform",
        }


def fit_train_split_standardizer(train_split: Mapping[str, Any], *, epsilon: float = 1e-8) -> SplitTransformStats:
    train_x = np.asarray(train_split["X"], dtype=np.float64)
    if train_x.size == 0:
        raise ValueError("Cannot fit transform statistics on an empty training split.")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=0)
    std = np.where(std < float(epsilon), 1.0, std)
    return SplitTransformStats(mean=mean, std=std, source_split="train")


def transform_split_with_standardizer(
    split: Mapping[str, Any],
    stats: SplitTransformStats,
) -> dict[str, Any]:
    transformed = dict(split)
    transformed["X"] = (np.asarray(split["X"], dtype=np.float64) - stats.mean) / stats.std
    split_meta = dict(transformed.get("meta", {}))
    split_meta["transform_provenance"] = {
        "source_split": stats.source_split,
        "policy": "split_before_fit_transform",
    }
    transformed["meta"] = split_meta
    return transformed


def apply_train_only_standardization(
    *,
    train: Mapping[str, Any],
    val: Mapping[str, Any],
    test: Mapping[str, Any],
    epsilon: float = 1e-8,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], SplitTransformStats]:
    stats = fit_train_split_standardizer(train, epsilon=epsilon)
    return (
        transform_split_with_standardizer(train, stats),
        transform_split_with_standardizer(val, stats),
        transform_split_with_standardizer(test, stats),
        stats,
    )


class NonlinearBuilder(ABC):
    """Abstract skeleton for one nonlinear benchmark.

    Concrete builders should only plug dataset-specific IO/transforms here; bundle
    assembly remains shared through ``NonlinearAdapter``.
    """

    def __init__(self, context: NonlinearBuilderContext):
        self.context = context

    @abstractmethod
    def load_splits(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Return train, val, test mappings with ``X`` / ``Y`` payloads."""

    def assemble_bundle(
        self,
        *,
        meta: Optional[Mapping[str, Any]] = None,
        artifacts: Optional[Mapping[str, Any]] = None,
    ):
        """Assemble a unified DatasetBundle via adapter.

        meta can override manifest-driven defaults, and artifacts can point to
        extra files or object references for experiment reproducibility.
        """

        train, val, test = self.load_splits()
        enriched_meta = dict(meta or {})
        extras = dict(enriched_meta.get("extras", {}))
        extras.setdefault(
            "transform_provenance",
            {
                "source_split": "train",
                "policy": "split_before_fit_transform",
                "builder": type(self).__name__,
            },
        )
        enriched_meta["extras"] = extras
        enriched_artifacts = dict(artifacts or {})
        artifact_extra = dict(enriched_artifacts.get("extra", {}))
        artifact_extra.setdefault(
            "transform_provenance",
            {
                "source_split": "train",
                "policy": "split_before_fit_transform",
                "builder": type(self).__name__,
            },
        )
        enriched_artifacts["extra"] = artifact_extra
        return NonlinearAdapter.build_bundle(
            dataset_name=self.context.dataset_name,
            train=train,
            val=val,
            test=test,
            meta=enriched_meta,
            artifacts=enriched_artifacts,
        )
