"""Shared builders for nonlinear benchmarks.

This module only defines reusable interfaces; concrete dataset builders must
implement raw loading and preprocessing in their own scripts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from data.adapters.nonlinear_adapter import NonlinearAdapter


@dataclass(frozen=True)
class NonlinearBuilderContext:
    """Canonical paths used across nonlinear builders."""

    dataset_name: str
    raw_root: Path
    interim_root: Path
    processed_root: Path
    splits_root: Path


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
        return NonlinearAdapter.build_bundle(
            dataset_name=self.context.dataset_name,
            train=train,
            val=val,
            test=test,
            meta=dict(meta or {}),
            artifacts=dict(artifacts or {}),
        )
