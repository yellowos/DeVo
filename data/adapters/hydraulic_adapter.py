"""Adapter contract for hydraulic datasets.

No IO is implemented in this file. Builders should parse/transform source data first
and then pass normalized fields to `build_bundle`.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .base import BaseDatasetAdapter, DataProtocolError, TaskFamily


class HydraulicAdapter(BaseDatasetAdapter):
    """Hydraulic task adapter."""

    task_family = TaskFamily.HYDRAULIC
    default_split_protocol = "hydraulic_system_temporal_holdout_v1"
    # Keep domain-open here to avoid inventing non-existent exact dataset names.
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
    ):
        if cls.task_family != TaskFamily.HYDRAULIC:
            raise DataProtocolError("hydraulic adapter task_family mismatch.")
        return super().build_bundle(
            dataset_name=dataset_name,
            train=train,
            val=val,
            test=test,
            meta=meta,
            artifacts=artifacts,
        )

