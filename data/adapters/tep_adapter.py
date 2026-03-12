"""Adapter contract for TEP datasets.

No IO is implemented in this file. Builders are responsible for source ingestion
and transformation; this adapter only standardizes the output contract.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .base import BaseDatasetAdapter, DataProtocolError, TaskFamily


class TEPAdapter(BaseDatasetAdapter):
    """TEP task adapter."""

    task_family = TaskFamily.TEP
    default_split_protocol = "tep_process_and_fault_holdout_v1"
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
        if cls.task_family != TaskFamily.TEP:
            raise DataProtocolError("tep adapter task_family mismatch.")
        return super().build_bundle(
            dataset_name=dataset_name,
            train=train,
            val=val,
            test=test,
            meta=meta,
            artifacts=artifacts,
        )

