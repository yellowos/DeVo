"""Adapter contract for TEP datasets.

No IO is implemented in this file. Builders are responsible for source ingestion
and transformation; this adapter only standardizes the output contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .base import BaseDatasetAdapter, DataProtocolError, TaskFamily


class TEPAdapter(BaseDatasetAdapter):
    """TEP task adapter."""

    task_family = TaskFamily.TEP
    default_split_protocol = "tep_process_and_fault_holdout_v1"
    # Keep domain-open here to avoid inventing non-existent exact dataset names.
    supported_dataset_names = None

    @classmethod
    def load_processed_bundle(cls, processed_root: str | Path):
        processed_root = Path(processed_root).expanduser().resolve()
        manifest_path = processed_root / "tep_processed_manifest.json"
        if not manifest_path.exists():
            raise DataProtocolError(f"tep processed manifest missing at {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        files = manifest.get("processed_files")
        if not isinstance(files, Mapping):
            raise DataProtocolError("tep processed manifest missing processed_files.")

        meta = manifest.get("bundle_meta") or manifest.get("meta")
        artifacts = manifest.get("bundle_artifacts") or manifest.get("artifacts")
        if not isinstance(meta, Mapping):
            raise DataProtocolError("tep processed manifest missing bundle_meta/meta.")

        splits: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "val", "test"):
            split_payload: dict[str, Any] = {}
            split_meta: dict[str, Any] = {}
            for key in (
                "X",
                "Y",
                "sample_id",
                "run_id",
                "timestamp",
                "mode",
                "fault_id",
                "scenario_id",
                "window_idx",
                "window_start",
                "window_end",
                "idv_aux",
                "idv_label",
            ):
                file_key = f"{split_name}_{key}"
                if file_key in files:
                    value = np.load(processed_root / str(files[file_key]), allow_pickle=True)
                    if key in {"X", "Y", "sample_id", "run_id", "timestamp"}:
                        split_payload[key] = value
                    else:
                        split_meta[key] = value
            if "X" not in split_payload or "Y" not in split_payload:
                raise DataProtocolError(f"tep processed manifest missing {split_name}_X or {split_name}_Y.")
            if split_meta:
                split_payload["meta"] = split_meta
            splits[split_name] = split_payload

        return cls.build_bundle(
            dataset_name=str(manifest.get("dataset_name", "tep")),
            train=splits["train"],
            val=splits["val"],
            test=splits["test"],
            meta=meta,
            artifacts=artifacts,
        )

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
