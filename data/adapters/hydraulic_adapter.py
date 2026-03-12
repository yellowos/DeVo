"""Adapter contract for hydraulic datasets.

No IO is implemented in this file. Builders should parse/transform source data first
and then pass normalized fields to `build_bundle`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .base import BaseDatasetAdapter, DataProtocolError, TaskFamily


class HydraulicAdapter(BaseDatasetAdapter):
    """Hydraulic task adapter."""

    task_family = TaskFamily.HYDRAULIC
    default_split_protocol = "hydraulic_cycle_bundle_holdout_v1"
    # Keep domain-open here to avoid inventing non-existent exact dataset names.
    supported_dataset_names = None

    @classmethod
    def load_processed_bundle(cls, processed_root: str | Path, representation: str | None = None):
        processed_root = Path(processed_root).expanduser().resolve()
        if representation is not None and (processed_root / representation).is_dir():
            processed_root = processed_root / representation

        manifest_path = processed_root / "hydraulic_processed_manifest.json"
        if not manifest_path.exists():
            index_path = processed_root / "hydraulic_representations_manifest.json"
            if not index_path.exists():
                raise DataProtocolError(f"hydraulic processed manifest missing at {manifest_path}")
            with index_path.open("r", encoding="utf-8") as f:
                index_payload = json.load(f)
            rep_name = representation or index_payload.get("default_representation")
            representations = index_payload.get("representations", {})
            if rep_name not in representations:
                raise DataProtocolError(f"unknown hydraulic representation `{rep_name}`.")
            manifest_path = Path(str(representations[rep_name]["processed_manifest"])).expanduser().resolve()
            processed_root = manifest_path.parent

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        files = manifest.get("processed_files")
        if not isinstance(files, Mapping):
            raise DataProtocolError("hydraulic processed manifest missing processed_files.")

        meta = manifest.get("bundle_meta") or manifest.get("meta")
        artifacts = manifest.get("bundle_artifacts") or manifest.get("artifacts")
        if not isinstance(meta, Mapping):
            raise DataProtocolError("hydraulic processed manifest missing bundle_meta/meta.")

        splits: dict[str, dict[str, Any]] = {}
        for split_name in ("train", "val", "test"):
            split_payload: dict[str, Any] = {}
            for key in ("X", "Y", "sample_id", "timestamp"):
                file_key = f"{split_name}_{key}"
                if file_key not in files:
                    raise DataProtocolError(f"hydraulic processed manifest missing {file_key}.")
                split_payload[key] = np.load(processed_root / str(files[file_key]), allow_pickle=True)
            splits[split_name] = split_payload

        return cls.build_bundle(
            dataset_name=str(manifest.get("dataset_name", "hydraulic")),
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
