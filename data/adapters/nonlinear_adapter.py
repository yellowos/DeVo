"""Adapter contract for nonlinear datasets.

No file loading is performed here. Builders assemble raw/interim data elsewhere and
reuse this adapter only to normalize into the shared DatasetBundle schema.

The Nonlinear adapter is the single assembly point for common protocol settings:

- benchmark-level manifest lookup
- defaults for meta fields used across methods
- kernel / GFRF truth registration references
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import BaseDatasetAdapter, DataProtocolError, TaskFamily


_ADAPTER_DIR = Path(__file__).resolve().parent
_METADATA_DIR = _ADAPTER_DIR.parent / "metadata" / "nonlinear"
_BENCHMARK_MANIFEST = _METADATA_DIR / "benchmark_manifest.json"
_KERNEL_MANIFEST = _METADATA_DIR / "kernel_truth_manifest.json"
_GFRF_MANIFEST = _METADATA_DIR / "gfrf_truth_manifest.json"
_PROTOCOL_DIR = _METADATA_DIR / "protocols"


_BENCHMARK_REQUIRED_KEYS = {
    "benchmark_name",
    "system_type",
    "task_usage",
    "input_channels",
    "output_channels",
    "default_window_length",
    "default_horizon",
    "has_ground_truth_kernel",
    "has_ground_truth_gfrf",
    "recommended_split_protocol",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise DataProtocolError(f"Manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_benchmark_entries(raw: Mapping[str, Any], *, expected_key: str = "benchmarks") -> dict[str, Any]:
    payload = dict(raw)
    entries = payload.get(expected_key, [])
    if not isinstance(entries, list):
        raise DataProtocolError(f"{expected_key} must be a list.")

    lookup: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise DataProtocolError("Each benchmark manifest entry must be a mapping.")
        name = str(entry.get("benchmark_name", "")).strip()
        if not name:
            raise DataProtocolError("Benchmark manifest entry missing benchmark_name.")
        lookup[name] = dict(entry)
    return lookup


def _load_manifests() -> dict[str, Any]:
    benchmark_payload = _read_json(_BENCHMARK_MANIFEST)
    kernel_payload = _read_json(_KERNEL_MANIFEST)
    gfrf_payload = _read_json(_GFRF_MANIFEST)

    benchmark_lookup = _normalise_benchmark_entries(benchmark_payload)
    kernel_lookup = _normalise_benchmark_entries(kernel_payload, expected_key="benchmarks")
    gfrf_lookup = _normalise_benchmark_entries(gfrf_payload, expected_key="benchmarks")

    return {
        "benchmark": benchmark_lookup,
        "kernel": kernel_lookup,
        "gfrf": gfrf_lookup,
        "benchmark_manifest": benchmark_payload,
        "kernel_manifest": kernel_payload,
        "gfrf_manifest": gfrf_payload,
    }


class NonlinearAdapter(BaseDatasetAdapter):
    """Nonlinear task adapter."""

    task_family = TaskFamily.NONLINEAR
    default_split_protocol = "nonlinear_temporal_grouped_holdout_v1"
    # Keep this open by default. Actual validation uses benchmark manifest records.
    supported_dataset_names = None
    _manifests: Optional[dict[str, Any]] = None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear manifest cache for dynamic manifest hot-reload in experiments."""
        cls._manifests = None

    @classmethod
    def _ensure_manifests(cls) -> dict[str, Any]:
        if cls._manifests is None:
            cls._manifests = _load_manifests()
        return cls._manifests

    @classmethod
    def _validate_manifest_entry(cls, entry: Mapping[str, Any]) -> None:
        missing = sorted(_BENCHMARK_REQUIRED_KEYS.difference(entry.keys()))
        if missing:
            raise DataProtocolError(
                f"Benchmark manifest entry [{entry.get('benchmark_name', 'unknown')}] "
                f"missing keys: {', '.join(missing)}."
            )
        if not isinstance(entry["task_usage"], list):
            raise DataProtocolError("task_usage must be a list.")
        if not isinstance(entry["input_channels"], list) or not entry["input_channels"]:
            raise DataProtocolError("input_channels must be a non-empty list.")
        if not isinstance(entry["output_channels"], list) or not entry["output_channels"]:
            raise DataProtocolError("output_channels must be a non-empty list.")
        if not isinstance(entry["default_window_length"], int) or entry["default_window_length"] <= 0:
            raise DataProtocolError("default_window_length must be a positive integer.")
        if not isinstance(entry["default_horizon"], int) or entry["default_horizon"] <= 0:
            raise DataProtocolError("default_horizon must be a positive integer.")
        if not isinstance(entry["has_ground_truth_kernel"], bool):
            raise DataProtocolError("has_ground_truth_kernel must be bool.")
        if not isinstance(entry["has_ground_truth_gfrf"], bool):
            raise DataProtocolError("has_ground_truth_gfrf must be bool.")
        if not str(entry["recommended_split_protocol"]).strip():
            raise DataProtocolError("recommended_split_protocol must be non-empty.")

    @classmethod
    def _get_manifest_entry(cls, dataset_name: str) -> dict[str, Any]:
        manifests = cls._ensure_manifests()
        benchmark_lookup = manifests["benchmark"]
        if dataset_name not in benchmark_lookup:
            available = ", ".join(sorted(benchmark_lookup.keys()))
            raise DataProtocolError(
                f"Unknown nonlinear dataset '{dataset_name}'. Available: {available}"
            )
        entry = benchmark_lookup[dataset_name]
        cls._validate_manifest_entry(entry)
        return dict(entry)

    @classmethod
    def _get_kernel_entry(cls, dataset_name: str) -> dict[str, Any]:
        manifests = cls._ensure_manifests()
        kernel_lookup = manifests["kernel"]
        entry = kernel_lookup.get(dataset_name)
        if entry is None:
            raise DataProtocolError(
                f"kernel_truth_manifest missing benchmark '{dataset_name}'."
            )
        if not isinstance(entry.get("has_ground_truth_kernel"), bool):
            raise DataProtocolError("kernel truth manifest: has_ground_truth_kernel must be bool.")
        return dict(entry)

    @classmethod
    def _get_gfrf_entry(cls, dataset_name: str) -> dict[str, Any]:
        manifests = cls._ensure_manifests()
        gfrf_lookup = manifests["gfrf"]
        entry = gfrf_lookup.get(dataset_name)
        if entry is None:
            raise DataProtocolError(
                f"gfrf_truth_manifest missing benchmark '{dataset_name}'."
            )
        if not isinstance(entry.get("has_ground_truth_gfrf"), bool):
            raise DataProtocolError("gfrf truth manifest: has_ground_truth_gfrf must be bool.")
        return dict(entry)

    @classmethod
    def _resolve_protocol_file(cls, split_protocol: str) -> Optional[str]:
        protocol_path = _PROTOCOL_DIR / f"{split_protocol}.json"
        if protocol_path.exists():
            return str(protocol_path)
        return None

    @classmethod
    def _merge_artifacts(
        cls,
        dataset_name: str,
        manifest_entry: Mapping[str, Any],
        *,
        artifacts: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        split_protocol = str(manifest_entry["recommended_split_protocol"])
        kernel_entry = cls._get_kernel_entry(dataset_name)
        gfrf_entry = cls._get_gfrf_entry(dataset_name)

        merged = {
            "protocol_file": cls._resolve_protocol_file(split_protocol),
            "truth_file": kernel_entry.get("kernel_reference"),
            "grouping_file": f"nonlinear://protocols/{split_protocol}",
            "extra": {
                "split_protocol": split_protocol,
                "split_protocol_name": split_protocol,
                "system_type": manifest_entry["system_type"],
                "task_usage": manifest_entry["task_usage"],
                "transform_provenance": {
                    "source_split": "train",
                    "policy": "split_before_fit_transform",
                    "adapter": cls.__name__,
                },
                "kernel_truth": kernel_entry,
                "gfrf_truth": gfrf_entry,
            },
        }

        # Keep artifact fields as explicit override points for builder-level customisation.
        if artifacts:
            merged.update(dict(artifacts))
            merged["extra"] = {
                **merged["extra"],
                **dict(artifacts).get("extra", {}),
            }
        return merged

    @classmethod
    def _merge_meta(
        cls,
        dataset_name: str,
        manifest_entry: Mapping[str, Any],
        *,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        meta = dict(meta or {})
        extras_override = dict(meta.get("extras", {}))
        transform_provenance = dict(extras_override.get("transform_provenance", {}))
        if transform_provenance and str(transform_provenance.get("source_split", "train")) != "train":
            raise DataProtocolError(
                f"{dataset_name}: transform provenance must declare source_split='train'."
            )
        merged_meta: dict[str, Any] = {
            "dataset_name": dataset_name,
            "task_family": cls.task_family.value,
            "input_dim": int(len(manifest_entry["input_channels"])),
            "output_dim": int(len(manifest_entry["output_channels"])),
            "window_length": int(manifest_entry["default_window_length"]),
            "horizon": int(manifest_entry["default_horizon"]),
            "split_protocol": manifest_entry["recommended_split_protocol"],
            "has_ground_truth_kernel": bool(manifest_entry["has_ground_truth_kernel"]),
            "has_ground_truth_gfrf": bool(manifest_entry["has_ground_truth_gfrf"]),
            "extras": {
                "benchmark_name": manifest_entry["benchmark_name"],
                "system_type": manifest_entry["system_type"],
                "task_usage": list(manifest_entry["task_usage"]),
                "input_channels": list(manifest_entry["input_channels"]),
                "output_channels": list(manifest_entry["output_channels"]),
                "protocol_name": manifest_entry["recommended_split_protocol"],
                "transform_provenance": {
                    "source_split": "train",
                    "policy": "split_before_fit_transform",
                    "adapter": cls.__name__,
                },
            },
        }
        if meta:
            merged_meta.update(meta)
            meta_extras = dict(merged_meta.get("extras", {}))
            extra_overrides = dict(meta.get("extras", {}))
            meta_extras.update(extra_overrides)
            merged_meta["extras"] = meta_extras
        return merged_meta

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
        if cls.task_family != TaskFamily.NONLINEAR:
            raise DataProtocolError("nonlinear adapter task_family mismatch.")

        manifest_entry = cls._get_manifest_entry(dataset_name)
        merged_meta = cls._merge_meta(dataset_name, manifest_entry, meta=meta)
        merged_artifacts = cls._merge_artifacts(
            dataset_name, manifest_entry, artifacts=artifacts
        )

        # Fail fast if manifests conflict with explicit builder-provided booleans.
        if (
            "has_ground_truth_kernel" in meta
            and bool(meta["has_ground_truth_kernel"])
            and not manifest_entry["has_ground_truth_kernel"]
        ):
            raise DataProtocolError(
                f"{dataset_name}: builder sets has_ground_truth_kernel but benchmark manifest is false."
            )
        if (
            "has_ground_truth_gfrf" in meta
            and bool(meta["has_ground_truth_gfrf"])
            and not manifest_entry["has_ground_truth_gfrf"]
        ):
            raise DataProtocolError(
                f"{dataset_name}: builder sets has_ground_truth_gfrf but benchmark manifest is false."
            )

        return super().build_bundle(
            dataset_name=dataset_name,
            train=train,
            val=val,
            test=test,
            meta=merged_meta,
            artifacts=merged_artifacts,
        )
