"""Build hydraulic dataset artifacts from the cycle-wise raw text matrices.

Outputs two unified temporal representations:
- ``cycle_60``: main experiment representation, [N, 60, 17]
- ``cycle_600``: sensitivity representation, [N, 600, 17]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from data.adapters.base import DataProtocolError, DatasetBundle
from data.adapters.hydraulic_adapter import HydraulicAdapter
from data.checks.bundle_checks import check_dataset_bundle


class HydraulicBuilderError(DataProtocolError):
    """Raised when the hydraulic raw data cannot satisfy the protocol."""


@dataclass(frozen=True)
class HydraulicBuilderConfig:
    dataset_name: str = "hydraulic"
    profile_file: str = "profile.txt"
    default_representation: str = "cycle_60"

    def validate(self) -> None:
        if not self.dataset_name:
            raise HydraulicBuilderError("dataset_name must be non-empty.")


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    target_cycle_length: int
    role: str
    description: str

    @property
    def target_sampling_rate_hz(self) -> int:
        if self.target_cycle_length % 60 != 0:
            raise HydraulicBuilderError(
                f"{self.name}: target_cycle_length must divide evenly over 60 s."
            )
        return self.target_cycle_length // 60


REPRESENTATION_SPECS: tuple[RepresentationSpec, ...] = (
    RepresentationSpec(
        name="cycle_60",
        target_cycle_length=60,
        role="main_experiment",
        description="Main experiment representation with 60 points per cycle.",
    ),
    RepresentationSpec(
        name="cycle_600",
        target_cycle_length=600,
        role="sensitivity_analysis",
        description="Sensitivity representation with 600 points per cycle.",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_matrix(path: Path, *, expected_rows: int | None, expected_cols: int, name: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float32)
    except Exception as exc:
        raise HydraulicBuilderError(f"failed to read {name} from {path}") from exc

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise HydraulicBuilderError(f"{name} must be a 2D matrix, got shape={arr.shape}.")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise HydraulicBuilderError(
            f"{name} row count mismatch: expected {expected_rows}, got {arr.shape[0]}."
        )
    if arr.shape[1] != expected_cols:
        raise HydraulicBuilderError(
            f"{name} column count mismatch: expected {expected_cols}, got {arr.shape[1]}."
        )
    return arr


def _as_object_array(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def _write_cycle_label_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise HydraulicBuilderError("cycle label table cannot be empty.")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _transform_channel(
    value: np.ndarray,
    *,
    target_cycle_length: int,
    source_samples_per_cycle: int,
    channel_name: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if source_samples_per_cycle == target_cycle_length:
        return np.asarray(value, dtype=np.float32), {
            "transform": "identity",
            "factor": 1,
            "notes": "Retain source samples without modification.",
        }

    if source_samples_per_cycle > target_cycle_length:
        if source_samples_per_cycle % target_cycle_length != 0:
            raise HydraulicBuilderError(
                f"{channel_name}: cannot aggregate {source_samples_per_cycle} -> {target_cycle_length} evenly."
            )
        factor = source_samples_per_cycle // target_cycle_length
        if value.shape[1] % factor != 0:
            raise HydraulicBuilderError(
                f"{channel_name}: sample axis {value.shape[1]} is not divisible by {factor}."
            )
        transformed = value.reshape(value.shape[0], target_cycle_length, factor).mean(axis=2)
        return np.asarray(transformed, dtype=np.float32), {
            "transform": "mean_pool",
            "factor": int(factor),
            "notes": f"Aggregate contiguous blocks of {factor} samples by mean.",
        }

    if target_cycle_length % source_samples_per_cycle != 0:
        raise HydraulicBuilderError(
            f"{channel_name}: cannot repeat-expand {source_samples_per_cycle} -> {target_cycle_length} evenly."
        )
    factor = target_cycle_length // source_samples_per_cycle
    transformed = np.repeat(value, repeats=factor, axis=1)
    return np.asarray(transformed, dtype=np.float32), {
        "transform": "repeat_hold",
        "factor": int(factor),
        "notes": f"Repeat each source sample {factor} times without interpolation.",
    }


def discover_raw_files(raw_root: Path, channel_map: Mapping[str, Any], cfg: HydraulicBuilderConfig) -> Dict[str, Any]:
    channels = channel_map.get("channels")
    details = channel_map.get("channel_details")
    if not isinstance(channels, list) or not isinstance(details, Mapping):
        raise HydraulicBuilderError("channel_map must define `channels` and `channel_details`.")

    sensor_files: Dict[str, str] = {}
    missing: List[str] = []
    for channel_name in channels:
        spec = details.get(channel_name)
        if not isinstance(spec, Mapping):
            raise HydraulicBuilderError(f"channel_map.channel_details missing spec for {channel_name}.")
        source_file = spec.get("source_file")
        if not isinstance(source_file, str) or not source_file.endswith(".txt"):
            raise HydraulicBuilderError(f"{channel_name}: source_file must be a `.txt` filename.")
        path = raw_root / source_file
        if not path.exists():
            missing.append(source_file)
            continue
        sensor_files[channel_name] = source_file

    profile_path = raw_root / cfg.profile_file
    if not profile_path.exists():
        missing.append(cfg.profile_file)
    if missing:
        raise HydraulicBuilderError(
            "missing required hydraulic raw files: " + ", ".join(sorted(missing))
        )

    ignored_docs = [
        str(path.name)
        for path in sorted(raw_root.glob("*.txt"))
        if path.name not in set(sensor_files.values()) | {cfg.profile_file}
    ]

    return {
        "sensor_files": sensor_files,
        "profile_file": cfg.profile_file,
        "ignored_text_files": ignored_docs,
    }


def load_raw(
    raw_root: Path,
    channel_map: Mapping[str, Any],
    cfg: HydraulicBuilderConfig,
) -> Dict[str, Any]:
    discovery = discover_raw_files(raw_root, channel_map, cfg)
    channels = channel_map["channels"]
    details = channel_map["channel_details"]

    sensor_data: Dict[str, np.ndarray] = {}
    expected_rows: int | None = None
    row_counts: Dict[str, int] = {}
    column_counts: Dict[str, int] = {}
    sampling_rates: Dict[str, int] = {}

    for channel_name in channels:
        spec = details[channel_name]
        path = raw_root / discovery["sensor_files"][channel_name]
        arr = _load_matrix(
            path,
            expected_rows=expected_rows,
            expected_cols=int(spec["samples_per_cycle"]),
            name=channel_name,
        )
        if expected_rows is None:
            expected_rows = int(arr.shape[0])
        sensor_data[channel_name] = arr
        row_counts[channel_name] = int(arr.shape[0])
        column_counts[channel_name] = int(arr.shape[1])
        sampling_rates[channel_name] = int(spec["sampling_rate_hz"])

    if expected_rows is None:
        raise HydraulicBuilderError("no hydraulic sensor matrices were loaded.")

    profile = _load_matrix(
        raw_root / discovery["profile_file"],
        expected_rows=expected_rows,
        expected_cols=5,
        name="profile",
    )

    return {
        "sensor_data": sensor_data,
        "profile": profile,
        "row_count": expected_rows,
        "row_counts": row_counts,
        "column_counts": column_counts,
        "sampling_rates": sampling_rates,
        "discovery": discovery,
    }


def _derive_protocol_rows(profile: np.ndarray, protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    nominal = protocol.get("nominal_conditions", {})
    if not isinstance(nominal, Mapping):
        raise HydraulicBuilderError("single_fault_protocol.nominal_conditions must be a mapping.")

    rows: List[Dict[str, Any]] = []
    for idx, raw_row in enumerate(profile, start=1):
        cooler_condition = int(raw_row[0])
        valve_condition = int(raw_row[1])
        pump_leakage = int(raw_row[2])
        accumulator_pressure = int(raw_row[3])
        stable_flag = int(raw_row[4])

        degraded_components: List[str] = []
        if cooler_condition != int(nominal["cooler_condition"]):
            degraded_components.append("Cooler")
        if valve_condition != int(nominal["valve_condition"]):
            degraded_components.append("Valve")
        if pump_leakage != int(nominal["pump_leakage"]):
            degraded_components.append("Pump")
        if accumulator_pressure != int(nominal["accumulator_pressure"]):
            degraded_components.append("Accumulator")

        is_healthy = len(degraded_components) == 0
        is_single_component_fault = len(degraded_components) == 1
        if is_healthy:
            condition_type = "healthy"
            subsystem_label = "healthy"
            fault_label = "healthy"
            fault_subsystem = ""
        elif is_single_component_fault:
            condition_type = "single_component_degraded"
            subsystem_label = degraded_components[0].lower()
            fault_label = degraded_components[0]
            fault_subsystem = degraded_components[0]
        else:
            condition_type = "multi_component_degraded"
            subsystem_label = "multi_component"
            fault_label = "multi_component"
            fault_subsystem = ""

        rows.append(
            {
                "cycle_id": idx,
                "sample_id": idx,
                "cooler_condition": cooler_condition,
                "valve_condition": valve_condition,
                "pump_leakage": pump_leakage,
                "accumulator_pressure": accumulator_pressure,
                "stable_flag": stable_flag,
                "is_stable_cycle": int(stable_flag == 0),
                "is_healthy": int(is_healthy),
                "num_degraded_components": len(degraded_components),
                "degraded_components": "|".join(degraded_components),
                "is_single_component_fault": int(is_single_component_fault),
                "fault_subsystem": fault_subsystem,
                "fault_label": fault_label,
                "subsystem_label": subsystem_label,
                "condition_type": condition_type,
            }
        )
    return rows


def _encode_labels(rows: Sequence[Mapping[str, Any]], class_names: Sequence[str]) -> np.ndarray:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    encoded = np.zeros((len(rows), len(class_names)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        label = str(row["subsystem_label"])
        if label not in class_to_idx:
            raise HydraulicBuilderError(f"unknown subsystem_label `{label}` in derived rows.")
        encoded[row_idx, class_to_idx[label]] = 1.0
    return encoded


def build_cycle_tensor(
    sensor_data: Mapping[str, np.ndarray],
    channel_map: Mapping[str, Any],
    spec: RepresentationSpec,
) -> tuple[np.ndarray, Dict[str, Any]]:
    channels = channel_map["channels"]
    details = channel_map["channel_details"]
    resampled_channels: List[np.ndarray] = []
    transforms: Dict[str, Any] = {}

    for channel_name in channels:
        channel_spec = details[channel_name]
        source = np.asarray(sensor_data[channel_name], dtype=np.float32)
        transformed, transform_info = _transform_channel(
            source,
            target_cycle_length=spec.target_cycle_length,
            source_samples_per_cycle=int(channel_spec["samples_per_cycle"]),
            channel_name=channel_name,
        )
        if transformed.shape[1] != spec.target_cycle_length:
            raise HydraulicBuilderError(
                f"{channel_name}: transformed cycle length {transformed.shape[1]} != {spec.target_cycle_length}."
            )
        resampled_channels.append(transformed)
        transforms[channel_name] = {
            "source_file": channel_spec["source_file"],
            "source_sampling_rate_hz": int(channel_spec["sampling_rate_hz"]),
            "source_samples_per_cycle": int(channel_spec["samples_per_cycle"]),
            "target_sampling_rate_hz": spec.target_sampling_rate_hz,
            "target_cycle_length": spec.target_cycle_length,
            **transform_info,
        }

    tensor = np.stack(resampled_channels, axis=-1).astype(np.float32, copy=False)
    summary = {
        "representation_name": spec.name,
        "representation_role": spec.role,
        "description": spec.description,
        "target_cycle_length": spec.target_cycle_length,
        "target_sampling_rate_hz": spec.target_sampling_rate_hz,
        "channels": transforms,
        "output_shape": [int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2])],
    }
    return tensor, summary


def _split_indices(n_samples: int, split_cfg: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise HydraulicBuilderError("invalid hydraulic split ratios.")
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise HydraulicBuilderError("hydraulic bundle split produced an empty split.")

    return {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test": np.arange(n_train + n_val, n_samples, dtype=np.int64),
    }


def _write_split_manifest(
    split_root: Path,
    *,
    dataset_name: str,
    protocol: Mapping[str, Any],
    split_indices: Mapping[str, np.ndarray],
    representation_names: Sequence[str],
) -> Path:
    split_root.mkdir(parents=True, exist_ok=True)
    split_path = split_root / "hydraulic_split_manifest.json"
    _write_json(
        split_path,
        {
            "dataset_name": dataset_name,
            "protocol_name": protocol["protocol_name"],
            "split_protocol": protocol["split_protocol"],
            "split_kind": protocol.get("split_protocol_kind", "bundle_scaffold_only"),
            "generated_at": _utc_now(),
            "split_indices": {name: idx.tolist() for name, idx in split_indices.items()},
            "counts": {name: int(len(idx)) for name, idx in split_indices.items()},
            "grouping": protocol.get("grouping", {}),
            "compatible_representations": list(representation_names),
            "notes": "Cycle-contiguous deterministic split for bundle scaffolding; not the paper experiment split.",
        },
    )
    return split_path


def _make_split_payload(
    cycle_tensor: np.ndarray,
    labels: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    indices: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    cycle_ids = np.asarray([int(row["cycle_id"]) for row in rows], dtype=np.int64)
    sample_ids = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
    timestamps = cycle_ids.astype(np.float64, copy=False)
    fault_labels = _as_object_array([str(row["fault_label"]) for row in rows])
    subsystem_labels = _as_object_array([str(row["subsystem_label"]) for row in rows])
    condition_types = _as_object_array([str(row["condition_type"]) for row in rows])

    payload: Dict[str, Dict[str, Any]] = {}
    for split_name, split_idx in indices.items():
        payload[split_name] = {
            "X": cycle_tensor[split_idx],
            "Y": labels[split_idx],
            "sample_id": sample_ids[split_idx],
            "cycle_id": cycle_ids[split_idx],
            "timestamp": timestamps[split_idx],
            "fault_label": fault_labels[split_idx],
            "subsystem_label": subsystem_labels[split_idx],
            "condition_type": condition_types[split_idx],
        }
    return payload


def _save_split_arrays(version_root: Path, splits: Mapping[str, Mapping[str, Any]]) -> Dict[str, str]:
    processed_files: Dict[str, str] = {}
    object_keys = {"fault_label", "subsystem_label", "condition_type"}
    for split_name, payload in splits.items():
        for key in ("X", "Y", "sample_id", "cycle_id", "timestamp", "fault_label", "subsystem_label", "condition_type"):
            value = payload.get(key)
            if value is None:
                continue
            filename = f"{split_name}_{key}.npy"
            array = np.asarray(value, dtype=object if key in object_keys else None)
            np.save(version_root / filename, array)
            processed_files[f"{split_name}_{key}"] = filename
    return processed_files


def _make_root_relative_manifest(
    version_manifest: Mapping[str, Any],
    *,
    representation_name: str,
) -> Dict[str, Any]:
    root_manifest = dict(version_manifest)
    root_manifest["representation_name"] = representation_name
    root_manifest["default_representation"] = representation_name
    files = version_manifest.get("processed_files", {})
    if isinstance(files, Mapping):
        root_manifest["processed_files"] = {
            key: f"{representation_name}/{value}" for key, value in files.items()
        }
    cycle_tensor_file = version_manifest.get("cycle_tensor_file")
    if isinstance(cycle_tensor_file, str):
        root_manifest["cycle_tensor_file"] = f"{representation_name}/{cycle_tensor_file}"
    label_table_file = version_manifest.get("label_table_file")
    if isinstance(label_table_file, str):
        root_manifest["label_table_file"] = f"{representation_name}/{label_table_file}"
    return root_manifest


def export_bundle(
    *,
    cfg: HydraulicBuilderConfig,
    processed_root: Path,
    split_root: Path,
    metadata_root: Path,
    channel_map: Mapping[str, Any],
    subsystem_groups: Mapping[str, Any],
    protocol: Mapping[str, Any],
    raw_summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
) -> DatasetBundle:
    processed_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    split_indices = _split_indices(len(rows), protocol.get("split", {}))
    split_path = _write_split_manifest(
        split_root,
        dataset_name=cfg.dataset_name,
        protocol=protocol,
        split_indices=split_indices,
        representation_names=[spec.name for spec in REPRESENTATION_SPECS],
    )

    class_names = list(protocol["labeling"]["subsystem_class_order"])
    healthy_count = int(sum(int(row["is_healthy"]) for row in rows))
    single_fault_count = int(sum(int(row["is_single_component_fault"]) for row in rows))
    multi_component_count = int(sum(1 for row in rows if row["condition_type"] == "multi_component_degraded"))

    representation_index: Dict[str, Any] = {}
    version_manifests: Dict[str, Dict[str, Any]] = {}
    bundles: Dict[str, DatasetBundle] = {}

    for spec in REPRESENTATION_SPECS:
        version_root = processed_root / spec.name
        version_root.mkdir(parents=True, exist_ok=True)
        cycle_tensor, resampling_summary = build_cycle_tensor(raw_summary["sensor_data"], channel_map, spec)
        split_payload = _make_split_payload(cycle_tensor, labels, rows, split_indices)
        processed_files = _save_split_arrays(version_root, split_payload)

        cycle_tensor_file = "hydraulic_cycle_tensor.npy"
        np.save(version_root / cycle_tensor_file, cycle_tensor)

        label_table_name = "hydraulic_cycle_labels.csv"
        _write_cycle_label_table(version_root / label_table_name, rows)

        bundle = HydraulicAdapter.build_bundle(
            dataset_name=cfg.dataset_name,
            train=split_payload["train"],
            val=split_payload["val"],
            test=split_payload["test"],
            meta={
                "dataset_name": cfg.dataset_name,
                "task_family": "hydraulic",
                "input_dim": int(cycle_tensor.shape[-1]),
                "output_dim": len(class_names),
                "window_length": int(spec.target_cycle_length),
                "horizon": 1,
                "split_protocol": str(protocol["split_protocol"]),
                "has_ground_truth_kernel": False,
                "has_ground_truth_gfrf": False,
                "extras": {
                    "sample_granularity": "cycle",
                    "cycle_duration_seconds": int(protocol["cycle_duration_seconds"]),
                    "representation_name": spec.name,
                    "representation_role": spec.role,
                    "representation_description": spec.description,
                    "target_cycle_length": int(spec.target_cycle_length),
                    "target_sampling_rate_hz": int(spec.target_sampling_rate_hz),
                    "canonical_channel_order": list(channel_map["channels"]),
                    "channel_map": channel_map,
                    "subsystem_groups": subsystem_groups,
                    "single_fault_protocol": protocol,
                    "resampling": resampling_summary,
                    "label_table_file": str(version_root / label_table_name),
                    "stable_flag_retained": True,
                    "class_names": class_names,
                    "available_representations": [item.name for item in REPRESENTATION_SPECS],
                    "task_note": "Y stores protocol support labels for bundle compatibility; the paper task remains unsupervised.",
                },
            },
            artifacts={
                "truth_file": None,
                "grouping_file": str(metadata_root / "subsystem_groups.json"),
                "protocol_file": str(metadata_root / "single_fault_protocol.json"),
                "extra": {
                    "channel_map_file": str(metadata_root / "channel_map.json"),
                    "split_file": str(split_path),
                    "label_table_file": str(version_root / label_table_name),
                    "representation_name": spec.name,
                },
            },
        )
        check_dataset_bundle(bundle)
        bundles[spec.name] = bundle

        version_manifest = {
            "dataset_name": cfg.dataset_name,
            "generated_at": _utc_now(),
            "representation_name": spec.name,
            "representation_role": spec.role,
            "representation_description": spec.description,
            "target_cycle_length": int(spec.target_cycle_length),
            "target_sampling_rate_hz": int(spec.target_sampling_rate_hz),
            "cycle_tensor_file": cycle_tensor_file,
            "cycle_tensor_shape": [int(cycle_tensor.shape[0]), int(cycle_tensor.shape[1]), int(cycle_tensor.shape[2])],
            "label_table_file": label_table_name,
            "split_file": str(split_path),
            "processed_files": processed_files,
            "counts": {name: int(len(payload["X"])) for name, payload in split_payload.items()},
            "raw_summary": {
                "discovery": dict(raw_summary["discovery"]),
                "row_count": int(raw_summary["row_count"]),
                "row_counts": dict(raw_summary["row_counts"]),
                "column_counts": dict(raw_summary["column_counts"]),
                "sampling_rates": dict(raw_summary["sampling_rates"]),
            },
            "derived_protocol_summary": {
                "healthy_count": healthy_count,
                "single_component_fault_count": single_fault_count,
                "multi_component_count": multi_component_count,
                "fault_subsystems_present": sorted(
                    {str(row["fault_subsystem"]) for row in rows if str(row["fault_subsystem"])}
                ),
            },
            "resampling": resampling_summary,
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
        }
        _write_json(version_root / "hydraulic_processed_manifest.json", version_manifest)
        version_manifests[spec.name] = version_manifest
        representation_index[spec.name] = {
            "representation_role": spec.role,
            "target_cycle_length": int(spec.target_cycle_length),
            "target_sampling_rate_hz": int(spec.target_sampling_rate_hz),
            "processed_manifest": str(version_root / "hydraulic_processed_manifest.json"),
            "cycle_tensor_file": str(version_root / cycle_tensor_file),
            "label_table_file": str(version_root / label_table_name),
        }

    representations_manifest_path = processed_root / "hydraulic_representations_manifest.json"
    _write_json(
        representations_manifest_path,
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": _utc_now(),
            "default_representation": cfg.default_representation,
            "representations": representation_index,
        },
    )

    default_manifest = _make_root_relative_manifest(
        version_manifests[cfg.default_representation],
        representation_name=cfg.default_representation,
    )
    default_manifest["representations_manifest_file"] = "hydraulic_representations_manifest.json"
    default_manifest["available_representations"] = representation_index
    _write_json(processed_root / "hydraulic_processed_manifest.json", default_manifest)

    _write_json(
        metadata_root / "hydraulic_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": _utc_now(),
            "raw_discovery": dict(raw_summary["discovery"]),
            "cycle_count": int(raw_summary["row_count"]),
            "split_file": str(split_path),
            "default_representation": cfg.default_representation,
            "representations": {
                spec.name: {
                    "target_cycle_length": int(spec.target_cycle_length),
                    "target_sampling_rate_hz": int(spec.target_sampling_rate_hz),
                    "role": spec.role,
                    "description": spec.description,
                }
                for spec in REPRESENTATION_SPECS
            },
            "protocol_name": protocol["protocol_name"],
        },
    )

    return bundles[cfg.default_representation]


def run_build(project_root: Path) -> DatasetBundle:
    cfg = HydraulicBuilderConfig()
    cfg.validate()

    project_root = project_root.expanduser().resolve()
    raw_root = project_root / "data" / "raw" / "hydraulic"
    processed_root = project_root / "data" / "processed" / "hydraulic"
    split_root = project_root / "data" / "splits" / "hydraulic"
    metadata_root = project_root / "data" / "metadata" / "hydraulic"

    channel_map = _load_json(metadata_root / "channel_map.json")
    subsystem_groups = _load_json(metadata_root / "subsystem_groups.json")
    protocol = _load_json(metadata_root / "single_fault_protocol.json")

    raw = load_raw(raw_root, channel_map, cfg)
    cycle_rows = _derive_protocol_rows(np.asarray(raw["profile"], dtype=np.float32), protocol)
    labels = _encode_labels(cycle_rows, protocol["labeling"]["subsystem_class_order"])

    if len(cycle_rows) != int(raw["row_count"]):
        raise HydraulicBuilderError(
            f"cycle label row count {len(cycle_rows)} != raw cycle count {raw['row_count']}."
        )

    return export_bundle(
        cfg=cfg,
        processed_root=processed_root,
        split_root=split_root,
        metadata_root=metadata_root,
        channel_map=channel_map,
        subsystem_groups=subsystem_groups,
        protocol=protocol,
        raw_summary=raw,
        rows=cycle_rows,
        labels=labels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hydraulic dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_build(args.project_root)


if __name__ == "__main__":
    main()
