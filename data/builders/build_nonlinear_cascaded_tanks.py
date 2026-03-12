"""Build Cascaded Tanks dataset bundle for nonlinear benchmark.

Scope:
- raw -> interim -> processed
- produce unified DatasetBundle with train/val/test/meta/artifacts
- no model, no training, no experiment runner logic
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from data.adapters.base import DataProtocolError, DatasetBundle
from data.adapters.nonlinear_adapter import NonlinearAdapter
from data.builders.nonlinear_builder import NonlinearBuilder, NonlinearBuilderContext
from data.checks.bundle_checks import check_dataset_bundle


class CascadedTanksBuilderError(DataProtocolError):
    """Raised when Cascaded Tanks build steps fail."""


@dataclass(frozen=True)
class CascadedTanksBuilderConfig:
    dataset_name: str = "cascaded_tanks"
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "nonlinear_temporal_grouped_holdout_v1"
    raw_file_candidates: Sequence[str] = (
        "CascadedTanksFiles/dataBenchmark.mat",
        "dataBenchmark.mat",
        "CascadedTanksFiles/dataBenchmark.csv",
        "dataBenchmark.csv",
    )
    interim_prefix: str = "cascaded_tanks"

    def validate(self) -> None:
        if self.window_length <= 0:
            raise CascadedTanksBuilderError("window_length must be a positive integer.")
        if self.horizon <= 0:
            raise CascadedTanksBuilderError("horizon must be a positive integer.")


@dataclass(frozen=True)
class RawDiscoveryResult:
    path: Path
    discovery_method: str
    raw_format: str
    matched_candidate: Optional[str]
    candidate_rank: Optional[int]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _to_float2d(value: Any, field: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise CascadedTanksBuilderError(f"{field} must be 2D after reshape.")
    if not np.issubdtype(arr.dtype, np.number):
        raise CascadedTanksBuilderError(f"{field} must be numeric.")
    return arr.astype(np.float64)


def _relative_to_root(path: Path, raw_root: Path) -> str:
    try:
        return str(path.relative_to(raw_root))
    except ValueError:
        return str(path)


def _fallback_sort_key(path: Path, raw_root: Path) -> tuple[int, int, int]:
    rel = _relative_to_root(path, raw_root)
    suffix_rank = 0 if path.suffix.lower() == ".mat" else 1
    return (suffix_rank, len(path.relative_to(raw_root).parts), len(rel))


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> RawDiscoveryResult:
    for rank, name in enumerate(candidates, start=1):
        candidate = raw_root / name
        if candidate.is_file():
            return RawDiscoveryResult(
                path=candidate.resolve(),
                discovery_method="whitelist",
                raw_format=candidate.suffix.lower().lstrip("."),
                matched_candidate=name,
                candidate_rank=rank,
            )

    fallback_matches: List[Path] = []
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".mat", ".csv"}:
            continue
        if path.stem.casefold() != "databenchmark".casefold():
            continue
        fallback_matches.append(path.resolve())

    if not fallback_matches:
        raise CascadedTanksBuilderError(
            "No Cascaded Tanks raw file found in "
            f"{raw_root}. Checked candidates {list(candidates)} and fallback stem 'dataBenchmark'."
        )

    fallback_matches.sort(key=lambda path: _fallback_sort_key(path, raw_root))
    best = fallback_matches[0]
    best_key = _fallback_sort_key(best, raw_root)
    tied = [path for path in fallback_matches if _fallback_sort_key(path, raw_root) == best_key]
    if len(tied) > 1:
        rels = [_relative_to_root(path, raw_root) for path in tied]
        raise CascadedTanksBuilderError(
            "Ambiguous Cascaded Tanks raw discovery after applying suffix/path-length priority: "
            f"{rels}"
        )

    return RawDiscoveryResult(
        path=best,
        discovery_method="fallback",
        raw_format=best.suffix.lower().lstrip("."),
        matched_candidate=_relative_to_root(best, raw_root),
        candidate_rank=None,
    )


def _extract_scalar_float(value: Any, field: str) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return None
    try:
        scalar = float(arr[0])
    except Exception as exc:
        raise CascadedTanksBuilderError(f"{field} must be numeric scalar when provided.") from exc
    if not np.isfinite(scalar):
        return None
    return scalar


def _clean_segment(
    *,
    segment_name: str,
    u_value: Any,
    y_value: Any,
    sampling_period: Optional[float],
) -> Dict[str, Any]:
    u_arr = _to_float2d(u_value, f"{segment_name}.u")
    y_arr = _to_float2d(y_value, f"{segment_name}.y")
    n = min(len(u_arr), len(y_arr))
    u_arr = u_arr[:n]
    y_arr = y_arr[:n]

    mask = np.isfinite(u_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
    u_arr = u_arr[mask]
    y_arr = y_arr[mask]
    if len(u_arr) == 0:
        raise CascadedTanksBuilderError(f"{segment_name} has no valid rows after cleaning.")

    local_index = np.arange(len(u_arr), dtype=np.int64)
    sample_id = np.asarray([f"{segment_name}:{idx}" for idx in local_index], dtype=object)
    run_id = np.full(len(u_arr), segment_name, dtype=object)
    timestamp = None
    if sampling_period is not None:
        timestamp = local_index.astype(np.float64) * float(sampling_period)

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": sample_id,
        "run_id": run_id,
        "timestamp": timestamp,
        "raw_samples": int(len(u_arr)),
    }


def _load_mat_segments(raw_path: Path) -> Dict[str, Any]:
    try:
        import scipy.io

        payload = scipy.io.loadmat(raw_path)
    except Exception as exc:
        raise CascadedTanksBuilderError("Cannot read .mat without scipy.") from exc

    required = ("uEst", "yEst", "uVal", "yVal")
    missing = [key for key in required if key not in payload]
    if missing:
        raise CascadedTanksBuilderError(
            f"{raw_path} missing required Cascaded Tanks fields: {missing}."
        )

    ts = _extract_scalar_float(payload.get("Ts"), "Ts")
    return {
        "uEst": payload["uEst"],
        "yEst": payload["yEst"],
        "uVal": payload["uVal"],
        "yVal": payload["yVal"],
        "Ts": ts,
    }


def _csv_required_column(rows: Sequence[Mapping[str, str]], column: str, raw_path: Path) -> np.ndarray:
    values: List[float] = []
    for idx, row in enumerate(rows):
        raw_value = row.get(column)
        if raw_value is None or str(raw_value).strip() == "":
            raise CascadedTanksBuilderError(
                f"{raw_path} has empty value in required column '{column}' at row {idx}."
            )
        try:
            values.append(float(raw_value))
        except Exception as exc:
            raise CascadedTanksBuilderError(
                f"{raw_path} column '{column}' must be numeric at row {idx}."
            ) from exc
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def _load_csv_segments(raw_path: Path) -> Dict[str, Any]:
    with raw_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise CascadedTanksBuilderError(f"{raw_path} has no rows.")

    required = ("uEst", "yEst", "uVal", "yVal")
    header = set(rows[0].keys())
    missing = [key for key in required if key not in header]
    if missing:
        raise CascadedTanksBuilderError(
            f"{raw_path} missing required Cascaded Tanks columns: {missing}."
        )

    ts = None
    for row in rows:
        value = row.get("Ts")
        if value is None or str(value).strip() == "":
            continue
        try:
            ts = float(value)
        except Exception as exc:
            raise CascadedTanksBuilderError(f"{raw_path} column 'Ts' must be numeric.") from exc
        break

    return {
        "uEst": _csv_required_column(rows, "uEst", raw_path),
        "yEst": _csv_required_column(rows, "yEst", raw_path),
        "uVal": _csv_required_column(rows, "uVal", raw_path),
        "yVal": _csv_required_column(rows, "yVal", raw_path),
        "Ts": ts,
    }


def load_raw(raw_root: Path, cfg: CascadedTanksBuilderConfig) -> Dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)

    discovered = _find_raw_file(raw_root, cfg.raw_file_candidates)
    raw_path = discovered.path
    suffix = raw_path.suffix.lower()
    if suffix == ".mat":
        payload = _load_mat_segments(raw_path)
    elif suffix == ".csv":
        payload = _load_csv_segments(raw_path)
    else:
        raise CascadedTanksBuilderError(f"Unsupported raw format for Cascaded Tanks: {suffix}")

    sampling_period = payload.get("Ts")
    segment_mapping = {
        "estimation": {"u": "uEst", "y": "yEst", "role": "train_source"},
        "validation": {"u": "uVal", "y": "yVal", "role": "holdout_source"},
    }
    segments = {
        segment_name: _clean_segment(
            segment_name=segment_name,
            u_value=payload[mapping["u"]],
            y_value=payload[mapping["y"]],
            sampling_period=sampling_period,
        )
        for segment_name, mapping in segment_mapping.items()
    }

    return {
        "raw_source": {
            "selected_path": str(raw_path),
            "relative_path": _relative_to_root(raw_path, raw_root),
            "discovery_method": discovered.discovery_method,
            "raw_format": discovered.raw_format,
            "matched_candidate": discovered.matched_candidate,
            "candidate_rank": discovered.candidate_rank,
        },
        "raw_field_mapping": {
            "estimation": {"u": "uEst", "y": "yEst"},
            "validation": {"u": "uVal", "y": "yVal"},
            "sampling_period": "Ts",
        },
        "sampling_period": sampling_period,
        "segments": segments,
        "segment_roles": {
            "estimation": {
                "role": "train_source",
                "split_targets": ["train"],
                "raw_fields": {"u": "uEst", "y": "yEst"},
                "raw_samples": segments["estimation"]["raw_samples"],
            },
            "validation": {
                "role": "holdout_source",
                "split_targets": ["val", "test"],
                "raw_fields": {"u": "uVal", "y": "yVal"},
                "raw_samples": segments["validation"]["raw_samples"],
            },
        },
    }


def build_windows(
    u: np.ndarray,
    y: np.ndarray,
    *,
    window_length: int,
    horizon: int,
    sample_id: Optional[np.ndarray],
    run_id: Optional[np.ndarray],
    timestamp: Optional[np.ndarray],
) -> Dict[str, Any]:
    n = min(len(u), len(y))
    if n <= window_length + horizon:
        raise CascadedTanksBuilderError(
            f"Not enough points: n={n}, window={window_length}, horizon={horizon}."
        )

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    sample_ids: List[Any] = []
    run_ids: List[Any] = []
    timestamps: List[Any] = []

    for idx in range(0, n - window_length - horizon + 1):
        anchor = idx + window_length - 1
        x_list.append(u[idx : idx + window_length])
        y_list.append(y[idx + window_length : idx + window_length + horizon])
        sample_ids.append(sample_id[anchor] if sample_id is not None and len(sample_id) == n else idx)
        run_ids.append(run_id[anchor] if run_id is not None and len(run_id) == n else None)
        timestamps.append(timestamp[anchor] if timestamp is not None and len(timestamp) == n else None)

    timestamp_array: np.ndarray
    if timestamps and all(value is not None for value in timestamps):
        timestamp_array = np.asarray(timestamps, dtype=np.float64)
    else:
        timestamp_array = np.asarray(timestamps, dtype=object)

    return {
        "X": np.asarray(x_list, dtype=np.float64),
        "Y": np.asarray(y_list, dtype=np.float64),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "run_id": np.asarray(run_ids, dtype=object),
        "timestamp": timestamp_array,
    }


def _slice_window_payload(window_payload: Mapping[str, Any], indices: np.ndarray, source_segment: str) -> Dict[str, Any]:
    payload = {
        "X": np.asarray(window_payload["X"])[indices],
        "Y": np.asarray(window_payload["Y"])[indices],
        "sample_id": np.asarray(window_payload["sample_id"], dtype=object)[indices],
        "run_id": np.asarray(window_payload["run_id"], dtype=object)[indices],
        "timestamp": np.asarray(window_payload["timestamp"])[indices],
        "meta": {"source_segment": source_segment},
    }
    return payload


def build_split(
    estimation_windows: Mapping[str, Any],
    validation_windows: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_path: Path,
    *,
    traceability: Mapping[str, Any],
) -> Dict[str, Any]:
    train_count = int(len(np.asarray(estimation_windows["X"])))
    holdout_count = int(len(np.asarray(validation_windows["X"])))
    if train_count == 0:
        raise CascadedTanksBuilderError("Estimation segment produced zero train windows.")
    if holdout_count < 2:
        raise CascadedTanksBuilderError(
            "Validation segment must produce at least two windows so val/test are both non-empty."
        )

    split_cfg = protocol.get("split", {})
    val_weight = float(split_cfg.get("val", 0.15))
    test_weight = float(split_cfg.get("test", 0.15))
    if val_weight <= 0 or test_weight <= 0:
        raise CascadedTanksBuilderError(
            "Cascaded Tanks split requires positive val/test ratios for validation holdout."
        )

    total_holdout_weight = val_weight + test_weight
    val_count = int(np.floor(holdout_count * val_weight / total_holdout_weight))
    val_count = min(max(1, val_count), holdout_count - 1)
    test_count = holdout_count - val_count
    if val_count <= 0 or test_count <= 0:
        raise CascadedTanksBuilderError(
            "Validation segment could not be split into non-empty val/test partitions."
        )

    train_indices = np.arange(train_count, dtype=int)
    val_indices = np.arange(val_count, dtype=int)
    test_indices = np.arange(val_count, holdout_count, dtype=int)

    split_payload = {
        "dataset_name": "cascaded_tanks",
        "protocol_name": protocol.get("protocol_name"),
        "split_strategy": "estimation_to_train_validation_holdout",
        "grouping": {"level": "official_segments", "notes": "train from estimation, val/test from validation"},
        "windowed_samples": int(train_count + holdout_count),
        "split_indices": {
            "train": train_indices.tolist(),
            "val": (train_count + val_indices).tolist(),
            "test": (train_count + test_indices).tolist(),
        },
        "counts": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "split_sources": {
            "train": "estimation",
            "val": "validation",
            "test": "validation",
        },
        "segment_roles": traceability["segment_roles"],
        "raw_source": traceability["raw_source"],
        "raw_field_mapping": traceability["raw_field_mapping"],
        "segment_window_counts": {
            "estimation": train_count,
            "validation": holdout_count,
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(split_path, split_payload)

    return {
        "train": _slice_window_payload(estimation_windows, train_indices, "estimation"),
        "val": _slice_window_payload(validation_windows, val_indices, "validation"),
        "test": _slice_window_payload(validation_windows, test_indices, "validation"),
        "split_sources": split_payload["split_sources"],
        "segment_window_counts": split_payload["segment_window_counts"],
    }


def _find_protocol_root(start: Path, protocol_name: str) -> Path:
    for candidate in (start, *start.parents):
        path = candidate / "data" / "metadata" / "nonlinear" / "protocols" / f"{protocol_name}.json"
        if path.exists():
            return path
    fallback = Path(__file__).resolve().parents[1] / "metadata" / "nonlinear" / "protocols" / f"{protocol_name}.json"
    if fallback.exists():
        return fallback
    raise CascadedTanksBuilderError(f"Cannot locate split protocol: {protocol_name}")


def _find_metadata_entry(metadata_root: Path, manifest_name: str, dataset_name: str) -> Mapping[str, Any]:
    manifest = _load_json(metadata_root / manifest_name)
    for item in manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return item
    raise CascadedTanksBuilderError(f"{manifest_name} missing dataset '{dataset_name}'.")


def _register_truth_reference(
    metadata_root: Path,
    dataset_name: str,
    truth_type: str,
    manifest_reference: Optional[str],
) -> Dict[str, str]:
    truth_root = metadata_root / "truth"
    truth_root.mkdir(parents=True, exist_ok=True)
    out = truth_root / f"{dataset_name}_{truth_type}_reference.json"
    payload = {
        "dataset_name": dataset_name,
        "truth_type": truth_type,
        "status": "not_available" if manifest_reference is None else "registered",
        "reference_uri": manifest_reference,
        "artifact_type": "truth_reference_registry",
        "notes": f"{truth_type} truth reference for {dataset_name} (protocol-driven). "
        "Explicitly keep as placeholder when not available.",
        "registered_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(out, payload)
    return {
        "reference_file": str(out),
        "reference_uri": manifest_reference,
    }


def export_bundle(
    cfg: CascadedTanksBuilderConfig,
    splits: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    benchmark_entry: Mapping[str, Any],
    kernel_manifest_entry: Mapping[str, Any],
    gfrf_manifest_entry: Mapping[str, Any],
    processed_root: Path,
    split_path: Path,
    metadata_root: Path,
    interim_root: Path,
    traceability: Mapping[str, Any],
) -> DatasetBundle:
    processed_root.mkdir(parents=True, exist_ok=True)
    processed_files: Dict[str, str] = {}

    for split_name, payload in splits.items():
        x = np.asarray(payload["X"], dtype=float)
        y = np.asarray(payload["Y"], dtype=float)
        np.save(processed_root / f"{split_name}_X.npy", x)
        np.save(processed_root / f"{split_name}_Y.npy", y)
        processed_files[f"{split_name}_X"] = f"{split_name}_X.npy"
        processed_files[f"{split_name}_Y"] = f"{split_name}_Y.npy"

        if payload.get("sample_id") is not None:
            np.save(processed_root / f"{split_name}_sample_id.npy", np.asarray(payload["sample_id"], dtype=object))
            processed_files[f"{split_name}_sample_id"] = f"{split_name}_sample_id.npy"
        if payload.get("run_id") is not None:
            np.save(processed_root / f"{split_name}_run_id.npy", np.asarray(payload["run_id"], dtype=object))
            processed_files[f"{split_name}_run_id"] = f"{split_name}_run_id.npy"
        if payload.get("timestamp") is not None:
            np.save(processed_root / f"{split_name}_timestamp.npy", np.asarray(payload["timestamp"]))
            processed_files[f"{split_name}_timestamp"] = f"{split_name}_timestamp.npy"

    train_x = np.asarray(splits["train"]["X"])
    train_y = np.asarray(splits["train"]["Y"])
    input_dim = int(train_x.shape[-1]) if train_x.size else 0
    output_dim = int(train_y.shape[-1]) if train_y.size else 0

    kernel_ref = benchmark_entry.get("artifacts", {}).get("kernel_reference")
    gfrf_ref = benchmark_entry.get("artifacts", {}).get("gfrf_reference")
    protocol_ref = str(metadata_root / "protocols" / f"{cfg.split_protocol}.json")

    kernel_entry = _register_truth_reference(metadata_root, cfg.dataset_name, "kernel", kernel_ref)
    gfrf_entry = _register_truth_reference(metadata_root, cfg.dataset_name, "gfrf", gfrf_ref)

    bundle = NonlinearAdapter.build_bundle(
        dataset_name=cfg.dataset_name,
        train={k: v for k, v in splits["train"].items()},
        val={k: v for k, v in splits["val"].items()},
        test={k: v for k, v in splits["test"].items()},
        meta={
            "dataset_name": cfg.dataset_name,
            "task_family": "nonlinear",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "window_length": cfg.window_length,
            "horizon": cfg.horizon,
            "split_protocol": cfg.split_protocol,
            "has_ground_truth_kernel": bool(benchmark_entry.get("has_ground_truth_kernel", False)),
            "has_ground_truth_gfrf": bool(benchmark_entry.get("has_ground_truth_gfrf", False)),
            "extras": {
                "system_type": benchmark_entry.get("system_type"),
                "task_usage": benchmark_entry.get("task_usage", ["prediction"]),
                "input_channels": benchmark_entry.get("input_channels", ["u"]),
                "output_channels": benchmark_entry.get("output_channels", ["y"]),
                "split_path": str(split_path),
                "protocol_file": protocol_ref,
                "protocol_name": protocol.get("protocol_name"),
                "grouping_reference": benchmark_entry.get("artifacts", {}).get("grouping_reference"),
                "kernel_truth_reference": kernel_entry,
                "gfrf_truth_reference": gfrf_entry,
                "kernel_manifest_entry": kernel_entry,
                "gfrf_manifest_entry": gfrf_entry,
                "raw_source": traceability["raw_source"],
                "raw_field_mapping": traceability["raw_field_mapping"],
                "segment_roles": traceability["segment_roles"],
                "split_sources": traceability["split_sources"],
            },
        },
        artifacts={
            "truth_file": kernel_entry["reference_file"],
            "grouping_file": str(split_path),
            "protocol_file": protocol_ref,
            "extra": {
                "kernel_reference": kernel_ref,
                "gfrf_reference": gfrf_ref,
                "kernel_reference_file": kernel_entry["reference_file"],
                "gfrf_reference_file": gfrf_entry["reference_file"],
                "input_file_candidates": list(cfg.raw_file_candidates),
                "raw_source": traceability["raw_source"],
                "raw_field_mapping": traceability["raw_field_mapping"],
                "segment_roles": traceability["segment_roles"],
                "split_sources": traceability["split_sources"],
            },
        },
    )

    _write_json(
        processed_root / f"{cfg.dataset_name}_processed_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "build_parameters": {
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
                "split_protocol": cfg.split_protocol,
            },
            "split_file": str(split_path),
            "protocol_file": protocol_ref,
            "files": processed_files,
            "counts": {
                "train": int(len(splits["train"]["X"])),
                "val": int(len(splits["val"]["X"])),
                "test": int(len(splits["test"]["X"])),
            },
            "raw_source": traceability["raw_source"],
            "raw_field_mapping": traceability["raw_field_mapping"],
            "segment_roles": traceability["segment_roles"],
            "split_sources": traceability["split_sources"],
            "segment_window_counts": traceability["segment_window_counts"],
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
        },
    )
    _write_json(
        metadata_root / f"{cfg.dataset_name}_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "interim_root": str(interim_root),
            "processed_root": str(processed_root),
            "split_path": str(split_path),
            "split_protocol": cfg.split_protocol,
            "raw_source": traceability["raw_source"],
            "raw_field_mapping": traceability["raw_field_mapping"],
            "segment_roles": traceability["segment_roles"],
            "segment_window_counts": traceability["segment_window_counts"],
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
            "has_ground_truth_kernel": bool(benchmark_entry.get("has_ground_truth_kernel", False)),
            "has_ground_truth_gfrf": bool(benchmark_entry.get("has_ground_truth_gfrf", False)),
            "kernel_manifest_entry": kernel_manifest_entry,
            "gfrf_manifest_entry": gfrf_manifest_entry,
        },
    )

    check_dataset_bundle(bundle, strict=False)
    return bundle


class CascadedTanksBuilder(NonlinearBuilder):
    def __init__(self, context: NonlinearBuilderContext):
        super().__init__(context)
        self.last_traceability: Dict[str, Any] = {}

    def load_splits(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cfg = CascadedTanksBuilderConfig()
        protocol_path = _find_protocol_root(self.context.interim_root, cfg.split_protocol)
        protocol = _load_json(protocol_path)

        raw = load_raw(self.context.raw_root, cfg)
        estimation = raw["segments"]["estimation"]
        validation = raw["segments"]["validation"]

        self.context.interim_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.context.interim_root / f"{cfg.interim_prefix}_raw.npz",
            estimation_u=estimation["u"],
            estimation_y=estimation["y"],
            estimation_sample_id=np.asarray(estimation["sample_id"], dtype=object),
            estimation_run_id=np.asarray(estimation["run_id"], dtype=object),
            estimation_timestamp=np.asarray(estimation["timestamp"], dtype=float)
            if estimation["timestamp"] is not None
            else np.array([], dtype=float),
            validation_u=validation["u"],
            validation_y=validation["y"],
            validation_sample_id=np.asarray(validation["sample_id"], dtype=object),
            validation_run_id=np.asarray(validation["run_id"], dtype=object),
            validation_timestamp=np.asarray(validation["timestamp"], dtype=float)
            if validation["timestamp"] is not None
            else np.array([], dtype=float),
        )

        estimation_windows = build_windows(
            estimation["u"],
            estimation["y"],
            window_length=cfg.window_length,
            horizon=cfg.horizon,
            sample_id=estimation["sample_id"],
            run_id=estimation["run_id"],
            timestamp=estimation["timestamp"],
        )
        validation_windows = build_windows(
            validation["u"],
            validation["y"],
            window_length=cfg.window_length,
            horizon=cfg.horizon,
            sample_id=validation["sample_id"],
            run_id=validation["run_id"],
            timestamp=validation["timestamp"],
        )

        raw["segment_roles"]["estimation"]["window_count"] = int(len(estimation_windows["X"]))
        raw["segment_roles"]["validation"]["window_count"] = int(len(validation_windows["X"]))

        _write_json(
            self.context.interim_root / f"{cfg.interim_prefix}_interim_manifest.json",
            {
                "dataset_name": cfg.dataset_name,
                "raw_root": str(self.context.raw_root),
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "raw_file_candidates": list(cfg.raw_file_candidates),
                "raw_source": raw["raw_source"],
                "raw_field_mapping": raw["raw_field_mapping"],
                "segment_roles": raw["segment_roles"],
                "sampling_period": raw["sampling_period"],
            },
        )

        split_path = self.context.splits_root / f"{cfg.dataset_name}_split_manifest.json"
        split_payload = build_split(
            estimation_windows,
            validation_windows,
            protocol,
            split_path,
            traceability=raw,
        )

        self.last_traceability = {
            "raw_source": raw["raw_source"],
            "raw_field_mapping": raw["raw_field_mapping"],
            "segment_roles": raw["segment_roles"],
            "split_sources": split_payload["split_sources"],
            "segment_window_counts": split_payload["segment_window_counts"],
        }
        return split_payload["train"], split_payload["val"], split_payload["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cascaded_tanks dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split-protocol", type=str, default="nonlinear_temporal_grouped_holdout_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CascadedTanksBuilderConfig(
        window_length=args.window_length,
        horizon=args.horizon,
        split_protocol=args.split_protocol,
    )
    cfg.validate()

    project_root = args.project_root.expanduser().resolve()
    context = NonlinearBuilderContext(
        dataset_name=cfg.dataset_name,
        raw_root=project_root / "data" / "raw" / "nonlinear" / cfg.dataset_name,
        interim_root=project_root / "data" / "interim" / "nonlinear" / cfg.dataset_name,
        processed_root=project_root / "data" / "processed" / "nonlinear" / cfg.dataset_name,
        splits_root=project_root / "data" / "splits" / "nonlinear",
    )
    metadata_root = project_root / "data" / "metadata" / "nonlinear"

    benchmark_entry = _find_metadata_entry(metadata_root, "benchmark_manifest.json", cfg.dataset_name)
    kernel_entry = _find_metadata_entry(metadata_root, "kernel_truth_manifest.json", cfg.dataset_name)
    gfrf_entry = _find_metadata_entry(metadata_root, "gfrf_truth_manifest.json", cfg.dataset_name)

    builder = CascadedTanksBuilder(context)
    train, val, test = builder.load_splits()

    protocol = _load_json(metadata_root / "protocols" / f"{cfg.split_protocol}.json")
    split_path = context.splits_root / f"{cfg.dataset_name}_split_manifest.json"

    export_bundle(
        cfg=cfg,
        splits={"train": train, "val": val, "test": test},
        protocol=protocol,
        benchmark_entry=benchmark_entry,
        kernel_manifest_entry=kernel_entry,
        gfrf_manifest_entry=gfrf_entry,
        processed_root=context.processed_root,
        split_path=split_path,
        metadata_root=metadata_root,
        interim_root=context.interim_root,
        traceability=builder.last_traceability,
    )


if __name__ == "__main__":
    main()
