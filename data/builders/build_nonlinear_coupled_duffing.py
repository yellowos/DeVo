"""Build Coupled Duffing dataset bundle for nonlinear benchmark.

Scope:
- raw -> interim -> processed
- export unified DatasetBundle (train/val/test/meta/artifacts)
- prediction-oriented dataset with explicit dimensional declarations
- no model / training / experiment runner implementation
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.adapters.base import DataProtocolError
from data.adapters.nonlinear_adapter import NonlinearAdapter
from data.builders.nonlinear_builder import NonlinearBuilder, NonlinearBuilderContext
from data.checks.bundle_checks import check_dataset_bundle


class CoupledDuffingBuilderError(DataProtocolError):
    """Raised when Coupled Duffing build steps fail."""


@dataclass(frozen=True)
class CoupledDuffingBuilderConfig:
    dataset_name: str = "coupled_duffing"
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "nonlinear_temporal_grouped_holdout_v1"
    raw_file_candidates: Tuple[str, ...] = (
        "DATAUNIF.MAT",
        "DATAPRBS.MAT",
        "DATAUNIF.csv",
        "DATAPRBS.csv",
    )
    sampling_period_seconds: float = 0.02
    interim_prefix: str = "coupled_duffing"

    def validate(self) -> None:
        if self.window_length <= 0:
            raise CoupledDuffingBuilderError("window_length must be a positive integer.")
        if self.horizon <= 0:
            raise CoupledDuffingBuilderError("horizon must be a positive integer.")
        if self.sampling_period_seconds <= 0:
            raise CoupledDuffingBuilderError("sampling_period_seconds must be positive.")


@dataclass(frozen=True)
class RawSourceDiscovery:
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


def _to_float2d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise CoupledDuffingBuilderError(f"{name} must be a 2D numeric matrix after reshape.")
    if not np.issubdtype(arr.dtype, np.number):
        raise CoupledDuffingBuilderError(f"{name} must contain numeric values.")
    return arr.astype(np.float64)


def _relative_to_root(path: Path, raw_root: Path) -> str:
    try:
        return str(path.relative_to(raw_root))
    except ValueError:
        return str(path)


def _fallback_sort_key(path: Path, raw_root: Path) -> tuple[int, int, int]:
    suffix_rank = 0 if path.suffix.lower() == ".mat" else 1
    rel = _relative_to_root(path, raw_root)
    return (suffix_rank, len(path.relative_to(raw_root).parts), len(rel))


def _find_named_raw_file(
    raw_root: Path,
    *,
    canonical_stem: str,
    candidates: Sequence[str],
) -> RawSourceDiscovery:
    for rank, name in enumerate(candidates, start=1):
        target = raw_root / name
        if target.is_file():
            return RawSourceDiscovery(
                path=target.resolve(),
                discovery_method="whitelist",
                raw_format=target.suffix.lower().lstrip("."),
                matched_candidate=name,
                candidate_rank=rank,
            )

    matches: List[Path] = []
    for found in raw_root.rglob("*"):
        if not found.is_file():
            continue
        if found.suffix.lower() not in {".mat", ".csv"}:
            continue
        if found.stem.casefold() != canonical_stem.casefold():
            continue
        matches.append(found.resolve())

    if not matches:
        raise CoupledDuffingBuilderError(
            f"Cannot locate required raw source '{canonical_stem}' under {raw_root}; "
            f"checked candidates {list(candidates)}."
        )

    matches.sort(key=lambda path: _fallback_sort_key(path, raw_root))
    best = matches[0]
    best_key = _fallback_sort_key(best, raw_root)
    tied = [path for path in matches if _fallback_sort_key(path, raw_root) == best_key]
    if len(tied) > 1:
        rels = [_relative_to_root(path, raw_root) for path in tied]
        raise CoupledDuffingBuilderError(
            f"Ambiguous raw discovery for '{canonical_stem}' after suffix/path priority: {rels}"
        )

    return RawSourceDiscovery(
        path=best,
        discovery_method="fallback",
        raw_format=best.suffix.lower().lstrip("."),
        matched_candidate=_relative_to_root(best, raw_root),
        candidate_rank=None,
    )


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _csv_required_column(rows: Sequence[Mapping[str, str]], column: str, raw_path: Path) -> np.ndarray:
    values: List[float] = []
    for row_idx, row in enumerate(rows):
        raw_value = row.get(column)
        if raw_value is None or str(raw_value).strip() == "":
            raise CoupledDuffingBuilderError(
                f"{raw_path} has empty value in required column '{column}' at row {row_idx}."
            )
        try:
            values.append(float(raw_value))
        except Exception as exc:
            raise CoupledDuffingBuilderError(
                f"{raw_path} column '{column}' must be numeric at row {row_idx}."
            ) from exc
    return np.asarray(values, dtype=np.float64).reshape(-1, 1)


def _clean_trajectory(
    *,
    trajectory_name: str,
    u_value: Any,
    y_value: Any,
    sampling_period_seconds: float,
) -> Dict[str, Any]:
    u_arr = _to_float2d(u_value, f"{trajectory_name}.u")
    y_arr = _to_float2d(y_value, f"{trajectory_name}.y")
    n = min(len(u_arr), len(y_arr))
    u_arr = u_arr[:n]
    y_arr = y_arr[:n]

    mask = np.isfinite(u_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
    u_arr = u_arr[mask]
    y_arr = y_arr[mask]
    if len(u_arr) == 0:
        raise CoupledDuffingBuilderError(f"{trajectory_name} has no valid rows after cleaning.")

    local_index = np.arange(len(u_arr), dtype=np.int64)
    sample_id = np.asarray([f"{trajectory_name}:{idx}" for idx in local_index], dtype=object)
    run_id = np.full(len(u_arr), trajectory_name, dtype=object)
    timestamp = local_index.astype(np.float64) * float(sampling_period_seconds)

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": sample_id,
        "run_id": run_id,
        "timestamp": timestamp,
        "raw_samples": int(len(u_arr)),
    }


def _load_source_trajectories(
    *,
    raw_path: Path,
    source_name: str,
    trajectory_specs: Sequence[tuple[str, str, str]],
    sampling_period_seconds: float,
) -> Dict[str, Dict[str, Any]]:
    suffix = raw_path.suffix.lower()
    if suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise CoupledDuffingBuilderError("Failed to read .mat without scipy support.") from exc
        getter = lambda key: payload.get(key)
    elif suffix == ".csv":
        rows = _read_csv_rows(raw_path)
        if not rows:
            raise CoupledDuffingBuilderError(f"{raw_path} has no rows.")

        def getter(key: str):
            return _csv_required_column(rows, key, raw_path)
    else:
        raise CoupledDuffingBuilderError(f"Unsupported raw file type for {source_name}: {suffix}")

    trajectories: Dict[str, Dict[str, Any]] = {}
    for trajectory_name, u_key, y_key in trajectory_specs:
        u_value = getter(u_key)
        y_value = getter(y_key)
        if u_value is None or y_value is None:
            raise CoupledDuffingBuilderError(
                f"{raw_path} is missing required trajectory fields '{u_key}'/'{y_key}'."
            )
        trajectories[trajectory_name] = _clean_trajectory(
            trajectory_name=trajectory_name,
            u_value=u_value,
            y_value=y_value,
            sampling_period_seconds=sampling_period_seconds,
        )
    return trajectories


def load_raw(raw_root: Path, cfg: CoupledDuffingBuilderConfig) -> Dict[str, Any]:
    raw_root = raw_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)

    uniform_source = _find_named_raw_file(
        raw_root,
        canonical_stem="DATAUNIF",
        candidates=("DATAUNIF.MAT", "dataunif.mat", "DATAUNIF.csv", "dataunif.csv"),
    )
    prbs_source = _find_named_raw_file(
        raw_root,
        canonical_stem="DATAPRBS",
        candidates=("DATAPRBS.MAT", "dataprbs.mat", "DATAPRBS.csv", "dataprbs.csv"),
    )

    uniform_specs = (
        ("uniform_1", "u11", "z11"),
        ("uniform_2", "u12", "z12"),
    )
    prbs_specs = (
        ("prbs_1", "u1", "z1"),
        ("prbs_2", "u2", "z2"),
        ("prbs_3", "u3", "z3"),
    )

    trajectories = {}
    trajectories.update(
        _load_source_trajectories(
            raw_path=uniform_source.path,
            source_name="uniform_source",
            trajectory_specs=uniform_specs,
            sampling_period_seconds=cfg.sampling_period_seconds,
        )
    )
    trajectories.update(
        _load_source_trajectories(
            raw_path=prbs_source.path,
            source_name="prbs_source",
            trajectory_specs=prbs_specs,
            sampling_period_seconds=cfg.sampling_period_seconds,
        )
    )

    raw_sources = {
        "uniform_source": {
            "selected_path": str(uniform_source.path),
            "relative_path": _relative_to_root(uniform_source.path, raw_root),
            "discovery_method": uniform_source.discovery_method,
            "raw_format": uniform_source.raw_format,
            "matched_candidate": uniform_source.matched_candidate,
            "candidate_rank": uniform_source.candidate_rank,
            "role": "identification_source",
            "trajectory_names": ["uniform_1", "uniform_2"],
        },
        "prbs_source": {
            "selected_path": str(prbs_source.path),
            "relative_path": _relative_to_root(prbs_source.path, raw_root),
            "discovery_method": prbs_source.discovery_method,
            "raw_format": prbs_source.raw_format,
            "matched_candidate": prbs_source.matched_candidate,
            "candidate_rank": prbs_source.candidate_rank,
            "role": "holdout_source",
            "trajectory_names": ["prbs_1", "prbs_2", "prbs_3"],
        },
    }

    trajectory_roles = {
        "uniform_1": {
            "source_file": raw_sources["uniform_source"]["relative_path"],
            "source_role": "identification_source",
            "excitation_type": "uniform",
            "input_field": "u11",
            "output_field": "z11",
            "split_role": "train",
            "raw_samples": trajectories["uniform_1"]["raw_samples"],
        },
        "uniform_2": {
            "source_file": raw_sources["uniform_source"]["relative_path"],
            "source_role": "identification_source",
            "excitation_type": "uniform",
            "input_field": "u12",
            "output_field": "z12",
            "split_role": "train",
            "raw_samples": trajectories["uniform_2"]["raw_samples"],
        },
        "prbs_1": {
            "source_file": raw_sources["prbs_source"]["relative_path"],
            "source_role": "holdout_source",
            "excitation_type": "prbs",
            "input_field": "u1",
            "output_field": "z1",
            "split_role": "val",
            "raw_samples": trajectories["prbs_1"]["raw_samples"],
        },
        "prbs_2": {
            "source_file": raw_sources["prbs_source"]["relative_path"],
            "source_role": "holdout_source",
            "excitation_type": "prbs",
            "input_field": "u2",
            "output_field": "z2",
            "split_role": "test",
            "raw_samples": trajectories["prbs_2"]["raw_samples"],
        },
        "prbs_3": {
            "source_file": raw_sources["prbs_source"]["relative_path"],
            "source_role": "holdout_source",
            "excitation_type": "prbs",
            "input_field": "u3",
            "output_field": "z3",
            "split_role": "test",
            "raw_samples": trajectories["prbs_3"]["raw_samples"],
        },
    }

    return {
        "raw_sources": raw_sources,
        "trajectory_roles": trajectory_roles,
        "sampling_period_seconds": cfg.sampling_period_seconds,
        "trajectories": trajectories,
        "split_sources": {
            "train": ["uniform_1", "uniform_2"],
            "val": ["prbs_1"],
            "test": ["prbs_2", "prbs_3"],
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
        raise CoupledDuffingBuilderError(
            f"Insufficient points: n={n}, window_length={window_length}, horizon={horizon}."
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


def _concat_window_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    source_trajectories: Sequence[str],
) -> Dict[str, Any]:
    if not payloads:
        raise CoupledDuffingBuilderError(f"{split_name} split has no source trajectories.")

    merged = {
        "X": np.concatenate([np.asarray(payload["X"], dtype=np.float64) for payload in payloads], axis=0),
        "Y": np.concatenate([np.asarray(payload["Y"], dtype=np.float64) for payload in payloads], axis=0),
        "sample_id": np.concatenate([np.asarray(payload["sample_id"], dtype=object) for payload in payloads], axis=0),
        "run_id": np.concatenate([np.asarray(payload["run_id"], dtype=object) for payload in payloads], axis=0),
        "timestamp": np.concatenate([np.asarray(payload["timestamp"]) for payload in payloads], axis=0),
        "meta": {"source_trajectories": list(source_trajectories)},
    }
    return merged


def build_split(
    windows_by_trajectory: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    split_path: Path,
    *,
    traceability: Mapping[str, Any],
) -> Dict[str, Any]:
    split_sources = traceability["split_sources"]
    train = _concat_window_payloads(
        [windows_by_trajectory[name] for name in split_sources["train"]],
        split_name="train",
        source_trajectories=split_sources["train"],
    )
    val = _concat_window_payloads(
        [windows_by_trajectory[name] for name in split_sources["val"]],
        split_name="val",
        source_trajectories=split_sources["val"],
    )
    test = _concat_window_payloads(
        [windows_by_trajectory[name] for name in split_sources["test"]],
        split_name="test",
        source_trajectories=split_sources["test"],
    )

    counts = {
        "train": int(len(train["X"])),
        "val": int(len(val["X"])),
        "test": int(len(test["X"])),
    }
    total = counts["train"] + counts["val"] + counts["test"]
    split_payload = {
        "dataset_name": "coupled_duffing",
        "protocol_name": protocol.get("protocol_name"),
        "split_strategy": "uniform_for_identification_prbs_for_holdout",
        "window_count": total,
        "split_indices": {
            "train": list(range(0, counts["train"])),
            "val": list(range(counts["train"], counts["train"] + counts["val"])),
            "test": list(range(counts["train"] + counts["val"], total)),
        },
        "split_sources": split_sources,
        "trajectory_roles": traceability["trajectory_roles"],
        "raw_sources": traceability["raw_sources"],
        "trajectory_window_counts": {
            name: int(len(np.asarray(windows["X"])))
            for name, windows in windows_by_trajectory.items()
        },
        "counts": counts,
        "protocol": protocol,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "notes": (
            "DATAUNIF trajectories are used for identification/train. "
            "DATAPRBS trajectories are reserved for holdout, with prbs_1 for validation "
            "and prbs_2/prbs_3 for test to avoid within-trajectory splitting."
        ),
    }
    _write_json(split_path, split_payload)

    return {
        "train": train,
        "val": val,
        "test": test,
        "split_sources": split_sources,
        "trajectory_window_counts": split_payload["trajectory_window_counts"],
    }


def _find_metadata_entry(metadata_root: Path, manifest_name: str, dataset_name: str) -> Mapping[str, Any]:
    manifest = _load_json(metadata_root / manifest_name)
    entries = manifest.get("benchmarks")
    if not isinstance(entries, list):
        raise CoupledDuffingBuilderError(f"Invalid manifest format: {manifest_name}")
    for item in entries:
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return item
    raise CoupledDuffingBuilderError(f"{dataset_name} missing from {manifest_name}")


def _ensure_truth_entry(metadata_root: Path, dataset_name: str, truth_type: str, uri: Optional[str]) -> Dict[str, Any]:
    truth_root = metadata_root / "truth"
    truth_root.mkdir(parents=True, exist_ok=True)
    out = truth_root / f"{dataset_name}_{truth_type}_reference.json"
    payload = {
        "benchmark_name": dataset_name,
        "truth_type": truth_type,
        "reference_uri": uri if uri else None,
        "status": "registered",
        "source": "nonlinear_benchmark_manifest",
        "build_time": datetime.utcnow().isoformat() + "Z",
        "notes": f"{truth_type} truth registration for {dataset_name}.",
    }
    _write_json(out, payload)
    return {
        "truth_file": str(out),
        "truth_uri": uri,
        "truth_payload": str(out),
    }


def export_bundle(
    cfg: CoupledDuffingBuilderConfig,
    splits: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    benchmark_entry: Mapping[str, Any],
    kernel_entry: Mapping[str, Any],
    gfrf_entry: Mapping[str, Any],
    processed_root: Path,
    split_path: Path,
    metadata_root: Path,
    traceability: Mapping[str, Any],
):
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
    input_dim = int(train_x.shape[-1]) if train_x.ndim >= 3 and train_x.size else 0
    output_dim = int(train_y.shape[-1]) if train_y.ndim >= 3 and train_y.size else 0

    kernel_refs = _ensure_truth_entry(
        metadata_root,
        cfg.dataset_name,
        "kernel",
        str(kernel_entry.get("kernel_reference")) if kernel_entry.get("kernel_reference") is not None else None,
    )
    gfrf_refs = _ensure_truth_entry(
        metadata_root,
        cfg.dataset_name,
        "gfrf",
        str(gfrf_entry.get("gfrf_reference")) if gfrf_entry.get("gfrf_reference") is not None else None,
    )

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
                "task_usage": benchmark_entry.get("task_usage", ["prediction"]),
                "input_channels": benchmark_entry.get("input_channels", ["u"]),
                "output_channels": benchmark_entry.get("output_channels", ["y"]),
                "system_type": benchmark_entry.get("system_type"),
                "split_protocol": cfg.split_protocol,
                "split_path": str(split_path),
                "protocol_path": str(metadata_root / "protocols" / f"{cfg.split_protocol}.json"),
                "sampling_period_seconds": cfg.sampling_period_seconds,
                "raw_sources": traceability["raw_sources"],
                "trajectory_roles": traceability["trajectory_roles"],
                "split_sources": traceability["split_sources"],
            },
        },
        artifacts={
            "truth_file": kernel_refs["truth_file"],
            "grouping_file": str(split_path),
            "protocol_file": str(metadata_root / "protocols" / f"{cfg.split_protocol}.json"),
            "extra": {
                "kernel_reference": kernel_refs["truth_uri"],
                "kernel_reference_file": kernel_refs["truth_payload"],
                "gfrf_reference": gfrf_refs["truth_uri"],
                "gfrf_reference_file": gfrf_refs["truth_payload"],
                "input_dim": input_dim,
                "output_dim": output_dim,
                "raw_sources": traceability["raw_sources"],
                "trajectory_roles": traceability["trajectory_roles"],
                "split_sources": traceability["split_sources"],
                "sampling_period_seconds": cfg.sampling_period_seconds,
            },
        },
    )

    processed_manifest = {
        "dataset_name": cfg.dataset_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "build_parameters": {
            "window_length": cfg.window_length,
            "horizon": cfg.horizon,
            "split_protocol": cfg.split_protocol,
            "sampling_period_seconds": cfg.sampling_period_seconds,
        },
        "counts": {
            "train": int(len(splits["train"]["X"])),
            "val": int(len(splits["val"]["X"])),
            "test": int(len(splits["test"]["X"])),
        },
        "files": processed_files,
        "raw_sources": traceability["raw_sources"],
        "trajectory_roles": traceability["trajectory_roles"],
        "split_sources": traceability["split_sources"],
        "trajectory_window_counts": traceability["trajectory_window_counts"],
        "bundle_meta": bundle.meta.to_dict(),
        "bundle_artifacts": bundle.artifacts.to_dict(),
    }
    _write_json(processed_root / "coupled_duffing_processed_manifest.json", processed_manifest)
    _write_json(
        metadata_root / "coupled_duffing_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
            "processed_root": str(processed_root),
            "split_path": str(split_path),
            "interim_root": str(metadata_root.parent.parent / "interim" / "nonlinear" / cfg.dataset_name),
            "has_ground_truth_kernel": bool(benchmark_entry.get("has_ground_truth_kernel", False)),
            "has_ground_truth_gfrf": bool(benchmark_entry.get("has_ground_truth_gfrf", False)),
            "raw_sources": traceability["raw_sources"],
            "trajectory_roles": traceability["trajectory_roles"],
            "split_sources": traceability["split_sources"],
            "trajectory_window_counts": traceability["trajectory_window_counts"],
        },
    )

    check_dataset_bundle(bundle, strict=False)
    return bundle


def _find_protocol_root(start: Path, protocol_name: str) -> Path:
    candidates = [
        parent / "data" / "metadata" / "nonlinear" / "protocols" / f"{protocol_name}.json"
        for parent in (start, *start.parents)
    ]
    candidates.extend(
        [
            parent / "metadata" / "nonlinear" / "protocols" / f"{protocol_name}.json"
            for parent in (start, *start.parents)
            if parent.name == "data" or (parent.parent / "data").exists()
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    fallback = Path(__file__).resolve().parents[1] / "metadata" / "nonlinear" / "protocols" / f"{protocol_name}.json"
    if fallback.exists():
        return fallback
    raise CoupledDuffingBuilderError(f"Cannot locate split protocol {protocol_name}")


class CoupledDuffingBuilder(NonlinearBuilder):
    def __init__(self, context: NonlinearBuilderContext):
        super().__init__(context)
        self.last_traceability: Dict[str, Any] = {}

    def load_splits(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cfg = CoupledDuffingBuilderConfig()
        protocol_path = _find_protocol_root(self.context.interim_root, cfg.split_protocol)
        protocol = _load_json(protocol_path)

        raw = load_raw(self.context.raw_root, cfg)
        windows_by_trajectory: Dict[str, Dict[str, Any]] = {}
        self.context.interim_root.mkdir(parents=True, exist_ok=True)

        interim_payload: Dict[str, Any] = {}
        for trajectory_name, payload in raw["trajectories"].items():
            interim_payload[f"{trajectory_name}_u"] = payload["u"]
            interim_payload[f"{trajectory_name}_y"] = payload["y"]
            interim_payload[f"{trajectory_name}_sample_id"] = np.asarray(payload["sample_id"], dtype=object)
            interim_payload[f"{trajectory_name}_run_id"] = np.asarray(payload["run_id"], dtype=object)
            interim_payload[f"{trajectory_name}_timestamp"] = np.asarray(payload["timestamp"], dtype=float)
            windows = build_windows(
                payload["u"],
                payload["y"],
                window_length=cfg.window_length,
                horizon=cfg.horizon,
                sample_id=payload["sample_id"],
                run_id=payload["run_id"],
                timestamp=payload["timestamp"],
            )
            windows_by_trajectory[trajectory_name] = windows
            raw["trajectory_roles"][trajectory_name]["window_count"] = int(len(windows["X"]))

        np.savez_compressed(self.context.interim_root / f"{cfg.interim_prefix}_raw.npz", **interim_payload)
        _write_json(
            self.context.interim_root / f"{cfg.interim_prefix}_interim_manifest.json",
            {
                "dataset_name": cfg.dataset_name,
                "raw_file_candidates": list(cfg.raw_file_candidates),
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "sampling_period_seconds": cfg.sampling_period_seconds,
                "raw_sources": raw["raw_sources"],
                "trajectory_roles": raw["trajectory_roles"],
                "split_sources": raw["split_sources"],
            },
        )

        split_path = self.context.splits_root / f"{cfg.dataset_name}_split_manifest.json"
        split_payload = build_split(windows_by_trajectory, protocol, split_path, traceability=raw)
        self.last_traceability = {
            "raw_sources": raw["raw_sources"],
            "trajectory_roles": raw["trajectory_roles"],
            "split_sources": raw["split_sources"],
            "trajectory_window_counts": split_payload["trajectory_window_counts"],
        }
        return split_payload["train"], split_payload["val"], split_payload["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coupled Duffing dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split-protocol", type=str, default="nonlinear_temporal_grouped_holdout_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CoupledDuffingBuilderConfig(
        window_length=args.window_length,
        horizon=args.horizon,
        split_protocol=args.split_protocol,
    )
    cfg.validate()

    project_root = args.project_root.expanduser().resolve()
    metadata_root = project_root / "data" / "metadata" / "nonlinear"
    context = NonlinearBuilderContext(
        dataset_name=cfg.dataset_name,
        raw_root=project_root / "data" / "raw" / "nonlinear" / "coupled_duffing",
        interim_root=project_root / "data" / "interim" / "nonlinear" / "coupled_duffing",
        processed_root=project_root / "data" / "processed" / "nonlinear" / "coupled_duffing",
        splits_root=project_root / "data" / "splits" / "nonlinear",
    )

    benchmark_entry = _find_metadata_entry(metadata_root, "benchmark_manifest.json", cfg.dataset_name)
    kernel_entry = _find_metadata_entry(metadata_root, "kernel_truth_manifest.json", cfg.dataset_name)
    gfrf_entry = _find_metadata_entry(metadata_root, "gfrf_truth_manifest.json", cfg.dataset_name)

    builder = CoupledDuffingBuilder(context)
    train, val, test = builder.load_splits()
    protocol = _load_json(metadata_root / "protocols" / f"{cfg.split_protocol}.json")
    split_path = context.splits_root / f"{cfg.dataset_name}_split_manifest.json"

    export_bundle(
        cfg=cfg,
        splits={"train": train, "val": val, "test": test},
        protocol=protocol,
        benchmark_entry=benchmark_entry,
        kernel_entry=kernel_entry,
        gfrf_entry=gfrf_entry,
        processed_root=context.processed_root,
        split_path=split_path,
        metadata_root=metadata_root,
        traceability=builder.last_traceability,
    )


if __name__ == "__main__":
    main()
