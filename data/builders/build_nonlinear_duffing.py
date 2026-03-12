"""Build Duffing oscillator dataset bundle for nonlinear benchmark.

Scope is intentionally narrow:
1. raw -> interim -> processed
2. define train/val/test splits
3. export standardized dataset manifest and artifacts
4. assemble dataset bundle with unified schema

The builder does not implement model logic, training logic, or experiment control.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.adapters.base import DatasetArtifacts, DatasetBundle, DataProtocolError
from data.adapters.nonlinear_adapter import NonlinearAdapter
from data.builders.nonlinear_builder import NonlinearBuilder, NonlinearBuilderContext
from data.checks.bundle_checks import check_dataset_bundle


class DuffingBuilderError(DataProtocolError):
    """Duffing-specific build-time errors."""


@dataclass(frozen=True)
class DuffingBuilderConfig:
    """Configuration for the Duffing builder."""

    dataset_name: str = "duffing"
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "nonlinear_temporal_grouped_holdout_v1"
    target_train_ratio: float = 0.70
    target_val_ratio: float = 0.15
    target_test_ratio: float = 0.15
    raw_file_candidates: Tuple[str, ...] = (
        "duffing.mat",
        "duffing_raw.npz",
        "duffing_processed.npz",
        "duffing.csv",
        "duffing.txt",
        "DATA_EMPS.mat",
        "DATA_EMPS_PULSES.mat",
    )
    processed_prefix: str = "duffing"

    def validate(self) -> None:
        if self.window_length <= 0:
            raise DuffingBuilderError("window_length must be > 0.")
        if self.horizon <= 0:
            raise DuffingBuilderError("horizon must be > 0.")
        total = self.target_train_ratio + self.target_val_ratio + self.target_test_ratio
        if not np.isclose(total, 1.0, atol=1e-9):
            raise DuffingBuilderError("split ratios must sum to 1.0.")


def _to_float2d(value: Any, field: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DuffingBuilderError(f"{field} must be 1D or 2D numeric data.")
    if not np.issubdtype(arr.dtype, np.number):
        raise DuffingBuilderError(f"{field} must be numeric.")
    return arr.astype(np.float64)


def _to_optional_1d(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    return arr.reshape(-1)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        found = raw_root / name
        if found.exists():
            return found
    for path in raw_root.glob("**/*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".npy", ".npz", ".mat"}:
            if any(token in path.name.lower() for token in {"duffing", "duff", "emps", "nonlinear"}):
                return path
    return None


def load_raw(raw_root: Path, cfg: DuffingBuilderConfig) -> Dict[str, np.ndarray]:
    """Load Duffing raw signal data with minimal assumptions.

    Supported layout:
    - .npz/.npy with keys `u`, `y`
    - .csv/.txt with columns `u`, `y` (optional: `run_id`, `timestamp`)
    - .mat with arrays named `u`, `y` (and optional metadata keys)

    Returns:
        {
          "u": ndarray (N, 1+),
          "y": ndarray (N, 1+),
          "sample_id": optional ndarray,
          "run_id": optional ndarray,
          "timestamp": optional ndarray,
        }
    """
    raw_root = raw_root.expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)

    raw_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if raw_path is None:
        raise DuffingBuilderError(
            f"No Duffing raw file found under {raw_root}. "
            "Expected one of {duffing.mat, duffing_raw.npz, duffing.csv, duffing.txt} "
            "or a filename containing `duffing`/`nonlinear`."
        )

    u = y = None
    sample_id = run_id = timestamp = None

    if raw_path.suffix.lower() == ".npz":
        payload = np.load(raw_path, allow_pickle=True)
        u = payload.get("u")
        y = payload.get("y")
        sample_id = payload.get("sample_id")
        run_id = payload.get("run_id")
        timestamp = payload.get("timestamp")
    elif raw_path.suffix.lower() == ".npy":
        arr = np.load(raw_path)
        if arr.ndim >= 2 and arr.shape[1] >= 2:
            u = arr[:, :1]
            y = arr[:, 1:2]
        else:
            raise DuffingBuilderError("duffing .npy file requires shape [T, >=2] with first two columns u,y.")
    elif raw_path.suffix.lower() in {".csv", ".txt"}:
        import csv

        with raw_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if not rows:
            raise DuffingBuilderError(f"{raw_path} is empty.")
        # Header-aware extraction
        header = rows[0].keys()
        u_key = next((k for k in ("u", "input", "u_in") if k in header), None)
        y_key = next((k for k in ("y", "output") if k in header), None)
        if u_key is None or y_key is None:
            raise DuffingBuilderError(
                f"{raw_path} must contain at least columns for u and y."
            )
        sample_key = next((k for k in ("sample_id", "sample") if k in header), None)
        run_key = next((k for k in ("run_id", "trajectory_id") if k in header), None)
        time_key = next((k for k in ("timestamp", "time", "t") if k in header), None)

        u = np.array([float(r[u_key]) for r in rows], dtype=float).reshape(-1, 1)
        y = np.array([float(r[y_key]) for r in rows], dtype=float).reshape(-1, 1)
        if sample_key:
            sample_id = np.array([r[sample_key] for r in rows], dtype=object)
        if run_key:
            run_id = np.array([r[run_key] for r in rows], dtype=object)
        if time_key:
            timestamp = np.array([float(r[time_key]) for r in rows], dtype=float)
    elif raw_path.suffix.lower() == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise DuffingBuilderError(
                f"Cannot read .mat file without scipy installed: {raw_path}"
            ) from exc
        candidates = {
            "u": ("u", "input", "u_in", "qg", "qm"),
            "y": ("y", "output", "y_out", "qm", "qn"),
        }
        for key in candidates["u"]:
            if key in payload and isinstance(payload[key], np.ndarray):
                u = payload[key]
                break
        for key in candidates["y"]:
            if key in payload and isinstance(payload[key], np.ndarray):
                y = payload[key]
                break
        for key in ("sample_id", "sample", "sid"):
            if key in payload:
                sample_id = payload[key]
        for key in ("run_id", "trajectory_id", "traj_id"):
            if key in payload:
                run_id = payload[key]
        for key in ("timestamp", "t", "time"):
            if key in payload and isinstance(payload[key], np.ndarray):
                timestamp = payload[key]
    else:
        raise DuffingBuilderError(f"Unsupported raw file format: {raw_path.suffix}")

    if u is None or y is None:
        raise DuffingBuilderError(f"Raw source missing u/y data: {raw_path}")

    u_arr = _to_float2d(u, "u")
    y_arr = _to_float2d(y, "y")
    if len(u_arr) != len(y_arr):
        m = min(len(u_arr), len(y_arr))
        u_arr = u_arr[:m]
        y_arr = y_arr[:m]

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": _to_optional_1d(sample_id),
        "run_id": _to_optional_1d(run_id),
        "timestamp": _to_optional_1d(timestamp),
    }


def preprocess(raw_data: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    u = raw_data["u"]
    y = raw_data["y"]
    sample_id = raw_data.get("sample_id")
    run_id = raw_data.get("run_id")
    timestamp = raw_data.get("timestamp")

    finite_mask = np.isfinite(u).all(axis=1) & np.isfinite(y).all(axis=1)
    if not finite_mask.all():
        u = u[finite_mask]
        y = y[finite_mask]
        if sample_id is not None:
            sample_id = sample_id[finite_mask]
        if run_id is not None:
            run_id = run_id[finite_mask]
        if timestamp is not None:
            timestamp = timestamp[finite_mask]

    # Simple canonical ordering check: if timestamp exists, sort by timestamp.
    if timestamp is not None and len(timestamp) == len(u) and len(np.unique(timestamp)) > 1:
        order = np.argsort(timestamp)
        u = u[order]
        y = y[order]
        sample_id = sample_id[order] if sample_id is not None else None
        run_id = run_id[order] if run_id is not None else None
        timestamp = timestamp[order]

    return {
        "u": u,
        "y": y,
        "sample_id": sample_id,
        "run_id": run_id,
        "timestamp": timestamp,
    }


def build_windows(
    u: np.ndarray,
    y: np.ndarray,
    *,
    window_length: int,
    horizon: int,
    sample_id: Optional[np.ndarray] = None,
    run_id: Optional[np.ndarray] = None,
    timestamp: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n = min(len(u), len(y))
    if n <= window_length + horizon:
        raise DuffingBuilderError(
            f"Not enough time steps ({n}) for window_length={window_length}, horizon={horizon}."
        )

    x_windows: List[np.ndarray] = []
    y_windows: List[np.ndarray] = []
    sample_ids: List[Any] = []
    run_ids: List[Any] = []
    timestamps: List[Any] = []

    for idx in range(0, n - window_length - horizon + 1):
        x = u[idx : idx + window_length]
        y_true = y[idx + window_length : idx + window_length + horizon]
        # Keep one-to-one mapping between windows and targets.
        x_windows.append(x)
        y_windows.append(y_true)

        if sample_id is not None and len(sample_id) == n:
            sample_ids.append(sample_id[idx + window_length - 1])
        else:
            sample_ids.append(idx)
        if run_id is not None and len(run_id) == n:
            run_ids.append(run_id[idx + window_length - 1])
        else:
            run_ids.append(None)
        if timestamp is not None and len(timestamp) == n:
            timestamps.append(timestamp[idx + window_length - 1])
        else:
            timestamps.append(None)

    X = np.asarray(x_windows, dtype=np.float64)
    Y = np.asarray(y_windows, dtype=np.float64)

    return {
        "X": X,
        "Y": Y,
        "sample_id": np.asarray(sample_ids, dtype=object),
        "run_id": np.asarray(run_ids, dtype=object),
        "timestamp": np.asarray(timestamps, dtype=object),
    }


def _build_split_indices_by_groups(
    window_run_ids: np.ndarray,
    split_path: Path,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    protocol_split = protocol.get("split", {})
    tr_r, va_r, te_r = (
        float(protocol_split.get("train", 0.7)),
        float(protocol_split.get("val", 0.15)),
        float(protocol_split.get("test", 0.15)),
    )

    if tr_r + va_r + te_r <= 0:
        raise DuffingBuilderError("Invalid split ratios in protocol.")
    if abs((tr_r + va_r + te_r) - 1.0) > 1e-8:
        total = tr_r + va_r + te_r
        tr_r, va_r, te_r = tr_r / total, va_r / total, te_r / total

    total_windows = len(window_run_ids)
    if total_windows == 0:
        raise DuffingBuilderError("No windowed samples to split.")

    grouped = protocol.get("grouping", {})
    if not grouped.get("level") == "trajectory":
        train_end = int(total_windows * tr_r)
        val_end = train_end + int(total_windows * va_r)
        splits = {
            "train": np.arange(0, train_end),
            "val": np.arange(train_end, min(val_end, total_windows)),
            "test": np.arange(min(val_end, total_windows), total_windows),
        }
        return {
            "protocol": str(protocol.get("protocol_name")),
            "train_ratio": tr_r,
            "val_ratio": va_r,
            "test_ratio": te_r,
            "grouped": False,
            "split_indices": {k: v.tolist() for k, v in splits.items()},
        }

    # grouped split: split by trajectory/run id boundaries to avoid leakage.
    valid_mask = window_run_ids != np.array(None)
    if valid_mask.any():
        run_values = window_run_ids[valid_mask]
        groups: List[str] = []
        positions: List[List[int]] = []
        current = run_values[0]
        groups.append(str(current))
        positions.append([])
        # reconstruct indices in original space
        window_pos = np.arange(total_windows)[:]
        grouped_idx = window_pos[valid_mask]
        for local_idx, rid in zip(grouped_idx, run_values):
            rid_key = str(rid)
            if rid_key != groups[-1]:
                groups.append(rid_key)
                positions.append([])
            positions[-1].append(int(local_idx))
    else:
        # fallback to contiguous split if run ids are unavailable
        train_end = int(total_windows * tr_r)
        val_end = train_end + int(total_windows * va_r)
        splits = {
            "train": np.arange(0, train_end),
            "val": np.arange(train_end, min(val_end, total_windows)),
            "test": np.arange(min(val_end, total_windows), total_windows),
        }
        return {
            "protocol": str(protocol.get("protocol_name")),
            "train_ratio": tr_r,
            "val_ratio": va_r,
            "test_ratio": te_r,
            "grouped": False,
            "split_indices": {k: v.tolist() for k, v in splits.items()},
        }

    group_count = len(positions)
    g_train = max(1, int(group_count * tr_r))
    g_val = max(0, int(group_count * va_r))
    g_test = max(0, group_count - g_train - g_val)
    if g_train + g_val + g_test > group_count:
        g_test = group_count - g_train - g_val

    train_groups = positions[:g_train]
    val_groups = positions[g_train : g_train + g_val]
    test_groups = positions[g_train + g_val : g_train + g_val + g_test]
    # if any split is empty, move one trajectory to guarantee coverage
    if g_train == 0 and group_count > 0:
        train_groups = positions[:1]
        val_groups = positions[1:2] if group_count > 1 else []
        test_groups = positions[2:] if group_count > 2 else []

    split_indices = {
        "train": np.array([idx for g in train_groups for idx in g], dtype=int),
        "val": np.array([idx for g in val_groups for idx in g], dtype=int),
        "test": np.array([idx for g in test_groups for idx in g], dtype=int),
    }
    # deduplicate / order
    for k in split_indices:
        split_indices[k] = np.array(sorted(set(int(i) for i in split_indices[k])), dtype=int)

    if len(split_indices["test"]) == 0 and len(split_indices["val"]) > 0:
        # move last val sample if no test trajectory was assigned
        split_indices["test"] = split_indices["val"][-1:].copy()
        split_indices["val"] = split_indices["val"][:-1]

    return {
        "protocol": str(protocol.get("protocol_name")),
        "train_ratio": tr_r,
        "val_ratio": va_r,
        "test_ratio": te_r,
        "grouped": True,
        "groups": groups,
        "split_indices": {k: v.tolist() for k, v in split_indices.items()},
        "time_gap": protocol.get("leakage_control", {}).get("time_gap", 0),
    }


def build_split(window_payload: Mapping[str, Any], protocol: Mapping[str, Any], split_path: Path) -> Dict[str, np.ndarray]:
    # Prefer trajectory-aware split when run_id exists.
    window_run_ids = np.asarray(window_payload.get("run_id", np.array([], dtype=object)), dtype=object)
    split_info = _build_split_indices_by_groups(window_run_ids, split_path, protocol)
    split_indices = split_info["split_indices"]
    if (
        len(split_indices["train"]) == 0
        or len(split_indices["val"]) == 0
        or len(split_indices["test"]) == 0
    ):
        # Soft fallback to ratio-based contiguous split to avoid empty folds.
        window_count = len(window_payload["X"])
        protocol_split = protocol.get("split", {})
        tr = float(protocol_split.get("train", 0.7))
        va = float(protocol_split.get("val", 0.15))
        te = float(protocol_split.get("test", 0.15))
        if tr + va + te <= 0:
            tr, va, te = 0.7, 0.15, 0.15
        if abs((tr + va + te) - 1.0) > 1e-8:
            total = tr + va + te
            tr, va, te = tr / total, va / total, te / total
        train_end = int(window_count * tr)
        val_end = train_end + int(window_count * va)
        split_indices = {
            "train": np.arange(0, train_end),
            "val": np.arange(train_end, min(val_end, window_count)),
            "test": np.arange(min(val_end, window_count), window_count),
        }
        split_info["grouped"] = False
        split_info["fallback"] = "ratio_contiguous"

    splits = {
        split_name: {
            "X": window_payload["X"][idxs],
            "Y": window_payload["Y"][idxs],
            "sample_id": (
                np.asarray(window_payload["sample_id"])[idxs]
                if "sample_id" in window_payload and len(window_payload["sample_id"]) > 0
                else None
            ),
            "run_id": (
                np.asarray(window_payload["run_id"])[idxs]
                if "run_id" in window_payload and len(window_payload["run_id"]) > 0
                else None
            ),
            "timestamp": (
                np.asarray(window_payload["timestamp"])[idxs]
                if "timestamp" in window_payload and len(window_payload["timestamp"]) > 0
                else None
            ),
        }
        for split_name, idxs in split_indices.items()
    }

    split_info["num_samples"] = {
        "all": len(window_payload["X"]),
        "train": len(splits["train"]["X"]),
        "val": len(splits["val"]["X"]),
        "test": len(splits["test"]["X"]),
    }

    split_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(split_path, split_info)
    return splits


def _ensure_split_protocol(protocol_root: Path, protocol_name: str) -> Mapping[str, Any]:
    protocol_path = protocol_root / f"{protocol_name}.json"
    if not protocol_path.exists():
        raise DuffingBuilderError(
            f"Split protocol file missing: {protocol_path}"
        )
    return _load_json(protocol_path)


def _load_nonlin_benchmark_entry(metadata_root: Path, dataset_name: str) -> Mapping[str, Any]:
    manifest = _load_json(metadata_root / "benchmark_manifest.json")
    for item in manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return item
    raise DuffingBuilderError(f"benchmark_manifest.json missing benchmark '{dataset_name}'.")


def _load_truth_reference(metadata_root: Path, truth_type: str, dataset_name: str) -> Mapping[str, Any]:
    filename = "kernel_truth_manifest.json" if truth_type == "kernel" else "gfrf_truth_manifest.json"
    manifest = _load_json(metadata_root / filename)
    for item in manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return item
    raise DuffingBuilderError(f"{filename} missing benchmark '{dataset_name}'.")


def _extract_scalar_from_script(script_path: Path, variable_name: str) -> float:
    if not script_path.exists():
        raise DuffingBuilderError(f"Missing reference script: {script_path}")
    pattern = re.compile(rf"^\s*{re.escape(variable_name)}\s*=\s*([-+0-9.eE]+)\s*;", re.MULTILINE)
    text = script_path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise DuffingBuilderError(f"Could not find {variable_name} in {script_path}")
    return float(match.group(1))


def _materialize_reference_truth(
    metadata_root: Path,
    raw_root: Path,
    cfg: DuffingBuilderConfig,
    dataset_name: str,
    benchmark_entry: Mapping[str, Any],
) -> Dict[str, Path]:
    try:
        import scipy.io
        from scipy import signal
    except Exception as exc:  # pragma: no cover - environment guard
        raise DuffingBuilderError("Duffing truth materialization requires scipy.") from exc

    truth_root = metadata_root / "truth"
    truth_root.mkdir(parents=True, exist_ok=True)

    data_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if data_path is None or data_path.suffix.lower() != ".mat":
        raise DuffingBuilderError(f"Could not locate Duffing reference .mat file under {raw_root}")
    payload = scipy.io.loadmat(data_path, squeeze_me=True)
    required_scalars = ("gtau", "kp", "kv")
    for key in required_scalars:
        if key not in payload:
            raise DuffingBuilderError(f"{data_path} is missing required scalar '{key}'.")

    gtau = float(np.asarray(payload["gtau"]).reshape(()))
    kp = float(np.asarray(payload["kp"]).reshape(()))
    kv = float(np.asarray(payload["kv"]).reshape(()))
    time = np.asarray(payload.get("t"))
    if time.ndim == 0 or time.size < 2:
        raise DuffingBuilderError(f"{data_path} is missing a valid time vector 't'.")
    dt = float(np.median(np.diff(time.reshape(-1))))
    if dt <= 0:
        raise DuffingBuilderError(f"Invalid Duffing sampling interval derived from {data_path}: {dt}")

    simulation_script = raw_root / "Simulation_EMPS.m"
    mass = _extract_scalar_from_script(simulation_script, "M1")
    damping = _extract_scalar_from_script(simulation_script, "Fv1")

    natural_gain = gtau * kp * kv / mass
    numerator = np.asarray([natural_gain], dtype=np.float64)
    denominator = np.asarray([1.0, (damping + gtau * kv) / mass, natural_gain], dtype=np.float64)
    discrete = signal.cont2discrete((numerator, denominator), dt, method="zoh")
    numerator_d = np.squeeze(np.asarray(discrete[0], dtype=np.float64))
    denominator_d = np.squeeze(np.asarray(discrete[1], dtype=np.float64))
    _, impulse = signal.dimpulse((numerator_d, denominator_d, discrete[2]), n=cfg.window_length + 1)
    impulse_response = np.asarray(impulse[0], dtype=np.float64).reshape(-1)
    if impulse_response.size < cfg.window_length + 1:
        raise DuffingBuilderError("Discrete impulse response is shorter than the configured window length.")

    order_1_kernel = np.asarray(impulse_response[1 : cfg.window_length + 1][::-1], dtype=np.float64)
    kernel_payload_path = truth_root / f"{dataset_name}_kernel_truth_payload.npz"
    gfrf_payload_path = truth_root / f"{dataset_name}_gfrf_truth_payload.npz"
    np.savez_compressed(kernel_payload_path, order_1=order_1_kernel)
    np.savez_compressed(gfrf_payload_path, order_1=np.fft.fft(order_1_kernel, n=cfg.window_length))

    protocol_reference = f"nonlinear://protocol/{benchmark_entry.get('recommended_split_protocol')}"
    kernel_reference = truth_root / f"{dataset_name}_kernel_bundle_reference.json"
    gfrf_reference = truth_root / f"{dataset_name}_gfrf_bundle_reference.json"
    common_derivation = {
        "source_dataset": str(data_path),
        "source_script": str(simulation_script),
        "window_length": int(cfg.window_length),
        "sampling_interval_seconds": dt,
        "parameters": {
            "gtau": gtau,
            "kp": kp,
            "kv": kv,
            "M1": mass,
            "Fv1": damping,
        },
        "model": "EMPS closed-loop second-order reference",
        "window_semantics": "oldest_to_most_recent",
    }
    _write_json(
        kernel_reference,
        {
            "benchmark_name": dataset_name,
            "truth_type": "kernel",
            "protocol_reference": protocol_reference,
            "registry_reference": benchmark_entry.get("artifacts", {}).get("kernel_reference"),
            "status": "materialized",
            "kernel_coefficients_path": str(kernel_payload_path),
            "orders_materialized": [1],
            "derivation": common_derivation,
            "notes": "Materialized order-1 reference kernel derived from the official EMPS reference model.",
        },
    )
    _write_json(
        gfrf_reference,
        {
            "benchmark_name": dataset_name,
            "truth_type": "gfrf",
            "protocol_reference": protocol_reference,
            "registry_reference": benchmark_entry.get("artifacts", {}).get("gfrf_reference"),
            "status": "materialized",
            "gfrf_coefficients_path": str(gfrf_payload_path),
            "orders_materialized": [1],
            "derivation": {**common_derivation, "nfft": int(cfg.window_length)},
            "notes": "Materialized order-1 reference GFRF derived from the official EMPS reference model.",
        },
    )
    return {
        "kernel_reference": kernel_reference,
        "gfrf_reference": gfrf_reference,
        "kernel_payload": kernel_payload_path,
        "gfrf_payload": gfrf_payload_path,
    }


def export_bundle(
    cfg: DuffingBuilderConfig,
    context: NonlinearBuilderContext,
    splits: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    benchmark_entry: Mapping[str, Any],
    kernel_entry: Mapping[str, Any],
    gfrf_entry: Mapping[str, Any],
    split_path: Path,
    processed_root: Path,
    metadata_root: Path,
) -> DatasetBundle:
    processed_root = processed_root
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_files = {}
    for name, payload in splits.items():
        for key in ("X", "Y"):
            arr = np.asarray(payload[key])
            if arr.ndim == 0:
                raise DuffingBuilderError(f"{name}/{key} is empty.")
            fname = f"{name}_{key.lower()}.npy"
            np.save(processed_root / fname, arr)
            processed_files[f"{name}_{key}"] = fname
        if payload.get("sample_id") is not None and len(payload["sample_id"]):
            fname = f"{name}_sample_id.npy"
            np.save(processed_root / fname, np.asarray(payload["sample_id"], dtype=object))
            processed_files[f"{name}_sample_id"] = fname

    input_dim = int(np.asarray(splits["train"]["X"])[0].shape[1]) if len(splits["train"]["X"]) else int(0)
    output_dim = int(np.asarray(splits["train"]["Y"])[0].shape[-1]) if len(splits["train"]["Y"]) else int(0)

    kernel_ref = benchmark_entry.get("artifacts", {}).get("kernel_reference")
    gfrf_ref = benchmark_entry.get("artifacts", {}).get("gfrf_reference")
    grouping_ref = benchmark_entry.get("artifacts", {}).get("grouping_reference")
    protocol_ref = str(metadata_root / "protocols" / f"{cfg.split_protocol}.json")

    truth_files = _materialize_reference_truth(
        metadata_root=metadata_root,
        raw_root=context.raw_root,
        cfg=cfg,
        dataset_name=cfg.dataset_name,
        benchmark_entry=benchmark_entry,
    )
    kernel_file = truth_files["kernel_reference"]
    gfrf_file = truth_files["gfrf_reference"]

    train_payload = {k: v for k, v in splits["train"].items() if k != "meta"}
    val_payload = {k: v for k, v in splits["val"].items() if k != "meta"}
    test_payload = {k: v for k, v in splits["test"].items() if k != "meta"}

    bundle = NonlinearAdapter.build_bundle(
        dataset_name=cfg.dataset_name,
        train=train_payload,
        val=val_payload,
        test=test_payload,
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
                "task_usage": benchmark_entry.get("task_usage", []),
                "input_channels": benchmark_entry.get("input_channels", ["u"]),
                "output_channels": benchmark_entry.get("output_channels", ["y"]),
                "split_path": str(split_path),
                "protocol_name": protocol.get("protocol_name"),
                "protocol_reference": protocol_ref,
                "artifacts": {
                    "kernel_reference": kernel_ref,
                    "gfrf_reference": gfrf_ref,
                    "grouping_reference": grouping_ref,
                },
                "kernel_entry": kernel_entry,
                "gfrf_entry": gfrf_entry,
            },
        },
        artifacts={
            "truth_file": str(gfrf_file),
            "grouping_file": str(split_path),
            "protocol_file": protocol_ref,
            "extra": {
                "kernel_reference_file": str(kernel_file),
                "gfrf_reference_file": str(gfrf_file),
                "input_file_candidates": list(cfg.raw_file_candidates),
            },
        },
    )

    manifest = {
        "dataset_name": cfg.dataset_name,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "processed_root": str(processed_root),
        "split_file": str(split_path),
        "protocol_file": protocol_ref,
        "processed_files": processed_files,
        "bundle_meta": bundle.meta.to_dict(),
        "bundle_artifacts": bundle.artifacts.to_dict(),
        "sample_counts": {
            "train": len(train_payload["X"]),
            "val": len(val_payload["X"]),
            "test": len(test_payload["X"]),
        },
        "input_dim": input_dim,
        "output_dim": output_dim,
    }
    _write_json(processed_root / f"{cfg.processed_prefix}_processed_manifest.json", manifest)
    _write_json(metadata_root / f"{cfg.processed_prefix}_benchmark_build_manifest.json", manifest)

    check_dataset_bundle(bundle)
    return bundle


class DuffingBuilder(NonlinearBuilder):
    """Concrete Duffing builder using explicit pipeline methods."""

    def load_splits(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        raw_root = self.context.raw_root
        interim_root = self.context.interim_root
        split_root = self.context.splits_root

        cfg = DuffingBuilderConfig()
        protocol_root = split_root
        for parent in [interim_root] + list(interim_root.parents):
            candidate = parent / "data" / "metadata" / "nonlinear" / "protocols"
            if candidate.exists():
                protocol_root = candidate
                break

        protocol = _ensure_split_protocol(protocol_root, cfg.split_protocol)
        raw = load_raw(raw_root, cfg)
        pre = preprocess(raw)

        interim_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            interim_root / f"{cfg.processed_prefix}_cleaned.npz",
            u=pre["u"],
            y=pre["y"],
            sample_id=np.asarray(pre["sample_id"], dtype=object) if pre["sample_id"] is not None else np.array([], dtype=object),
            run_id=np.asarray(pre["run_id"], dtype=object) if pre["run_id"] is not None else np.array([], dtype=object),
            timestamp=np.asarray(pre["timestamp"], dtype=float) if pre["timestamp"] is not None else np.array([], dtype=float),
        )
        _write_json(
            interim_root / f"{cfg.processed_prefix}_interim_manifest.json",
            {
                "dataset_name": cfg.dataset_name,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "raw_candidates": list(cfg.raw_file_candidates),
                "preprocessing": {
                    "window_length": cfg.window_length,
                    "horizon": cfg.horizon,
                },
            },
        )

        windows = build_windows(
            pre["u"],
            pre["y"],
            window_length=cfg.window_length,
            horizon=cfg.horizon,
            sample_id=pre["sample_id"],
            run_id=pre["run_id"],
            timestamp=pre["timestamp"],
        )

        split_path = split_root / f"{cfg.processed_prefix}_split_manifest.json"
        split_data = build_split(windows, protocol, split_path)
        return (
            split_data["train"],
            split_data["val"],
            split_data["test"],
        )


def _run_build(project_root: Path, cfg: DuffingBuilderConfig) -> DatasetBundle:
    cfg.validate()

    raw_root = project_root / "data" / "raw" / "nonlinear" / "duffing"
    interim_root = project_root / "data" / "interim" / "nonlinear" / "duffing"
    processed_root = project_root / "data" / "processed" / "nonlinear" / "duffing"
    splits_root = project_root / "data" / "splits" / "nonlinear"
    metadata_root = project_root / "data" / "metadata" / "nonlinear"

    context = NonlinearBuilderContext(
        dataset_name=cfg.dataset_name,
        raw_root=raw_root,
        interim_root=interim_root,
        processed_root=processed_root,
        splits_root=splits_root,
    )
    builder = DuffingBuilder(context)

    # 1) build splits (raw->interim->split)
    train, val, test = builder.load_splits()
    # 2) load metadata for unified bundle
    benchmark_entry = _load_nonlin_benchmark_entry(metadata_root, cfg.dataset_name)
    kernel_entry = _load_truth_reference(metadata_root, "kernel", cfg.dataset_name)
    gfrf_entry = _load_truth_reference(metadata_root, "gfrf", cfg.dataset_name)

    split_path = splits_root / f"{cfg.processed_prefix}_split_manifest.json"
    protocol = _ensure_split_protocol(
        metadata_root / "protocols", cfg.split_protocol
    )

    splits = {"train": train, "val": val, "test": test}
    bundle = export_bundle(
        cfg=cfg,
        context=context,
        splits=splits,
        protocol=protocol,
        benchmark_entry=benchmark_entry,
        kernel_entry=kernel_entry,
        gfrf_entry=gfrf_entry,
        split_path=split_path,
        processed_root=processed_root,
        metadata_root=metadata_root,
    )
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Duffing dataset bundle.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing data/ and metadata directories.",
    )
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--split-protocol",
        default="nonlinear_temporal_grouped_holdout_v1",
        help="Split protocol file name (without .json) in data/metadata/nonlinear/protocols.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DuffingBuilderConfig(
        window_length=args.window_length,
        horizon=args.horizon,
        split_protocol=args.split_protocol,
    )
    cfg.validate()
    project_root = args.project_root.expanduser().resolve()
    _run_build(project_root, cfg)


if __name__ == "__main__":
    main()
