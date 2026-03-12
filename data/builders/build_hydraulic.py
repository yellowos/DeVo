"""Build hydraulic dataset bundle.

This script only implements the data layer for the hydraulic benchmark:
raw -> interim -> processed, plus DatasetBundle export.
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

from data.adapters.base import DataProtocolError, DatasetBundle
from data.adapters.hydraulic_adapter import HydraulicAdapter
from data.checks.bundle_checks import check_dataset_bundle


class HydraulicBuilderError(DataProtocolError):
    """Raised when hydraulic data construction fails."""


@dataclass(frozen=True)
class HydraulicBuilderConfig:
    dataset_name: str = "hydraulic"
    raw_file_candidates: Tuple[str, ...] = (
        "hydraulic.mat",
        "hydraulic_raw.npz",
        "hydraulic_processed.npz",
        "hydraulic.csv",
        "hydraulic.txt",
    )

    def validate(self) -> None:
        if not self.dataset_name:
            raise HydraulicBuilderError("dataset_name must be non-empty.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float2d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise HydraulicBuilderError(f"{name} must be 2D numeric data.")
    if not np.issubdtype(arr.dtype, np.number):
        raise HydraulicBuilderError(f"{name} contains non-numeric values.")
    return arr.astype(np.float64)


def _to_text_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([], dtype=object)
    return arr.reshape(-1).astype(object)


def _as_scalar_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        iv = int(np.asarray(value).reshape(-1)[0])
    except Exception:
        return None
    return iv


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = raw_root / name
        if candidate.exists():
            return candidate
    for path in raw_root.glob("**/*"):
        if path.is_file() and path.suffix.lower() in {".mat", ".npz", ".npy", ".csv", ".txt"}:
            if "hydraulic" in path.name.lower():
                return path
    return None


def _load_channel_map(metadata_root: Path) -> Dict[str, Any]:
    payload = _load_json(metadata_root / "channel_map.json")
    channels = payload.get("channels")
    if not isinstance(channels, list) or len(channels) != 17:
        raise HydraulicBuilderError("channel_map.json must define exactly 17 channels.")
    payload["channels"] = [str(c) for c in channels]
    return payload


def _load_subsystem_groups(metadata_root: Path) -> Dict[str, Any]:
    payload = _load_json(metadata_root / "subsystem_groups.json")
    subs = payload.get("subsystems")
    if not isinstance(subs, Mapping):
        raise HydraulicBuilderError("subsystem_groups.json must contain a 'subsystems' mapping.")
    required = {"cooler", "valve", "pump", "accumulator"}
    missing = required.difference(set(subs.keys()))
    if missing:
        raise HydraulicBuilderError(f"subsystem_groups missing keys: {', '.join(sorted(missing))}")
    return payload


def _load_single_fault_protocol(metadata_root: Path) -> Dict[str, Any]:
    payload = _load_json(metadata_root / "single_fault_protocol.json")
    required = {
        "window_length",
        "horizon",
        "split_protocol",
        "split",
        "data_fields",
        "labeling",
        "single_fault_filter",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise HydraulicBuilderError(f"single_fault_protocol missing fields: {', '.join(sorted(missing))}")
    return payload


def _select_x_matrix_from_columns(rows: Sequence[Mapping[str, Any]], channels: Sequence[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(channels)), dtype=float)
    row0 = rows[0]
    missing = [c for c in channels if c not in row0]
    if missing:
        raise HydraulicBuilderError(
            f"Raw CSV header missing channel columns: {', '.join(missing)}. "
            f"Expected canonical 17-channel order from metadata/channel_map.json."
        )
    x = np.asarray(
        [[float(r.get(c)) for c in channels] for r in rows],
        dtype=float,
    )
    return x


def load_raw(raw_root: Path, protocol: Mapping[str, Any], channel_map: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = HydraulicBuilderConfig()
    raw_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if raw_path is None:
        raise HydraulicBuilderError(
            f"No hydraulic raw file found under {raw_root}. Checked candidates: {list(cfg.raw_file_candidates)}."
        )

    channels = channel_map["channels"]
    data = fields_cfg = protocol["data_fields"]

    u = None
    cycle_id = None
    subsystem_label = None
    fault_label = None
    condition_type = None
    fault_count = None
    sample_id = None

    suffix = raw_path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(raw_path, allow_pickle=True)
        if "X" in payload:
            u = _to_float2d(payload.get("X"), "X")
        elif "data" in payload:
            u = _to_float2d(payload.get("data"), "data")
        elif "signals" in payload:
            u = _to_float2d(payload.get("signals"), "signals")
        else:
            # fallback key scan for numeric arrays
            arr_candidates = [
                payload[k] for k in payload.files if isinstance(payload[k], np.ndarray) and np.asarray(payload[k]).ndim >= 2
            ]
            if arr_candidates:
                u = _to_float2d(arr_candidates[0], "array")
            else:
                raise HydraulicBuilderError(f"{raw_path} does not include 2D numeric signal array.")

        if u.shape[1] < len(channels):
            raise HydraulicBuilderError(
                f"Hydraulic signal array has {u.shape[1]} channels, expected at least {len(channels)}."
            )
        u = u[:, : len(channels)]

        cycle_id = payload.get(data["cycle_field"]) if isinstance(payload, Mapping) else None
        subsystem_label = payload.get(data["subsystem_field"]) if isinstance(payload, Mapping) else None
        fault_label = payload.get(data["fault_label_field"]) if isinstance(payload, Mapping) else None
        condition_type = payload.get(data["condition_field"]) if isinstance(payload, Mapping) else None
        fault_count = payload.get(data["fault_count_field"]) if isinstance(payload, Mapping) else None
        sample_id = payload.get(data.get("sample_id_field", "sample_id")) if isinstance(payload, Mapping) else None

    elif suffix == ".npy":
        arr = np.load(raw_path)
        if arr.ndim == 1:
            raise HydraulicBuilderError(".npy input must be 2D [time, channels] for hydraulic.")
        u = _to_float2d(arr, "hydraulic npy")
        if u.shape[1] < len(channels):
            raise HydraulicBuilderError(
                f"Hydraulic npy has {u.shape[1]} channels, expected at least {len(channels)}."
            )
        u = u[:, : len(channels)]
        n = len(u)
        cycle_id = np.array([f"cycle_{i}" for i in range(n)], dtype=object)
        subsystem_label = np.full(n, "healthy", dtype=object)
        fault_label = np.full(n, "healthy", dtype=object)
        condition_type = np.full(n, "healthy", dtype=object)
        fault_count = np.zeros(n, dtype=np.int64)
        sample_id = np.arange(n, dtype=object)

    elif suffix in {".csv", ".txt"}:
        with raw_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise HydraulicBuilderError(f"{raw_path} is empty.")

        u = _select_x_matrix_from_columns(rows, channels)

        n = len(rows)
        cycle_key = data.get("cycle_field", "cycle_id")
        subsystem_key = data.get("subsystem_field", "subsystem_label")
        fault_key = data.get("fault_label_field", "fault_label")
        condition_key = data.get("condition_field", "condition_type")
        fault_count_key = data.get("fault_count_field", "fault_count")
        sample_key = data.get("sample_id_field", "sample_id")

        cycle_id = np.asarray([r.get(cycle_key) for r in rows], dtype=object)
        subsystem_label = np.asarray([r.get(subsystem_key) for r in rows], dtype=object)
        fault_label = np.asarray([r.get(fault_key) for r in rows], dtype=object)
        condition_type = np.asarray([r.get(condition_key) for r in rows], dtype=object)
        fault_count = np.asarray([r.get(fault_count_key) for r in rows], dtype=object)
        sample_id = np.asarray([r.get(sample_key) for r in rows], dtype=object)

    elif suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise HydraulicBuilderError("Cannot read .mat without scipy." ) from exc

        if "signals" in payload:
            u = _to_float2d(payload["signals"], "signals")
        elif "X" in payload:
            u = _to_float2d(payload["X"], "X")
        elif "data" in payload:
            u = _to_float2d(payload["data"], "data")
        else:
            # first numeric 2d matrix
            for key, value in payload.items():
                if key.startswith("__"):
                    continue
                arr = np.asarray(value)
                if arr.ndim == 2 and arr.shape[1] >= len(channels):
                    u = arr
                    break
            if u is None:
                raise HydraulicBuilderError(f"{raw_path} does not include usable hydraulic matrix.")
            u = _to_float2d(u, "matrix")

        if u.shape[1] < len(channels):
            raise HydraulicBuilderError(
                f"Hydraulic mat has {u.shape[1]} channels, expected at least {len(channels)}."
            )
        u = u[:, : len(channels)]

        cycle_id = payload.get(data["cycle_field"]) if isinstance(payload, Mapping) else None
        subsystem_label = payload.get(data["subsystem_field"]) if isinstance(payload, Mapping) else None
        fault_label = payload.get(data["fault_label_field"]) if isinstance(payload, Mapping) else None
        condition_type = payload.get(data["condition_field"]) if isinstance(payload, Mapping) else None
        fault_count = payload.get(data["fault_count_field"]) if isinstance(payload, Mapping) else None
        sample_id = payload.get(data.get("sample_id_field", "sample_id")) if isinstance(payload, Mapping) else None
    else:
        raise HydraulicBuilderError(f"Unsupported raw file format: {suffix}")

    if cycle_id is None:
        cycle_id = np.arange(len(u), dtype=object)
    if sample_id is None:
        sample_id = np.arange(len(u), dtype=object)
    if fault_label is None:
        fault_label = np.full(len(u), "healthy", dtype=object)
    if subsystem_label is None:
        subsystem_label = np.full(len(u), "healthy", dtype=object)
    if condition_type is None:
        condition_type = np.array(["healthy" for _ in range(len(u))], dtype=object)
    if fault_count is None:
        fault_count = np.zeros(len(u), dtype=np.int64)

    if isinstance(fault_count, np.ndarray) and fault_count.dtype.kind in {"U", "S", "O"}:
        fault_count = np.array([_as_scalar_int(v) if v not in (None, "") else 0 for v in fault_count], dtype=object)

    return {
        "u": _to_float2d(u, "u"),
        "cycle_id": _to_text_array(cycle_id),
        "subsystem_label": _to_text_array(subsystem_label),
        "fault_label": _to_text_array(fault_label),
        "condition_type": _to_text_array(condition_type),
        "fault_count": np.asarray(fault_count, dtype=object),
        "sample_id": _to_text_array(sample_id),
    }


def _encode_labels(
    subsystem_label: np.ndarray,
    condition_type: np.ndarray,
    fault_count: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    n = len(subsystem_label)
    out = np.zeros((n, len(classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for i, (subsys, cond) in enumerate(zip(subsystem_label, condition_type)):
        key = "healthy"
        if cond is not None and str(cond).lower() in {"degraded", "fault", "abnormal", "anomaly"}:
            if subsys is not None and str(subsys).strip() and str(subsys) in class_to_idx and str(subsys) != "healthy":
                key = str(subsys)
            else:
                fc = fault_count[i]
                if fc is None:
                    key = "healthy"
                else:
                    try:
                        k = int(fc)
                        if k > 1:
                            key = "healthy"
                    except Exception:
                        key = "healthy"
        else:
            key = "healthy"

        idx = class_to_idx.get(key)
        if idx is None:
            idx = class_to_idx.get("healthy", 0)
        out[i, idx] = 1.0

    return out


def select_single_fault_rows(
    protocol: Mapping[str, Any],
    cycle_id: np.ndarray,
    subsystem_label: np.ndarray,
    fault_label: np.ndarray,
    condition_type: np.ndarray,
    fault_count: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    sf = protocol["single_fault_filter"]
    labeling = protocol["labeling"]

    enable = bool(sf.get("enable", True))
    if not enable:
        return np.ones(len(cycle_id), dtype=bool)

    min_fault = int(sf.get("min_active_fault_count", 1))
    max_fault = int(sf.get("max_active_fault_count", 1))
    include_healthy = bool(sf.get("allow_healthy", True))
    degraded_value = str(labeling.get("degraded_value", "degraded"))
    healthy_value = str(labeling.get("healthy_value", "healthy"))

    cond_ok = np.array([str(v).lower() == healthy_value for v in condition_type], dtype=bool)
    deg_ok = np.array([str(v).lower() == degraded_value for v in condition_type], dtype=bool)

    # if fault_count not available, infer from subsystem label string.
    counts = np.full(len(fault_count), 1, dtype=int)
    parsed_count = np.zeros(len(fault_count), dtype=int)
    has_count = 0
    for i, c in enumerate(fault_count):
        if c is None:
            continue
        if isinstance(c, (int, np.integer)):
            parsed_count[i] = int(c)
            has_count += 1
        else:
            try:
                parsed_count[i] = int(str(c))
                has_count += 1
            except Exception:
                parsed_count[i] = 0
    if has_count > 0:
        counts = parsed_count
    else:
        allowed = set(classes)
        counts = np.array([0 if str(s).lower() == healthy_value else 1 for s in subsystem_label], dtype=int)

    degraded_single = (
        deg_ok
        & (counts >= min_fault)
        & (counts <= max_fault)
        & np.array([str(ss) in set(labeling.get("allowed_fault_classes", classes[1:])) or str(ss) == "" for ss in subsystem_label], dtype=bool)
    )

    subsys_ok = np.array([str(ss).strip() != "" for ss in subsystem_label], dtype=bool)

    return (cond_ok | degraded_single) if include_healthy else (degraded_single & subsys_ok)


def build_windows(
    x: np.ndarray,
    y_onehot: np.ndarray,
    cycle_id: np.ndarray,
    fault_label: np.ndarray,
    subsystem_label: np.ndarray,
    condition_type: np.ndarray,
    sample_id: np.ndarray,
    window_length: int,
    horizon: int,
) -> Dict[str, Any]:
    n = len(x)
    if n <= window_length + horizon:
        raise HydraulicBuilderError(
            f"Not enough points for windowing: n={n}, window_length={window_length}, horizon={horizon}."
        )

    X_list = []
    Y_list = []
    sample_ids = []
    cycle_list = []
    fault_list = []
    subsystem_list = []
    condition_list = []

    for idx in range(0, n - window_length - horizon + 1):
        X_list.append(x[idx : idx + window_length])
        Y_list.append(y_onehot[idx + window_length : idx + window_length + horizon])

        t = idx + window_length - 1
        sample_ids.append(sample_id[t])
        cycle_list.append(cycle_id[t])
        fault_list.append(fault_label[t])
        subsystem_list.append(subsystem_label[t])
        condition_list.append(condition_type[t])

    return {
        "X": np.asarray(X_list, dtype=np.float64),
        "Y": np.asarray(Y_list, dtype=np.float64),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "cycle_id": np.asarray(cycle_list, dtype=object),
        "fault_label": np.asarray(fault_list, dtype=object),
        "subsystem_label": np.asarray(subsystem_list, dtype=object),
        "condition_type": np.asarray(condition_list, dtype=object),
    }


def _split_indices_by_cycle_or_ratio(
    windows: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    x = np.asarray(windows["X"])
    cycle_id = np.asarray(windows.get("cycle_id", np.array([], dtype=object)), dtype=object)

    split_cfg = protocol.get("split", {})
    tr = float(split_cfg.get("train", 0.70))
    va = float(split_cfg.get("val", 0.15))
    te = float(split_cfg.get("test", 0.15))
    total = tr + va + te
    if total <= 0:
        raise HydraulicBuilderError("invalid split ratio sum <=0")
    if abs(total - 1.0) > 1e-9:
        tr, va, te = tr / total, va / total, te / total

    if protocol.get("grouping", {}).get("level") == "cycle" and cycle_id.size == len(x):
        valid = np.array([c is not None and str(c) != "" for c in cycle_id], dtype=bool)
        if valid.any():
            cycles = cycle_id[valid]
            unique_cycles = [c for c in dict.fromkeys(cycles.tolist())]
            if unique_cycles:
                n_c = len(unique_cycles)
                n_tr = max(1, int(n_c * tr))
                n_va = max(0, int(n_c * va))
                n_te = max(0, n_c - n_tr - n_va)
                if n_tr + n_va + n_te > n_c:
                    n_te = n_c - n_tr - n_va

                train_cycles = set(unique_cycles[:n_tr])
                val_cycles = set(unique_cycles[n_tr : n_tr + n_va])
                test_cycles = set(unique_cycles[n_tr + n_va : n_tr + n_va + n_te])

                idx_train = np.array([i for i, c in enumerate(cycle_id) if c in train_cycles], dtype=int)
                idx_val = np.array([i for i, c in enumerate(cycle_id) if c in val_cycles], dtype=int)
                idx_test = np.array([i for i, c in enumerate(cycle_id) if c in test_cycles], dtype=int)

                if len(idx_train) > 0 and len(idx_val) > 0 and len(idx_test) > 0:
                    return {"train": idx_train, "val": idx_val, "test": idx_test}

    n = len(x)
    tr_end = int(n * tr)
    va_end = tr_end + int(n * va)
    idx_train = np.arange(0, max(1, tr_end), dtype=int)
    idx_val = np.arange(max(1, tr_end), min(n, max(1, va_end)), dtype=int)
    idx_test = np.arange(min(n, max(1, va_end)), n, dtype=int)
    if len(idx_val) == 0 and n - idx_val.size > idx_train.size:
        idx_val = np.array([idx_train[-1]], dtype=int)
    if len(idx_test) == 0 and n > idx_val.size:
        idx_test = np.array([n - 1], dtype=int)
    return {"train": idx_train, "val": idx_val, "test": idx_test}


def build_split(window_payload: Mapping[str, Any], protocol: Mapping[str, Any], split_path: Path) -> Dict[str, Dict[str, Any]]:
    x = np.asarray(window_payload["X"])
    idx_map = _split_indices_by_cycle_or_ratio(window_payload, protocol)

    split_payload = {
        "dataset_name": "hydraulic",
        "protocol_name": protocol.get("protocol_name"),
        "split_indices": {k: v.tolist() for k, v in idx_map.items()},
        "counts": {k: int(len(v)) for k, v in idx_map.items()},
        "protocol": protocol,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window_count": int(len(x)),
        "grouping": protocol.get("grouping", {}),
    }
    _write_json(split_path, split_payload)

    return {
        "train": {
            "X": x[idx_map["train"]],
            "Y": np.asarray(window_payload["Y"])[idx_map["train"]],
            "sample_id": np.asarray(window_payload["sample_id"])[idx_map["train"]],
            "cycle_id": np.asarray(window_payload["cycle_id"])[idx_map["train"]],
            "fault_label": np.asarray(window_payload["fault_label"])[idx_map["train"]],
            "subsystem_label": np.asarray(window_payload["subsystem_label"])[idx_map["train"]],
            "condition_type": np.asarray(window_payload["condition_type"])[idx_map["train"]],
        },
        "val": {
            "X": x[idx_map["val"]],
            "Y": np.asarray(window_payload["Y"])[idx_map["val"]],
            "sample_id": np.asarray(window_payload["sample_id"])[idx_map["val"]],
            "cycle_id": np.asarray(window_payload["cycle_id"])[idx_map["val"]],
            "fault_label": np.asarray(window_payload["fault_label"])[idx_map["val"]],
            "subsystem_label": np.asarray(window_payload["subsystem_label"])[idx_map["val"]],
            "condition_type": np.asarray(window_payload["condition_type"])[idx_map["val"]],
        },
        "test": {
            "X": x[idx_map["test"]],
            "Y": np.asarray(window_payload["Y"])[idx_map["test"]],
            "sample_id": np.asarray(window_payload["sample_id"])[idx_map["test"]],
            "cycle_id": np.asarray(window_payload["cycle_id"])[idx_map["test"]],
            "fault_label": np.asarray(window_payload["fault_label"])[idx_map["test"]],
            "subsystem_label": np.asarray(window_payload["subsystem_label"])[idx_map["test"]],
            "condition_type": np.asarray(window_payload["condition_type"])[idx_map["test"]],
        },
    }


def export_bundle(
    cfg: HydraulicBuilderConfig,
    splits: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    channel_map: Mapping[str, Any],
    subsystem_groups: Mapping[str, Any],
    processed_root: Path,
    split_path: Path,
    metadata_root: Path,
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

        for key in ("sample_id", "cycle_id", "fault_label", "subsystem_label", "condition_type"):
            if key in payload and payload[key] is not None:
                np.save(processed_root / f"{split_name}_{key}.npy", np.asarray(payload[key], dtype=object))
                processed_files[f"{split_name}_{key}"] = f"{split_name}_{key}.npy"

    train_x = np.asarray(splits["train"]["X"])
    train_y = np.asarray(splits["train"]["Y"])
    input_dim = int(train_x.shape[-1]) if train_x.size else int(0)
    output_dim = int(train_y.shape[-1]) if train_y.size else int(0)

    bundle = HydraulicAdapter.build_bundle(
        dataset_name=cfg.dataset_name,
        train={k: v for k, v in splits["train"].items()},
        val={k: v for k, v in splits["val"].items()},
        test={k: v for k, v in splits["test"].items()},
        meta={
            "dataset_name": cfg.dataset_name,
            "task_family": "hydraulic",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "window_length": int(protocol["window_length"]),
            "horizon": int(protocol["horizon"]),
            "split_protocol": protocol["split_protocol"],
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
            "extras": {
                "channel_map": channel_map,
                "subsystem_groups": subsystem_groups,
                "single_fault_protocol": protocol,
                "class_names": protocol["labeling"].get("subsystem_class_order", []),
            },
        },
        artifacts={
            "truth_file": None,
            "grouping_file": str(metadata_root / "subsystem_groups.json"),
            "protocol_file": str(metadata_root / "single_fault_protocol.json"),
            "extra": {
                "split_file": str(split_path),
                "condition_field": protocol["data_fields"].get("condition_field"),
                "subsystem_field": protocol["data_fields"].get("subsystem_field"),
                "cycle_field": protocol["data_fields"].get("cycle_field"),
                "labeling": protocol["labeling"],
            },
        },
    )

    _write_json(
        processed_root / "hydraulic_processed_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "protocol": protocol,
            "split_file": str(split_path),
            "processed_files": processed_files,
            "counts": {
                "train": int(len(splits["train"]["X"])),
                "val": int(len(splits["val"]["X"])),
                "test": int(len(splits["test"]["X"])),
            },
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
        },
    )

    _write_json(
        metadata_root / "hydraulic_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "protocol": protocol["protocol_name"],
            "single_fault_mode": "single_degraded_or_healthy",
            "filter_summary": {
                "allowed_fault_classes": protocol["labeling"].get("allowed_fault_classes", []),
                "window_length": int(protocol["window_length"]),
                "horizon": int(protocol["horizon"]),
            },
            "source_channels": channel_map["channels"],
            "subsystem_groups": subsystem_groups,
            "bundle_meta": bundle.meta.to_dict(),
        },
    )

    check_dataset_bundle(bundle)
    return bundle


def run_build(project_root: Path) -> DatasetBundle:
    metadata_root = project_root / "data" / "metadata" / "hydraulic"
    raw_root = project_root / "data" / "raw" / "hydraulic"
    interim_root = project_root / "data" / "interim" / "hydraulic"
    processed_root = project_root / "data" / "processed" / "hydraulic"
    splits_root = project_root / "data" / "splits" / "hydraulic"

    cfg = HydraulicBuilderConfig()
    cfg.validate()

    channel_map = _load_channel_map(metadata_root)
    subsystem_groups = _load_subsystem_groups(metadata_root)
    protocol = _load_single_fault_protocol(metadata_root)

    raw = load_raw(raw_root, protocol, channel_map)

    cycle_id = raw["cycle_id"]
    subsystem_label = raw["subsystem_label"]
    fault_label = raw["fault_label"]
    condition_type = raw["condition_type"]
    fault_count = np.asarray(raw["fault_count"], dtype=object)
    sample_id = raw["sample_id"]
    x_raw = raw["u"]

    classes = protocol["labeling"].get("subsystem_class_order", ["healthy", "cooler", "valve", "pump", "accumulator"])
    y_onehot = _encode_labels(subsystem_label, condition_type, fault_count, classes)

    # filter to single-fault + healthy samples.
    keep = select_single_fault_rows(
        protocol,
        cycle_id,
        subsystem_label,
        fault_label,
        condition_type,
        fault_count,
        classes,
    )

    x = x_raw[keep]
    y_onehot = y_onehot[keep]
    cycle_id = cycle_id[keep]
    fault_label = fault_label[keep]
    subsystem_label = subsystem_label[keep]
    condition_type = condition_type[keep]
    sample_id = sample_id[keep]

    # save interim for traceability
    interim_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        interim_root / "hydraulic_preprocessed.npz",
        X=x,
        Y=y_onehot,
        cycle_id=cycle_id,
        fault_label=fault_label,
        subsystem_label=subsystem_label,
        condition_type=condition_type,
        sample_id=sample_id,
        window_length=np.array([protocol["window_length"]], dtype=np.int64),
        horizon=np.array([protocol["horizon"]], dtype=np.int64),
    )
    _write_json(
        interim_root / "hydraulic_interim_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "raw_files_checked": cfg.raw_file_candidates,
            "kept_samples": int(len(x)),
            "raw_samples": int(len(x_raw)),
            "filter_kept_ratio": float(len(x)) / max(1, len(x_raw)),
            "filter_rule": protocol["single_fault_filter"],
            "labeling": protocol["labeling"],
            "window_length": int(protocol["window_length"]),
            "horizon": int(protocol["horizon"]),
            "channels": channel_map["channels"],
            "n_channels": len(channel_map["channels"]),
        },
    )

    windows = build_windows(
        x=x,
        y_onehot=y_onehot,
        cycle_id=cycle_id,
        fault_label=fault_label,
        subsystem_label=subsystem_label,
        condition_type=condition_type,
        sample_id=sample_id,
        window_length=int(protocol["window_length"]),
        horizon=int(protocol["horizon"]),
    )

    splits_path = splits_root / "hydraulic_split_manifest.json"
    split_payload = build_split(windows, protocol, splits_path)

    return export_bundle(
        cfg=cfg,
        splits={"train": split_payload["train"], "val": split_payload["val"], "test": split_payload["test"]},
        protocol=protocol,
        channel_map=channel_map,
        subsystem_groups=subsystem_groups,
        processed_root=processed_root,
        split_path=splits_path,
        metadata_root=metadata_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hydraulic dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_build(args.project_root.expanduser().resolve())


if __name__ == "__main__":
    main()
