"""Build Tennessee Eastman Process (TEP) dataset bundle.

Scope:
- data layer only: raw -> interim -> processed
- unified dataset bundle for TEP adapter
- mode-holdout split + five-unit/fault truth metadata protocol

No model, training, metric, or experiment-runner logic is implemented here.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.adapters.base import DataProtocolError, DatasetBundle
from data.adapters.tep_adapter import TEPAdapter
from data.checks.bundle_checks import check_dataset_bundle


class TEPBuilderError(DataProtocolError):
    """Raised when TEP build cannot satisfy protocol constraints."""


@dataclass(frozen=True)
class TEPBuilderConfig:
    dataset_name: str = "tep"
    raw_file_candidates: Tuple[str, ...] = (
        "tep.mat",
        "tep_raw.npz",
        "tep_processed.npz",
        "tep.csv",
        "tep.txt",
    )
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "tep_mode_holdout_v1"

    def validate(self) -> None:
        if self.window_length <= 0:
            raise TEPBuilderError("window_length must be > 0")
        if self.horizon <= 0:
            raise TEPBuilderError("horizon must be > 0")


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
        raise TEPBuilderError(f"{name} must be a 2D numeric array.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(np.float64)
        except Exception as exc:
            raise TEPBuilderError(f"{name} must be numeric.") from exc
    return np.asarray(arr, dtype=np.float64)


def _to_optional_1d(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value).reshape(-1)


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = raw_root / name
        if candidate.exists():
            return candidate
    for path in raw_root.glob("**/*"):
        if path.is_file() and path.suffix.lower() in {".mat", ".npz", ".npy", ".csv", ".txt"}:
            if "tep" in path.name.lower():
                return path
    return None


def _safe_timestamp_value(value: Any, fallback: int) -> float:
    try:
        v = float(value)
        return float(v) if np.isfinite(v) else float(fallback)
    except Exception:
        return float(fallback)


def _safe_str_array(value: Any, default: Any, n: int) -> Optional[np.ndarray]:
    if value is None:
        return None if default is None else np.array([default] * n, dtype=object)
    arr = np.asarray(value).reshape(-1)
    if arr.size != n:
        return None if default is None else np.array([default] * n, dtype=object)
    return arr.astype(str, copy=False)
    return np.asarray(value).reshape(-1).astype(str)


def _safe_num_array(value: Any, default_len: int) -> np.ndarray:
    if value is None:
        return np.arange(default_len, dtype=np.float64)
    arr = np.asarray(value).reshape(-1)
    if arr.size != default_len:
        return np.arange(default_len, dtype=np.float64)
    return arr.astype(np.float64, copy=False)


def load_raw(raw_root: Path, cfg: TEPBuilderConfig, channel_map: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    raw_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if raw_path is None:
        raise TEPBuilderError(
            f"No TEP raw file found in {raw_root}. Checked candidates: {list(cfg.raw_file_candidates)}."
        )

    obs_names = list(channel_map["observable_variables"]["names"])
    idv_names = list(channel_map["idv_variables"]["names"])
    all_names = obs_names + idv_names

    suffix = raw_path.suffix.lower()
    if suffix in {".npz", ".npy"}:
        payload = np.load(raw_path, allow_pickle=True) if suffix == ".npz" else None
        raw_matrix = None
        mode = fault_id = scenario_id = run_id = timestamp = None

        if suffix == ".npz":
            raw_matrix = payload.get("X") if payload is not None else None
            if raw_matrix is None:
                for key in ("signals", "data", "values", "raw", "arr"):
                    if key in payload:
                        raw_matrix = payload[key]
                        break
            mode = payload.get("mode") if payload is not None else None
            fault_id = payload.get("fault_id") if payload is not None else None
            scenario_id = payload.get("scenario_id") if payload is not None else None
            run_id = payload.get("run_id") if payload is not None else None
            timestamp = payload.get("timestamp") if payload is not None else None
            if fault_id is None and payload is not None and "label" in payload:
                fault_id = payload["label"]
        else:
            raw_matrix = np.load(raw_path)

    elif suffix in {".csv", ".txt"}:
        with raw_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise TEPBuilderError(f"{raw_path} has no rows.")

        header = rows[0].keys()
        missing = [c for c in all_names if c not in header]
        if missing:
            raise TEPBuilderError(f"Missing expected TEP signal columns: {missing}")

        raw_matrix = np.asarray([[float(r[name]) for name in all_names] for r in rows], dtype=float)
        mode = _safe_str_array(np.asarray([r.get("mode") for r in rows]), "UNKNOWN", len(rows))
        fault_id = _safe_str_array(np.asarray([r.get("fault_id") for r in rows]), "healthy", len(rows))
        scenario_id = _safe_str_array(np.asarray([r.get("scenario_id") for r in rows]), "N/A", len(rows))
        run_id = _safe_str_array(np.asarray([r.get("run_id") for r in rows]), default=None, n=len(rows))
        if run_id is None:
            run_id = np.array([str(i) for i in range(len(rows))], dtype=object)
        else:
            run_id = run_id.astype(object)
        raw_t = []
        for r in rows:
            ts = r.get("timestamp")
            if ts is None or ts == "":
                raw_t.append(np.nan)
            else:
                raw_t.append(ts)
        timestamp = np.asarray(raw_t)

    elif suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise TEPBuilderError("Cannot read .mat file without scipy.") from exc

        raw_matrix = None
        for key in ("signals", "X", "data", "raw"):
            if key in payload and isinstance(payload[key], np.ndarray):
                raw_matrix = payload[key]
                break

        if raw_matrix is None:
            for key, val in payload.items():
                if key.startswith("__"):
                    continue
                if isinstance(val, np.ndarray) and np.asarray(val).ndim == 2 and np.asarray(val).shape[1] >= 81:
                    raw_matrix = val
                    break
        if raw_matrix is None:
            raise TEPBuilderError(f"No usable (T,>=81) signal array in {raw_path}.")

        mode = payload.get("mode") if isinstance(payload.get("mode"), np.ndarray) else payload.get("run_mode")
        fault_id = payload.get("fault_id")
        scenario_id = payload.get("scenario_id")
        run_id = payload.get("run_id")
        timestamp = payload.get("timestamp")
        if fault_id is None and "label" in payload:
            fault_id = payload["label"]
    else:
        raise TEPBuilderError(f"Unsupported raw format: {suffix}")

    signals = _to_float2d(raw_matrix, "tep signals")
    if signals.shape[1] < 81:
        raise TEPBuilderError(f"TEP raw signal matrix must contain at least 81 columns, got {signals.shape[1]}.")

    n = len(signals)
    x_obs = signals[:, :53]
    x_idv = signals[:, 53:81]

    mode = _safe_str_array(mode, "UNKNOWN", n)
    fault_id = _safe_str_array(fault_id, "healthy", n)
    scenario_id = _safe_str_array(scenario_id, "N/A", n)
    run_id = _safe_str_array(run_id, default=None, n=n)
    if run_id is None or run_id.size != n:
        run_id = np.array([str(i) for i in range(n)], dtype=object)
    timestamp = _to_optional_1d(timestamp)
    timestamp = _safe_num_array(timestamp, n)

    return {
        "X_obs": x_obs,
        "X_idv": x_idv,
        "mode": mode,
        "fault_id": fault_id,
        "scenario_id": scenario_id,
        "run_id": run_id,
        "timestamp": timestamp,
        "obs_names": np.asarray(obs_names, dtype=object),
        "idv_names": np.asarray(idv_names, dtype=object),
    }


def build_windows(
    x_obs: np.ndarray,
    x_idv: np.ndarray,
    *,
    window_length: int,
    horizon: int,
    mode: np.ndarray,
    fault_id: np.ndarray,
    scenario_id: np.ndarray,
    run_id: np.ndarray,
    timestamp: np.ndarray,
) -> Dict[str, Any]:
    n = len(x_obs)
    if n <= window_length + horizon:
        raise TEPBuilderError(
            f"Not enough steps for windowing: n={n}, window_length={window_length}, horizon={horizon}."
        )

    X_list = []
    Y_list = []
    mode_list = []
    fault_list = []
    scenario_list = []
    run_list = []
    timestamp_list = []
    window_idx = []
    idv_list = []

    for start in range(0, n - window_length - horizon + 1):
        end = start + window_length
        t = end - 1
        X_list.append(x_obs[start:end])
        Y_list.append(x_obs[t + 1 : t + 1 + horizon])
        idv_list.append(x_idv[t])
        mode_list.append(mode[t])
        fault_list.append(fault_id[t])
        scenario_list.append(scenario_id[t])
        run_list.append(run_id[t])
        timestamp_list.append(_safe_timestamp_value(timestamp[t], t))
        window_idx.append(start)

    return {
        "X": np.asarray(X_list, dtype=np.float64),
        "Y": np.asarray(Y_list, dtype=np.float64),
        "idv_aux": np.asarray(idv_list, dtype=np.float64),
        "mode": np.asarray(mode_list, dtype=object),
        "fault_id": np.asarray(fault_list, dtype=object),
        "scenario_id": np.asarray(scenario_list, dtype=object),
        "run_id": np.asarray(run_list, dtype=object),
        "timestamp": np.asarray(timestamp_list, dtype=np.float64),
        "window_idx": np.asarray(window_idx, dtype=np.int64),
    }


def _to_index_set(values: np.ndarray, allowed: Sequence[str]) -> np.ndarray:
    if not allowed:
        return np.array([], dtype=int)
    allowed_set = {str(v).strip() for v in allowed}
    mask = np.array([str(v).strip() in allowed_set for v in values], dtype=bool)
    return np.where(mask)[0]


def _normalize_mode(value: Any) -> str:
    return str(value).strip()


def _split_indices_by_mode(
    payload: Mapping[str, np.ndarray],
    protocol: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    modes = payload["mode"]
    n = len(modes)
    if n == 0:
        raise TEPBuilderError("Cannot split empty payload.")

    protocol_modes = protocol["modes"]
    train_idx = _to_index_set(modes, protocol_modes.get("train_modes", []))
    val_idx = _to_index_set(modes, protocol_modes.get("val_modes", []))

    if train_idx.size == 0 and val_idx.size == 0:
        split_cfg = protocol.get("split", {})
        tr = float(split_cfg.get("train", 0.70))
        va = float(split_cfg.get("val", 0.15))
        te = float(split_cfg.get("test", 0.15))
        s = tr + va + te
        if s <= 0:
            tr, va, te = 0.70, 0.15, 0.15
        else:
            tr, va, te = tr / s, va / s, te / s
        n_train = max(1, int(n * tr))
        n_val = max(0, int(n * va))
        return {
            "train": np.arange(0, n_train, dtype=int),
            "val": np.arange(n_train, min(n, n_train + n_val), dtype=int),
            "test": np.arange(min(n, n_train + n_val), n, dtype=int),
            "fallback": True,
        }

    all_fault = []
    if protocol_modes.get("test_fault_mode_pattern", "").lower().find("fault") >= 0:
        all_fault = [m for m in modes if "FAULT" in _normalize_mode(m).upper()]
    fault_idx = np.array([i for i, m in enumerate(modes) if str(m) in set(protocol_modes.get("test_fault_modes", []))], dtype=int)
    if all_fault:
        fault_mask = np.array(["FAULT" in _normalize_mode(m).upper() for m in modes], dtype=bool)
        fault_idx = np.where(fault_mask)[0]
    normal_modes = protocol_modes.get("test_normal_modes", [])
    normal_idx = _to_index_set(modes, normal_modes)

    test_idx = np.union1d(fault_idx, normal_idx)

    if test_idx.size == 0:
        test_idx = np.setdiff1d(np.arange(n), np.union1d(train_idx, val_idx))

    if train_idx.size == 0:
        train_idx = np.array([0], dtype=int)
    if val_idx.size == 0:
        # keep at least one val if possible
        cand = np.setdiff1d(np.arange(n), train_idx)
        if cand.size >= 1:
            val_idx = cand[: min(1, cand.size)]

    if train_idx.size > 0 and val_idx.size > 0 and test_idx.size > 0:
        return {
            "train": np.array(sorted(np.unique(train_idx)), dtype=int),
            "val": np.array(sorted(np.unique(val_idx)), dtype=int),
            "test": np.array(sorted(np.unique(test_idx)), dtype=int),
            "fallback": False,
        }

    # final hard safeguard
    n_train = max(1, int(0.7 * n))
    n_val = max(0, int(0.15 * n))
    return {
        "train": np.arange(0, n_train, dtype=int),
        "val": np.arange(n_train, min(n, n_train + n_val), dtype=int),
        "test": np.arange(min(n, n_train + n_val), n, dtype=int),
        "fallback": True,
    }


def build_split(window_payload: Mapping[str, Any], protocol: Mapping[str, Any], split_path: Path) -> Dict[str, Any]:
    idx = _split_indices_by_mode(window_payload, protocol)

    split_payload = {
        "dataset_name": "tep",
        "protocol_name": protocol.get("protocol_name", "tep_mode_holdout_v1"),
        "split_protocol": protocol.get("split_protocol", "tep_mode_holdout_v1"),
        "mode_protocol": protocol.get("modes", {}),
        "split_indices": {
            k: v.tolist() for k, v in {"train": idx["train"], "val": idx["val"], "test": idx["test"]}.items()
        },
        "counts": {
            "train": int(len(idx["train"])),
            "val": int(len(idx["val"])),
            "test": int(len(idx["test"])),
        },
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "fallback": bool(idx.get("fallback", False)),
    }
    _write_json(split_path, split_payload)

    X = np.asarray(window_payload["X"], dtype=np.float64)
    Y = np.asarray(window_payload["Y"], dtype=np.float64)

    return {
        "train": {
            "X": X[idx["train"]],
            "Y": Y[idx["train"]],
            "sample_id": np.asarray(window_payload["window_idx"], dtype=np.int64)[idx["train"]],
            "run_id": np.asarray(window_payload["run_id"], dtype=object)[idx["train"]],
            "timestamp": np.asarray(window_payload["timestamp"], dtype=np.float64)[idx["train"]],
            "mode": np.asarray(window_payload["mode"], dtype=object)[idx["train"]],
            "fault_id": np.asarray(window_payload["fault_id"], dtype=object)[idx["train"]],
            "scenario_id": np.asarray(window_payload["scenario_id"], dtype=object)[idx["train"]],
            "window_idx": np.asarray(window_payload["window_idx"], dtype=np.int64)[idx["train"]],
            "idv_aux": np.asarray(window_payload["idv_aux"], dtype=np.float64)[idx["train"]],
            "meta_fields": {
                "obs_names": window_payload.get("obs_names"),
                "idv_names": window_payload.get("idv_names"),
            },
        },
        "val": {
            "X": X[idx["val"]],
            "Y": Y[idx["val"]],
            "sample_id": np.asarray(window_payload["window_idx"], dtype=np.int64)[idx["val"]],
            "run_id": np.asarray(window_payload["run_id"], dtype=object)[idx["val"]],
            "timestamp": np.asarray(window_payload["timestamp"], dtype=np.float64)[idx["val"]],
            "mode": np.asarray(window_payload["mode"], dtype=object)[idx["val"]],
            "fault_id": np.asarray(window_payload["fault_id"], dtype=object)[idx["val"]],
            "scenario_id": np.asarray(window_payload["scenario_id"], dtype=object)[idx["val"]],
            "window_idx": np.asarray(window_payload["window_idx"], dtype=np.int64)[idx["val"]],
            "idv_aux": np.asarray(window_payload["idv_aux"], dtype=np.float64)[idx["val"]],
            "meta_fields": {
                "obs_names": window_payload.get("obs_names"),
                "idv_names": window_payload.get("idv_names"),
            },
        },
        "test": {
            "X": X[idx["test"]],
            "Y": Y[idx["test"]],
            "sample_id": np.asarray(window_payload["window_idx"], dtype=np.int64)[idx["test"]],
            "run_id": np.asarray(window_payload["run_id"], dtype=object)[idx["test"]],
            "timestamp": np.asarray(window_payload["timestamp"], dtype=np.float64)[idx["test"]],
            "mode": np.asarray(window_payload["mode"], dtype=object)[idx["test"]],
            "fault_id": np.asarray(window_payload["fault_id"], dtype=object)[idx["test"]],
            "scenario_id": np.asarray(window_payload["scenario_id"], dtype=object)[idx["test"]],
            "window_idx": np.asarray(window_payload["window_idx"], dtype=np.int64)[idx["test"]],
            "idv_aux": np.asarray(window_payload["idv_aux"], dtype=np.float64)[idx["test"]],
            "meta_fields": {
                "obs_names": window_payload.get("obs_names"),
                "idv_names": window_payload.get("idv_names"),
            },
        },
    }


def export_bundle(
    cfg: TEPBuilderConfig,
    split_payload: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    five_unit: Mapping[str, Any],
    truth_table: Mapping[str, Any],
    channel_map: Mapping[str, Any],
    processed_root: Path,
    split_path: Path,
    metadata_root: Path,
) -> DatasetBundle:
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_files: Dict[str, str] = {}
    for split_name, payload in split_payload.items():
        for key in ("X", "Y", "sample_id", "run_id", "timestamp", "mode", "fault_id", "scenario_id", "window_idx"):
            if key not in payload or payload[key] is None:
                continue
            arr = np.asarray(payload[key])
            if arr.size == 0:
                continue
            fname = f"{split_name}_{key}.npy"
            np.save(processed_root / fname, arr)
            processed_files[f"{split_name}_{key}"] = fname

        idv_aux = np.asarray(payload.get("idv_aux"), dtype=np.float64)
        if idv_aux.size > 0:
            fname = f"{split_name}_idv_aux.npy"
            np.save(processed_root / fname, idv_aux)
            processed_files[f"{split_name}_idv_aux"] = fname

    train_x = np.asarray(split_payload["train"]["X"])
    train_y = np.asarray(split_payload["train"]["Y"])
    if train_x.size == 0:
        raise TEPBuilderError("train split is empty.")

    bundle = TEPAdapter.build_bundle(
        dataset_name=cfg.dataset_name,
        train=split_payload["train"],
        val=split_payload["val"],
        test=split_payload["test"],
        meta={
            "dataset_name": cfg.dataset_name,
            "task_family": "tep",
            "input_dim": int(train_x.shape[-1]),
            "output_dim": int(train_y.shape[-1]),
            "window_length": int(cfg.window_length),
            "horizon": int(cfg.horizon),
            "split_protocol": cfg.split_protocol,
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
            "extras": {
                "channel_map": channel_map,
                "five_unit_definition": five_unit,
                "fault_truth_table": truth_table,
                "mode_holdout_protocol": protocol,
                "feed_side_unit_not_scored": True,
            },
        },
        artifacts={
            "truth_file": None,
            "grouping_file": str(metadata_root / "five_unit_definition.json"),
            "protocol_file": str(metadata_root / "mode_holdout_protocol.json"),
            "extra": {
                "fault_truth_table_file": str(metadata_root / "fault_truth_table.json"),
                "split_file": str(split_path),
            },
        },
    )

    _write_json(
        processed_root / "tep_processed_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "protocol": protocol,
            "split_file": str(split_path),
            "window_length": int(cfg.window_length),
            "horizon": int(cfg.horizon),
            "sample_counts": {
                "train": int(len(split_payload["train"]["X"])),
                "val": int(len(split_payload["val"]["X"])),
                "test": int(len(split_payload["test"]["X"])),
            },
            "processed_files": processed_files,
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
        },
    )

    _write_json(
        metadata_root / "tep_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "mode_holdout_protocol_file": str(metadata_root / "mode_holdout_protocol.json"),
            "five_unit_definition_file": str(metadata_root / "five_unit_definition.json"),
            "fault_truth_table_file": str(metadata_root / "fault_truth_table.json"),
            "channel_map_file": str(metadata_root / "channel_map.json"),
            "split_file": str(split_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "protocol_name": protocol.get("protocol_name"),
        },
    )

    check_dataset_bundle(bundle)
    return bundle


def run_build(project_root: Path) -> DatasetBundle:
    cfg = TEPBuilderConfig()
    cfg.validate()

    metadata_root = project_root / "data" / "metadata" / "tep"
    raw_root = project_root / "data" / "raw" / "tep"
    interim_root = project_root / "data" / "interim" / "tep"
    processed_root = project_root / "data" / "processed" / "tep"
    splits_root = project_root / "data" / "splits" / "tep"

    channel_map = _load_json(metadata_root / "channel_map.json")
    five_unit = _load_json(metadata_root / "five_unit_definition.json")
    truth_table = _load_json(metadata_root / "fault_truth_table.json")
    protocol = _load_json(metadata_root / "mode_holdout_protocol.json")

    total_vars = int(channel_map["observable_variables"].get("count", 0)) + int(
        channel_map["idv_variables"].get("count", 0)
    )
    if total_vars != 81:
        raise TEPBuilderError("channel_map must define exactly 81 variables (v01-v81).")

    raw = load_raw(raw_root, cfg, channel_map)

    interim_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        interim_root / "tep_preprocessed.npz",
        X_obs=np.asarray(raw["X_obs"], dtype=np.float64),
        X_idv=np.asarray(raw["X_idv"], dtype=np.float64),
        mode=np.asarray(raw["mode"], dtype=object),
        fault_id=np.asarray(raw["fault_id"], dtype=object),
        scenario_id=np.asarray(raw["scenario_id"], dtype=object),
        run_id=np.asarray(raw["run_id"], dtype=object),
        timestamp=np.asarray(raw["timestamp"], dtype=np.float64),
    )
    _write_json(
        interim_root / "tep_interim_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "raw_signal_shape": [int(raw["X_obs"].shape[0]), int(raw["X_obs"].shape[1] + raw["X_idv"].shape[1])],
            "x_obs_shape": [int(raw["X_obs"].shape[0]), int(raw["X_obs"].shape[1])],
            "x_idv_shape": [int(raw["X_idv"].shape[0]), int(raw["X_idv"].shape[1])],
            "window_length": int(cfg.window_length),
            "horizon": int(cfg.horizon),
            "mode_protocol_name": protocol.get("protocol_name"),
        },
    )

    windows = build_windows(
        x_obs=np.asarray(raw["X_obs"], dtype=np.float64),
        x_idv=np.asarray(raw["X_idv"], dtype=np.float64),
        window_length=cfg.window_length,
        horizon=cfg.horizon,
        mode=np.asarray(raw["mode"], dtype=object),
        fault_id=np.asarray(raw["fault_id"], dtype=object),
        scenario_id=np.asarray(raw["scenario_id"], dtype=object),
        run_id=np.asarray(raw["run_id"], dtype=object),
        timestamp=np.asarray(raw["timestamp"], dtype=np.float64),
    )
    windows["obs_names"] = np.asarray(raw["obs_names"], dtype=object)
    windows["idv_names"] = np.asarray(raw["idv_names"], dtype=object)

    split_path = splits_root / "tep_mode_holdout_split_manifest.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_payload = build_split(windows, protocol, split_path)

    return export_bundle(
        cfg=cfg,
        split_payload=split_payload,
        protocol=protocol,
        five_unit=five_unit,
        truth_table=truth_table,
        channel_map=channel_map,
        processed_root=processed_root,
        split_path=split_path,
        metadata_root=metadata_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TEP dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_build(args.project_root.expanduser().resolve())


if __name__ == "__main__":
    main()
