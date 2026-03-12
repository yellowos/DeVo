"""Build Coupled Duffing dataset bundle for nonlinear benchmark.

Scope:
- raw -> interim -> processed
- export unified DatasetBundle (train/val/test/meta/artifacts)
- prediction-oriented dataset with explicit dimensional declarations
- no model / training / experiment runner implementation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
        "coupled_duffing.mat",
        "coupled_duffing_raw.npz",
        "coupled_duffing_processed.npz",
        "coupled_duffing.csv",
        "coupled_duffing.txt",
    )
    interim_prefix: str = "coupled_duffing"

    def validate(self) -> None:
        if self.window_length <= 0:
            raise CoupledDuffingBuilderError("window_length must be a positive integer.")
        if self.horizon <= 0:
            raise CoupledDuffingBuilderError("horizon must be a positive integer.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        target = raw_root / name
        if target.exists():
            return target
    for found in raw_root.glob("**/*"):
        if not found.is_file():
            continue
        if found.suffix.lower() in {".mat", ".npz", ".npy", ".csv", ".txt"}:
            if "coupled" in found.name.lower() and "duffing" in found.stem.lower():
                return found
    return None


def _to_float2d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise CoupledDuffingBuilderError(f"{name} must be a 2D numeric matrix after reshape.")
    if not np.issubdtype(arr.dtype, np.number):
        raise CoupledDuffingBuilderError(f"{name} must contain numeric values.")
    return arr.astype(np.float64)


def _resolve_csv_matrix(
    rows: List[Dict[str, str]],
    *,
    u_cols: Sequence[str],
    y_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise CoupledDuffingBuilderError("CSV/TSV file has no rows.")
    header = rows[0].keys()
    u_cols = [c for c in u_cols if c in header]
    y_cols = [c for c in y_cols if c in header]
    if not u_cols:
        raise CoupledDuffingBuilderError("No input columns found (expected u-related columns).")
    if not y_cols:
        raise CoupledDuffingBuilderError("No output columns found (expected y-related columns).")
    u = np.asarray([[float(r[c]) for c in u_cols] for r in rows], dtype=float)
    y = np.asarray([[float(r[c]) for c in y_cols] for r in rows], dtype=float)
    return u, y


def load_raw(raw_root: Path, cfg: CoupledDuffingBuilderConfig) -> Dict[str, Any]:
    raw_root = raw_root.expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if raw_path is None:
        raise CoupledDuffingBuilderError(
            f"Cannot locate Coupled Duffing raw file under {raw_root}; "
            f"checked candidates {list(cfg.raw_file_candidates)}."
        )

    u = y = None
    sample_id = None
    run_id = None
    timestamp = None

    suffix = raw_path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(raw_path, allow_pickle=True)
        u = payload.get("u") or payload.get("u_in") or payload.get("input")
        y = payload.get("y") or payload.get("output") or payload.get("y_out")
        if u is None:
            raise CoupledDuffingBuilderError(f"{raw_path} missing input matrix u/u_in/input.")
        if y is None:
            raise CoupledDuffingBuilderError(f"{raw_path} missing output matrix y/output/y_out.")
        sample_id = payload.get("sample_id")
        run_id = payload.get("run_id")
        timestamp = payload.get("timestamp")
    elif suffix == ".npy":
        arr = np.load(raw_path)
        if arr.ndim < 2 or arr.shape[1] < 2:
            raise CoupledDuffingBuilderError(f"{raw_path} must be shaped [T, >=2] for coupled signals.")
        # keep full width for possible multi-input/multi-output; split 50/50 by default only when clearly unavailable.
        half = arr.shape[1] // 2
        if half == 0:
            raise CoupledDuffingBuilderError("Cannot infer input/output split from single-column data.")
        u = arr[:, :half]
        y = arr[:, half:]
    elif suffix in {".csv", ".txt"}:
        import csv

        with raw_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        header = rows[0].keys() if rows else []
        # explicit coupling case: prefer named u1..uN and y1..yM
        input_columns = [c for c in header if c.lower().startswith("u")]
        output_columns = [c for c in header if c.lower().startswith("y")]
        if not input_columns or not output_columns:
            # fallback: first half / second half
            if rows:
                col_names = list(header)
                if len(col_names) < 2:
                    raise CoupledDuffingBuilderError(f"{raw_path} has less than 2 columns.")
                split_idx = len(col_names) // 2
                input_columns = col_names[:split_idx]
                output_columns = col_names[split_idx:]
            else:
                input_columns = []
                output_columns = []
        u, y = _resolve_csv_matrix(rows, u_cols=input_columns, y_cols=output_columns)
        if "sample_id" in header:
            sample_id = np.asarray([row.get("sample_id") for row in rows], dtype=object)
        if "run_id" in header:
            run_id = np.asarray([row.get("run_id") for row in rows], dtype=object)
        if "trajectory_id" in header:
            run_id = np.asarray([row.get("trajectory_id") for row in rows], dtype=object)
        if "timestamp" in header or "time" in header or "t" in header:
            tcol = "timestamp" if "timestamp" in header else ("time" if "time" in header else "t")
            timestamp = np.asarray([float(r[tcol]) for r in rows], dtype=float)
    elif suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise CoupledDuffingBuilderError("Failed to read .mat without scipy support.") from exc

        for key in ("u", "u_in", "input", "U", "X"):
            if key in payload and isinstance(payload[key], np.ndarray):
                u = payload[key]
                break
        for key in ("y", "output", "y_out", "Y"):
            if key in payload and isinstance(payload[key], np.ndarray):
                y = payload[key]
                break
        if u is None:
            raise CoupledDuffingBuilderError(f"{raw_path} missing input variable.")
        if y is None:
            raise CoupledDuffingBuilderError(f"{raw_path} missing output variable.")
        for key in ("sample_id", "sample"):
            if key in payload:
                sample_id = payload[key]
        for key in ("run_id", "trajectory_id", "traj_id"):
            if key in payload:
                run_id = payload[key]
        for key in ("timestamp", "time", "t"):
            if key in payload and isinstance(payload[key], np.ndarray):
                timestamp = payload[key]
    else:
        raise CoupledDuffingBuilderError(f"Unsupported file type: {suffix}")

    if u is None or y is None:
        raise CoupledDuffingBuilderError(f"Raw source missing u/y fields: {raw_path}")

    u_arr = _to_float2d(u, "u")
    y_arr = _to_float2d(y, "y")
    n = min(len(u_arr), len(y_arr))
    u_arr = u_arr[:n]
    y_arr = y_arr[:n]

    mask = np.isfinite(u_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
    u_arr = u_arr[mask]
    y_arr = y_arr[mask]
    if sample_id is not None and len(sample_id) == len(mask):
        sample_id = np.asarray(sample_id).reshape(-1)[mask]
    if run_id is not None and len(run_id) == len(mask):
        run_id = np.asarray(run_id).reshape(-1)[mask]
    if timestamp is not None and len(timestamp) == len(mask):
        timestamp = np.asarray(timestamp).reshape(-1)[mask]

    if len(u_arr) == 0:
        raise CoupledDuffingBuilderError(f"No valid sample after preprocessing: {raw_path}")

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": np.asarray(sample_id).reshape(-1) if sample_id is not None else None,
        "run_id": np.asarray(run_id).reshape(-1) if run_id is not None else None,
        "timestamp": np.asarray(timestamp).reshape(-1) if timestamp is not None else None,
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

    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    sample_ids: List[Any] = []
    run_ids: List[Any] = []
    timestamps: List[Any] = []
    for idx in range(0, n - window_length - horizon + 1):
        X_list.append(u[idx : idx + window_length])
        Y_list.append(y[idx + window_length : idx + window_length + horizon])
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

    return {
        "X": np.asarray(X_list, dtype=np.float64),
        "Y": np.asarray(Y_list, dtype=np.float64),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "run_id": np.asarray(run_ids, dtype=object),
        "timestamp": np.asarray(timestamps, dtype=object),
    }


def _derive_split_indices(n: int, protocol: Mapping[str, Any], run_ids: np.ndarray) -> Dict[str, np.ndarray]:
    split_cfg = protocol.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.70))
    val_ratio = float(split_cfg.get("val", 0.15))
    test_ratio = float(split_cfg.get("test", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise CoupledDuffingBuilderError("Split protocol ratios invalid.")
    if abs(total - 1.0) > 1e-8:
        train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total

    grouped = protocol.get("grouping", {}).get("level") == "trajectory"
    if grouped and run_ids is not None and len(run_ids) == n:
        has_id = np.array([rid is not None for rid in run_ids], dtype=bool)
        if has_id.any():
            valid_idx = np.where(has_id)[0]
            runs = []
            positions = []
            current = run_ids[valid_idx[0]]
            buf: List[int] = []
            for index in valid_idx:
                rid = run_ids[index]
                if rid != current:
                    runs.append(str(current))
                    positions.append(buf)
                    current = rid
                    buf = [int(index)]
                else:
                    buf.append(int(index))
            runs.append(str(current))
            positions.append(buf)

            if positions:
                k = len(positions)
                n_tr = max(1, int(k * train_ratio))
                n_va = max(0, int(k * val_ratio))
                n_te = max(0, k - n_tr - n_va)
                if n_tr == 0:
                    n_tr = 1
                split_indices = {
                    "train": np.array([j for chunk in positions[:n_tr] for j in chunk], dtype=int),
                    "val": np.array([j for chunk in positions[n_tr : n_tr + n_va] for j in chunk], dtype=int),
                    "test": np.array([j for chunk in positions[n_tr + n_va : n_tr + n_va + n_te] for j in chunk], dtype=int),
                }
                split_indices = {k2: np.array(sorted(set(map(int, v))), dtype=int) for k2, v in split_indices.items()}
                if len(split_indices["test"]) == 0 and len(split_indices["val"]) > 0:
                    split_indices["test"] = split_indices["val"][-1:]
                    split_indices["val"] = split_indices["val"][:-1]
                return split_indices

    # fallback contiguous split
    tr_end = int(n * train_ratio)
    va_end = tr_end + int(n * val_ratio)
    return {
        "train": np.arange(0, tr_end),
        "val": np.arange(tr_end, min(va_end, n)),
        "test": np.arange(min(va_end, n), n),
    }


def build_split(
    window_payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_path: Path,
) -> Dict[str, Dict[str, Any]]:
    x = np.asarray(window_payload["X"])
    y = np.asarray(window_payload["Y"])
    sample_ids = np.asarray(window_payload.get("sample_id", np.array([], dtype=object)), dtype=object)
    run_ids = np.asarray(window_payload.get("run_id", np.array([], dtype=object)), dtype=object)
    timestamps = np.asarray(window_payload.get("timestamp", np.array([], dtype=object)), dtype=object)

    split_indices = _derive_split_indices(len(x), protocol, run_ids)
    split_payload = {
        "dataset_name": "coupled_duffing",
        "protocol_name": protocol.get("protocol_name"),
        "window_count": int(len(x)),
        "indices": {
            "train": split_indices["train"].tolist(),
            "val": split_indices["val"].tolist(),
            "test": split_indices["test"].tolist(),
        },
        "counts": {
            "all": int(len(x)),
            "train": int(len(split_indices["train"])),
            "val": int(len(split_indices["val"])),
            "test": int(len(split_indices["test"])),
        },
        "protocol": protocol,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(split_path, split_payload)

    def _pick(d: np.ndarray, arr: np.ndarray) -> Optional[np.ndarray]:
        return arr[d] if arr is not None and len(arr) else None

    return {
        "train": {
            "X": x[split_indices["train"]],
            "Y": y[split_indices["train"]],
            "sample_id": _pick(split_indices["train"], sample_ids),
            "run_id": _pick(split_indices["train"], run_ids),
            "timestamp": _pick(split_indices["train"], timestamps),
        },
        "val": {
            "X": x[split_indices["val"]],
            "Y": y[split_indices["val"]],
            "sample_id": _pick(split_indices["val"], sample_ids),
            "run_id": _pick(split_indices["val"], run_ids),
            "timestamp": _pick(split_indices["val"], timestamps),
        },
        "test": {
            "X": x[split_indices["test"]],
            "Y": y[split_indices["test"]],
            "sample_id": _pick(split_indices["test"], sample_ids),
            "run_id": _pick(split_indices["test"], run_ids),
            "timestamp": _pick(split_indices["test"], timestamps),
        },
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
            np.save(processed_root / f"{split_name}_timestamp.npy", np.asarray(payload["timestamp"], dtype=object))
            processed_files[f"{split_name}_timestamp"] = f"{split_name}_timestamp.npy"

    train_x = np.asarray(splits["train"]["X"])
    train_y = np.asarray(splits["train"]["Y"])
    input_dim = int(train_x.shape[-1]) if train_x.ndim >= 3 and train_x.size else int(0)
    output_dim = int(train_y.shape[-1]) if train_y.ndim >= 3 and train_y.size else int(0)

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
        },
        "counts": {
            "train": int(len(splits["train"]["X"])),
            "val": int(len(splits["val"]["X"])),
            "test": int(len(splits["test"]["X"])),
        },
        "files": processed_files,
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
            if parent.name == "data"
            or (parent.parent / "data").exists()
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
    def load_splits(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cfg = CoupledDuffingBuilderConfig()
        protocol_path = _find_protocol_root(self.context.interim_root, cfg.split_protocol)
        protocol = _load_json(protocol_path)

        raw = load_raw(self.context.raw_root, cfg)
        self.context.interim_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.context.interim_root / f"{cfg.interim_prefix}_raw.npz",
            u=raw["u"],
            y=raw["y"],
            sample_id=np.asarray(raw["sample_id"], dtype=object) if raw["sample_id"] is not None else np.array([], dtype=object),
            run_id=np.asarray(raw["run_id"], dtype=object) if raw["run_id"] is not None else np.array([], dtype=object),
            timestamp=np.asarray(raw["timestamp"], dtype=float) if raw["timestamp"] is not None else np.array([], dtype=float),
        )
        _write_json(
            self.context.interim_root / f"{cfg.interim_prefix}_interim_manifest.json",
            {
                "dataset_name": cfg.dataset_name,
                "raw_file_candidates": list(cfg.raw_file_candidates),
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
        )

        windows = build_windows(
            raw["u"],
            raw["y"],
            window_length=cfg.window_length,
            horizon=cfg.horizon,
            sample_id=raw["sample_id"],
            run_id=raw["run_id"],
            timestamp=raw["timestamp"],
        )

        split_path = self.context.splits_root / f"{cfg.dataset_name}_split_manifest.json"
        split_payload = build_split(windows, protocol, split_path)
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
    )


if __name__ == "__main__":
    main()
