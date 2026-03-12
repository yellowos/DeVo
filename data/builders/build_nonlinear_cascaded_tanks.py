"""Build Cascaded Tanks dataset bundle for nonlinear benchmark.

Scope:
- raw -> interim -> processed
- produce unified DatasetBundle with train/val/test/meta/artifacts
- no model, no training, no experiment runner logic
"""

from __future__ import annotations

import argparse
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
        "cascaded_tanks.mat",
        "cascaded_tanks_raw.npz",
        "cascaded_tanks_processed.npz",
        "cascaded_tanks.csv",
        "cascaded_tanks.txt",
    )
    interim_prefix: str = "cascaded_tanks"

    def validate(self) -> None:
        if self.window_length <= 0:
            raise CascadedTanksBuilderError("window_length must be a positive integer.")
        if self.horizon <= 0:
            raise CascadedTanksBuilderError("horizon must be a positive integer.")


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


def _to_optional_1d(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return np.array([], dtype=object)
    return arr.reshape(-1)


def _pick_array_key(payload: Mapping[str, Any], keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        arr = payload.get(key) if isinstance(payload, Mapping) else None
        if arr is not None and isinstance(arr, np.ndarray):
            return arr
    return None


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = raw_root / name
        if candidate.exists():
            return candidate
    for path in raw_root.glob("**/*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".mat", ".npz", ".npy", ".csv", ".txt"} and "cascaded" in path.name.lower() and "tanks" in path.name.lower():
            return path
    return None


def load_raw(raw_root: Path, cfg: CascadedTanksBuilderConfig) -> Dict[str, Any]:
    raw_root = raw_root.expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)

    raw_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if raw_path is None:
        raise CascadedTanksBuilderError(
            f"No Cascaded Tanks raw file found in {raw_root}. Checked candidates {list(cfg.raw_file_candidates)}."
        )

    u = y = None
    sample_id = None
    run_id = None
    timestamp = None

    suffix = raw_path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(raw_path, allow_pickle=True)
        u = _pick_array_key(payload, ("u", "u_in", "input"))
        y = _pick_array_key(payload, ("y", "output", "y_out"))
        sample_id = payload.get("sample_id")
        run_id = payload.get("run_id")
        timestamp = payload.get("timestamp")
    elif suffix == ".npy":
        arr = np.load(raw_path)
        if arr.ndim < 2 or arr.shape[1] < 2:
            raise CascadedTanksBuilderError(
                f"{raw_path} must be at least shape [T,2] for input/output fields."
            )
        if arr.shape[1] == 2:
            u = arr[:, :1]
            y = arr[:, 1:2]
        else:
            split = max(1, arr.shape[1] // 2)
            u = arr[:, :split]
            y = arr[:, split:]
    elif suffix in {".csv", ".txt"}:
        import csv

        with raw_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise CascadedTanksBuilderError(f"{raw_path} has no rows.")
        header = rows[0].keys()
        u_key = next((k for k in ("u", "input", "u_in") if k in header), None)
        y_key = next((k for k in ("y", "output", "y_out") if k in header), None)
        if u_key is None or y_key is None:
            raise CascadedTanksBuilderError(
                f"{raw_path} must include both input and output columns."
            )
        u = np.asarray([float(r[u_key]) for r in rows], dtype=float).reshape(-1, 1)
        y = np.asarray([float(r[y_key]) for r in rows], dtype=float).reshape(-1, 1)

        if "sample_id" in header:
            sample_id = np.asarray([r["sample_id"] for r in rows], dtype=object)
        if "run_id" in header or "trajectory_id" in header:
            key = "run_id" if "run_id" in header else "trajectory_id"
            run_id = np.asarray([r[key] for r in rows], dtype=object)
        if "timestamp" in header or "time" in header or "t" in header:
            t_key = "timestamp" if "timestamp" in header else ("time" if "time" in header else "t")
            timestamp = np.asarray([float(r[t_key]) for r in rows], dtype=float)
    elif suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise CascadedTanksBuilderError("Cannot read .mat without scipy.") from exc

        for key in ("u", "input", "u_in", "U"):
            if key in payload and isinstance(payload[key], np.ndarray):
                u = payload[key]
                break
        for key in ("y", "output", "y_out", "Y"):
            if key in payload and isinstance(payload[key], np.ndarray):
                y = payload[key]
                break
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
        raise CascadedTanksBuilderError(f"Unsupported raw format: {suffix}")

    if u is None or y is None:
        raise CascadedTanksBuilderError(f"Raw file missing input/output in {raw_path}.")

    u_arr = _to_float2d(u, "u")
    y_arr = _to_float2d(y, "y")
    n = min(len(u_arr), len(y_arr))
    u_arr = u_arr[:n]
    y_arr = y_arr[:n]

    if sample_id is not None and len(sample_id) >= n:
        sample_id = np.asarray(sample_id).reshape(-1)[:n]
    if run_id is not None and len(run_id) >= n:
        run_id = np.asarray(run_id).reshape(-1)[:n]
    if timestamp is not None and len(timestamp) >= n:
        timestamp = np.asarray(timestamp).reshape(-1)[:n]

    mask = np.isfinite(u_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
    u_arr = u_arr[mask]
    y_arr = y_arr[mask]
    if sample_id is not None and len(sample_id) == len(mask):
        sample_id = sample_id[mask]
    if run_id is not None and len(run_id) == len(mask):
        run_id = run_id[mask]
    if timestamp is not None and len(timestamp) == len(mask):
        timestamp = timestamp[mask]

    if len(u_arr) == 0:
        raise CascadedTanksBuilderError(f"No valid rows after cleaning in {raw_path}.")

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": _to_optional_1d(sample_id),
        "run_id": _to_optional_1d(run_id),
        "timestamp": _to_optional_1d(timestamp),
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
        x_list.append(u[idx : idx + window_length])
        y_list.append(y[idx + window_length : idx + window_length + horizon])
        sample_ids.append(sample_id[idx + window_length - 1] if sample_id is not None and len(sample_id) == n else idx)
        run_ids.append(run_id[idx + window_length - 1] if run_id is not None and len(run_id) == n else None)
        timestamps.append(timestamp[idx + window_length - 1] if timestamp is not None and len(timestamp) == n else None)

    return {
        "X": np.asarray(x_list, dtype=np.float64),
        "Y": np.asarray(y_list, dtype=np.float64),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "run_id": np.asarray(run_ids, dtype=object),
        "timestamp": np.asarray(timestamps, dtype=object),
    }


def _build_split_indices(window_count: int, run_ids: np.ndarray, protocol: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    split_cfg = protocol.get("split", {})
    tr = float(split_cfg.get("train", 0.70))
    va = float(split_cfg.get("val", 0.15))
    te = float(split_cfg.get("test", 0.15))
    total = tr + va + te
    if total <= 0:
        raise CascadedTanksBuilderError("Invalid split ratios in protocol.")
    if abs(total - 1.0) > 1e-9:
        tr, va, te = tr / total, va / total, te / total

    grouped = protocol.get("grouping", {}).get("level") == "trajectory"
    if grouped and run_ids is not None and len(run_ids) == window_count:
        has_id = np.array([rid is not None for rid in run_ids], dtype=bool)
        if has_id.any():
            valid_idx = np.where(has_id)[0]
            groups: List[List[int]] = []
            current = run_ids[valid_idx[0]]
            bucket: List[int] = []
            for i in valid_idx:
                rid = run_ids[i]
                if rid != current:
                    groups.append(bucket)
                    bucket = [int(i)]
                    current = rid
                else:
                    bucket.append(int(i))
            groups.append(bucket)

            g_n = len(groups)
            n_tr = max(1, int(g_n * tr))
            n_va = max(0, int(g_n * va))
            n_te = max(0, g_n - n_tr - n_va)
            if n_tr + n_va + n_te > g_n:
                n_te = g_n - n_tr - n_va

            split_indices = {
                "train": np.array([j for g in groups[:n_tr] for j in g], dtype=int),
                "val": np.array([j for g in groups[n_tr : n_tr + n_va] for j in g], dtype=int),
                "test": np.array([j for g in groups[n_tr + n_va : n_tr + n_va + n_te] for j in g], dtype=int),
            }
            # Ensure stable deterministic order.
            for k in split_indices:
                split_indices[k] = np.array(sorted(set(int(v) for v in split_indices[k])), dtype=int)
            if len(split_indices["test"]) == 0 and len(split_indices["val"]) > 0:
                split_indices["test"] = split_indices["val"][-1:].copy()
                split_indices["val"] = split_indices["val"][:-1]
            return split_indices

    # fallback contiguous split
    train_end = int(window_count * tr)
    val_end = train_end + int(window_count * va)
    return {
        "train": np.arange(0, max(1, train_end)),
        "val": np.arange(max(1, train_end), min(window_count, max(1, val_end))),
        "test": np.arange(min(window_count, max(1, val_end)), window_count),
    }


def build_split(
    window_payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_path: Path,
) -> Dict[str, Dict[str, Any]]:
    x = np.asarray(window_payload["X"])
    y = np.asarray(window_payload["Y"])
    run_ids = np.asarray(window_payload.get("run_id", np.array([], dtype=object)), dtype=object)

    split_indices = _build_split_indices(len(x), run_ids, protocol)

    split_payload = {
        "dataset_name": "cascaded_tanks",
        "protocol_name": protocol.get("protocol_name"),
        "grouping": protocol.get("grouping", {}),
        "windowed_samples": int(len(x)),
        "split_indices": {k: v.tolist() for k, v in split_indices.items()},
        "counts": {k: int(len(v)) for k, v in split_indices.items()},
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(split_path, split_payload)

    sample_ids = np.asarray(window_payload.get("sample_id", np.array([], dtype=object)), dtype=object)
    timestamps = np.asarray(window_payload.get("timestamp", np.array([], dtype=object)), dtype=object)

    return {
        "train": {
            "X": x[split_indices["train"]],
            "Y": y[split_indices["train"]],
            "sample_id": sample_ids[split_indices["train"]] if sample_ids.size else None,
            "run_id": run_ids[split_indices["train"]] if run_ids.size else None,
            "timestamp": timestamps[split_indices["train"]] if timestamps.size else None,
        },
        "val": {
            "X": x[split_indices["val"]],
            "Y": y[split_indices["val"]],
            "sample_id": sample_ids[split_indices["val"]] if sample_ids.size else None,
            "run_id": run_ids[split_indices["val"]] if run_ids.size else None,
            "timestamp": timestamps[split_indices["val"]] if timestamps.size else None,
        },
        "test": {
            "X": x[split_indices["test"]],
            "Y": y[split_indices["test"]],
            "sample_id": sample_ids[split_indices["test"]] if sample_ids.size else None,
            "run_id": run_ids[split_indices["test"]] if run_ids.size else None,
            "timestamp": timestamps[split_indices["test"]] if timestamps.size else None,
        },
    }


def _find_protocol_root(start: Path, protocol_name: str) -> Path:
    for candidate in (start, *start.parents):
        p = candidate / "data" / "metadata" / "nonlinear" / "protocols" / f"{protocol_name}.json"
        if p.exists():
            return p
    # fallback local fallback
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


def _register_truth_reference(metadata_root: Path, dataset_name: str, truth_type: str, manifest_reference: Optional[str]) -> Dict[str, str]:
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
            np.save(processed_root / f"{split_name}_timestamp.npy", np.asarray(payload["timestamp"], dtype=object))
            processed_files[f"{split_name}_timestamp"] = f"{split_name}_timestamp.npy"

    train_x = np.asarray(splits["train"]["X"])
    train_y = np.asarray(splits["train"]["Y"])
    input_dim = int(train_x.shape[-1]) if train_x.size else int(0)
    output_dim = int(train_y.shape[-1]) if train_y.size else int(0)

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
    def load_splits(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cfg = CascadedTanksBuilderConfig()
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
                "raw_root": str(self.context.raw_root),
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "raw_file_candidates": list(cfg.raw_file_candidates),
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
    # keep calls explicit for this dataset, though values are usually null/false.
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
    )


if __name__ == "__main__":
    main()
