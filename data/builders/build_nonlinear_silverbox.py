"""Build Silverbox dataset bundle for nonlinear benchmark.

Scope:
- only data-layer construction for Silverbox
- supports raw -> interim -> processed pipeline
- outputs unified DatasetBundle for downstream methods
- no model / training / experiment runner logic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from data.adapters.base import DataProtocolError, DatasetBundle
from data.adapters.nonlinear_adapter import NonlinearAdapter
from data.builders.nonlinear_builder import NonlinearBuilderContext, NonlinearBuilder
from data.checks.bundle_checks import check_dataset_bundle


class SilverboxBuilderError(DataProtocolError):
    """Raised when silverbox source data cannot satisfy required schema."""


@dataclass(frozen=True)
class SilverboxBuilderConfig:
    """Silverbox builder configuration."""

    dataset_name: str = "silverbox"
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "nonlinear_temporal_grouped_holdout_v1"
    raw_file_candidates: Tuple[str, ...] = (
        "silverbox.mat",
        "silverbox_raw.npz",
        "silverbox.csv",
        "silverbox.txt",
        "silverbox_processed.npz",
        "SNLS80mV.mat",
        "Schroeder80mV.mat",
    )

    def validate(self) -> None:
        if self.window_length <= 0:
            raise SilverboxBuilderError("window_length must be a positive integer.")
        if self.horizon <= 0:
            raise SilverboxBuilderError("horizon must be a positive integer.")


def _to_float2d(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise SilverboxBuilderError(f"{name} must be 1D or 2D numeric array.")
    if not np.issubdtype(arr.dtype, np.number):
        raise SilverboxBuilderError(f"{name} must be numeric.")
    return arr.astype(np.float64)


def _to_optional_1d(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    return arr


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_raw_file(raw_root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        candidate = raw_root / name
        if candidate.exists():
            return candidate
    for path in raw_root.glob("**/*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".mat", ".npz", ".npy", ".csv", ".txt"}:
            name = path.name.lower()
            if any(token in name for token in {"silverbox", "silver", "snls", "schroeder"}):
                return path
    return None


def load_raw(raw_root: Path, cfg: SilverboxBuilderConfig) -> Dict[str, np.ndarray]:
    """Load Silverbox raw arrays.

    Supported:
    - .npz/.npy arrays with keys u, y
    - .csv/.txt with headers (u/y or input/output)
    - .mat with keys/fields u/u_in and y/y_out
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = _find_raw_file(raw_root, cfg.raw_file_candidates)
    if raw_path is None:
        raise SilverboxBuilderError(
            f"No Silverbox raw file found in {raw_root}. "
            "Expected candidate names include silverbox.mat / silverbox_raw.npz / silverbox.csv."
        )

    u = y = None
    sample_id = run_id = timestamp = None

    suffix = raw_path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(raw_path, allow_pickle=True)
        u = payload.get("u")
        y = payload.get("y")
        if u is None:
            u = payload.get("input") or payload.get("u_in")
        if y is None:
            y = payload.get("output") or payload.get("y_out")
        sample_id = payload.get("sample_id")
        run_id = payload.get("run_id")
        timestamp = payload.get("timestamp")
    elif suffix == ".npy":
        data = np.load(raw_path)
        if data.ndim >= 2 and data.shape[1] >= 2:
            u = data[:, :1]
            y = data[:, 1:2]
        else:
            raise SilverboxBuilderError("silverbox .npy requires at least 2 columns [u, y].")
    elif suffix in {".csv", ".txt"}:
        import csv

        with raw_path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise SilverboxBuilderError(f"{raw_path} contains no rows.")
        headers = rows[0].keys()
        u_key = next((k for k in ("u", "input", "u_in") if k in headers), None)
        y_key = next((k for k in ("y", "output", "y_out") if k in headers), None)
        if u_key is None or y_key is None:
            raise SilverboxBuilderError(f"{raw_path} must provide u and y columns.")

        u = np.array([float(r[u_key]) for r in rows], dtype=float).reshape(-1, 1)
        y = np.array([float(r[y_key]) for r in rows], dtype=float).reshape(-1, 1)

        if "sample_id" in headers:
            sample_id = np.array([r["sample_id"] for r in rows], dtype=object)
        if "run_id" in headers:
            run_id = np.array([r["run_id"] for r in rows], dtype=object)
        if "timestamp" in headers:
            timestamp = np.array([float(r["timestamp"]) for r in rows], dtype=float)
    elif suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(raw_path)
        except Exception as exc:
            raise SilverboxBuilderError("Cannot read .mat without scipy.") from exc

        for key in ("u", "input", "u_in", "V1"):
            if key in payload and isinstance(payload[key], np.ndarray):
                u = payload[key]
                break
        for key in ("y", "output", "y_out", "V2"):
            if key in payload and isinstance(payload[key], np.ndarray):
                y = payload[key]
                break
        for key in ("sample_id", "sample", "sid"):
            if key in payload:
                sample_id = payload[key]
        for key in ("run_id", "trajectory_id", "traj_id"):
            if key in payload:
                run_id = payload[key]
        for key in ("timestamp", "time", "t"):
            if key in payload:
                timestamp = payload[key]
    else:
        raise SilverboxBuilderError(f"Unsupported raw format {suffix}.")

    if u is None or y is None:
        raise SilverboxBuilderError(f"Missing u/y in raw file: {raw_path}")

    u_arr = _to_float2d(u, "u")
    y_arr = _to_float2d(y, "y")
    if u_arr.shape[0] == 1 and u_arr.shape[1] > 1:
        u_arr = u_arr.T
    if y_arr.shape[0] == 1 and y_arr.shape[1] > 1:
        y_arr = y_arr.T

    n = min(len(u_arr), len(y_arr))
    u_arr = u_arr[:n]
    y_arr = y_arr[:n]

    if len(u_arr) == 0:
        raise SilverboxBuilderError(f"Empty raw signal after alignment: {raw_path}")

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": _to_optional_1d(sample_id),
        "run_id": _to_optional_1d(run_id),
        "timestamp": _to_optional_1d(timestamp),
    }


def preprocess(raw: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    u = _to_float2d(raw["u"], "u")
    y = _to_float2d(raw["y"], "y")
    sample_id = raw.get("sample_id")
    run_id = raw.get("run_id")
    timestamp = raw.get("timestamp")

    mask = np.isfinite(u).all(axis=1) & np.isfinite(y).all(axis=1)
    u = u[mask]
    y = y[mask]
    sample_id = sample_id[mask] if sample_id is not None and len(sample_id) == len(mask) else sample_id
    run_id = run_id[mask] if run_id is not None and len(run_id) == len(mask) else run_id
    timestamp = timestamp[mask] if timestamp is not None and len(timestamp) == len(mask) else timestamp

    if timestamp is not None and len(timestamp) == len(u):
        order = np.argsort(timestamp)
        u = u[order]
        y = y[order]
        if sample_id is not None:
            sample_id = sample_id[order]
        if run_id is not None:
            run_id = run_id[order]
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
    sample_id: Optional[np.ndarray],
    run_id: Optional[np.ndarray],
    timestamp: Optional[np.ndarray],
) -> Dict[str, Any]:
    n = min(len(u), len(y))
    if n <= window_length + horizon:
        raise SilverboxBuilderError(
            f"Need n > window_length + horizon, got n={n}, window={window_length}, horizon={horizon}"
        )

    xs = []
    ys = []
    sample_ids = []
    run_ids = []
    timestamps = []

    for idx in range(0, n - window_length - horizon + 1):
        xs.append(u[idx : idx + window_length])
        ys.append(y[idx + window_length : idx + window_length + horizon])
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
        "X": np.asarray(xs, dtype=float),
        "Y": np.asarray(ys, dtype=float),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "run_id": np.asarray(run_ids, dtype=object),
        "timestamp": np.asarray(timestamps, dtype=object),
    }


def _protocol_split_indices(window_count: int, protocol: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    split = protocol.get("split", {})
    tr = float(split.get("train", 0.70))
    va = float(split.get("val", 0.15))
    te = float(split.get("test", 0.15))
    total = tr + va + te
    if total <= 0:
        raise SilverboxBuilderError("Invalid split configuration.")
    if abs(total - 1.0) > 1e-8:
        tr, va, te = tr / total, va / total, te / total

    train_end = int(window_count * tr)
    val_end = train_end + int(window_count * va)
    return {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, min(val_end, window_count)),
        "test": np.arange(min(val_end, window_count), window_count),
    }


def build_split(
    window_payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_path: Path,
) -> Dict[str, Dict[str, Any]]:
    x = np.asarray(window_payload["X"])
    y = np.asarray(window_payload["Y"])
    run_id = np.asarray(window_payload.get("run_id", []), dtype=object)

    indices = _protocol_split_indices(len(x), protocol)
    # trajectory-aware split only when run IDs are explicitly available
    can_group = (
        len(run_id) == len(x)
        and run_id.size > 0
        and "trajectory" in str(protocol.get("grouping", {}).get("level", ""))
    )
    if can_group:
        nonempty_runs = [str(r) for r in run_id if r is not None and str(r).strip() != ""]
        can_group = len(set(nonempty_runs)) > 1

    if can_group:
        unique_runs = []
        positions = []
        current_run = run_id[0]
        current_pos = []
        for i, rid in enumerate(run_id):
            if str(rid) != str(current_run):
                unique_runs.append(str(current_run))
                positions.append(current_pos)
                current_run = rid
                current_pos = [i]
            else:
                current_pos.append(i)
        unique_runs.append(str(current_run))
        positions.append(current_pos)

        ratio = protocol.get("split", {})
        tr = float(ratio.get("train", 0.70))
        va = float(ratio.get("val", 0.15))
        te = float(ratio.get("test", 0.15))
        s = tr + va + te
        if s <= 0:
            raise SilverboxBuilderError("Invalid split ratios.")
        if abs(s - 1.0) > 1e-8:
            tr, va, te = tr / s, va / s, te / s
        group_total = len(positions)
        gtr = int(group_total * tr)
        gva = int(group_total * va)
        if gtr <= 0:
            gtr = 1
        if gtr >= group_total:
            gtr = group_total - 1
        gva_start = gtr
        gva_end = gtr + gva
        if va > 0 and gva <= 0 and gva_start < group_total:
            gva_end = gva_start + 1
        if gva_end < gva_start:
            gva_end = gva_start
        if gva_end > group_total:
            gva_end = group_total
        split_run = {
            "train": positions[:gtr],
            "val": positions[gva_start:gva_end],
            "test": positions[gva_end:],
        }
        indices = {
            k: np.array([item for sub in v for item in sub], dtype=int)
            for k, v in split_run.items()
        }

    split_payload = {
        "protocol_name": protocol.get("protocol_name"),
        "split_indices": {k: np.asarray(v).astype(int).tolist() for k, v in indices.items()},
        "grouped": "run_id" in window_payload,
        "num_samples": {
            "all": int(len(x)),
            "train": int(len(indices["train"])),
            "val": int(len(indices["val"])),
            "test": int(len(indices["test"])),
        },
    }
    split_payload.update(protocol)
    _write_json(split_path, split_payload)

    return {
        "train": {
            "X": x[indices["train"]],
            "Y": y[indices["train"]],
            "sample_id": np.asarray(window_payload["sample_id"])[indices["train"]],
            "run_id": np.asarray(window_payload["run_id"])[indices["train"]],
            "timestamp": np.asarray(window_payload["timestamp"])[indices["train"]],
        },
        "val": {
            "X": x[indices["val"]],
            "Y": y[indices["val"]],
            "sample_id": np.asarray(window_payload["sample_id"])[indices["val"]],
            "run_id": np.asarray(window_payload["run_id"])[indices["val"]],
            "timestamp": np.asarray(window_payload["timestamp"])[indices["val"]],
        },
        "test": {
            "X": x[indices["test"]],
            "Y": y[indices["test"]],
            "sample_id": np.asarray(window_payload["sample_id"])[indices["test"]],
            "run_id": np.asarray(window_payload["run_id"])[indices["test"]],
            "timestamp": np.asarray(window_payload["timestamp"])[indices["test"]],
        },
    }


def _find_benchmark_entry(metadata_root: Path, dataset_name: str) -> Mapping[str, Any]:
    manifest = _load_json(metadata_root / "benchmark_manifest.json")
    for item in manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return item
    raise SilverboxBuilderError(f"benchmark_manifest missing {dataset_name}.")


def export_bundle(
    cfg: SilverboxBuilderConfig,
    splits: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    benchmark_entry: Mapping[str, Any],
    processed_root: Path,
    split_path: Path,
    metadata_root: Path,
) -> DatasetBundle:
    processed_root.mkdir(parents=True, exist_ok=True)

    processed_files: Dict[str, str] = {}
    for split_name, payload in splits.items():
        np.save(processed_root / f"{split_name}_X.npy", np.asarray(payload["X"]))
        np.save(processed_root / f"{split_name}_Y.npy", np.asarray(payload["Y"]))
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
    input_dim = int(train_x.shape[-1]) if train_x.size else 0
    output_dim = int(train_y.shape[-1]) if train_y.size else 0

    artifacts = {
        "truth_file": None,
        "grouping_file": str(split_path),
        "protocol_file": str((metadata_root / "protocols" / f"{cfg.split_protocol}.json")),
        "extra": {
            "source": str(metadata_root / "benchmark_manifest.json"),
            "task_usage": benchmark_entry.get("task_usage", []),
            "system_type": benchmark_entry.get("system_type"),
        },
    }

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
                "protocol_reference": str(metadata_root / "protocols" / f"{cfg.split_protocol}.json"),
                "processed_root": str(processed_root),
                "split_file": str(split_path),
            },
        },
        artifacts=artifacts,
    )

    manifest = {
        "dataset_name": cfg.dataset_name,
        "generated_at": "placeholder_not_run",
        "window_length": cfg.window_length,
        "horizon": cfg.horizon,
        "split_protocol": cfg.split_protocol,
        "split_file": str(split_path),
        "processed_files": processed_files,
        "split_summary": {
            "train": int(len(splits["train"]["X"])),
            "val": int(len(splits["val"]["X"])),
            "test": int(len(splits["test"]["X"])),
        },
        "meta": bundle.meta.to_dict(),
        "artifacts": bundle.artifacts.to_dict(),
    }
    _write_json(processed_root / f"{cfg.dataset_name}_processed_manifest.json", manifest)
    _write_json(
        metadata_root / f"{cfg.dataset_name}_build_manifest.json",
        {
            **manifest,
            "task_family": "nonlinear",
            "has_ground_truth_kernel": bool(benchmark_entry.get("has_ground_truth_kernel", False)),
            "has_ground_truth_gfrf": bool(benchmark_entry.get("has_ground_truth_gfrf", False)),
        },
    )
    check_dataset_bundle(bundle, strict=False)
    return bundle


class SilverboxBuilder(NonlinearBuilder):
    """Concrete Silverbox builder."""

    def load_splits(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cfg = SilverboxBuilderConfig()
        project_root = self.context.interim_root
        for candidate in project_root.parents:
            if (candidate / "data" / "metadata" / "nonlinear" / "protocols" / f"{cfg.split_protocol}.json").exists():
                project_root = candidate
                break
        protocol = _load_json(
            project_root / "data" / "metadata" / "nonlinear" / "protocols" / f"{cfg.split_protocol}.json"
        )
        # load raw
        raw = load_raw(self.context.raw_root, cfg)
        pre = preprocess(raw)
        # interim export
        self.context.interim_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.context.interim_root / f"{cfg.dataset_name}_cleaned.npz",
            u=pre["u"],
            y=pre["y"],
            sample_id=np.asarray(pre["sample_id"], dtype=object) if pre["sample_id"] is not None else np.array([], dtype=object),
            run_id=np.asarray(pre["run_id"], dtype=object) if pre["run_id"] is not None else np.array([], dtype=object),
            timestamp=np.asarray(pre["timestamp"], dtype=float) if pre["timestamp"] is not None else np.array([], dtype=float),
        )
        _write_json(
            self.context.interim_root / f"{cfg.dataset_name}_interim_manifest.json",
            {
                "dataset_name": cfg.dataset_name,
                "raw_candidates": list(cfg.raw_file_candidates),
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
            },
        )

        windows = build_windows(
            pre["u"], pre["y"],
            window_length=cfg.window_length,
            horizon=cfg.horizon,
            sample_id=pre["sample_id"],
            run_id=pre["run_id"],
            timestamp=pre["timestamp"],
        )
        split_path = self.context.splits_root / f"{cfg.dataset_name}_split_manifest.json"
        splits = build_split(windows, protocol, split_path)
        return splits["train"], splits["val"], splits["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Silverbox dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split-protocol", type=str, default="nonlinear_temporal_grouped_holdout_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SilverboxBuilderConfig(
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

    builder = SilverboxBuilder(context)
    train, val, test = builder.load_splits()
    split_file = context.splits_root / f"{cfg.dataset_name}_split_manifest.json"
    protocol = _load_json(project_root / "data" / "metadata" / "nonlinear" / "protocols" / f"{cfg.split_protocol}.json")
    metadata_root = project_root / "data" / "metadata" / "nonlinear"
    benchmark_entry = _find_benchmark_entry(metadata_root, cfg.dataset_name)

    export_bundle(
        cfg=cfg,
        splits={"train": train, "val": val, "test": test},
        protocol=protocol,
        benchmark_entry=benchmark_entry,
        processed_root=context.processed_root,
        split_path=split_file,
        metadata_root=metadata_root,
    )


if __name__ == "__main__":
    main()
