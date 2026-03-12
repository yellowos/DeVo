"""Build Volterra–Wiener dataset bundle for nonlinear benchmark.

Scope:
- raw -> interim -> processed pipeline
- unified DatasetBundle output (train/val/test/meta/artifacts)
- explicit kernel truth registration entry for downstream kernel recovery
- no model/training/experiment logic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from data.adapters.nonlinear_adapter import NonlinearAdapter
from data.adapters.base import DataProtocolError
from data.builders.nonlinear_builder import NonlinearBuilder, NonlinearBuilderContext
from data.checks.bundle_checks import check_dataset_bundle


class VolterraWienerBuilderError(DataProtocolError):
    """Raised when Volterra-Wiener build steps cannot be completed."""


@dataclass(frozen=True)
class VolterraWienerBuilderConfig:
    dataset_name: str = "volterra_wiener"
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "nonlinear_temporal_grouped_holdout_v1"
    raw_file_candidates: tuple[str, ...] = (
        "volterra_wiener.mat",
        "volterra_wiener_raw.npz",
        "volterra_wiener_processed.npz",
        "volterra_wiener.csv",
        "volterra_wiener.txt",
    )

    def validate(self) -> None:
        if self.window_length <= 0:
            raise VolterraWienerBuilderError("window_length must be > 0.")
        if self.horizon <= 0:
            raise VolterraWienerBuilderError("horizon must be > 0.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_first_existing(root: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        p = root / name
        if p.exists():
            return p
    for candidate in root.glob("**/*"):
        if not candidate.is_file():
            continue
        stem = candidate.stem.lower()
        suffix = candidate.suffix.lower()
        if suffix in {".mat", ".npz", ".npy", ".csv", ".txt"} and "volterra" in stem:
            return candidate
    return None


def _to_float2d(value: Any, key: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise VolterraWienerBuilderError(f"{key} must be 1D or 2D numeric array.")
    if not np.issubdtype(arr.dtype, np.number):
        raise VolterraWienerBuilderError(f"{key} must be numeric.")
    return arr.astype(np.float64)


def _to_optional_vector(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value).reshape(-1)


def load_raw(raw_root: Path, cfg: VolterraWienerBuilderConfig) -> Dict[str, Any]:
    raw_root.mkdir(parents=True, exist_ok=True)
    path = _find_first_existing(raw_root, cfg.raw_file_candidates)
    if path is None:
        raise VolterraWienerBuilderError(
            f"No Volterra-Wiener raw file found under {raw_root}. "
            f"Expected one of: {list(cfg.raw_file_candidates)}"
        )

    u = y = None
    sample_id = None
    run_id = None
    timestamp = None

    suffix = path.suffix.lower()
    if suffix == ".npz":
        payload = np.load(path, allow_pickle=True)
        u = payload.get("u") or payload.get("input")
        y = payload.get("y") or payload.get("output")
        if u is None:
            raise VolterraWienerBuilderError(f"{path} missing u/input array.")
        if y is None:
            raise VolterraWienerBuilderError(f"{path} missing y/output array.")
        sample_id = payload.get("sample_id")
        run_id = payload.get("run_id")
        timestamp = payload.get("timestamp")
    elif suffix == ".npy":
        arr = np.load(path)
        if arr.ndim < 2 or arr.shape[1] < 2:
            raise VolterraWienerBuilderError(
                f"{path} must be shaped [T, >=2] to infer input/output."
            )
        u = arr[:, :1]
        y = arr[:, 1:2]
    elif suffix in {".csv", ".txt"}:
        import csv

        with path.open("r", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise VolterraWienerBuilderError(f"{path} is empty.")
        keys = rows[0].keys()
        u_key = next((k for k in ("u", "input", "u_in") if k in keys), None)
        y_key = next((k for k in ("y", "output", "y_out") if k in keys), None)
        if u_key is None or y_key is None:
            raise VolterraWienerBuilderError(f"{path} lacks required u/y columns.")
        u = np.asarray([float(r[u_key]) for r in rows], dtype=float).reshape(-1, 1)
        y = np.asarray([float(r[y_key]) for r in rows], dtype=float).reshape(-1, 1)
        sample_id = np.asarray([r.get("sample_id", i) for i, r in enumerate(rows)], dtype=object) if "sample_id" in keys else None
        run_id = np.asarray([r.get("run_id", None) for r in rows], dtype=object) if "run_id" in keys else None
        if "timestamp" in keys:
            timestamp = np.asarray([float(r["timestamp"]) for r in rows], dtype=float)
    elif suffix == ".mat":
        try:
            import scipy.io

            payload = scipy.io.loadmat(path)
        except Exception as exc:
            raise VolterraWienerBuilderError(
                "Could not parse .mat without scipy support."
            ) from exc

        for key in ("u", "input", "u_in"):
            if key in payload and isinstance(payload[key], np.ndarray):
                u = payload[key]
                break
        for key in ("y", "output", "y_out"):
            if key in payload and isinstance(payload[key], np.ndarray):
                y = payload[key]
                break
        if u is None:
            raise VolterraWienerBuilderError(f"{path} missing u/input array.")
        if y is None:
            raise VolterraWienerBuilderError(f"{path} missing y/output array.")
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
        raise VolterraWienerBuilderError(f"Unsupported raw format: {suffix}")

    u_arr = _to_float2d(u, "u")
    y_arr = _to_float2d(y, "y")
    n = min(len(u_arr), len(y_arr))
    u_arr = u_arr[:n]
    y_arr = y_arr[:n]
    if n == 0:
        raise VolterraWienerBuilderError(f"No time steps after align on {path}.")

    mask = np.isfinite(u_arr).all(axis=1) & np.isfinite(y_arr).all(axis=1)
    u_arr = u_arr[mask]
    y_arr = y_arr[mask]
    if sample_id is not None and len(sample_id) == len(mask):
        sample_id = sample_id[mask]
    if run_id is not None and len(run_id) == len(mask):
        run_id = run_id[mask]
    if timestamp is not None and len(timestamp) == len(mask):
        timestamp = timestamp[mask]

    return {
        "u": u_arr,
        "y": y_arr,
        "sample_id": _to_optional_vector(sample_id),
        "run_id": _to_optional_vector(run_id),
        "timestamp": _to_optional_vector(timestamp),
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
        raise VolterraWienerBuilderError(
            f"Not enough points: n={n}, window_length={window_length}, horizon={horizon}."
        )
    X_list = []
    Y_list = []
    sid_list = []
    rid_list = []
    ts_list = []
    for i in range(0, n - window_length - horizon + 1):
        X_list.append(u[i : i + window_length])
        Y_list.append(y[i + window_length : i + window_length + horizon])
        sid_list.append(sample_id[i + window_length - 1] if sample_id is not None and len(sample_id) == n else i)
        rid_list.append(run_id[i + window_length - 1] if run_id is not None and len(run_id) == n else None)
        ts_list.append(timestamp[i + window_length - 1] if timestamp is not None and len(timestamp) == n else None)
    return {
        "X": np.asarray(X_list, dtype=np.float64),
        "Y": np.asarray(Y_list, dtype=np.float64),
        "sample_id": np.asarray(sid_list, dtype=object),
        "run_id": np.asarray(rid_list, dtype=object),
        "timestamp": np.asarray(ts_list, dtype=object),
    }


def _build_split_indices(
    window_count: int,
    run_id: np.ndarray,
    protocol: Mapping[str, Any],
    window_payload: Mapping[str, Any],
    split_path: Path,
) -> Dict[str, np.ndarray]:
    split_cfg = protocol.get("split", {})
    tr = float(split_cfg.get("train", 0.7))
    va = float(split_cfg.get("val", 0.15))
    te = float(split_cfg.get("test", 0.15))
    total = tr + va + te
    if total <= 0:
        raise VolterraWienerBuilderError("Split protocol ratios sum to zero.")
    if abs(total - 1.0) > 1e-9:
        tr, va, te = tr / total, va / total, te / total

    grouped = protocol.get("grouping", {}).get("level") == "trajectory"
    if grouped and run_id is not None and len(run_id) == window_count:
        valid = np.array([r is not None for r in run_id], dtype=bool)
        order = np.arange(window_count)[valid]
        if order.size:
            groups: list[list[int]] = []
            current = run_id[order[0]]
            current_ids: list[int] = []
            groups.append(current_ids)
            for idx in order:
                rid = run_id[idx]
                if rid != current:
                    current = rid
                    current_ids = []
                    groups.append(current_ids)
                current_ids.append(int(idx))

            g_n = len(groups)
            gtr = max(1, int(g_n * tr))
            gva = max(0, int(g_n * va))
            gte = max(1, g_n - gtr - gva)
            if gtr + gva + gte > g_n:
                gte = g_n - gtr - gva

            split_indices = {
                "train": np.array([j for g in groups[:gtr] for j in g], dtype=int),
                "val": np.array([j for g in groups[gtr : gtr + gva] for j in g], dtype=int),
                "test": np.array([j for g in groups[gtr + gva : gtr + gva + gte] for j in g], dtype=int),
            }
            for key in split_indices:
                split_indices[key] = np.array(sorted(set(split_indices[key].tolist())), dtype=int)
        else:
            split_indices = {
                "train": np.array([], dtype=int),
                "val": np.array([], dtype=int),
                "test": np.array([], dtype=int),
            }
    else:
        split_indices = {
            "train": np.arange(0, int(window_count * tr)),
            "val": np.arange(int(window_count * tr), int(window_count * (tr + va))),
            "test": np.arange(int(window_count * (tr + va)), window_count),
        }

    if any(len(v) == 0 for v in split_indices.values()):
        # conservative fallback to non-empty contiguous split
        train_end = int(window_count * tr)
        val_end = train_end + int(window_count * va)
        split_indices = {
            "train": np.arange(0, max(1, train_end)),
            "val": np.arange(max(1, train_end), min(window_count, max(1, val_end))),
            "test": np.arange(min(window_count, max(1, val_end)), window_count),
        }
        if len(split_indices["test"]) == 0 and len(split_indices["val"]) > 1:
            split_indices["test"] = split_indices["val"][-1:]
            split_indices["val"] = split_indices["val"][:-1]

    # Persist split metadata now.
    payload = {
        "dataset_name": "volterra_wiener",
        "protocol_name": protocol.get("protocol_name"),
        "windowed_samples": int(window_count),
        "split_indices": {k: v.tolist() for k, v in split_indices.items()},
        "counts": {k: int(len(v)) for k, v in split_indices.items()},
        "protocol": protocol,
        "grouping": protocol.get("grouping", {}),
        "run_id_present": bool(run_id is not None and len(run_id) == window_count),
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(split_path, payload)

    return {
        "train": split_indices["train"],
        "val": split_indices["val"],
        "test": split_indices["test"],
    }


def build_split(
    window_payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_path: Path,
) -> Dict[str, Dict[str, Any]]:
    run_id = np.asarray(window_payload.get("run_id", np.array([], dtype=object)), dtype=object)
    if run_id.size:
        run_id = run_id.reshape(-1)
    x = np.asarray(window_payload["X"])
    y = np.asarray(window_payload["Y"])
    sample_id = np.asarray(window_payload.get("sample_id", np.array([], dtype=object)), dtype=object)
    timestamp = np.asarray(window_payload.get("timestamp", np.array([], dtype=object)), dtype=object)

    split_indices = _build_split_indices(len(x), run_id, protocol, window_payload, split_path)

    train_idx = split_indices["train"]
    val_idx = split_indices["val"]
    test_idx = split_indices["test"]

    return {
        "train": {
            "X": x[train_idx],
            "Y": y[train_idx],
            "sample_id": sample_id[train_idx] if len(sample_id) else None,
            "run_id": run_id[train_idx] if len(run_id) else None,
            "timestamp": timestamp[train_idx] if len(timestamp) else None,
        },
        "val": {
            "X": x[val_idx],
            "Y": y[val_idx],
            "sample_id": sample_id[val_idx] if len(sample_id) else None,
            "run_id": run_id[val_idx] if len(run_id) else None,
            "timestamp": timestamp[val_idx] if len(timestamp) else None,
        },
        "test": {
            "X": x[test_idx],
            "Y": y[test_idx],
            "sample_id": sample_id[test_idx] if len(sample_id) else None,
            "run_id": run_id[test_idx] if len(run_id) else None,
            "timestamp": timestamp[test_idx] if len(timestamp) else None,
        },
    }


def _resolve_manifest_entry(metadata_root: Path, manifest_name: str, dataset_name: str) -> Mapping[str, Any]:
    manifest_path = metadata_root / manifest_name
    if not manifest_path.exists():
        raise VolterraWienerBuilderError(f"Missing metadata file: {manifest_path}")
    manifest = _load_json(manifest_path)
    items = manifest.get("benchmarks")
    if not isinstance(items, list):
        raise VolterraWienerBuilderError(f"Invalid manifest structure in {manifest_path}")

    for item in items:
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return item
    raise VolterraWienerBuilderError(
        f"Dataset '{dataset_name}' missing in manifest {manifest_path}"
    )


def ensure_kernel_reference_artifact(
    metadata_root: Path,
    benchmark_entry: Mapping[str, Any],
    kernel_entry: Mapping[str, Any],
) -> Mapping[str, str]:
    truth_dir = metadata_root / "truth"
    truth_dir.mkdir(parents=True, exist_ok=True)

    kernel_ref = kernel_entry.get("kernel_reference")
    # keep protocol-defined uri when empty and build local manifest path deterministically.
    artifact_path = truth_dir / f"{benchmark_entry.get('benchmark_name', 'volterra_wiener')}_kernel_reference.json"

    # Keep registry explicit and non-empty.
    payload = {
        "benchmark_name": benchmark_entry.get("benchmark_name", "volterra_wiener"),
        "truth_type": "kernel",
        "source": kernel_ref or "",
        "status": "registered",
        "registration_type": "builder_ledger_reference",
        "artifact_created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Ground-truth kernel object reference used by kernel recovery experiments.",
        "registration": {
            "can_be_loaded_by_methods": bool(benchmark_entry.get("has_ground_truth_kernel", False)),
            "reference_protocol": "nonlinear.kernel_truth_manifest.v1",
            "reference_entry": kernel_entry,
        },
    }

    # If a reference file already exists, keep existing payload and enrich if needed.
    if artifact_path.exists():
        try:
            existing = _load_json(artifact_path)
            merged = dict(existing)
            merged.update({"artifact_updated_at": datetime.utcnow().isoformat() + "Z"})
            if isinstance(existing, Mapping):
                _write_json(artifact_path, merged)
            else:
                _write_json(artifact_path, payload)
        except Exception:
            _write_json(artifact_path, payload)
    else:
        _write_json(artifact_path, payload)

    return {
        "kernel_reference_uri": str(kernel_ref) if kernel_ref else str(artifact_path),
        "kernel_reference_file": str(artifact_path),
        "kernel_reference_payload": str(artifact_path),
    }


def _gfrf_truth_available(metadata_root: Path, dataset_name: str) -> bool:
    gfrf_manifest = _load_json(metadata_root / "gfrf_truth_manifest.json")
    for item in gfrf_manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset_name:
            return bool(item.get("has_ground_truth_gfrf", False) and item.get("gfrf_reference"))
    return False


def export_bundle(
    cfg: VolterraWienerBuilderConfig,
    splits: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    benchmark_entry: Mapping[str, Any],
    kernel_entry: Mapping[str, Any],
    processed_root: Path,
    split_path: Path,
    metadata_root: Path,
) -> Any:
    processed_root.mkdir(parents=True, exist_ok=True)
    processed_files: Dict[str, str] = {}

    for split_name, payload in splits.items():
        np.save(processed_root / f"{split_name}_X.npy", np.asarray(payload["X"], dtype=float))
        np.save(processed_root / f"{split_name}_Y.npy", np.asarray(payload["Y"], dtype=float))
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

    kernel_refs = ensure_kernel_reference_artifact(metadata_root, benchmark_entry, kernel_entry)
    gfrf_available = _gfrf_truth_available(metadata_root, cfg.dataset_name)

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
            "has_ground_truth_gfrf": bool(gfrf_available),
            "extras": {
                "task_usage": benchmark_entry.get("task_usage", ["prediction", "kernel_recovery"]),
                "system_type": benchmark_entry.get("system_type"),
                "split_file": str(split_path),
                "protocol_file": str(metadata_root / "protocols" / f"{cfg.split_protocol}.json"),
                "truth_artifacts": {
                    "kernel_reference": kernel_refs["kernel_reference_uri"],
                    "kernel_reference_file": kernel_refs["kernel_reference_file"],
                    "gfrf_reference_available": bool(gfrf_available),
                },
            },
        },
        artifacts={
            "truth_file": kernel_refs["kernel_reference_file"],
            "grouping_file": str(split_path),
            "protocol_file": str(metadata_root / "protocols" / f"{cfg.split_protocol}.json"),
            "extra": {
                "kernel_reference_uri": kernel_refs["kernel_reference_uri"],
                "gfrf_available": bool(gfrf_available),
                "task_family": "nonlinear",
            },
        },
    )

    # Build-level manifest for reproducibility.
    _write_json(
        processed_root / f"{cfg.dataset_name}_processed_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "window_length": cfg.window_length,
            "horizon": cfg.horizon,
            "split_protocol": cfg.split_protocol,
            "split_file": str(split_path),
            "processed_files": processed_files,
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
            "counts": {
                "train": len(splits["train"]["X"]),
                "val": len(splits["val"]["X"]),
                "test": len(splits["test"]["X"]),
            },
        },
    )
    _write_json(
        metadata_root / f"{cfg.dataset_name}_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "build_manifest": "data/metadata/nonlinear/" + f"{cfg.dataset_name}_processed_manifest.json",
            "bundle_artifacts": bundle.artifacts.to_dict(),
            "kernel_truth_reference": kernel_refs,
            "has_ground_truth_kernel": bool(benchmark_entry.get("has_ground_truth_kernel", False)),
            "has_ground_truth_gfrf": bool(gfrf_available),
        },
    )

    check_dataset_bundle(bundle, strict=False)
    return bundle


class VolterraWienerBuilder(NonlinearBuilder):
    """Concrete Volterra-Wiener builder."""

    def load_splits(self) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        cfg = VolterraWienerBuilderConfig()
        # locate protocol by walking upward from interim root
        protocol_path = None
        for parent in self.context.interim_root.parents:
            candidate = parent / "data" / "metadata" / "nonlinear" / "protocols" / f"{cfg.split_protocol}.json"
            if candidate.exists():
                protocol_path = candidate
                break
        if protocol_path is None:
            raise VolterraWienerBuilderError(
                f"Cannot locate split protocol {cfg.split_protocol} from {self.context.interim_root}."
            )
        protocol = _load_json(protocol_path)

        raw = load_raw(self.context.raw_root, cfg)
        interim_root = self.context.interim_root
        interim_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            interim_root / f"{cfg.dataset_name}_preprocessed.npz",
            u=raw["u"],
            y=raw["y"],
            sample_id=np.asarray(raw["sample_id"], dtype=object) if raw["sample_id"] is not None else np.array([], dtype=object),
            run_id=np.asarray(raw["run_id"], dtype=object) if raw["run_id"] is not None else np.array([], dtype=object),
            timestamp=np.asarray(raw["timestamp"], dtype=float) if raw["timestamp"] is not None else np.array([], dtype=float),
        )
        _write_json(
            interim_root / f"{cfg.dataset_name}_interim_manifest.json",
            {
                "dataset_name": cfg.dataset_name,
                "raw_source_root": str(self.context.raw_root),
                "window_length": cfg.window_length,
                "horizon": cfg.horizon,
                "generated_at": datetime.utcnow().isoformat() + "Z",
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
        split_data = build_split(windows, protocol, split_path)
        return split_data["train"], split_data["val"], split_data["test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Volterra-Wiener dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--window-length", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--split-protocol", type=str, default="nonlinear_temporal_grouped_holdout_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VolterraWienerBuilderConfig(
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

    builder = VolterraWienerBuilder(context)
    benchmark_entry = _resolve_manifest_entry(
        project_root / "data" / "metadata" / "nonlinear",
        "benchmark_manifest.json",
        cfg.dataset_name,
    )
    kernel_entry = _resolve_manifest_entry(
        project_root / "data" / "metadata" / "nonlinear",
        "kernel_truth_manifest.json",
        cfg.dataset_name,
    )

    train, val, test = builder.load_splits()
    protocol = _load_json(
        project_root / "data" / "metadata" / "nonlinear" / "protocols" / f"{cfg.split_protocol}.json"
    )
    export_bundle(
        cfg=cfg,
        splits={"train": train, "val": val, "test": test},
        protocol=protocol,
        benchmark_entry=benchmark_entry,
        kernel_entry=kernel_entry,
        processed_root=context.processed_root,
        split_path=context.splits_root / f"{cfg.dataset_name}_split_manifest.json",
        metadata_root=project_root / "data" / "metadata" / "nonlinear",
    )


if __name__ == "__main__":
    main()
