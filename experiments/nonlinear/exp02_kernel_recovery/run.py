#!/usr/bin/env python3
"""Nonlinear benchmark experiment 02: kernel recovery."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methods.base import KernelRecoveryResult, get_method_class
from methods.devo import DeVoConfig
from methods.utils import load_processed_dataset_bundle, slice_dataset_bundle


EXPERIMENT_NAME = "exp02_kernel_recovery"
METHOD_ORDER = ["narmax", "tt_volterra", "cp_volterra", "laguerre_volterra", "devo"]
DATASET_ORDER = ["volterra_wiener", "duffing"]
METHOD_DISPLAY_NAMES = {
    "narmax": "NARMAX",
    "tt_volterra": "TT-Volterra",
    "cp_volterra": "CP-Volterra",
    "laguerre_volterra": "Laguerre-Volterra",
    "devo": "DeVo",
}
DATASET_DISPLAY_NAMES = {
    "volterra_wiener": "Volterra-Wiener",
    "duffing": "Duffing",
}
METRIC_BY_DATASET = {
    "volterra_wiener": "knmse",
    "duffing": "gfrf_re",
}


@dataclass
class NormalizedKernelSet:
    """Experiment-layer dense kernel schema.

    For this benchmark we only compute metrics on SISO, horizon-1 kernels.
    Dense order tensors are therefore stored as order-dimensional lag tensors:
    - order 1 -> [L]
    - order 2 -> [L, L]
    - ...
    """

    representation: str
    dense_orders: dict[int, np.ndarray]
    bias: Optional[np.ndarray]
    warnings: list[str]
    source_summary: dict[str, Any]


def project_root() -> Path:
    return PROJECT_ROOT


def experiment_root() -> Path:
    return Path(__file__).resolve().parent


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=True)


def _resolve_path(value: str | Path, *, base: Optional[Path] = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = base or project_root()
    return (anchor / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML payload must be a mapping: {path}")
    return dict(payload)


def _latest_run_dir(results_root: Path) -> Path:
    run_dirs = sorted([path for path in results_root.iterdir() if path.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {results_root}")
    return run_dirs[-1]


def _selected_names(
    requested: Optional[Iterable[str]],
    *,
    default_order: list[str],
    available: Optional[set[str]] = None,
) -> list[str]:
    if requested is None:
        names = list(default_order)
    else:
        names = [str(name).strip() for name in requested if str(name).strip()]
    if available is None:
        return names
    return [name for name in names if name in available]


def _bundle_reference_candidates(bundle: Any, truth_kind: str) -> list[Path]:
    candidates: list[Path] = []
    artifacts = getattr(bundle, "artifacts", None)
    extra = dict(getattr(artifacts, "extra", {}) or {})
    truth_file = getattr(artifacts, "truth_file", None)
    if truth_kind == "kernel":
        for item in (
            extra.get("kernel_reference_file"),
            truth_file if truth_file and "kernel" in str(truth_file).lower() else None,
            getattr(bundle.meta, "extras", {}).get("truth_artifacts", {}).get("kernel_reference_file"),
        ):
            if item:
                candidates.append(_resolve_path(item))
    elif truth_kind == "gfrf":
        for item in (
            extra.get("gfrf_reference_file"),
            truth_file if truth_file and "gfrf" in str(truth_file).lower() else None,
            getattr(bundle.meta, "extras", {}).get("truth_artifacts", {}).get("gfrf_reference_file"),
        ):
            if item:
                candidates.append(_resolve_path(item))
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _read_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _materialized_payload_path(reference_path: Path, payload: Mapping[str, Any], truth_kind: str) -> Optional[Path]:
    keys = {
        "kernel": (
            "kernel_reference_file",
            "kernel_coefficients_path",
            "kernel_metadata_path",
            "payload_path",
            "artifact_path",
            "path",
        ),
        "gfrf": (
            "gfrf_reference_file",
            "gfrf_coefficients_path",
            "gfrf_metadata_path",
            "payload_path",
            "artifact_path",
            "path",
        ),
    }
    for key in keys[truth_kind]:
        candidate = payload.get(key)
        if not candidate:
            continue
        resolved = _resolve_path(candidate, base=reference_path.parent)
        if resolved != reference_path:
            return resolved
    return None


def _siso_lag_tensor(value: Any, order: int, *, lag_axes_recent_first: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    squeezed = np.squeeze(array)
    if order == 0:
        return np.asarray(squeezed, dtype=np.float64)
    if squeezed.ndim != order:
        raise ValueError(
            f"Expected a squeezed order-{order} lag tensor, got shape {tuple(array.shape)} -> {tuple(squeezed.shape)}."
        )
    if lag_axes_recent_first:
        squeezed = np.flip(squeezed, axis=tuple(range(order)))
    return np.asarray(squeezed, dtype=np.float64)


def _normalize_devo_recovery(recovery: KernelRecoveryResult) -> NormalizedKernelSet:
    kernels = recovery.kernels
    dense_orders: dict[int, np.ndarray] = {}
    warnings = list(recovery.summary.get("warnings", [])) if isinstance(recovery.summary, Mapping) else []
    for order_key, payload in dict(kernels.get("orders", {})).items():
        order = int(order_key)
        full_tensor = payload.get("full_tensor")
        if full_tensor is None:
            warnings.append(f"DeVo order {order} full_tensor is not materialized.")
            continue
        dense_orders[order] = _siso_lag_tensor(full_tensor, order)
    return NormalizedKernelSet(
        representation="devo_dense_volterra",
        dense_orders=dense_orders,
        bias=np.asarray(kernels.get("bias"), dtype=np.float64) if kernels.get("bias") is not None else None,
        warnings=warnings,
        source_summary=dict(recovery.summary),
    )


def _normalize_cp_recovery(recovery: KernelRecoveryResult) -> NormalizedKernelSet:
    kernels = recovery.kernels
    dense_orders: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    for payload in list(kernels.get("orders", [])):
        order = int(payload.get("order"))
        full_kernel = payload.get("full_kernel")
        if full_kernel is None:
            warnings.append(f"CP-Volterra order {order} full_kernel is not materialized.")
            continue
        dense_orders[order] = _siso_lag_tensor(full_kernel, order)
    return NormalizedKernelSet(
        representation="cp_dense_volterra",
        dense_orders=dense_orders,
        bias=np.asarray(kernels.get("bias_raw"), dtype=np.float64) if kernels.get("bias_raw") is not None else None,
        warnings=warnings,
        source_summary=dict(recovery.summary),
    )


def _normalize_tt_recovery(recovery: KernelRecoveryResult) -> NormalizedKernelSet:
    kernels = recovery.kernels
    dense_orders: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    for order_key, payload in dict(kernels.get("orders", {})).items():
        order = int(order_key)
        dense = payload.get("dense")
        if dense is None:
            warnings.append(f"TT-Volterra order {order} dense kernel is not materialized.")
            continue
        dense_orders[order] = _siso_lag_tensor(dense, order)
    return NormalizedKernelSet(
        representation="tt_dense_volterra",
        dense_orders=dense_orders,
        bias=np.asarray(kernels.get("bias"), dtype=np.float64) if kernels.get("bias") is not None else None,
        warnings=warnings,
        source_summary=dict(recovery.summary),
    )


def _normalize_laguerre_recovery(recovery: KernelRecoveryResult) -> NormalizedKernelSet:
    kernels = recovery.kernels
    dense_orders: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    for key, value in dict(kernels.get("time_domain_kernels", {})).items():
        if not str(key).startswith("order_"):
            continue
        order = int(str(key).split("_", 1)[1])
        if order == 0:
            continue
        dense_orders[order] = _siso_lag_tensor(value, order, lag_axes_recent_first=True)
    if not dense_orders:
        warnings.append("Laguerre-Volterra time-domain kernels are unavailable.")
    return NormalizedKernelSet(
        representation="laguerre_time_domain_volterra",
        dense_orders=dense_orders,
        bias=np.asarray(kernels.get("time_domain_kernels", {}).get("order_0"), dtype=np.float64)
        if kernels.get("time_domain_kernels", {}).get("order_0") is not None
        else None,
        warnings=warnings,
        source_summary=dict(recovery.summary),
    )


def _normalize_narmax_recovery(recovery: KernelRecoveryResult) -> NormalizedKernelSet:
    kernels = recovery.kernels
    warnings = [
        "NARMAX recover_kernels() returns structural coefficients over normalized regressors.",
        "Exact dense Volterra kernels are not exposed by the methods layer, so KNMSE/GFRF-RE are skipped.",
    ]
    exact = bool(kernels.get("exact_volterra_kernels", False))
    if exact:
        warnings.append("NARMAX reported exact_volterra_kernels=True, but no dense kernel adapter is implemented here.")
    return NormalizedKernelSet(
        representation="narmax_structural_coefficients",
        dense_orders={},
        bias=None,
        warnings=warnings,
        source_summary=dict(recovery.summary),
    )


def normalize_recovery(method_name: str, recovery: KernelRecoveryResult) -> NormalizedKernelSet:
    normalized = method_name.strip().lower().replace("-", "_")
    if normalized == "devo":
        return _normalize_devo_recovery(recovery)
    if normalized == "cp_volterra":
        return _normalize_cp_recovery(recovery)
    if normalized == "tt_volterra":
        return _normalize_tt_recovery(recovery)
    if normalized == "laguerre_volterra":
        return _normalize_laguerre_recovery(recovery)
    if normalized == "narmax":
        return _normalize_narmax_recovery(recovery)
    raise KeyError(f"No kernel normalization adapter for method '{method_name}'.")


def _load_truth_kernel_from_file(path: Path) -> tuple[Optional[dict[int, np.ndarray]], str]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        dense_orders = {
            int(key.split("_", 1)[1]): np.asarray(payload[key], dtype=np.float64)
            for key in payload.files
            if key.startswith("order_")
        }
        if dense_orders:
            return dense_orders, "loaded from npz"
        return None, f"No order_* arrays found in {path}"
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=True)
        if array.ndim == 1:
            return {1: np.asarray(array, dtype=np.float64)}, "loaded order-1 kernel from npy"
        return None, f"Cannot infer order from standalone npy shape {tuple(array.shape)}"
    if path.suffix.lower() == ".json":
        payload = _read_mapping(path)
        if "orders" in payload and isinstance(payload["orders"], Mapping):
            dense_orders = {
                int(key): np.asarray(value, dtype=np.float64)
                for key, value in payload["orders"].items()
            }
            return dense_orders, "loaded from json orders mapping"
        if "kernels" in payload and isinstance(payload["kernels"], Mapping):
            kernels = payload["kernels"]
            if "orders" in kernels and isinstance(kernels["orders"], Mapping):
                dense_orders = {
                    int(key): np.asarray(value, dtype=np.float64)
                    for key, value in kernels["orders"].items()
                }
                return dense_orders, "loaded from json kernels.orders mapping"
        return None, f"Reference json {path} does not contain materialized kernel orders."
    return None, f"Unsupported truth file suffix for kernel loading: {path.suffix}"


def load_truth_kernel(bundle: Any) -> dict[str, Any]:
    candidates = _bundle_reference_candidates(bundle, "kernel")
    if not candidates:
        return {
            "status": "failed",
            "reference_path": None,
            "resolved_payload_path": None,
            "message": "Kernel truth reference path is not available in the dataset bundle.",
            "orders": None,
        }

    reference_path = candidates[0]
    if not reference_path.exists():
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": None,
            "message": "Kernel truth reference file does not exist.",
            "orders": None,
        }

    try:
        payload = _read_mapping(reference_path)
    except Exception as exc:  # pragma: no cover - defensive.
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": None,
            "message": f"Failed to parse kernel truth reference: {exc}",
            "orders": None,
        }

    # Direct materialized payload embedded in the reference json.
    embedded_orders, embedded_message = _load_truth_kernel_from_file(reference_path)
    if embedded_orders:
        return {
            "status": "completed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(reference_path),
            "message": embedded_message,
            "orders": embedded_orders,
            "payload": payload,
        }

    materialized = _materialized_payload_path(reference_path, payload, "kernel")
    if materialized is None:
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": None,
            "message": (
                "Kernel truth reference is only a registration placeholder and does not point to a "
                "materialized kernel object."
            ),
            "orders": None,
            "payload": payload,
        }
    if not materialized.exists():
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(materialized),
            "message": "Kernel truth payload path is registered but missing on disk.",
            "orders": None,
            "payload": payload,
        }

    try:
        orders, message = _load_truth_kernel_from_file(materialized)
    except Exception as exc:  # pragma: no cover - defensive.
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(materialized),
            "message": f"Failed to load kernel truth payload: {exc}",
            "orders": None,
            "payload": payload,
        }
    if not orders:
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(materialized),
            "message": message,
            "orders": None,
            "payload": payload,
        }
    return {
        "status": "completed",
        "reference_path": str(reference_path),
        "resolved_payload_path": str(materialized),
        "message": message,
        "orders": orders,
        "payload": payload,
    }


def _load_truth_gfrf_from_file(path: Path) -> tuple[Optional[dict[int, np.ndarray]], str]:
    if path.suffix.lower() == ".npz":
        payload = np.load(path, allow_pickle=True)
        dense_orders = {
            int(key.split("_", 1)[1]): np.asarray(payload[key])
            for key in payload.files
            if key.startswith("order_")
        }
        if dense_orders:
            return dense_orders, "loaded from npz"
        return None, f"No order_* arrays found in {path}"
    if path.suffix.lower() == ".json":
        payload = _read_mapping(path)
        if "orders" in payload and isinstance(payload["orders"], Mapping):
            dense_orders = {int(key): np.asarray(value) for key, value in payload["orders"].items()}
            return dense_orders, "loaded from json orders mapping"
        if "gfrf" in payload and isinstance(payload["gfrf"], Mapping):
            dense_orders = {int(key): np.asarray(value) for key, value in payload["gfrf"].items()}
            return dense_orders, "loaded from json gfrf mapping"
        return None, f"Reference json {path} does not contain materialized GFRF orders."
    return None, f"Unsupported truth file suffix for GFRF loading: {path.suffix}"


def load_truth_gfrf(bundle: Any) -> dict[str, Any]:
    candidates = _bundle_reference_candidates(bundle, "gfrf")
    if not candidates:
        return {
            "status": "failed",
            "reference_path": None,
            "resolved_payload_path": None,
            "message": "GFRF truth reference path is not available in the dataset bundle.",
            "orders": None,
        }

    reference_path = candidates[0]
    if not reference_path.exists():
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": None,
            "message": "GFRF truth reference file does not exist.",
            "orders": None,
        }

    try:
        payload = _read_mapping(reference_path)
    except Exception as exc:  # pragma: no cover - defensive.
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": None,
            "message": f"Failed to parse GFRF truth reference: {exc}",
            "orders": None,
        }

    embedded_orders, embedded_message = _load_truth_gfrf_from_file(reference_path)
    if embedded_orders:
        return {
            "status": "completed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(reference_path),
            "message": embedded_message,
            "orders": embedded_orders,
            "payload": payload,
        }

    materialized = _materialized_payload_path(reference_path, payload, "gfrf")
    if materialized is None:
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": None,
            "message": (
                "GFRF truth reference is only a registration placeholder and does not point to a "
                "materialized GFRF object."
            ),
            "orders": None,
            "payload": payload,
        }
    if not materialized.exists():
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(materialized),
            "message": "GFRF truth payload path is registered but missing on disk.",
            "orders": None,
            "payload": payload,
        }

    try:
        orders, message = _load_truth_gfrf_from_file(materialized)
    except Exception as exc:  # pragma: no cover - defensive.
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(materialized),
            "message": f"Failed to load GFRF truth payload: {exc}",
            "orders": None,
            "payload": payload,
        }
    if not orders:
        return {
            "status": "failed",
            "reference_path": str(reference_path),
            "resolved_payload_path": str(materialized),
            "message": message,
            "orders": None,
            "payload": payload,
        }
    return {
        "status": "completed",
        "reference_path": str(reference_path),
        "resolved_payload_path": str(materialized),
        "message": message,
        "orders": orders,
        "payload": payload,
    }


def kernels_to_gfrf(kernel_orders: Mapping[int, np.ndarray], *, nfft: int) -> dict[int, np.ndarray]:
    gfrf: dict[int, np.ndarray] = {}
    for order, tensor in sorted(kernel_orders.items()):
        if order <= 0:
            continue
        array = np.asarray(tensor, dtype=np.float64)
        gfrf[order] = np.fft.fftn(array, s=(int(nfft),) * int(order))
    return gfrf


def compute_knmse(truth_orders: Mapping[int, np.ndarray], recovered_orders: Mapping[int, np.ndarray]) -> dict[str, Any]:
    common_orders = sorted(set(truth_orders).intersection(recovered_orders))
    if not common_orders:
        return {
            "status": "failed",
            "value": None,
            "message": "No common kernel orders are available for KNMSE.",
            "per_order": {},
        }
    numerator = 0.0
    denominator = 0.0
    per_order: dict[str, Any] = {}
    for order in common_orders:
        truth = np.asarray(truth_orders[order], dtype=np.float64)
        recovered = np.asarray(recovered_orders[order], dtype=np.float64)
        if truth.shape != recovered.shape:
            return {
                "status": "failed",
                "value": None,
                "message": (
                    f"Kernel shape mismatch at order {order}: truth {tuple(truth.shape)} vs "
                    f"recovered {tuple(recovered.shape)}."
                ),
                "per_order": per_order,
            }
        diff_norm_sq = float(np.linalg.norm(recovered - truth) ** 2)
        truth_norm_sq = float(np.linalg.norm(truth) ** 2)
        numerator += diff_norm_sq
        denominator += truth_norm_sq
        per_order[str(order)] = {
            "nmse": None if truth_norm_sq <= 0.0 else diff_norm_sq / truth_norm_sq,
            "truth_norm_sq": truth_norm_sq,
            "diff_norm_sq": diff_norm_sq,
        }
    if denominator <= 0.0:
        return {
            "status": "failed",
            "value": None,
            "message": "Kernel truth norm is zero, cannot compute KNMSE.",
            "per_order": per_order,
        }
    return {
        "status": "completed",
        "value": numerator / denominator,
        "message": "KNMSE computed over the common dense kernel orders.",
        "per_order": per_order,
    }


def compute_gfrf_re(truth_orders: Mapping[int, np.ndarray], recovered_orders: Mapping[int, np.ndarray]) -> dict[str, Any]:
    common_orders = sorted(set(truth_orders).intersection(recovered_orders))
    if not common_orders:
        return {
            "status": "failed",
            "value": None,
            "message": "No common GFRF orders are available for GFRF-RE.",
            "per_order": {},
        }
    numerator = 0.0
    denominator = 0.0
    per_order: dict[str, Any] = {}
    for order in common_orders:
        truth = np.asarray(truth_orders[order])
        recovered = np.asarray(recovered_orders[order])
        if truth.shape != recovered.shape:
            return {
                "status": "failed",
                "value": None,
                "message": (
                    f"GFRF shape mismatch at order {order}: truth {tuple(truth.shape)} vs "
                    f"recovered {tuple(recovered.shape)}."
                ),
                "per_order": per_order,
            }
        diff_norm_sq = float(np.linalg.norm(recovered - truth) ** 2)
        truth_norm_sq = float(np.linalg.norm(truth) ** 2)
        numerator += diff_norm_sq
        denominator += truth_norm_sq
        per_order[str(order)] = {
            "re": None if truth_norm_sq <= 0.0 else math.sqrt(diff_norm_sq / truth_norm_sq),
            "truth_norm_sq": truth_norm_sq,
            "diff_norm_sq": diff_norm_sq,
        }
    if denominator <= 0.0:
        return {
            "status": "failed",
            "value": None,
            "message": "GFRF truth norm is zero, cannot compute GFRF-RE.",
            "per_order": per_order,
        }
    return {
        "status": "completed",
        "value": math.sqrt(numerator / denominator),
        "message": "GFRF-RE computed over the common dense GFRF orders.",
        "per_order": per_order,
    }


def _metric_payload(metric_name: str, *, status: str, value: Optional[float], message: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "metric_name": metric_name,
        "status": status,
        "value": value,
        "message": message,
    }
    payload.update(extra)
    return payload


def _recovery_kwargs(method_cfg: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = method_cfg.get("recovery_kwargs", {})
    if not isinstance(kwargs, Mapping):
        return {}
    return dict(kwargs)


def _method_config(method_name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    methods_cfg = dict(cfg.get("methods", {}))
    method_cfg = dict(methods_cfg.get(method_name, {}))
    raw = method_cfg.get("method_config", {})
    if not isinstance(raw, Mapping):
        return {}
    return dict(raw)


def instantiate_method(method_name: str, cfg: Mapping[str, Any]) -> Any:
    method_cls = get_method_class(method_name)
    config = _method_config(method_name, cfg)
    if method_name == "devo":
        return method_cls(config=DeVoConfig(**config))
    return method_cls(config=config)


def _slice_sizes(dataset_name: str, cfg: Mapping[str, Any], smoke: bool) -> Optional[dict[str, int]]:
    if not smoke:
        return None
    smoke_cfg = dict(cfg.get("smoke", {}))
    dataset_cfg = dict(cfg.get("datasets", {})).get(dataset_name, {})
    dataset_smoke = dict(dataset_cfg.get("smoke", {})) if isinstance(dataset_cfg, Mapping) else {}
    merged = {
        "train_size": smoke_cfg.get("train_size"),
        "val_size": smoke_cfg.get("val_size"),
        "test_size": smoke_cfg.get("test_size"),
    }
    merged.update({key: dataset_smoke.get(key, merged[key]) for key in merged})
    if all(value is None for value in merged.values()):
        return None
    return {key: int(value) for key, value in merged.items() if value is not None}


def load_bundle(dataset_name: str, cfg: Mapping[str, Any], *, smoke: bool) -> Any:
    dataset_cfg = dict(cfg.get("datasets", {})).get(dataset_name)
    if not isinstance(dataset_cfg, Mapping):
        raise KeyError(f"Dataset '{dataset_name}' is not configured.")
    bundle_path = dataset_cfg.get("bundle")
    if not bundle_path:
        raise ValueError(f"Dataset '{dataset_name}' is missing bundle path in config.")
    bundle = load_processed_dataset_bundle(_resolve_path(bundle_path))
    sizes = _slice_sizes(dataset_name, cfg, smoke)
    if sizes:
        bundle = slice_dataset_bundle(bundle, **sizes)
    return bundle


def _overall_status(*, recovery_status: str, metric_status: str) -> str:
    if recovery_status == "skipped":
        return "skipped"
    if recovery_status == "failed":
        return "failed"
    if metric_status == "completed":
        return "completed"
    if metric_status in {"failed", "skipped"}:
        return "warning"
    return "warning"


def _save_normalized_kernels(root: Path, normalized: NormalizedKernelSet) -> dict[str, Optional[str]]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "recovered_kernels_manifest.json"
    npz_path = root / "recovered_kernels.npz"
    npz_payload = {f"order_{order}": tensor for order, tensor in sorted(normalized.dense_orders.items())}
    if normalized.bias is not None:
        npz_payload["bias"] = np.asarray(normalized.bias, dtype=np.float64)
    np.savez_compressed(npz_path, **npz_payload)
    _write_json(
        manifest_path,
        {
            "representation": normalized.representation,
            "orders": {str(order): list(tensor.shape) for order, tensor in sorted(normalized.dense_orders.items())},
            "bias_shape": None if normalized.bias is None else list(np.asarray(normalized.bias).shape),
            "warnings": list(normalized.warnings),
            "source_summary": dict(normalized.source_summary),
            "npz_path": str(npz_path),
        },
    )
    return {
        "recovered_kernel_manifest": str(manifest_path),
        "recovered_kernel_npz": str(npz_path),
    }


def _save_gfrf(root: Path, gfrf_orders: Mapping[int, np.ndarray]) -> dict[str, Optional[str]]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "recovered_gfrf_manifest.json"
    npz_path = root / "recovered_gfrf.npz"
    np.savez_compressed(npz_path, **{f"order_{order}": value for order, value in sorted(gfrf_orders.items())})
    _write_json(
        manifest_path,
        {
            "orders": {
                str(order): {
                    "shape": list(np.asarray(value).shape),
                    "dtype": str(np.asarray(value).dtype),
                }
                for order, value in sorted(gfrf_orders.items())
            },
            "npz_path": str(npz_path),
        },
    )
    return {
        "recovered_gfrf_manifest": str(manifest_path),
        "recovered_gfrf_npz": str(npz_path),
    }


def run_one(
    *,
    dataset_name: str,
    method_name: str,
    cfg: Mapping[str, Any],
    bundle: Any,
    run_root: Path,
) -> dict[str, Any]:
    result_root = run_root / dataset_name / method_name
    result_root.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_name": dataset_name,
        "dataset_display_name": DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name),
        "method_name": method_name,
        "method_display_name": METHOD_DISPLAY_NAMES.get(method_name, method_name),
        "metric_name": METRIC_BY_DATASET[dataset_name],
        "supports_kernel_recovery": None,
        "status": "failed",
        "recovery_status": "failed",
        "metric_status": "failed",
        "warnings": [],
        "errors": [],
        "artifacts": {
            "truth_reference_path": None,
            "truth_payload_path": None,
            "recovered_kernel_manifest": None,
            "recovered_kernel_npz": None,
            "recovered_gfrf_manifest": None,
            "recovered_gfrf_npz": None,
            "method_exports_dir": None,
        },
        "metrics": {},
        "training_summary": {},
        "run_root": str(run_root),
        "result_root": str(result_root),
        "smoke_mode": bool(cfg.get("_runtime_smoke", False)),
    }

    try:
        method = instantiate_method(method_name, cfg)
        record["supports_kernel_recovery"] = bool(method.supports_kernel_recovery())
        if not record["supports_kernel_recovery"]:
            record["status"] = "skipped"
            record["recovery_status"] = "skipped"
            record["metric_status"] = "skipped"
            record["warnings"].append("Method does not support kernel recovery and was skipped explicitly.")
            record["metrics"][METRIC_BY_DATASET[dataset_name]] = _metric_payload(
                METRIC_BY_DATASET[dataset_name],
                status="skipped",
                value=None,
                message="Method does not support kernel recovery.",
            )
            return record

        fit_result = method.fit(bundle)
        record["training_summary"] = dict(getattr(fit_result, "training_summary", {}) or {})

        exports_dir = result_root / "method_exports"
        try:
            method.export_artifacts(exports_dir)
            record["artifacts"]["method_exports_dir"] = str(exports_dir)
        except Exception as exc:
            record["warnings"].append(f"export_artifacts() failed: {exc}")

        recovery = method.recover_kernels(**_recovery_kwargs(dict(cfg.get("methods", {})).get(method_name, {})))
        normalized = normalize_recovery(method_name, recovery)
        record["recovery_status"] = "completed"
        record["warnings"].extend(normalized.warnings)
        record["artifacts"].update(_save_normalized_kernels(result_root, normalized))

        metric_name = METRIC_BY_DATASET[dataset_name]
        if not normalized.dense_orders:
            metric_payload = _metric_payload(
                metric_name,
                status="failed",
                value=None,
                message="Recovered kernels could not be normalized into dense Volterra tensors.",
            )
            record["metrics"][metric_name] = metric_payload
            record["metric_status"] = metric_payload["status"]
            record["status"] = _overall_status(
                recovery_status=record["recovery_status"],
                metric_status=record["metric_status"],
            )
            return record

        if dataset_name == "volterra_wiener":
            truth = load_truth_kernel(bundle)
            record["artifacts"]["truth_reference_path"] = truth.get("reference_path")
            record["artifacts"]["truth_payload_path"] = truth.get("resolved_payload_path")
            if truth["status"] != "completed":
                metric_payload = _metric_payload(
                    metric_name,
                    status="failed",
                    value=None,
                    message=truth["message"],
                )
            else:
                metric_payload = compute_knmse(truth["orders"], normalized.dense_orders)
                metric_payload["metric_name"] = metric_name
            record["metrics"][metric_name] = metric_payload
            record["metric_status"] = metric_payload["status"]

        elif dataset_name == "duffing":
            gfrf_cfg = dict(cfg.get("gfrf", {}))
            recovered_gfrf = kernels_to_gfrf(
                normalized.dense_orders,
                nfft=int(gfrf_cfg.get("nfft", 128)),
            )
            record["artifacts"].update(_save_gfrf(result_root, recovered_gfrf))

            truth_gfrf = load_truth_gfrf(bundle)
            record["artifacts"]["truth_reference_path"] = truth_gfrf.get("reference_path")
            record["artifacts"]["truth_payload_path"] = truth_gfrf.get("resolved_payload_path")
            if truth_gfrf["status"] != "completed":
                truth_kernel = load_truth_kernel(bundle)
                if truth_kernel["status"] == "completed":
                    record["warnings"].append(
                        "Direct Duffing GFRF truth is unavailable; using truth kernels to derive GFRF truth."
                    )
                    truth_gfrf = {
                        "status": "completed",
                        "reference_path": truth_kernel.get("reference_path"),
                        "resolved_payload_path": truth_kernel.get("resolved_payload_path"),
                        "message": "Derived GFRF truth from truth kernels.",
                        "orders": kernels_to_gfrf(
                            truth_kernel["orders"],
                            nfft=int(gfrf_cfg.get("nfft", 128)),
                        ),
                    }
                    record["artifacts"]["truth_reference_path"] = truth_gfrf.get("reference_path")
                    record["artifacts"]["truth_payload_path"] = truth_gfrf.get("resolved_payload_path")

            if truth_gfrf["status"] != "completed":
                metric_payload = _metric_payload(
                    metric_name,
                    status="failed",
                    value=None,
                    message=truth_gfrf["message"],
                )
            else:
                metric_payload = compute_gfrf_re(truth_gfrf["orders"], recovered_gfrf)
                metric_payload["metric_name"] = metric_name
            record["metrics"][metric_name] = metric_payload
            record["metric_status"] = metric_payload["status"]

        record["status"] = _overall_status(
            recovery_status=record["recovery_status"],
            metric_status=record["metric_status"],
        )
        return record

    except Exception as exc:
        record["status"] = "failed"
        record["recovery_status"] = "failed"
        record["metric_status"] = "failed"
        record["errors"].append(f"{type(exc).__name__}: {exc}")
        record["traceback"] = traceback.format_exc()
        metric_name = METRIC_BY_DATASET[dataset_name]
        record["metrics"][metric_name] = _metric_payload(
            metric_name,
            status="failed",
            value=None,
            message=f"Experiment failed before metric computation: {exc}",
        )
        return record


def run_experiment(
    *,
    cfg: Mapping[str, Any],
    datasets: list[str],
    methods: list[str],
    smoke: bool,
    run_name: Optional[str] = None,
) -> Path:
    results_cfg = dict(cfg.get("results", {}))
    results_root = _resolve_path(results_cfg.get("root", experiment_root() / "results"))
    run_dir = results_root / (run_name or _timestamp())
    run_dir.mkdir(parents=True, exist_ok=True)

    runtime_cfg = dict(cfg)
    runtime_cfg["_runtime_smoke"] = bool(smoke)

    bundles = {dataset_name: load_bundle(dataset_name, runtime_cfg, smoke=smoke) for dataset_name in datasets}
    records: list[dict[str, Any]] = []
    for dataset_name in datasets:
        for method_name in methods:
            record = run_one(
                dataset_name=dataset_name,
                method_name=method_name,
                cfg=runtime_cfg,
                bundle=bundles[dataset_name],
                run_root=run_dir,
            )
            records.append(record)
            _write_json(run_dir / dataset_name / method_name / "result.json", record)

    _write_json(
        run_dir / "run_manifest.json",
        {
            "experiment_name": EXPERIMENT_NAME,
            "generated_at": datetime.now().isoformat(),
            "datasets": datasets,
            "methods": methods,
            "smoke": bool(smoke),
            "records": [
                {
                    "dataset_name": item["dataset_name"],
                    "method_name": item["method_name"],
                    "status": item["status"],
                    "metric_status": item["metric_status"],
                }
                for item in records
            ],
        },
    )
    _write_json(run_dir / "results.json", {"records": records})
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nonlinear benchmark exp02 kernel recovery.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(experiment_root() / "config.yaml"),
        help="Path to the experiment config YAML.",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional explicit run directory name.")
    parser.add_argument("--smoke", action="store_true", help="Run on the configured smoke slices.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset subset. Defaults to config order.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method subset. Defaults to config order.",
    )
    parser.add_argument(
        "--print-run-dir",
        action="store_true",
        help="Print the output run directory after completion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(_resolve_path(args.config))

    configured_datasets = {
        name for name, dataset_cfg in dict(cfg.get("datasets", {})).items() if isinstance(dataset_cfg, Mapping)
    }
    configured_methods = {
        name
        for name, method_cfg in dict(cfg.get("methods", {})).items()
        if isinstance(method_cfg, Mapping) and bool(method_cfg.get("enabled", True))
    }

    datasets = _selected_names(args.datasets, default_order=DATASET_ORDER, available=configured_datasets)
    methods = _selected_names(args.methods, default_order=METHOD_ORDER, available=configured_methods)
    if not datasets:
        raise SystemExit("No datasets selected.")
    if not methods:
        raise SystemExit("No methods selected.")

    run_dir = run_experiment(
        cfg=cfg,
        datasets=datasets,
        methods=methods,
        smoke=bool(args.smoke),
        run_name=args.run_name,
    )
    if args.print_run_dir:
        print(run_dir)


if __name__ == "__main__":
    main()
