"""Run nonlinear benchmark experiment 03: ablation and hyperparameter analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - script dependency guard.
    raise RuntimeError("PyYAML is required to run experiment scripts.") from exc

from data.adapters.base import DatasetMeta
from methods.base import MethodDatasetBundle, MethodDatasetSplit
from methods.devo import DeVoConfig, DeVoMethod
from methods.utils import load_processed_dataset_bundle, slice_dataset_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")
EPS = 1e-12


@dataclass(frozen=True)
class RunJob:
    """One concrete experiment run."""

    experiment_kind: str
    benchmark: str
    suite: str
    run_group: str
    label: str
    seed: int
    method_config: dict[str, Any]
    data_config: dict[str, Any]
    dataset_spec: dict[str, Any]
    evaluate: dict[str, Any]
    variant: Optional[str] = None
    hyperparameter: Optional[str] = None
    hyperparameter_value: Any = None
    sweep_label: Optional[str] = None

    @property
    def run_id(self) -> str:
        parts = [
            self.experiment_kind,
            self.benchmark,
            self.run_group,
            f"seed{self.seed}",
        ]
        return "__".join(_safe_slug(part) for part in parts if part)


def _safe_slug(value: Any) -> str:
    text = str(value).strip().lower()
    cleaned = []
    for char in text:
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "item"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config at {path} must be a mapping.")
    return dict(payload)


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON mapping at {path}.")
    return payload


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=True)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _config_output_root(config: Mapping[str, Any], *, base_dir: Path) -> Path:
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment config must be a mapping.")
    output_root = experiment.get("output_root", "results/nonlinear/exp03_ablation_hparam")
    return _resolve_path(str(output_root), base_dir=base_dir)


def _build_method_config(
    *,
    base_config: Mapping[str, Any],
    suite_overrides: Mapping[str, Any],
    variant_overrides: Mapping[str, Any],
    sweep_overrides: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    merged = _deep_merge(base_config, suite_overrides)
    merged = _deep_merge(merged, variant_overrides)
    merged = _deep_merge(merged, sweep_overrides)
    merged["seed"] = int(seed)
    return merged


def _load_bundle(dataset_spec: Mapping[str, Any], *, base_dir: Path) -> MethodDatasetBundle:
    manifest = dataset_spec.get("manifest")
    if not isinstance(manifest, str):
        raise ValueError("dataset manifest must be configured as a string path.")
    return load_processed_dataset_bundle(_resolve_path(manifest, base_dir=base_dir))


def _clone_split_with_payload(split: MethodDatasetSplit, *, X: Any) -> MethodDatasetSplit:
    return MethodDatasetSplit(
        X=X,
        Y=split.Y,
        sample_id=split.sample_id,
        run_id=split.run_id,
        timestamp=split.timestamp,
        meta=dict(split.meta),
        extra_fields=dict(split.extra_fields),
    )


def _apply_memory_depth(bundle: MethodDatasetBundle, memory_depth: Optional[int]) -> MethodDatasetBundle:
    if memory_depth is None:
        return bundle
    memory_depth = int(memory_depth)
    if memory_depth <= 0:
        raise ValueError("memory_depth must be positive.")
    if memory_depth > int(bundle.meta.window_length):
        raise ValueError(
            f"memory_depth={memory_depth} exceeds bundle window_length={bundle.meta.window_length}."
        )
    if memory_depth == int(bundle.meta.window_length):
        return bundle

    def crop(split: MethodDatasetSplit) -> MethodDatasetSplit:
        array = np.asarray(split.X)
        if array.ndim < 2:
            raise ValueError("Expected windowed X with at least 2 dimensions.")
        cropped = array[:, -memory_depth:, ...]
        return _clone_split_with_payload(split, X=cropped)

    meta_payload = bundle.meta.to_dict()
    meta_payload["window_length"] = memory_depth
    meta = DatasetMeta.from_mapping(meta_payload)
    return MethodDatasetBundle(
        train=crop(bundle.train),
        val=crop(bundle.val),
        test=crop(bundle.test),
        meta=meta,
        artifacts=bundle.artifacts,
        source_manifest=bundle.source_manifest,
        source_root=bundle.source_root,
    )


def _prepare_bundle(bundle: MethodDatasetBundle, *, suite_spec: Mapping[str, Any], data_config: Mapping[str, Any]) -> MethodDatasetBundle:
    slice_spec = suite_spec.get("slice", {})
    if isinstance(slice_spec, Mapping):
        bundle = slice_dataset_bundle(
            bundle,
            train_size=_maybe_int(slice_spec.get("train")),
            val_size=_maybe_int(slice_spec.get("val")),
            test_size=_maybe_int(slice_spec.get("test")),
        )
    return _apply_memory_depth(bundle, _maybe_int(data_config.get("memory_depth")))


def _maybe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _predict_nmse(y_true: Any, y_pred: Any) -> float:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError(f"NMSE shape mismatch: truth={truth.shape}, pred={pred.shape}.")
    denom = float(np.var(truth))
    if denom <= EPS:
        denom = float(np.mean(truth ** 2))
    denom = max(denom, EPS)
    return float(np.mean((pred - truth) ** 2) / denom)


def _metric_payload(value: Optional[float], *, status: str, notes: Optional[str] = None) -> dict[str, Any]:
    return {
        "value": None if value is None or not np.isfinite(value) else float(value),
        "status": status,
        "notes": notes,
    }


def _reference_path_candidates(bundle: MethodDatasetBundle, reference_type: str) -> list[Path]:
    candidates: list[Optional[str]] = []
    extras = dict(getattr(bundle.artifacts, "extra", {}) or {})
    meta_extras = dict(getattr(bundle.meta, "extras", {}) or {})
    truth_artifacts = meta_extras.get("truth_artifacts", {})
    if isinstance(truth_artifacts, Mapping):
        candidates.append(truth_artifacts.get(f"{reference_type}_reference_file"))
    candidates.append(extras.get(f"{reference_type}_reference_file"))
    if reference_type == "gfrf":
        candidates.append(bundle.artifacts.truth_file)
    elif reference_type == "kernel":
        candidates.append(bundle.artifacts.truth_file)

    resolved: list[Path] = []
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if path.exists() and path not in resolved:
            resolved.append(path)
    return resolved


def _load_json_or_array(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    raise ValueError(f"Unsupported reference file type: {path}")


def _load_array_like(value: Any, *, base_dir: Path) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if isinstance(value, str):
        path = Path(value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            return None
        payload = _load_json_or_array(path)
        if isinstance(payload, Mapping):
            return None
        return np.asarray(payload)
    return None


def _payload_contains_material_reference(payload: Mapping[str, Any]) -> bool:
    material_keys = {
        "orders",
        "kernel_coefficients_path",
        "kernel_metadata_path",
        "gfrf_path",
        "full_tensor",
        "symmetric_coefficients",
        "effective_coefficients",
        "coefficients",
    }
    return any(key in payload for key in material_keys)


def _load_reference_payload(bundle: MethodDatasetBundle, reference_type: str) -> tuple[Optional[Any], str, Path]:
    paths = _reference_path_candidates(bundle, reference_type)
    if not paths:
        return None, "reference_missing", PROJECT_ROOT
    for path in paths:
        payload = _load_json_or_array(path)
        if isinstance(payload, Mapping):
            status = str(payload.get("status", "")).strip().lower()
            if status in {"not_available", "placeholder_registered", "registered"} and not _payload_contains_material_reference(payload):
                continue
        return payload, "ok", path.parent
    return None, "reference_unavailable", PROJECT_ROOT


def _lookup_order_entry(payload: Any, order: int) -> Optional[Any]:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        orders = payload.get("orders")
        if isinstance(orders, Mapping):
            if str(order) in orders:
                return orders[str(order)]
            if order in orders:
                return orders[order]
        if isinstance(orders, list):
            for item in orders:
                if isinstance(item, Mapping) and int(item.get("order", -1)) == int(order):
                    return item
        for key in (f"order_{order}", f"order{order}"):
            if key in payload:
                return payload[key]
    return None


def _flat_full_tensor_shape(order_payload: Mapping[str, Any]) -> tuple[int, ...]:
    response_dim = int(order_payload["horizon"]) * int(order_payload["output_dim"])
    flattened_dim = int(order_payload["window_length"]) * int(order_payload["input_dim"])
    order = int(order_payload["order"])
    return (response_dim,) + (flattened_dim,) * order


def _full_tensor_to_canonical_symmetric(reference_full_tensor: Any, order_payload: Mapping[str, Any]) -> Optional[np.ndarray]:
    full = np.asarray(reference_full_tensor)
    structured_shape = tuple(int(value) for value in order_payload["full_tensor_shape"])
    flat_shape = _flat_full_tensor_shape(order_payload)
    if tuple(full.shape) == structured_shape:
        full = full.reshape(flat_shape)
    elif tuple(full.shape) != flat_shape:
        return None

    canonical_indices = np.asarray(order_payload["canonical_indices"], dtype=np.int64)
    gathered = np.empty((flat_shape[0], canonical_indices.shape[0]), dtype=np.float64)
    for feature_idx, tuple_indices in enumerate(canonical_indices):
        gathered[:, feature_idx] = full[(slice(None),) + tuple(int(value) for value in tuple_indices)]
    horizon = int(order_payload["horizon"])
    output_dim = int(order_payload["output_dim"])
    return gathered.reshape(horizon, output_dim, canonical_indices.shape[0])


def _extract_kernel_reference_order(payload: Any, order_payload: Mapping[str, Any], *, base_dir: Path) -> Optional[np.ndarray]:
    if isinstance(payload, np.ndarray):
        return _full_tensor_to_canonical_symmetric(payload, order_payload)
    entry = _lookup_order_entry(payload, int(order_payload["order"]))
    if entry is None and isinstance(payload, Mapping):
        entry = payload
    if isinstance(entry, Mapping):
        for key in ("symmetric_coefficients", "coefficients", "effective_coefficients"):
            array = _load_array_like(entry.get(key), base_dir=base_dir)
            if array is not None:
                return np.asarray(array, dtype=np.float64)
        full_tensor = _load_array_like(entry.get("full_tensor"), base_dir=base_dir)
        if full_tensor is not None:
            return _full_tensor_to_canonical_symmetric(full_tensor, order_payload)
    return None


def _compute_knmse(recovery_payload: Mapping[str, Any], bundle: MethodDatasetBundle) -> dict[str, Any]:
    payload, status, base_dir = _load_reference_payload(bundle, "kernel")
    if payload is None:
        return _metric_payload(None, status=status, notes="No numeric kernel reference available.")
    estimates: list[np.ndarray] = []
    references: list[np.ndarray] = []
    missing_orders: list[int] = []
    for order_key, order_payload in sorted(recovery_payload["orders"].items(), key=lambda item: int(item[0])):
        reference = _extract_kernel_reference_order(payload, order_payload, base_dir=base_dir)
        if reference is None:
            missing_orders.append(int(order_key))
            continue
        estimate = np.asarray(order_payload["symmetric_coefficients"], dtype=np.float64)
        if estimate.shape != reference.shape:
            return _metric_payload(
                None,
                status="shape_mismatch",
                notes=f"order {order_key}: estimate {estimate.shape} vs reference {reference.shape}",
            )
        estimates.append(estimate.reshape(-1))
        references.append(reference.reshape(-1))
    if not estimates or missing_orders:
        notes = "Missing numeric kernel reference orders: " + ", ".join(str(order) for order in missing_orders)
        return _metric_payload(None, status="reference_incomplete", notes=notes)
    estimate = np.concatenate(estimates)
    reference = np.concatenate(references)
    denominator = float(np.sum(reference ** 2))
    value = float(np.sum((estimate - reference) ** 2) / max(denominator, EPS))
    return _metric_payload(value, status="ok")


def _extract_siso_kernel(full_tensor: Any, order_payload: Mapping[str, Any]) -> Optional[np.ndarray]:
    array = np.asarray(full_tensor, dtype=np.float64)
    if tuple(array.shape) != tuple(int(value) for value in order_payload["full_tensor_shape"]):
        return None
    if int(order_payload["horizon"]) != 1 or int(order_payload["output_dim"]) != 1 or int(order_payload["input_dim"]) != 1:
        return None
    array = array[0, 0]
    order = int(order_payload["order"])
    for axis in reversed(range(1, 2 * order, 2)):
        if array.shape[axis] != 1:
            return None
        array = np.take(array, 0, axis=axis)
    return array


def _kernel_order_to_gfrf(order_payload: Mapping[str, Any]) -> Optional[np.ndarray]:
    full_tensor = order_payload.get("full_tensor")
    if full_tensor is None:
        return None
    kernel = _extract_siso_kernel(full_tensor, order_payload)
    if kernel is None:
        return None
    return np.fft.fftn(kernel, axes=tuple(range(kernel.ndim)))


def _extract_gfrf_reference_order(payload: Any, order_payload: Mapping[str, Any], *, base_dir: Path) -> Optional[np.ndarray]:
    entry = _lookup_order_entry(payload, int(order_payload["order"]))
    if entry is None and isinstance(payload, Mapping):
        entry = payload
    if isinstance(entry, Mapping):
        real = _load_array_like(entry.get("real"), base_dir=base_dir)
        imag = _load_array_like(entry.get("imag"), base_dir=base_dir)
        if real is not None and imag is not None and real.shape == imag.shape:
            return np.asarray(real, dtype=np.float64) + 1j * np.asarray(imag, dtype=np.float64)
        for key in ("gfrf", "response", "coefficients"):
            array = _load_array_like(entry.get(key), base_dir=base_dir)
            if array is not None:
                return np.asarray(array)
        full_tensor = _load_array_like(entry.get("full_tensor"), base_dir=base_dir)
        if full_tensor is not None:
            kernel = _extract_siso_kernel(full_tensor, order_payload)
            if kernel is not None:
                return np.fft.fftn(kernel, axes=tuple(range(kernel.ndim)))
    return None


def _compute_gfrf_re(recovery_payload: Mapping[str, Any], bundle: MethodDatasetBundle) -> dict[str, Any]:
    reference_payload, status, gfrf_base_dir = _load_reference_payload(bundle, "gfrf")
    kernel_reference_payload = None
    kernel_base_dir = PROJECT_ROOT
    if reference_payload is None:
        kernel_reference_payload, _, kernel_base_dir = _load_reference_payload(bundle, "kernel")
        if kernel_reference_payload is None:
            return _metric_payload(None, status=status, notes="No numeric GFRF or kernel reference available.")

    errors: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    for order_key, order_payload in sorted(recovery_payload["orders"].items(), key=lambda item: int(item[0])):
        estimate = _kernel_order_to_gfrf(order_payload)
        if estimate is None:
            return _metric_payload(
                None,
                status="full_tensor_missing",
                notes=f"Order {order_key} does not have a materialized full tensor for GFRF.",
            )
        reference = None
        if reference_payload is not None:
            reference = _extract_gfrf_reference_order(reference_payload, order_payload, base_dir=gfrf_base_dir)
        if reference is None and kernel_reference_payload is not None:
            kernel_reference = _extract_kernel_reference_order(kernel_reference_payload, order_payload, base_dir=kernel_base_dir)
            if kernel_reference is not None:
                full_tensor = _full_tensor_from_canonical(kernel_reference, order_payload)
                if full_tensor is not None:
                    reference = np.fft.fftn(full_tensor, axes=tuple(range(full_tensor.ndim)))
        if reference is None:
            return _metric_payload(None, status="reference_incomplete", notes=f"Missing GFRF reference for order {order_key}.")
        if estimate.shape != reference.shape:
            return _metric_payload(
                None,
                status="shape_mismatch",
                notes=f"order {order_key}: estimate {estimate.shape} vs reference {reference.shape}",
            )
        errors.append((estimate - reference).reshape(-1))
        refs.append(reference.reshape(-1))
    estimate_error = np.concatenate(errors)
    reference = np.concatenate(refs)
    denominator = float(np.sum(np.abs(reference) ** 2))
    value = float(np.sum(np.abs(estimate_error) ** 2) / max(denominator, EPS))
    return _metric_payload(math.sqrt(value), status="ok")


def _full_tensor_from_canonical(canonical_coefficients: np.ndarray, order_payload: Mapping[str, Any]) -> Optional[np.ndarray]:
    coefficients = np.asarray(canonical_coefficients, dtype=np.float64)
    if coefficients.shape != tuple(np.asarray(order_payload["symmetric_coefficients"]).shape):
        return None
    horizon = int(order_payload["horizon"])
    output_dim = int(order_payload["output_dim"])
    flattened_dim = int(order_payload["window_length"]) * int(order_payload["input_dim"])
    order = int(order_payload["order"])
    flat = np.zeros((horizon * output_dim,) + (flattened_dim,) * order, dtype=np.float64)
    coefficient_rows = coefficients.reshape(horizon * output_dim, -1)
    for feature_idx, tuple_indices in enumerate(np.asarray(order_payload["canonical_indices"], dtype=np.int64)):
        value = coefficient_rows[:, feature_idx]
        for permutation in set(tuple(item) for item in itertools.permutations(tuple_indices.tolist())):
            flat[(slice(None),) + permutation] = value
    return _extract_siso_kernel(flat.reshape(tuple(int(value) for value in order_payload["full_tensor_shape"])), order_payload)


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for name, payload in metrics.items():
        if not isinstance(payload, Mapping):
            continue
        flat[f"{name}_value"] = payload.get("value")
        flat[f"{name}_status"] = payload.get("status")
        flat[f"{name}_notes"] = payload.get("notes")
    return flat


def _history_has_nonfinite(history: Iterable[Mapping[str, Any]]) -> bool:
    for item in history:
        for key in ("train_loss", "val_loss"):
            value = item.get(key)
            if value is None:
                continue
            if not np.isfinite(float(value)):
                return True
    return False


def _run_job(job: RunJob, *, output_root: Path, suite_spec: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    run_dir = output_root / job.suite / "runs" / job.experiment_kind / job.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = _prepare_bundle(_load_bundle(job.dataset_spec, base_dir=base_dir), suite_spec=suite_spec, data_config=job.data_config)
    effective_method_config = dict(job.method_config)
    effective_method_config["window_length"] = int(bundle.meta.window_length)

    payload = {
        "run_id": job.run_id,
        "experiment_kind": job.experiment_kind,
        "benchmark": job.benchmark,
        "suite": job.suite,
        "label": job.label,
        "variant": job.variant,
        "hyperparameter": job.hyperparameter,
        "hyperparameter_value": job.hyperparameter_value,
        "sweep_label": job.sweep_label,
        "seed": int(job.seed),
        "bundle_window_length": int(bundle.meta.window_length),
        "bundle_source_manifest": bundle.source_manifest,
        "method_config": effective_method_config,
        "data_config": dict(job.data_config),
        "evaluate": dict(job.evaluate),
    }
    _write_json(run_dir / "resolved_config.json", payload)

    fit_seconds = None
    exception_message = None
    predictions = None
    recovery = None
    method: Optional[DeVoMethod] = None
    fit_summary: dict[str, Any] = {}
    training_history: list[dict[str, Any]] = []
    started_at = time.time()
    try:
        method = DeVoMethod(config=DeVoConfig(**job.method_config))
        method.fit(bundle)
        fit_seconds = time.time() - started_at
        fit_summary = dict(method.training_summary)
        training_history = list(method.training_history)
        needs_predict = bool(job.evaluate.get("predict")) or bool(job.evaluate.get("compute_nmse"))
        needs_recover = bool(job.evaluate.get("recover_kernels")) or bool(job.evaluate.get("compute_knmse")) or bool(job.evaluate.get("compute_gfrf_re"))
        if method.is_fitted and needs_predict:
            predictions = method.predict(bundle.test.X)
        if method.is_fitted and needs_recover:
            recovery_result = method.recover_kernels(
                materialize_full=bool(job.evaluate.get("compute_gfrf_re")),
                max_full_elements=int(job.method_config.get("max_full_recovery_elements", 4_000_000)),
            )
            recovery = recovery_result.kernels
    except Exception as exc:  # pragma: no cover - experiment runtime guard.
        fit_seconds = time.time() - started_at
        exception_message = f"{exc.__class__.__name__}: {exc}"

    metrics: dict[str, Any] = {}
    if exception_message is not None:
        metrics["nmse"] = _metric_payload(None, status="fit_failed", notes=exception_message)
        metrics["knmse"] = _metric_payload(None, status="fit_failed", notes=exception_message)
        metrics["gfrf_re"] = _metric_payload(None, status="fit_failed", notes=exception_message)
    else:
        if bool(job.evaluate.get("compute_nmse")) and predictions is not None:
            metrics["nmse"] = _metric_payload(_predict_nmse(bundle.test.Y, predictions), status="ok")
        elif bool(job.evaluate.get("compute_nmse")):
            metrics["nmse"] = _metric_payload(None, status="training_not_converged")
        else:
            metrics["nmse"] = _metric_payload(None, status="skipped")

        if bool(job.evaluate.get("compute_knmse")) and recovery is not None:
            metrics["knmse"] = _compute_knmse(recovery, bundle)
        elif bool(job.evaluate.get("compute_knmse")):
            metrics["knmse"] = _metric_payload(None, status="training_not_converged")
        else:
            metrics["knmse"] = _metric_payload(None, status="skipped")

        if bool(job.evaluate.get("compute_gfrf_re")) and recovery is not None:
            metrics["gfrf_re"] = _compute_gfrf_re(recovery, bundle)
        elif bool(job.evaluate.get("compute_gfrf_re")):
            metrics["gfrf_re"] = _metric_payload(None, status="training_not_converged")
        else:
            metrics["gfrf_re"] = _metric_payload(None, status="skipped")

    stability = {
        "success": bool(exception_message is None and method is not None and method.is_fitted),
        "fit_completed": bool(exception_message is None),
        "converged": bool(fit_summary.get("converged", False)),
        "had_nonfinite": bool(fit_summary.get("had_nonfinite", False) or _history_has_nonfinite(training_history)),
        "early_stopped": bool(fit_summary.get("early_stopped", False)),
        "epochs_completed": int(fit_summary.get("epochs_completed", 0)) if fit_summary else 0,
        "fit_seconds": fit_seconds,
        "exception": exception_message,
    }

    order_summaries = []
    if isinstance(recovery, Mapping):
        for order_key, order_payload in sorted(recovery.get("orders", {}).items(), key=lambda item: int(item[0])):
            order_summaries.append(
                {
                    "order": int(order_key),
                    "feature_count": int(order_payload["feature_count"]),
                    "full_tensor_materialized": bool(order_payload.get("full_tensor") is not None),
                    "feature_mode": str(recovery.get("metadata", {}).get("feature_mode", effective_method_config.get("feature_mode", "canonical"))),
                }
            )

    result = {
        **payload,
        "metrics": metrics,
        "stability": stability,
        "training_summary": fit_summary,
        "recovery_summary": order_summaries,
    }
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "training_history.json", training_history)
    return result


def _ablation_jobs(config: Mapping[str, Any], *, suite: str, base_dir: Path) -> list[RunJob]:
    experiment = config["experiment"]
    datasets = config["datasets"]
    ablation = config["ablation"]
    base_method = config["method"]["base"]
    suite_overrides = config["suites"][suite].get("method_overrides", {})
    suite_data_overrides = dict(config["suites"][suite].get("data_overrides", {}))
    seeds = list(experiment.get("seeds", [7]))

    jobs: list[RunJob] = []
    for variant_name, variant_spec in ablation["variants"].items():
        method_variant_overrides = dict(variant_spec.get("method_overrides", {}))
        label = str(variant_spec.get("label", variant_name))
        for benchmark in ablation["benchmarks"]:
            dataset_spec = dict(datasets[benchmark])
            evaluate = dict(dataset_spec.get("evaluate", {}))
            for seed in seeds:
                method_config = _build_method_config(
                    base_config=base_method,
                    suite_overrides=suite_overrides,
                    variant_overrides=method_variant_overrides,
                    sweep_overrides={},
                    seed=int(seed),
                )
                jobs.append(
                    RunJob(
                        experiment_kind="ablation",
                        benchmark=str(benchmark),
                        suite=suite,
                        run_group=variant_name,
                        label=label,
                        seed=int(seed),
                        method_config=method_config,
                        data_config=dict(suite_data_overrides),
                        dataset_spec=dataset_spec,
                        evaluate=evaluate,
                        variant=variant_name,
                    )
                )
    return jobs


def _hyperparameter_setting_overrides(parameter: str, value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    method_overrides: dict[str, Any] = {}
    data_overrides: dict[str, Any] = {}
    if parameter == "kernel_order":
        max_order = int(value)
        method_overrides["orders"] = list(range(1, max_order + 1))
    elif parameter == "memory_depth":
        data_overrides["memory_depth"] = int(value)
    elif parameter == "branch_number":
        method_overrides["num_branches"] = int(value)
    else:
        raise ValueError(f"Unsupported hyperparameter sweep: {parameter}")
    return method_overrides, data_overrides


def _hyperparameter_jobs(config: Mapping[str, Any], *, suite: str, base_dir: Path) -> list[RunJob]:
    experiment = config["experiment"]
    datasets = config["datasets"]
    ablation = config["ablation"]
    hyper = config["hyperparameter"]
    base_method = config["method"]["base"]
    suite_overrides = config["suites"][suite].get("method_overrides", {})
    suite_data_overrides = dict(config["suites"][suite].get("data_overrides", {}))
    seeds = list(experiment.get("seeds", [7]))

    base_variant_name = str(hyper["base_variant"])
    base_variant_spec = dict(ablation["variants"][base_variant_name])
    base_variant_overrides = dict(base_variant_spec.get("method_overrides", {}))

    jobs: list[RunJob] = []
    for sweep_name, sweep_spec in hyper["sweeps"].items():
        parameter = str(sweep_spec.get("parameter", sweep_name))
        if suite == "smoke" and "smoke_values" in sweep_spec:
            values = list(sweep_spec["smoke_values"])
        elif "grid" in sweep_spec:
            values = list(sweep_spec["grid"])
        else:
            values = list(sweep_spec["values"])
        sweep_label = str(sweep_spec.get("label", parameter))
        for value in values:
            if isinstance(value, Mapping):
                method_sweep_overrides = dict(value.get("method_overrides", {}))
                data_sweep_overrides = dict(value.get("data_overrides", {}))
                display_value = value.get("label") or value.get(parameter)
            else:
                method_sweep_overrides, data_sweep_overrides = _hyperparameter_setting_overrides(parameter, value)
                display_value = value
            for benchmark in hyper["benchmarks"]:
                dataset_spec = dict(datasets[benchmark])
                evaluate = dict(dataset_spec.get("evaluate", {}))
                for seed in seeds:
                    method_config = _build_method_config(
                        base_config=base_method,
                        suite_overrides=suite_overrides,
                        variant_overrides=base_variant_overrides,
                        sweep_overrides=method_sweep_overrides,
                        seed=int(seed),
                    )
                    data_config = dict(suite_data_overrides)
                    data_config.update(data_sweep_overrides)
                    jobs.append(
                        RunJob(
                            experiment_kind="hyperparameter",
                            benchmark=str(benchmark),
                            suite=suite,
                            run_group=f"{sweep_name}_{display_value}",
                            label=f"{sweep_label}={display_value}",
                            seed=int(seed),
                            method_config=method_config,
                            data_config=data_config,
                            dataset_spec=dataset_spec,
                            evaluate=evaluate,
                            variant=base_variant_name,
                            hyperparameter=parameter,
                            hyperparameter_value=display_value,
                            sweep_label=sweep_label,
                        )
                    )
    return jobs


def _select_jobs(config: Mapping[str, Any], *, suite: str, mode: str, base_dir: Path) -> list[RunJob]:
    jobs: list[RunJob] = []
    if mode in {"ablation", "all"}:
        jobs.extend(_ablation_jobs(config, suite=suite, base_dir=base_dir))
    if mode in {"hyperparameter", "all"}:
        jobs.extend(_hyperparameter_jobs(config, suite=suite, base_dir=base_dir))
    return jobs


def _write_index(rows: list[Mapping[str, Any]], *, output_root: Path, suite: str) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    target = output_root / suite / "run_index.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--mode", choices=("ablation", "hyperparameter", "all"), default="all")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of jobs to run.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_yaml(config_path)
    base_dir = config_path.parent
    output_root = _config_output_root(config, base_dir=base_dir)
    suite_spec = config["suites"][args.suite]

    jobs = _select_jobs(config, suite=args.suite, mode=args.mode, base_dir=base_dir)
    if args.limit is not None:
        jobs = jobs[: int(args.limit)]

    results: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[exp03] ({index}/{len(jobs)}) {job.experiment_kind} {job.benchmark} {job.label} seed={job.seed}")
        result = _run_job(job, output_root=output_root, suite_spec=suite_spec, base_dir=base_dir)
        results.append(result)
        index_rows.append(
            {
                "run_id": result["run_id"],
                "experiment_kind": result["experiment_kind"],
                "benchmark": result["benchmark"],
                "label": result["label"],
                "variant": result.get("variant"),
                "hyperparameter": result.get("hyperparameter"),
                "hyperparameter_value": result.get("hyperparameter_value"),
                "seed": result["seed"],
                **_flatten_metrics(result["metrics"]),
                "success": result["stability"]["success"],
                "converged": result["stability"]["converged"],
                "had_nonfinite": result["stability"]["had_nonfinite"],
                "early_stopped": result["stability"]["early_stopped"],
            }
        )

    run_manifest = {
        "config": str(config_path),
        "suite": args.suite,
        "mode": args.mode,
        "job_count": len(jobs),
        "results_root": str(output_root / args.suite),
        "jobs_hash": hashlib.sha1(json.dumps([asdict(job) for job in jobs], sort_keys=True, default=str).encode("utf-8")).hexdigest(),
    }
    _write_json(output_root / args.suite / "run_manifest.json", run_manifest)
    _write_index(index_rows, output_root=output_root, suite=args.suite)


if __name__ == "__main__":
    main()
