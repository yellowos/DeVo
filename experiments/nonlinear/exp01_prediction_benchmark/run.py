from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.base import create_method, get_method_class, load_dataset_bundle
from methods.devo import DeVoConfig
from methods.utils import set_random_seed, slice_dataset_bundle


EXPERIMENT_NAME = "exp01_prediction_benchmark"
DATASET_ORDER = [
    "duffing",
    "silverbox",
    "volterra_wiener",
    "coupled_duffing",
    "cascaded_tanks",
]
METHOD_ORDER = [
    "narmax",
    "tt_volterra",
    "cp_volterra",
    "laguerre_volterra",
    "mlp",
    "lstm",
    "devo",
]
_INPUT_CONTRACT_ERROR_FRAGMENTS = (
    "lstm baseline expects sequence inputs with shape [n, m, d]",
    "dataset meta input_dim=",
    "dataset meta window_length=",
    "devo expects x with shape [n, m, d] or [n, m].",
    "expected x with shape [n, m, d] or [n, m].",
    "1d y only supports horizon=1 and output_dim=1.",
    "expected y with shape [n, h, o], [n, o], [n, h], or [n].",
    "expected y shape [n,",
    "training x must be shaped as (num_samples, window_length, input_dim).",
    "1d x is only supported for single-input models.",
    "2d x must be shaped as (window_length, input_dim) for one sample",
    "x must be 1d, 2d, or 3d.",
    "expected window_length=",
    "expected input_dim=",
    "2d x does not match fitted window_length/input_dim and cannot be reshaped safely.",
    "x window_length mismatch:",
    "x input_dim mismatch:",
    "x must be a 1d, 2d, or 3d numeric array.",
    "1d x length ",
    "does not match expected feature size ",
    "expected x trailing shape ",
    "x must have shape [n, ...].",
    "expected y trailing shape ",
    "y must have shape [n, ...].",
)
_METHOD_BATCH_KEYS: dict[str, tuple[str, ...]] = {
    "cp_volterra": ("batch_size",),
    "devo": ("batch_size",),
    "lstm": ("batch_size",),
    "mlp": ("batch_size",),
    "tt_volterra": ("batch_size",),
}
_METHOD_EVAL_BATCH_KEYS: dict[str, tuple[str, ...]] = {
    "devo": ("eval_batch_size",),
    "tt_volterra": ("predict_batch_size",),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolve_path(base_dir: Path, raw: str | None, *, fallback: str) -> Path:
    target = Path(raw or fallback)
    if not target.is_absolute():
        target = (base_dir / target).resolve()
    return target


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, ensure_ascii=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return payload


def _resolve_config(config_path: Path, profile: str) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    config_dir = config_path.parent.resolve()

    defaults = dict(raw.get("defaults", {}) or {})
    profiles = dict(raw.get("profiles", {}) or {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile '{profile}'. Available: {available}")

    profile_payload = dict(profiles[profile] or {})
    runtime_defaults = dict(raw.get("runtime", {}) or {})
    runtime_overrides = dict(profile_payload.get("runtime", {}) or {})
    runtime = {**runtime_defaults, **runtime_overrides}

    resolved = {
        "experiment_name": str(raw.get("experiment_name", EXPERIMENT_NAME)),
        "profile": profile,
        "config_path": str(config_path.resolve()),
        "processed_root": _resolve_path(
            config_dir,
            str((raw.get("paths", {}) or {}).get("processed_root", "")) or None,
            fallback="../../../data/processed/nonlinear",
        ),
        "output_dir": _resolve_path(
            config_dir,
            str(profile_payload.get("output_dir", "")) or None,
            fallback=f"./outputs/{profile}",
        ),
        "datasets": list(profile_payload.get("datasets", [])),
        "methods": list(profile_payload.get("methods", [])),
        "seeds": [int(seed) for seed in profile_payload.get("seeds", [])],
        "skip_existing": bool(profile_payload.get("skip_existing", defaults.get("skip_existing", False))),
        "save_running_state": bool(defaults.get("save_running_state", True)),
        "max_train_samples": profile_payload.get("max_train_samples", defaults.get("max_train_samples")),
        "max_val_samples": profile_payload.get("max_val_samples", defaults.get("max_val_samples")),
        "max_test_samples": profile_payload.get("max_test_samples", defaults.get("max_test_samples")),
        "runtime": runtime,
        "method_configs": dict(raw.get("method_configs", {}) or {}),
        "method_overrides": dict(profile_payload.get("method_overrides", {}) or {}),
    }
    return resolved


def _override_config_with_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = dict(config)
    if args.datasets:
        resolved["datasets"] = list(args.datasets)
    if args.methods:
        resolved["methods"] = list(args.methods)
    if args.seeds:
        resolved["seeds"] = [int(seed) for seed in args.seeds]
    if args.output_dir:
        resolved["output_dir"] = Path(args.output_dir).expanduser().resolve()
    if args.max_train_samples is not None:
        resolved["max_train_samples"] = int(args.max_train_samples)
    if args.max_val_samples is not None:
        resolved["max_val_samples"] = int(args.max_val_samples)
    if args.max_test_samples is not None:
        resolved["max_test_samples"] = int(args.max_test_samples)
    if args.force:
        resolved["skip_existing"] = False
    if args.skip_existing:
        resolved["skip_existing"] = True
    return resolved


def _validate_selection(selected: list[str], allowed: list[str], *, kind: str) -> None:
    invalid = [item for item in selected if item not in allowed]
    if invalid:
        raise ValueError(f"Unknown {kind}: {', '.join(invalid)}")


def _load_bundle(processed_root: Path, dataset_name: str, config: Mapping[str, Any]) -> Any:
    manifest_path = processed_root / dataset_name / f"{dataset_name}_processed_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Processed manifest does not exist: {manifest_path}")

    bundle = load_dataset_bundle(manifest_path)
    max_train = config.get("max_train_samples")
    max_val = config.get("max_val_samples")
    max_test = config.get("max_test_samples")
    if any(limit is not None for limit in (max_train, max_val, max_test)):
        bundle = slice_dataset_bundle(
            bundle,
            train_size=None if max_train is None else int(max_train),
            val_size=None if max_val is None else int(max_val),
            test_size=None if max_test is None else int(max_test),
        )
    return bundle


def _resolve_method_config(method_name: str, config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    method_config = copy.deepcopy(dict(config.get("method_configs", {}).get(method_name, {}) or {}))
    method_config.update(copy.deepcopy(dict(config.get("method_overrides", {}).get(method_name, {}) or {})))
    runtime = dict(config.get("runtime", {}) or {})

    batch_pref = runtime.get("batch_size_preference")
    if batch_pref is not None:
        for key in _METHOD_BATCH_KEYS.get(method_name, ()):
            method_config.setdefault(key, int(batch_pref))

    eval_batch_pref = runtime.get("eval_batch_size_preference")
    if eval_batch_pref is not None:
        for key in _METHOD_EVAL_BATCH_KEYS.get(method_name, ()):
            method_config.setdefault(key, int(eval_batch_pref))

    if runtime.get("device_preference") is not None:
        method_config.setdefault("device", runtime.get("device_preference"))
    if runtime.get("dtype") is not None:
        method_config.setdefault("dtype", runtime.get("dtype"))

    if method_name == "tt_volterra":
        method_config.setdefault("random_seed", int(seed))
    else:
        method_config.setdefault("seed", int(seed))

    return method_config


def _resolve_torch_dtype(dtype_name: str | None) -> Any:
    if dtype_name is None:
        return None
    import torch

    if isinstance(dtype_name, str) and hasattr(torch, dtype_name):
        return getattr(torch, dtype_name)
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def _instantiate_method(method_name: str, method_config: Mapping[str, Any], runtime: Mapping[str, Any]) -> Any:
    device_preference = runtime.get("device_preference")
    dtype_preference = runtime.get("dtype")

    if method_name == "devo":
        devo_kwargs = dict(method_config)
        if devo_kwargs.get("dtype") is not None:
            devo_kwargs["dtype"] = _resolve_torch_dtype(str(devo_kwargs["dtype"]))
        devo_config = DeVoConfig(**devo_kwargs)
        method_class = get_method_class(method_name)
        return method_class(config=devo_config)

    return create_method(
        method_name,
        config=dict(method_config),
        device=device_preference,
        dtype=dtype_preference,
    )


def _align_predictions(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    true_array = np.asarray(y_true, dtype=np.float64)
    pred_array = np.asarray(y_pred, dtype=np.float64)

    if true_array.shape[0] != pred_array.shape[0]:
        raise ValueError(
            f"Prediction batch mismatch: expected {true_array.shape[0]} samples, got {pred_array.shape[0]}."
        )

    if true_array.shape == pred_array.shape:
        return true_array, pred_array

    true_flat = true_array.reshape(true_array.shape[0], -1)
    pred_flat = pred_array.reshape(pred_array.shape[0], -1)
    if true_flat.shape != pred_flat.shape:
        raise ValueError(
            f"Prediction shape mismatch: expected flattened shape {true_flat.shape}, got {pred_flat.shape}."
        )
    return true_flat, pred_flat


def _compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    true_array, pred_array = _align_predictions(y_true, y_pred)
    error = pred_array - true_array
    mse = float(np.mean(np.square(error)))
    rmse = float(np.sqrt(mse))

    true_flat = true_array.reshape(true_array.shape[0], -1)
    centered = true_flat - true_flat.mean(axis=0, keepdims=True)
    variance = float(np.mean(np.square(centered)))
    if variance <= 1e-12:
        signal_power = float(np.mean(np.square(true_flat)))
        variance = signal_power if signal_power > 1e-12 else float("nan")
    nmse = float(mse / variance) if np.isfinite(variance) else float("nan")

    return {
        "nmse": nmse,
        "rmse": rmse,
        "mse": mse,
    }


def _classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (TypeError, ValueError)) and any(
        fragment in text for fragment in _INPUT_CONTRACT_ERROR_FRAGMENTS
    ):
        return "skipped"
    return "failed"


def _run_result_base(
    *,
    config: Mapping[str, Any],
    dataset_name: str,
    method_name: str,
    seed: int,
    run_dir: Path,
    bundle: Any | None,
    method_config: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_meta = None
    split_sizes = None
    if bundle is not None:
        dataset_meta = {
            "dataset_name": bundle.meta.dataset_name,
            "task_family": bundle.meta.task_family.value,
            "input_dim": int(bundle.meta.input_dim),
            "output_dim": int(bundle.meta.output_dim),
            "window_length": int(bundle.meta.window_length),
            "horizon": int(bundle.meta.horizon),
            "split_protocol": bundle.meta.split_protocol,
            "task_usage": list(bundle.meta.extras.get("task_usage", [])),
        }
        split_sizes = {
            "train": int(bundle.train.num_samples),
            "val": int(bundle.val.num_samples),
            "test": int(bundle.test.num_samples),
        }

    return {
        "experiment_name": str(config["experiment_name"]),
        "profile": str(config["profile"]),
        "dataset": dataset_name,
        "method": method_name,
        "seed": int(seed),
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "duration_seconds": None,
        "run_dir": str(run_dir),
        "config_path": str(config["config_path"]),
        "output_dir": str(config["output_dir"]),
        "method_config": dict(method_config),
        "dataset_meta": dataset_meta,
        "split_sizes": split_sizes,
        "metrics": None,
        "method_result": None,
        "error": None,
        "notes": [],
    }


def _finalize_result(
    payload: dict[str, Any],
    *,
    status: str,
    started_at: datetime,
    metrics: Mapping[str, Any] | None = None,
    method_result: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    finished_at = datetime.now(UTC)
    payload["status"] = status
    payload["finished_at"] = finished_at.isoformat()
    payload["duration_seconds"] = round((finished_at - started_at).total_seconds(), 6)
    payload["metrics"] = None if metrics is None else dict(metrics)
    payload["method_result"] = None if method_result is None else dict(method_result)
    payload["error"] = None if error is None else dict(error)
    payload["notes"] = list(notes or [])
    return payload


def _existing_result_is_terminal(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") in {"success", "failed", "skipped"}


def _build_run_manifest(config: Mapping[str, Any], summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_name": str(config["experiment_name"]),
        "profile": str(config["profile"]),
        "generated_at": _utc_now(),
        "config_path": str(config["config_path"]),
        "processed_root": str(config["processed_root"]),
        "output_dir": str(config["output_dir"]),
        "datasets": list(config["datasets"]),
        "methods": list(config["methods"]),
        "seeds": [int(seed) for seed in config["seeds"]],
        "summary": dict(summary),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nonlinear benchmark experiment 01 prediction benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the experiment YAML config.",
    )
    parser.add_argument("--profile", default="full", help="Config profile to execute.")
    parser.add_argument("--datasets", nargs="*", help="Optional dataset override.")
    parser.add_argument("--methods", nargs="*", help="Optional method override.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional seed override.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    parser.add_argument("--max-train-samples", type=int, help="Optional train split cap.")
    parser.add_argument("--max-val-samples", type=int, help="Optional val split cap.")
    parser.add_argument("--max-test-samples", type=int, help="Optional test split cap.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip finished runs already on disk.")
    parser.add_argument("--force", action="store_true", help="Re-run even if result.json already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _resolve_config(args.config.resolve(), args.profile)
    config = _override_config_with_cli(config, args)

    _validate_selection(config["datasets"], DATASET_ORDER, kind="datasets")
    _validate_selection(config["methods"], METHOD_ORDER, kind="methods")

    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", config)

    bundle_cache: dict[str, Any] = {}
    bundle_errors: dict[str, Exception] = {}
    counters = {"success": 0, "failed": 0, "skipped": 0, "reused": 0}

    for dataset_name in config["datasets"]:
        try:
            bundle_cache[dataset_name] = _load_bundle(Path(config["processed_root"]), dataset_name, config)
        except Exception as exc:  # pragma: no cover - exercised in broken-data scenarios.
            bundle_errors[dataset_name] = exc

    for dataset_name in config["datasets"]:
        bundle = bundle_cache.get(dataset_name)
        bundle_error = bundle_errors.get(dataset_name)
        for method_name in config["methods"]:
            for seed in config["seeds"]:
                run_dir = output_dir / "runs" / dataset_name / method_name / f"seed_{int(seed):03d}"
                result_path = run_dir / "result.json"
                if bool(config["skip_existing"]) and _existing_result_is_terminal(result_path):
                    counters["reused"] += 1
                    continue

                method_config = _resolve_method_config(method_name, config, int(seed))
                result_payload = _run_result_base(
                    config=config,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    seed=int(seed),
                    run_dir=run_dir,
                    bundle=bundle,
                    method_config=method_config,
                )
                started_at = datetime.now(UTC)
                if bool(config.get("save_running_state", True)):
                    _write_json(result_path, result_payload)

                if bundle_error is not None:
                    final_payload = _finalize_result(
                        result_payload,
                        status="failed",
                        started_at=started_at,
                        error={
                            "type": type(bundle_error).__name__,
                            "message": str(bundle_error),
                            "traceback": None,
                        },
                        notes=["dataset_bundle_load_failed"],
                    )
                    counters["failed"] += 1
                    _write_json(result_path, final_payload)
                    continue

                set_random_seed(int(seed))

                try:
                    method = _instantiate_method(method_name, method_config, config["runtime"])
                    fit_result = method.fit(bundle)
                    predictions = method.predict(bundle.test.X)
                    metrics = _compute_metrics(bundle.test.Y, predictions)
                    method_result = {
                        "training_summary": dict(getattr(fit_result, "training_summary", {}) or {}),
                        "metadata": dict(getattr(fit_result, "metadata", {}) or {}),
                    }
                    final_payload = _finalize_result(
                        result_payload,
                        status="success",
                        started_at=started_at,
                        metrics=metrics,
                        method_result=method_result,
                    )
                    counters["success"] += 1
                except Exception as exc:  # pragma: no cover - exercised via smoke/error handling.
                    status = _classify_exception(exc)
                    final_payload = _finalize_result(
                        result_payload,
                        status=status,
                        started_at=started_at,
                        error={
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    counters[status] += 1

                _write_json(result_path, final_payload)

    _write_json(output_dir / "run_manifest.json", _build_run_manifest(config, counters))
    print(json.dumps(_to_jsonable(counters), indent=2))


if __name__ == "__main__":
    main()
