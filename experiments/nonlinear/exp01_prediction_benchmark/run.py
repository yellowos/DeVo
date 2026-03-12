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
from methods.metrics import compute_prediction_metrics
from methods.utils import set_random_seed, slice_dataset_bundle
from experiments.common import compute_config_hash, compute_split_hash, deep_merge_configs, make_run_id, write_resolved_config


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
    runtime = deep_merge_configs(raw.get("runtime", {}) or {}, profile_payload.get("runtime", {}) or {})

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
    return copy.deepcopy(resolved)


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
    method_config = deep_merge_configs(
        config.get("method_configs", {}).get(method_name, {}) or {},
        config.get("method_overrides", {}).get(method_name, {}) or {},
    )
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


def _compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    metrics = compute_prediction_metrics(y_true, y_pred, domain="native")
    return {
        "nmse": float(metrics["nmse"]),
        "rmse": float(metrics["rmse"]),
        "mse": float(metrics["mse"]),
    }


def _classify_exception(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (TypeError, ValueError)) and any(
        fragment in text for fragment in _INPUT_CONTRACT_ERROR_FRAGMENTS
    ):
        return "skipped"
    return "failed"


def _flatten_for_hash(value: Any) -> list[Any]:
    if value is None:
        return []
    return np.asarray(value, dtype=object).reshape(-1).tolist()


def _compute_bundle_split_hash(bundle: Any) -> str:
    payload: dict[str, Any] = {
        "dataset_name": bundle.meta.dataset_name,
        "source_manifest": getattr(bundle, "source_manifest", None),
        "source_root": getattr(bundle, "source_root", None),
        "splits": {},
    }
    for split_name in ("train", "val", "test"):
        split = bundle.get_split(split_name)
        payload["splits"][split_name] = {
            "num_samples": int(split.num_samples),
            "sample_id": _flatten_for_hash(split.sample_id),
            "run_id": _flatten_for_hash(split.run_id),
            "timestamp": _flatten_for_hash(split.timestamp),
            "window_start": _flatten_for_hash(split.extra_fields.get("window_start")),
            "window_end": _flatten_for_hash(split.extra_fields.get("window_end")),
            "target_index": _flatten_for_hash(split.extra_fields.get("target_index")),
        }
    return compute_split_hash(payload)


def _iter_valid_run_dirs(results_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    if not results_dir.exists():
        return run_dirs
    for result_path in sorted(results_dir.rglob("result.json")):
        run_dir = result_path.parent
        if _is_valid_exp01_run_dir(run_dir):
            run_dirs.append(run_dir)
    return run_dirs


def _find_existing_terminal_run(seed_dir: Path, *, config_hash: str) -> Path | None:
    if not seed_dir.exists():
        return None
    for run_dir in sorted(seed_dir.iterdir()):
        if not run_dir.is_dir() or not _is_valid_exp01_run_dir(run_dir):
            continue
        result_path = run_dir / "result.json"
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("config_hash") != config_hash:
            continue
        if payload.get("status") in {"success", "failed", "skipped"}:
            return run_dir
    return None


def _is_valid_exp01_run_dir(run_dir: Path) -> bool:
    required = (
        run_dir / "result.json",
        run_dir / "status.json",
        run_dir / "run_context.json",
        run_dir / "resolved_config.json",
        run_dir / "metrics.json",
        run_dir / "artifacts_manifest.json",
    )
    if any(not path.is_file() for path in required):
        return False
    try:
        result_payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        config_payload = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(result_payload, dict) or not isinstance(status_payload, dict) or not isinstance(config_payload, dict):
        return False
    if result_payload.get("run_id") != status_payload.get("run_id"):
        return False
    return result_payload.get("config_hash") == compute_config_hash(config_payload)


def _run_result_base(
    *,
    config: Mapping[str, Any],
    dataset_name: str,
    method_name: str,
    seed: int,
    run_id: str,
    run_dir: Path,
    bundle: Any | None,
    method_config: Mapping[str, Any],
    config_hash: str,
    split_hash: str | None,
    resolved_config_path: Path,
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
        "run_id": run_id,
        "config_hash": config_hash,
        "resolved_config_path": str(resolved_config_path),
        "split_hash": split_hash,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "duration_seconds": None,
        "run_dir": str(run_dir),
        "run_context_path": str(run_dir / "run_context.json"),
        "status_path": str(run_dir / "status.json"),
        "metrics_path": str(run_dir / "metrics.json"),
        "artifacts_manifest_path": str(run_dir / "artifacts_manifest.json"),
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


def _write_run_sidecars(
    run_dir: Path,
    result_payload: Mapping[str, Any],
    *,
    run_context: Mapping[str, Any],
) -> None:
    status_value = str(result_payload.get("status", "running"))
    error_payload = result_payload.get("error")
    error_message = error_payload.get("message") if isinstance(error_payload, Mapping) else None
    _write_json(
        run_dir / "run_context.json",
        run_context,
    )
    _write_json(
        run_dir / "status.json",
        {
            "run_id": result_payload.get("run_id"),
            "state": status_value,
            "message": None if status_value == "success" else error_message,
            "config_hash": result_payload.get("config_hash"),
            "resolved_config_path": result_payload.get("resolved_config_path"),
            "split_hash": result_payload.get("split_hash"),
        },
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "run_id": result_payload.get("run_id"),
            "metrics": dict(result_payload.get("metrics") or {}),
            "domain": "native",
        },
    )
    _write_json(
        run_dir / "artifacts_manifest.json",
        {
            "run_id": result_payload.get("run_id"),
            "artifacts": dict(result_payload.get("artifacts") or {}),
            "method_result": dict(result_payload.get("method_result") or {}),
        },
    )


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
    config_hash = compute_config_hash(config)
    write_resolved_config(config, output_dir / "resolved_config.json")

    bundle_cache: dict[str, Any] = {}
    bundle_errors: dict[str, Exception] = {}
    bundle_split_hashes: dict[str, str] = {}
    counters = {"success": 0, "failed": 0, "skipped": 0, "reused": 0}

    for dataset_name in config["datasets"]:
        try:
            bundle = _load_bundle(Path(config["processed_root"]), dataset_name, config)
            bundle_cache[dataset_name] = bundle
            bundle_split_hashes[dataset_name] = _compute_bundle_split_hash(bundle)
        except Exception as exc:  # pragma: no cover - exercised in broken-data scenarios.
            bundle_errors[dataset_name] = exc

    for dataset_name in config["datasets"]:
        bundle = bundle_cache.get(dataset_name)
        bundle_error = bundle_errors.get(dataset_name)
        split_hash = bundle_split_hashes.get(dataset_name)
        for method_name in config["methods"]:
            for seed in config["seeds"]:
                seed_dir = output_dir / "runs" / dataset_name / method_name / f"seed_{int(seed):03d}"
                existing_run_dir = None
                if bool(config["skip_existing"]):
                    existing_run_dir = _find_existing_terminal_run(seed_dir, config_hash=config_hash)
                if existing_run_dir is not None:
                    counters["reused"] += 1
                    continue

                method_config = _resolve_method_config(method_name, config, int(seed))
                run_id = make_run_id(
                    experiment_name=str(config["experiment_name"]),
                    dataset=dataset_name,
                    method=method_name,
                    seed=int(seed),
                )
                run_dir = seed_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=False)
                resolved_config_path = write_resolved_config(config, run_dir / "resolved_config.json")
                result_path = run_dir / "result.json"
                result_payload = _run_result_base(
                    config=config,
                    dataset_name=dataset_name,
                    method_name=method_name,
                    seed=int(seed),
                    run_id=run_id,
                    run_dir=run_dir,
                    bundle=bundle,
                    method_config=method_config,
                    config_hash=config_hash,
                    split_hash=split_hash,
                    resolved_config_path=resolved_config_path,
                )
                started_at = datetime.now(UTC)
                run_context = {
                    "run_id": run_id,
                    "seed": int(seed),
                    "dataset": dataset_name,
                    "method": method_name,
                    "config_hash": config_hash,
                    "resolved_config_path": str(resolved_config_path),
                    "split_hash": split_hash,
                }
                _write_run_sidecars(run_dir, result_payload, run_context=run_context)
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
                    _write_run_sidecars(run_dir, final_payload, run_context=run_context)
                    _write_json(result_path, final_payload)
                    continue

                set_random_seed(int(seed))

                try:
                    method = _instantiate_method(method_name, method_config, config["runtime"])
                    fit_result = method.fit(bundle, run_context=run_context)
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

                _write_run_sidecars(run_dir, final_payload, run_context=run_context)
                _write_json(result_path, final_payload)

    _write_json(
        output_dir / "run_manifest.json",
        {
            **_build_run_manifest(config, counters),
            "config_hash": config_hash,
            "valid_run_dirs": [str(path) for path in _iter_valid_run_dirs(output_dir / "runs")],
        },
    )
    print(json.dumps(_to_jsonable(counters), indent=2))


if __name__ == "__main__":
    main()
