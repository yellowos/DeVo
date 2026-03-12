"""TEP case study experiment 01: five-unit fault isolation.

This runner stays inside the experiment layer:
- no data-layer refactor
- no methods-layer refactor
- no propagation analysis

The implementation follows the requested protocol:
- train normal: M1-M4 d00
- val normal: M5 d00
- normal test: M6 d00
- fault evaluation: fault runs
- model input: v01-v53 only
- v54-v81 are metadata only and never enter model input
- feed-side variables remain inputs but are excluded from five-unit scoring

The current repository metadata has no curated five-unit truth rows enabled for
main evaluation. This runner still processes fault runs and records scores, but
aggregate metrics are computed only for rows explicitly marked
`included_in_main_eval=true` in the truth table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover - config loading is runtime-only
    raise RuntimeError("PyYAML is required to run this experiment.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.adapters.base import DatasetArtifacts, DatasetBundle, DatasetMeta, DatasetSplit, TaskFamily
from methods.baselines.lstm import LSTMMethod
from methods.devo import DeVoConfig, DeVoMethod


@dataclass(frozen=True)
class Scaler:
    enabled: bool
    mean: np.ndarray
    std: np.ndarray

    def transform(self, value: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.asarray(value, dtype=np.float32)
        return ((np.asarray(value, dtype=np.float32) - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mean": self.mean.astype(np.float32).tolist(),
            "std": self.std.astype(np.float32).tolist(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TEP five-unit fault isolation experiment runner.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile", type=str, default="full")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional subset of method ids from config.yaml.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=None,
        help="Optional subset of horizons from config.yaml.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root from config.yaml.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected YAML mapping at {path}.")
    return payload


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["empty"])
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_config(raw: Mapping[str, Any], *, config_path: Path, output_override: Optional[Path]) -> Dict[str, Any]:
    config = deepcopy(dict(raw))
    base_dir = config_path.parent.resolve()
    paths = dict(config.get("paths", {}))
    required = ("processed_root", "metadata_root", "output_root")
    missing = [name for name in required if name not in paths]
    if missing:
        raise ValueError(f"config.yaml missing paths: {', '.join(missing)}")
    for key in ("processed_root", "metadata_root", "output_root"):
        paths[key] = resolve_path(base_dir, paths[key])
    if output_override is not None:
        paths["output_root"] = output_override.expanduser().resolve()
    config["paths"] = paths
    return config


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def sort_method_ids(methods_cfg: Mapping[str, Any], selected: Optional[Sequence[str]]) -> List[str]:
    known = list(methods_cfg.keys())
    if selected is None:
        return known
    invalid = [method_id for method_id in selected if method_id not in methods_cfg]
    if invalid:
        raise ValueError(f"Unknown method ids: {', '.join(invalid)}")
    return list(selected)


def sort_horizons(horizons_cfg: Sequence[Any], selected: Optional[Sequence[int]]) -> List[int]:
    base = [int(item) for item in horizons_cfg]
    if selected is None:
        return base
    invalid = [str(item) for item in selected if int(item) not in base]
    if invalid:
        raise ValueError(f"Unknown horizons: {', '.join(invalid)}")
    return [int(item) for item in selected]


def load_experiment_metadata(config: Mapping[str, Any]) -> Dict[str, Any]:
    metadata_root = Path(config["paths"]["metadata_root"])
    processed_root = Path(config["paths"]["processed_root"])
    processed_manifest = load_json(processed_root / "tep_processed_manifest.json")
    fault_eval_manifest = load_json(processed_root / "tep_fault_eval_manifest.json")
    run_manifest = load_json(processed_root / "tep_run_manifest.json")
    mode_protocol = load_json(metadata_root / "mode_holdout_protocol.json")
    five_unit_definition = load_json(metadata_root / "five_unit_definition.json")
    feed_side_definition = load_json(metadata_root / "feed_side_definition.json")
    fault_truth_table = load_json(metadata_root / "fault_truth_table.json")
    channel_map = load_json(metadata_root / "channel_map.json")
    scenario_to_idv_map = load_json(metadata_root / "scenario_to_idv_map.json")
    return {
        "processed_manifest": processed_manifest,
        "fault_eval_manifest": fault_eval_manifest,
        "run_manifest": run_manifest,
        "mode_protocol": mode_protocol,
        "five_unit_definition": five_unit_definition,
        "feed_side_definition": feed_side_definition,
        "fault_truth_table": fault_truth_table,
        "channel_map": channel_map,
        "scenario_to_idv_map": scenario_to_idv_map,
    }


def validate_channel_policy(channel_map: Mapping[str, Any]) -> None:
    model_inputs = list(channel_map.get("model_input_columns", []))
    metadata_only = list(channel_map.get("metadata_only_columns", []))
    if model_inputs != [f"v{i:02d}" for i in range(1, 54)]:
        raise ValueError("channel_map.model_input_columns must be exactly v01-v53.")
    if metadata_only != [f"v{i:02d}" for i in range(54, 82)]:
        raise ValueError("channel_map.metadata_only_columns must be exactly v54-v81.")


def build_run_lookup(run_manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    runs = run_manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("run_manifest.runs must be a list.")
    lookup: Dict[str, Mapping[str, Any]] = {}
    for entry in runs:
        if not isinstance(entry, Mapping):
            continue
        run_key = str(entry.get("run_key", "")).strip()
        if run_key:
            lookup[run_key] = entry
    return lookup


def build_truth_lookup(fault_truth_table: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = fault_truth_table.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("fault_truth_table.rows must be a list.")
    lookup: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        scenario = str(row.get("scenario", "")).strip()
        if scenario:
            lookup[scenario] = row
    return lookup


def build_unit_indices(
    *,
    model_input_columns: Sequence[str],
    five_unit_definition: Mapping[str, Any],
    feed_side_definition: Mapping[str, Any],
) -> tuple[list[str], Dict[str, np.ndarray], np.ndarray]:
    variable_to_index = {name: idx for idx, name in enumerate(model_input_columns)}
    units_payload = ensure_mapping(five_unit_definition.get("five_units", {}), name="five_unit_definition.five_units")
    feed_side_variables = list(feed_side_definition.get("feed_side_variables", []))
    feed_side_indices = []
    for variable in feed_side_variables:
        if variable not in variable_to_index:
            raise ValueError(f"Unknown feed-side variable: {variable}")
        feed_side_indices.append(variable_to_index[variable])
    unit_order: list[str] = []
    unit_indices: Dict[str, np.ndarray] = {}
    for unit_name, payload in units_payload.items():
        unit_order.append(str(unit_name))
        payload_mapping = ensure_mapping(payload, name=f"five_units[{unit_name}]")
        variables = payload_mapping.get("variables", [])
        if not isinstance(variables, list) or not variables:
            raise ValueError(f"{unit_name} must define a non-empty variables list.")
        indices = []
        for variable in variables:
            if variable not in variable_to_index:
                raise ValueError(f"Unknown variable {variable} in unit {unit_name}.")
            indices.append(variable_to_index[variable])
        unit_indices[str(unit_name)] = np.asarray(indices, dtype=np.int64)
    return unit_order, unit_indices, np.asarray(feed_side_indices, dtype=np.int64)


def compute_scaler(
    *,
    processed_root: Path,
    train_run_keys: Sequence[str],
    run_lookup: Mapping[str, Mapping[str, Any]],
    enabled: bool,
    epsilon: float,
) -> Scaler:
    if not enabled:
        return Scaler(enabled=False, mean=np.zeros(53, dtype=np.float32), std=np.ones(53, dtype=np.float32))

    total_sum = np.zeros(53, dtype=np.float64)
    total_sumsq = np.zeros(53, dtype=np.float64)
    total_count = 0

    for run_key in train_run_keys:
        entry = run_lookup[run_key]
        observable_file = str(entry["observable_file"])
        values = np.load(processed_root / observable_file, allow_pickle=True).astype(np.float64, copy=False)
        total_sum += values.sum(axis=0)
        total_sumsq += np.square(values, dtype=np.float64).sum(axis=0)
        total_count += int(values.shape[0])

    if total_count <= 0:
        raise ValueError("Cannot compute scaler without training samples.")

    mean = total_sum / total_count
    variance = np.maximum(total_sumsq / total_count - np.square(mean), float(epsilon) ** 2)
    std = np.sqrt(variance)
    return Scaler(enabled=True, mean=mean.astype(np.float32), std=std.astype(np.float32))


def build_windows(
    *,
    run_values: np.ndarray,
    time_index: np.ndarray,
    window_length: int,
    horizon: int,
) -> Dict[str, np.ndarray]:
    if run_values.ndim != 2:
        raise ValueError(f"run_values must be rank-2, got shape {run_values.shape}.")
    if time_index.ndim != 1:
        raise ValueError(f"time_index must be rank-1, got shape {time_index.shape}.")
    if run_values.shape[0] != time_index.shape[0]:
        raise ValueError("run_values and time_index must share the same first dimension.")

    n_steps, input_dim = run_values.shape
    window_count = n_steps - window_length - horizon + 1
    if window_count <= 0:
        empty_X = np.empty((0, window_length, input_dim), dtype=np.float32)
        empty_Y = np.empty((0, horizon, input_dim), dtype=np.float32)
        empty_idx = np.empty((0,), dtype=np.int32)
        return {
            "X": empty_X,
            "Y": empty_Y,
            "window_idx": empty_idx,
            "window_start": empty_idx,
            "window_end": empty_idx,
            "timestamp": np.empty((0,), dtype=time_index.dtype),
        }

    input_view = np.lib.stride_tricks.sliding_window_view(run_values, (window_length, input_dim))[:, 0]
    input_view = input_view[:window_count]
    target_view = np.lib.stride_tricks.sliding_window_view(run_values[window_length:], (horizon, input_dim))[:, 0]
    target_view = target_view[:window_count]

    window_idx = np.arange(window_count, dtype=np.int32)
    window_start = window_idx
    window_end = window_idx + window_length - 1
    timestamp = time_index[window_length : window_length + window_count]
    return {
        "X": np.asarray(input_view, dtype=np.float32),
        "Y": np.asarray(target_view, dtype=np.float32),
        "window_idx": window_idx,
        "window_start": window_start,
        "window_end": window_end,
        "timestamp": timestamp,
    }


def sample_indices(total: int, max_count: Optional[int], *, seed: int) -> np.ndarray:
    if max_count is None or max_count >= total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total, size=max_count, replace=False)
    chosen.sort()
    return chosen.astype(np.int64, copy=False)


def load_scaled_run(
    *,
    processed_root: Path,
    run_entry: Mapping[str, Any],
    scaler: Scaler,
) -> tuple[np.ndarray, np.ndarray]:
    observable_file = str(run_entry["observable_file"])
    time_index_file = str(run_entry["time_index_file"])
    run_values = np.load(processed_root / observable_file, allow_pickle=True)
    time_index = np.load(processed_root / time_index_file, allow_pickle=True)
    return scaler.transform(run_values), np.asarray(time_index)


def build_split_from_runs(
    *,
    processed_root: Path,
    run_keys: Sequence[str],
    run_lookup: Mapping[str, Mapping[str, Any]],
    scaler: Scaler,
    window_length: int,
    horizon: int,
    max_windows: Optional[int],
    seed: int,
    split_name: str,
) -> DatasetSplit:
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    run_id_chunks: list[np.ndarray] = []

    run_keys = list(run_keys)
    per_run_cap = None if max_windows is None else max(1, math.ceil(max_windows / max(len(run_keys), 1)))

    for run_offset, run_key in enumerate(run_keys):
        run_entry = run_lookup[run_key]
        run_values, time_index = load_scaled_run(processed_root=processed_root, run_entry=run_entry, scaler=scaler)
        windows = build_windows(
            run_values=run_values,
            time_index=time_index,
            window_length=window_length,
            horizon=horizon,
        )
        if windows["X"].shape[0] == 0:
            continue
        indices = sample_indices(windows["X"].shape[0], per_run_cap, seed=seed + run_offset)
        x_chunks.append(windows["X"][indices])
        y_chunks.append(windows["Y"][indices])
        time_chunks.append(windows["timestamp"][indices])
        run_id_chunks.append(np.asarray([run_key] * len(indices), dtype=object))

    if not x_chunks:
        return DatasetSplit(
            X=np.empty((0, window_length, 53), dtype=np.float32),
            Y=np.empty((0, horizon, 53), dtype=np.float32),
            run_id=np.empty((0,), dtype=object),
            timestamp=np.empty((0,), dtype=np.int32),
            meta={"split_name": split_name},
        )

    X = np.concatenate(x_chunks, axis=0)
    Y = np.concatenate(y_chunks, axis=0)
    timestamps = np.concatenate(time_chunks, axis=0)
    run_ids = np.concatenate(run_id_chunks, axis=0)

    final_indices = sample_indices(X.shape[0], max_windows, seed=seed + 10_000)
    return DatasetSplit(
        X=X[final_indices],
        Y=Y[final_indices],
        run_id=run_ids[final_indices],
        timestamp=timestamps[final_indices],
        meta={"split_name": split_name},
    )


def build_dataset_bundle(
    *,
    processed_manifest: Mapping[str, Any],
    processed_root: Path,
    mode_protocol: Mapping[str, Any],
    run_lookup: Mapping[str, Mapping[str, Any]],
    scaler: Scaler,
    window_length: int,
    horizon: int,
    profile_cfg: Mapping[str, Any],
) -> DatasetBundle:
    seed = int(profile_cfg.get("seed", 42))
    run_sets = ensure_mapping(mode_protocol.get("run_sets", {}), name="mode_protocol.run_sets")

    train = build_split_from_runs(
        processed_root=processed_root,
        run_keys=list(run_sets["train_normal_run_keys"]),
        run_lookup=run_lookup,
        scaler=scaler,
        window_length=window_length,
        horizon=horizon,
        max_windows=profile_cfg.get("max_train_windows"),
        seed=seed,
        split_name="train_normal",
    )
    val = build_split_from_runs(
        processed_root=processed_root,
        run_keys=list(run_sets["val_normal_run_keys"]),
        run_lookup=run_lookup,
        scaler=scaler,
        window_length=window_length,
        horizon=horizon,
        max_windows=profile_cfg.get("max_val_windows"),
        seed=seed + 1_000,
        split_name="val_normal",
    )
    test = build_split_from_runs(
        processed_root=processed_root,
        run_keys=list(run_sets["normal_test_run_keys"]),
        run_lookup=run_lookup,
        scaler=scaler,
        window_length=window_length,
        horizon=horizon,
        max_windows=profile_cfg.get("max_normal_test_windows"),
        seed=seed + 2_000,
        split_name="normal_test",
    )

    template_meta = ensure_mapping(processed_manifest.get("bundle_meta", {}), name="processed_manifest.bundle_meta")
    extras = dict(template_meta.get("extras", {}))
    extras["requested_horizon"] = int(horizon)
    meta = DatasetMeta(
        dataset_name=str(template_meta["dataset_name"]),
        task_family=TaskFamily(str(template_meta["task_family"])),
        input_dim=int(template_meta["input_dim"]),
        output_dim=int(template_meta["output_dim"]),
        window_length=int(window_length),
        horizon=int(horizon),
        split_protocol=str(template_meta["split_protocol"]),
        has_ground_truth_kernel=bool(template_meta["has_ground_truth_kernel"]),
        has_ground_truth_gfrf=bool(template_meta["has_ground_truth_gfrf"]),
        extras=extras,
    )
    artifacts = DatasetArtifacts.from_mapping(processed_manifest.get("bundle_artifacts", {}))
    return DatasetBundle(train=train, val=val, test=test, meta=meta, artifacts=artifacts)


def build_method(method_id: str, method_cfg: Mapping[str, Any]) -> Any:
    backend = str(method_cfg.get("backend", "")).strip()
    training_cfg = dict(method_cfg.get("training", {}))
    if backend == "devo":
        return DeVoMethod(DeVoConfig(**training_cfg))
    if backend == "lstm":
        return LSTMMethod(config=training_cfg)
    raise ValueError(f"Unsupported backend '{backend}' for method {method_id}.")


def fit_method(method: Any, bundle: DatasetBundle) -> Mapping[str, Any]:
    result = method.fit(bundle)
    return {
        "training_summary": dict(result.training_summary or {}),
        "metadata": dict(result.metadata or {}),
    }


def compute_normal_test_summary(method: Any, bundle: DatasetBundle, method_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if np.asarray(bundle.test.X).shape[0] == 0:
        return {"window_count": 0, "mse": None}
    batch_size = int(method_cfg.get("prediction_batch_size", method_cfg.get("attribution", {}).get("batch_size", 128)))
    predictions = method.predict(bundle.test.X, batch_size=batch_size)
    residual = predictions - np.asarray(bundle.test.Y)
    mse = float(np.mean(np.square(residual)))
    return {
        "window_count": int(np.asarray(bundle.test.X).shape[0]),
        "mse": mse,
    }


def lstm_prediction_error_attribution(
    method: LSTMMethod,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    batch_size: int,
    mode: str,
) -> np.ndarray:
    if mode not in {"gradient", "grad_x_input"}:
        raise ValueError("LSTM attribution mode must be 'gradient' or 'grad_x_input'.")

    outputs: list[np.ndarray] = []
    target_tensor = torch.as_tensor(Y, dtype=method.runtime.dtype, device=method.runtime.device)
    for start in range(0, int(X.shape[0]), batch_size):
        stop = min(start + batch_size, int(X.shape[0]))
        inputs = method.prepare_inputs(X[start:stop], requires_grad=True)
        predictions = method.forward_tensor(inputs)
        objective = (predictions - target_tensor[start:stop]).pow(2).mean(dim=(1, 2))
        grads = torch.autograd.grad(objective.sum(), inputs, retain_graph=False, create_graph=False)[0]
        if mode == "grad_x_input":
            grads = grads * inputs
        outputs.append(grads.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def predict_and_attribute(
    *,
    method: Any,
    method_id: str,
    method_cfg: Mapping[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not X.flags.writeable:
        X = np.array(X, dtype=np.float32, copy=True)
    if not Y.flags.writeable:
        Y = np.array(Y, dtype=np.float32, copy=True)
    prediction_batch_size = int(method_cfg.get("prediction_batch_size", 128))
    attribution_cfg = dict(method_cfg.get("attribution", {}))
    attribution_batch_size = int(attribution_cfg.get("batch_size", prediction_batch_size))
    attribution_mode = str(attribution_cfg.get("mode", "gradient"))
    predictions = method.predict(X, batch_size=prediction_batch_size)

    if method_id == "devo_error_attribution":
        attributions = method.attribute(
            X,
            Y,
            batch_size=attribution_batch_size,
            mode=attribution_mode,
        )
    elif method_id == "lstm_gradient":
        attributions = lstm_prediction_error_attribution(
            method,
            X,
            Y,
            batch_size=attribution_batch_size,
            mode=attribution_mode,
        )
    else:
        raise ValueError(f"Attribution pipeline not implemented for method {method_id}.")
    return np.asarray(predictions, dtype=np.float32), np.asarray(attributions, dtype=np.float32)


def compute_variable_scores(attributions: np.ndarray, *, reduce_mode: str) -> np.ndarray:
    if attributions.ndim != 3:
        raise ValueError(f"Expected attributions [N, T, D], got {attributions.shape}.")
    if reduce_mode != "mean_abs":
        raise ValueError(f"Unsupported variable score reduction: {reduce_mode}")
    return np.mean(np.abs(attributions), axis=1)


def aggregate_unit_scores(
    variable_scores: np.ndarray,
    *,
    unit_order: Sequence[str],
    unit_indices: Mapping[str, np.ndarray],
    feed_side_indices: np.ndarray,
    member_reduce: str,
) -> tuple[np.ndarray, np.ndarray]:
    if member_reduce not in {"mean", "sum"}:
        raise ValueError(f"Unsupported unit reduce mode: {member_reduce}")
    unit_score_chunks: list[np.ndarray] = []
    for unit_name in unit_order:
        values = variable_scores[:, unit_indices[unit_name]]
        if member_reduce == "mean":
            unit_score_chunks.append(np.mean(values, axis=1))
        else:
            unit_score_chunks.append(np.sum(values, axis=1))
    unit_scores = np.stack(unit_score_chunks, axis=1).astype(np.float32, copy=False)
    if feed_side_indices.size == 0:
        feed_side_score = np.zeros((variable_scores.shape[0],), dtype=np.float32)
    elif member_reduce == "mean":
        feed_side_score = np.mean(variable_scores[:, feed_side_indices], axis=1).astype(np.float32, copy=False)
    else:
        feed_side_score = np.sum(variable_scores[:, feed_side_indices], axis=1).astype(np.float32, copy=False)
    return unit_scores, feed_side_score


def rank_units(score_by_unit: Mapping[str, float], unit_order: Sequence[str]) -> List[str]:
    return sorted(unit_order, key=lambda unit: (-float(score_by_unit[unit]), unit))


def canonical_expected_units(primary_unit: Optional[str], expected_units: Sequence[Any]) -> List[str]:
    values = [str(item) for item in expected_units if str(item).strip()]
    if primary_unit and primary_unit not in values:
        values.insert(0, primary_unit)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def truth_row_is_evaluable(truth_row: Optional[Mapping[str, Any]]) -> bool:
    if not truth_row:
        return False
    primary_unit_raw = truth_row.get("primary_unit")
    primary_unit = str(primary_unit_raw) if primary_unit_raw else None
    expected_units = canonical_expected_units(primary_unit, truth_row.get("expected_units", []))
    return bool(truth_row.get("included_in_main_eval", False) and expected_units)


def count_selected_evaluable_runs(
    selected_fault_runs: Sequence[Mapping[str, Any]],
    truth_lookup: Mapping[str, Mapping[str, Any]],
) -> int:
    total = 0
    for run_entry in selected_fault_runs:
        truth_row = truth_lookup.get(str(run_entry.get("scenario")))
        total += int(truth_row_is_evaluable(truth_row))
    return total


def compute_run_metrics(
    *,
    ranking: Sequence[str],
    early_ranking: Sequence[str],
    truth_row: Optional[Mapping[str, Any]],
) -> Dict[str, Optional[float]]:
    if not truth_row:
        return {"top1": None, "top3": None, "soft_precision_at_3": None, "early_hit": None}

    primary_unit_raw = truth_row.get("primary_unit")
    primary_unit = str(primary_unit_raw) if primary_unit_raw else None
    expected_units = canonical_expected_units(primary_unit, truth_row.get("expected_units", []))
    if not truth_row_is_evaluable(truth_row):
        return {"top1": None, "top3": None, "soft_precision_at_3": None, "early_hit": None}

    top3 = list(ranking[:3])
    top1_value = float(primary_unit is not None and ranking[:1] == [primary_unit])
    top3_value = float(primary_unit is not None and primary_unit in top3)
    soft_precision = float(len(set(top3).intersection(expected_units)) / min(3, len(expected_units)))
    early_hit = float(bool(early_ranking) and early_ranking[0] in set(expected_units))
    return {
        "top1": top1_value,
        "top3": top3_value,
        "soft_precision_at_3": soft_precision,
        "early_hit": early_hit,
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(item) for item in values if item is not None]
    if not finite:
        return None
    return float(np.mean(finite))


def build_metric_summary(run_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "top1": [],
        "top3": [],
        "soft_precision_at_3": [],
        "early_hit": [],
    }
    for record in run_records:
        record_metrics = ensure_mapping(record.get("metrics", {}), name="run_record.metrics")
        for key in metrics:
            metrics[key].append(record_metrics.get(key))
    return {
        "processed_runs": int(len(run_records)),
            "evaluable_runs": int(sum(record.get("evaluable", False) for record in run_records)),
        "top1": {"mean": mean_or_none(metrics["top1"]), "count": sum(item is not None for item in metrics["top1"])},
        "top3": {"mean": mean_or_none(metrics["top3"]), "count": sum(item is not None for item in metrics["top3"])},
        "soft_precision_at_3": {
            "mean": mean_or_none(metrics["soft_precision_at_3"]),
            "count": sum(item is not None for item in metrics["soft_precision_at_3"]),
        },
        "early_hit": {
            "mean": mean_or_none(metrics["early_hit"]),
            "count": sum(item is not None for item in metrics["early_hit"]),
        },
    }


def select_fault_runs(
    *,
    fault_eval_manifest: Mapping[str, Any],
    run_lookup: Mapping[str, Mapping[str, Any]],
    profile_cfg: Mapping[str, Any],
) -> List[Mapping[str, Any]]:
    runs = fault_eval_manifest.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError("fault_eval_manifest.runs must be a list.")
    selected: list[Mapping[str, Any]] = []
    for entry in runs:
        if not isinstance(entry, Mapping):
            continue
        run_key = str(entry.get("run_key", ""))
        merged = dict(run_lookup.get(run_key, {}))
        merged.update(dict(entry))
        selected.append(merged)

    scenario_allowlist = {str(item) for item in profile_cfg.get("scenario_allowlist", [])}
    if scenario_allowlist:
        selected = [entry for entry in selected if str(entry.get("scenario")) in scenario_allowlist]

    run_key_allowlist = {str(item) for item in profile_cfg.get("run_key_allowlist", [])}
    if run_key_allowlist:
        selected = [entry for entry in selected if str(entry.get("run_key")) in run_key_allowlist]

    fault_run_limit = profile_cfg.get("fault_run_limit")
    if fault_run_limit is not None:
        selected = selected[: int(fault_run_limit)]

    selected.sort(key=lambda entry: str(entry.get("run_key", "")))
    return selected


def process_fault_runs(
    *,
    processed_root: Path,
    selected_fault_runs: Sequence[Mapping[str, Any]],
    truth_lookup: Mapping[str, Mapping[str, Any]],
    scaler: Scaler,
    method: Any,
    method_id: str,
    method_cfg: Mapping[str, Any],
    window_length: int,
    horizon: int,
    unit_order: Sequence[str],
    unit_indices: Mapping[str, np.ndarray],
    feed_side_indices: np.ndarray,
    scoring_cfg: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    run_records: list[Dict[str, Any]] = []
    early_window_count = int(scoring_cfg.get("early_window_count", 20))
    variable_reduce = str(scoring_cfg.get("variable_score_reduce", "mean_abs"))
    member_reduce = str(scoring_cfg.get("unit_member_reduce", "mean"))
    save_run_traces = bool(scoring_cfg.get("save_run_traces", False))

    for run_entry in selected_fault_runs:
        run_values, time_index = load_scaled_run(processed_root=processed_root, run_entry=run_entry, scaler=scaler)
        windows = build_windows(
            run_values=run_values,
            time_index=time_index,
            window_length=window_length,
            horizon=horizon,
        )
        truth_row = truth_lookup.get(str(run_entry.get("scenario")))
        expected_units = canonical_expected_units(
            None if truth_row is None else truth_row.get("primary_unit"),
            [] if truth_row is None else truth_row.get("expected_units", []),
        )

        if windows["X"].shape[0] == 0:
            run_records.append(
                {
                    "run_key": str(run_entry.get("run_key")),
                    "mode": str(run_entry.get("mode")),
                    "scenario": str(run_entry.get("scenario")),
                    "idv": str(run_entry.get("idv")),
                    "n_steps": int(run_entry.get("n_steps", 0)),
                    "window_count": 0,
                    "evaluable": False,
                    "included_in_main_eval": bool(truth_row and truth_row.get("included_in_main_eval", False)),
                    "primary_unit": None if truth_row is None else truth_row.get("primary_unit"),
                    "expected_units": expected_units,
                    "metrics": {"top1": None, "top3": None, "soft_precision_at_3": None, "early_hit": None},
                    "skip_reason": f"run shorter than window_length + horizon ({window_length} + {horizon}).",
                }
            )
            continue

        predictions, attributions = predict_and_attribute(
            method=method,
            method_id=method_id,
            method_cfg=method_cfg,
            X=windows["X"],
            Y=windows["Y"],
        )
        residual = predictions - windows["Y"]
        residual_mse_per_window = np.mean(np.square(residual), axis=(1, 2))
        variable_scores = compute_variable_scores(attributions, reduce_mode=variable_reduce)
        unit_scores, feed_side_scores = aggregate_unit_scores(
            variable_scores,
            unit_order=unit_order,
            unit_indices=unit_indices,
            feed_side_indices=feed_side_indices,
            member_reduce=member_reduce,
        )

        run_mean_scores = unit_scores.mean(axis=0)
        early_count = min(early_window_count, int(unit_scores.shape[0]))
        early_mean_scores = unit_scores[:early_count].mean(axis=0)
        score_by_unit = {unit_name: float(run_mean_scores[idx]) for idx, unit_name in enumerate(unit_order)}
        early_score_by_unit = {unit_name: float(early_mean_scores[idx]) for idx, unit_name in enumerate(unit_order)}
        ranking = rank_units(score_by_unit, unit_order)
        early_ranking = rank_units(early_score_by_unit, unit_order)
        metrics = compute_run_metrics(ranking=ranking, early_ranking=early_ranking, truth_row=truth_row)

        record: Dict[str, Any] = {
            "run_key": str(run_entry.get("run_key")),
            "mode": str(run_entry.get("mode")),
            "scenario": str(run_entry.get("scenario")),
            "idv": str(run_entry.get("idv")),
            "n_steps": int(run_entry.get("n_steps", 0)),
            "window_count": int(windows["X"].shape[0]),
            "early_window_count_used": int(early_count),
            "evaluable": truth_row_is_evaluable(truth_row),
            "included_in_main_eval": bool(truth_row and truth_row.get("included_in_main_eval", False)),
            "primary_unit": None if truth_row is None else truth_row.get("primary_unit"),
            "expected_units": expected_units,
            "ranking": ranking,
            "early_ranking": early_ranking,
            "unit_scores": score_by_unit,
            "early_unit_scores": early_score_by_unit,
            "feed_side_score_mean": float(feed_side_scores.mean()),
            "residual_mse_mean": float(residual_mse_per_window.mean()),
            "metrics": metrics,
            "time_range": {
                "first_target_timestamp": None if windows["timestamp"].size == 0 else int(windows["timestamp"][0]),
                "last_target_timestamp": None if windows["timestamp"].size == 0 else int(windows["timestamp"][-1]),
            },
        }
        if save_run_traces:
            record["unit_score_trace_preview"] = {
                unit_name: unit_scores[: min(10, unit_scores.shape[0]), idx].astype(np.float32).tolist()
                for idx, unit_name in enumerate(unit_order)
            }
        run_records.append(record)

    return run_records


def skipped_result_payload(
    *,
    method_id: str,
    method_cfg: Mapping[str, Any],
    horizon: int,
    reason: str,
    selected_fault_runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "status": "skipped",
        "skip_reason": reason,
        "method_id": method_id,
        "display_name": str(method_cfg.get("display_name", method_id)),
        "horizon": int(horizon),
        "aggregate_metrics": {
            "processed_runs": int(len(selected_fault_runs)),
            "evaluable_runs": 0,
            "top1": {"mean": None, "count": 0},
            "top3": {"mean": None, "count": 0},
            "soft_precision_at_3": {"mean": None, "count": 0},
            "early_hit": {"mean": None, "count": 0},
        },
        "run_records": [],
    }


def write_method_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "result.json", payload)

    csv_rows: list[Dict[str, Any]] = []
    for record in payload.get("run_records", []):
        row = {
            "run_key": record.get("run_key"),
            "mode": record.get("mode"),
            "scenario": record.get("scenario"),
            "idv": record.get("idv"),
            "window_count": record.get("window_count"),
            "evaluable": record.get("evaluable"),
            "included_in_main_eval": record.get("included_in_main_eval"),
            "primary_unit": record.get("primary_unit"),
            "expected_units": "|".join(record.get("expected_units", [])),
            "top_prediction": (record.get("ranking") or [None])[0],
            "early_top_prediction": (record.get("early_ranking") or [None])[0],
            "top1": ensure_mapping(record.get("metrics", {}), name="record.metrics").get("top1"),
            "top3": ensure_mapping(record.get("metrics", {}), name="record.metrics").get("top3"),
            "soft_precision_at_3": ensure_mapping(record.get("metrics", {}), name="record.metrics").get("soft_precision_at_3"),
            "early_hit": ensure_mapping(record.get("metrics", {}), name="record.metrics").get("early_hit"),
            "feed_side_score_mean": record.get("feed_side_score_mean"),
            "residual_mse_mean": record.get("residual_mse_mean"),
        }
        unit_scores = ensure_mapping(record.get("unit_scores", {}), name="record.unit_scores") if "unit_scores" in record else {}
        for unit_name, value in unit_scores.items():
            row[f"unit_score_{unit_name.lower()}"] = value
        csv_rows.append(row)
    save_csv(output_dir / "run_metrics.csv", csv_rows)


def run_experiment(args: argparse.Namespace) -> Path:
    raw_config = load_yaml(args.config.resolve())
    config = resolve_config(raw_config, config_path=args.config.resolve(), output_override=args.output_root)
    experiment_cfg = ensure_mapping(config.get("experiment", {}), name="experiment")
    profiles_cfg = ensure_mapping(config.get("profiles", {}), name="profiles")
    if args.profile not in profiles_cfg:
        raise ValueError(f"Unknown profile '{args.profile}'. Available: {', '.join(profiles_cfg.keys())}")
    profile_cfg = ensure_mapping(profiles_cfg[args.profile], name=f"profiles.{args.profile}")
    methods_cfg = ensure_mapping(config.get("methods", {}), name="methods")
    scoring_cfg = ensure_mapping(config.get("scoring", {}), name="scoring")

    selected_method_ids = sort_method_ids(methods_cfg, args.methods)
    selected_horizons = sort_horizons(experiment_cfg.get("horizons", []), args.horizons)

    metadata = load_experiment_metadata(config)
    processed_manifest = ensure_mapping(metadata["processed_manifest"], name="processed_manifest")
    fault_eval_manifest = ensure_mapping(metadata["fault_eval_manifest"], name="fault_eval_manifest")
    mode_protocol = ensure_mapping(metadata["mode_protocol"], name="mode_protocol")
    five_unit_definition = ensure_mapping(metadata["five_unit_definition"], name="five_unit_definition")
    feed_side_definition = ensure_mapping(metadata["feed_side_definition"], name="feed_side_definition")
    truth_table = ensure_mapping(metadata["fault_truth_table"], name="fault_truth_table")
    channel_map = ensure_mapping(metadata["channel_map"], name="channel_map")
    validate_channel_policy(channel_map)

    run_lookup = build_run_lookup(ensure_mapping(metadata["run_manifest"], name="run_manifest"))
    truth_lookup = build_truth_lookup(truth_table)
    unit_order, unit_indices, feed_side_indices = build_unit_indices(
        model_input_columns=list(channel_map["model_input_columns"]),
        five_unit_definition=five_unit_definition,
        feed_side_definition=feed_side_definition,
    )

    processed_root = Path(config["paths"]["processed_root"])
    output_root = Path(config["paths"]["output_root"]) / str(args.profile)
    output_root.mkdir(parents=True, exist_ok=True)

    run_sets = ensure_mapping(mode_protocol.get("run_sets", {}), name="mode_protocol.run_sets")
    scaler_cfg = ensure_mapping(config.get("scaler", {}), name="scaler")
    scaler = compute_scaler(
        processed_root=processed_root,
        train_run_keys=list(run_sets["train_normal_run_keys"]),
        run_lookup=run_lookup,
        enabled=bool(scaler_cfg.get("enabled", True)),
        epsilon=float(scaler_cfg.get("epsilon", 1e-6)),
    )

    selected_fault_runs = select_fault_runs(
        fault_eval_manifest=fault_eval_manifest,
        run_lookup=run_lookup,
        profile_cfg=profile_cfg,
    )
    truth_included_count = sum(bool(row.get("included_in_main_eval", False)) for row in truth_lookup.values())
    selected_truth_count = count_selected_evaluable_runs(selected_fault_runs, truth_lookup)
    allow_empty_main_eval = bool(profile_cfg.get("allow_empty_main_eval", False))

    if not allow_empty_main_eval and truth_included_count == 0:
        raise RuntimeError(
            "fault_truth_table contains no included_in_main_eval=true rows; "
            "refusing to mark the experiment as completed."
        )
    if not allow_empty_main_eval and selected_truth_count == 0:
        raise RuntimeError(
            "Selected fault runs contain no evaluable main-eval scenes; "
            "adjust the profile allowlist or set allow_empty_main_eval=true for debug-only runs."
        )

    config_snapshot = {
        "profile": args.profile,
        "selected_methods": selected_method_ids,
        "selected_horizons": selected_horizons,
        "paths": {key: str(value) for key, value in config["paths"].items()},
        "scaler": scaler.to_dict(),
        "allow_empty_main_eval": allow_empty_main_eval,
        "truth_included_count": truth_included_count,
        "selected_truth_count": selected_truth_count,
        "selected_fault_run_count": len(selected_fault_runs),
        "selected_fault_runs_preview": [str(entry.get("run_key")) for entry in selected_fault_runs[:10]],
    }
    save_json(output_root / "resolved_config.json", config_snapshot)

    for horizon in selected_horizons:
        bundle = build_dataset_bundle(
            processed_manifest=processed_manifest,
            processed_root=processed_root,
            mode_protocol=mode_protocol,
            run_lookup=run_lookup,
            scaler=scaler,
            window_length=int(experiment_cfg.get("window_length", 128)),
            horizon=int(horizon),
            profile_cfg=profile_cfg,
        )

        for method_id in selected_method_ids:
            method_cfg = ensure_mapping(methods_cfg[method_id], name=f"methods.{method_id}")
            method_output_dir = output_root / f"h{horizon}" / method_id

            if not bool(method_cfg.get("enabled", False)):
                payload = skipped_result_payload(
                    method_id=method_id,
                    method_cfg=method_cfg,
                    horizon=horizon,
                    reason=str(method_cfg.get("skip_reason", "disabled in config")),
                    selected_fault_runs=selected_fault_runs,
                )
                payload.update(
                    {
                        "profile": args.profile,
                        "window_length": int(experiment_cfg.get("window_length", 128)),
                        "metadata_paths": {
                            "truth_table": str(Path(config["paths"]["metadata_root"]) / "fault_truth_table.json"),
                            "five_unit_definition": str(Path(config["paths"]["metadata_root"]) / "five_unit_definition.json"),
                            "feed_side_definition": str(Path(config["paths"]["metadata_root"]) / "feed_side_definition.json"),
                        },
                    }
                )
                write_method_outputs(method_output_dir, payload)
                continue

            method = build_method(method_id, method_cfg)
            fit_summary = fit_method(method, bundle)
            normal_test_summary = compute_normal_test_summary(method, bundle, method_cfg)
            run_records = process_fault_runs(
                processed_root=processed_root,
                selected_fault_runs=selected_fault_runs,
                truth_lookup=truth_lookup,
                scaler=scaler,
                method=method,
                method_id=method_id,
                method_cfg=method_cfg,
                window_length=int(experiment_cfg.get("window_length", 128)),
                horizon=int(horizon),
                unit_order=unit_order,
                unit_indices=unit_indices,
                feed_side_indices=feed_side_indices,
                scoring_cfg={**scoring_cfg, **{"save_run_traces": bool(profile_cfg.get("save_run_traces", False))}},
            )
            aggregate_metrics = build_metric_summary(run_records)
            if not allow_empty_main_eval and int(aggregate_metrics["evaluable_runs"]) == 0:
                payload = {
                    "status": "failed",
                    "failure_reason": (
                        "No evaluable runs remained after windowing and scoring. "
                        "This usually means the selected main-eval scenes were filtered out or too short."
                    ),
                    "method_id": method_id,
                    "display_name": str(method_cfg.get("display_name", method_id)),
                    "backend": str(method_cfg.get("backend")),
                    "profile": args.profile,
                    "horizon": int(horizon),
                    "window_length": int(experiment_cfg.get("window_length", 128)),
                    "aggregate_metrics": aggregate_metrics,
                    "selected_fault_run_count": int(len(selected_fault_runs)),
                    "selected_fault_runs": [str(entry.get("run_key")) for entry in selected_fault_runs],
                }
                write_method_outputs(method_output_dir, payload)
                raise RuntimeError(payload["failure_reason"])
            payload = {
                "status": "completed",
                "method_id": method_id,
                "display_name": str(method_cfg.get("display_name", method_id)),
                "backend": str(method_cfg.get("backend")),
                "profile": args.profile,
                "horizon": int(horizon),
                "window_length": int(experiment_cfg.get("window_length", 128)),
                "input_columns": list(channel_map["model_input_columns"]),
                "metadata_only_columns": list(channel_map["metadata_only_columns"]),
                "feed_side_variables": list(feed_side_definition.get("feed_side_variables", [])),
                "unit_order": list(unit_order),
                "unit_variable_map": {
                    unit_name: [channel_map["model_input_columns"][idx] for idx in unit_indices[unit_name].tolist()]
                    for unit_name in unit_order
                },
                "truth_table_summary": {
                    "row_count": int(len(truth_lookup)),
                    "included_in_main_eval_count": int(truth_included_count),
                    "source": str(Path(config["paths"]["metadata_root"]) / "fault_truth_table.json"),
                },
                "selected_fault_run_count": int(len(selected_fault_runs)),
                "selected_fault_runs": [str(entry.get("run_key")) for entry in selected_fault_runs],
                "scaler": scaler.to_dict(),
                "training": fit_summary,
                "normal_test_summary": normal_test_summary,
                "aggregate_metrics": aggregate_metrics,
                "run_records": run_records,
                "metadata_paths": {
                    "truth_table": str(Path(config["paths"]["metadata_root"]) / "fault_truth_table.json"),
                    "five_unit_definition": str(Path(config["paths"]["metadata_root"]) / "five_unit_definition.json"),
                    "feed_side_definition": str(Path(config["paths"]["metadata_root"]) / "feed_side_definition.json"),
                    "mode_holdout_protocol": str(Path(config["paths"]["metadata_root"]) / "mode_holdout_protocol.json"),
                    "channel_map": str(Path(config["paths"]["metadata_root"]) / "channel_map.json"),
                },
            }
            write_method_outputs(method_output_dir, payload)

    return output_root


def main() -> None:
    args = parse_args()
    output_root = run_experiment(args)
    print(f"[exp01] results written to: {output_root}")


if __name__ == "__main__":
    main()
