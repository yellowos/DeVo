#!/usr/bin/env python3
"""Hydraulic case study: unsupervised subsystem fault isolation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.adapters.base import DataProtocolError  # noqa: E402
from data.adapters.hydraulic_adapter import HydraulicAdapter  # noqa: E402
from methods.base import create_method  # noqa: E402
from methods.baselines.arx_var import ARXMethod, VARMethod  # noqa: E402
from methods.baselines.lstm import LSTMMethod  # noqa: E402
from methods.baselines.narmax import NARMAXMethod  # noqa: E402
from methods.baselines.tcn import TCNMethod  # noqa: E402
from methods.devo import DeVoConfig, DeVoMethod  # noqa: E402


STATUS_COMPLETED = "completed"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"

SUPPORTED_METHODS = ("arx", "var", "narmax", "lstm", "tcn", "devo")
DEFAULT_SUBSYSTEM_ORDER = ("Cooler", "Valve", "Pump", "Accumulator")
EPS = 1e-8


@dataclass(frozen=True)
class CycleRecord:
    row_index: int
    cycle_id: int
    sample_id: int
    fault_subsystem: str
    is_healthy: bool
    is_single_component_fault: bool
    condition_type: str
    raw: dict[str, str]


@dataclass(frozen=True)
class DataBundle:
    cycles: np.ndarray
    labels: list[CycleRecord]
    healthy_records: list[CycleRecord]
    eval_records: list[CycleRecord]
    channel_names: list[str]
    subsystem_channels: dict[str, list[str]]
    subsystem_indices: dict[str, list[int]]
    representation_name: str
    processed_manifest_path: Path
    cycle_tensor_path: Path
    label_table_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Hydraulic unsupervised subsystem fault isolation experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the experiment config YAML.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional fixed run directory name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Override the configured method list.",
    )
    parser.add_argument(
        "--max-eval-cycles",
        type=int,
        default=None,
        help="Override the number of evaluated single-fault cycles.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply the smoke overrides from config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run methods even if result files already exist.",
    )
    return parser.parse_args()


def _now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected YAML mapping at {path}.")
    return dict(payload)


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_path(path_like: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return dict(payload)


def _load_hydraulic_data(config: Mapping[str, Any]) -> tuple[Optional[DataBundle], Optional[str], dict[str, Any]]:
    experiment_cfg = dict(config.get("experiment", {}))
    processed_root = _resolve_path(
        experiment_cfg.get("processed_root", "data/processed/hydraulic"),
        base_dir=REPO_ROOT,
    )
    representation = str(experiment_cfg.get("representation", "cycle_60"))
    representation_root = processed_root / representation if (processed_root / representation).is_dir() else processed_root
    manifest_path = representation_root / "hydraulic_processed_manifest.json"
    subsystem_groups_path = _resolve_path(
        experiment_cfg.get("subsystem_groups_file", "data/metadata/hydraulic/subsystem_groups.json"),
        base_dir=REPO_ROOT,
    )

    diagnostics: dict[str, Any] = {
        "processed_root": str(processed_root),
        "representation": representation,
        "representation_root": str(representation_root),
        "manifest_path": str(manifest_path),
        "subsystem_groups_path": str(subsystem_groups_path),
    }

    try:
        _ = HydraulicAdapter.load_processed_bundle(processed_root, representation=representation)
    except Exception as exc:
        return None, f"Hydraulic processed bundle is not loadable: {exc}", diagnostics

    if not manifest_path.exists():
        return None, f"Hydraulic processed manifest is missing: {manifest_path}", diagnostics
    if not subsystem_groups_path.exists():
        return None, f"Subsystem grouping metadata is missing: {subsystem_groups_path}", diagnostics

    manifest = _load_json(manifest_path)
    cycle_tensor_ref = manifest.get("cycle_tensor_file")
    label_table_ref = manifest.get("label_table_file")
    if not isinstance(cycle_tensor_ref, str) or not cycle_tensor_ref.strip():
        return None, "Hydraulic processed manifest is missing cycle_tensor_file.", diagnostics
    if not isinstance(label_table_ref, str) or not label_table_ref.strip():
        return None, "Hydraulic processed manifest is missing label_table_file.", diagnostics

    cycle_tensor_path = _resolve_path(cycle_tensor_ref, base_dir=manifest_path.parent)
    label_table_path = _resolve_path(label_table_ref, base_dir=manifest_path.parent)
    diagnostics["cycle_tensor_path"] = str(cycle_tensor_path)
    diagnostics["label_table_path"] = str(label_table_path)

    if not cycle_tensor_path.exists():
        return None, f"Hydraulic cycle tensor is missing: {cycle_tensor_path}", diagnostics
    if not label_table_path.exists():
        return None, f"Hydraulic label table is missing: {label_table_path}", diagnostics

    try:
        cycle_tensor = np.load(cycle_tensor_path, allow_pickle=True)
    except Exception as exc:
        return None, f"Failed to load hydraulic cycle tensor: {exc}", diagnostics

    if cycle_tensor.ndim != 3:
        return None, f"Hydraulic cycle tensor must be 3D [cycles, time, channels], got {cycle_tensor.shape}.", diagnostics

    if representation != "cycle_60":
        return None, f"Main experiment requires representation cycle_60, got {representation}.", diagnostics

    if int(cycle_tensor.shape[1]) != 60:
        return None, f"Main experiment requires 60-second cycles with 60 samples, got {cycle_tensor.shape[1]}.", diagnostics

    subsystem_groups = _load_json(subsystem_groups_path)
    subsystem_payload = subsystem_groups.get("subsystems")
    if not isinstance(subsystem_payload, Mapping):
        return None, "subsystem_groups.json is missing the subsystems mapping.", diagnostics

    extras = dict((manifest.get("bundle_meta") or {}).get("extras", {}))
    channel_names = list(extras.get("canonical_channel_order", []))
    if len(channel_names) != int(cycle_tensor.shape[2]):
        return None, "Hydraulic metadata does not expose the canonical 17-channel order.", diagnostics

    subsystem_order = list(
        config.get("evaluation", {}).get("subsystem_order", DEFAULT_SUBSYSTEM_ORDER)
    )
    subsystem_channels: dict[str, list[str]] = {}
    subsystem_indices: dict[str, list[int]] = {}
    for display_name in subsystem_order:
        matched_key = None
        matched_payload = None
        for key, value in subsystem_payload.items():
            if not isinstance(value, Mapping):
                continue
            if str(value.get("display_name", key)).strip().lower() == display_name.lower():
                matched_key = str(key)
                matched_payload = dict(value)
                break
        if matched_key is None or matched_payload is None:
            return None, f"Subsystem grouping is missing required subsystem `{display_name}`.", diagnostics
        channels = matched_payload.get("channels", [])
        if not isinstance(channels, list) or not channels:
            return None, f"Subsystem `{display_name}` has no channels.", diagnostics
        unknown_channels = [channel for channel in channels if channel not in channel_names]
        if unknown_channels:
            return None, f"Subsystem `{display_name}` references unknown channels: {unknown_channels}", diagnostics
        subsystem_channels[display_name] = [str(channel) for channel in channels]
        subsystem_indices[display_name] = [channel_names.index(str(channel)) for channel in channels]

    with label_table_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None, "Hydraulic label table is empty.", diagnostics

    required_columns = {
        "cycle_id",
        "sample_id",
        "is_healthy",
        "is_single_component_fault",
        "fault_subsystem",
        "condition_type",
    }
    missing_columns = sorted(required_columns - set(rows[0].keys()))
    if missing_columns:
        return None, f"Hydraulic label table is missing required columns: {missing_columns}", diagnostics

    if len(rows) != int(cycle_tensor.shape[0]):
        return None, (
            "Hydraulic label table and cycle tensor are misaligned: "
            f"{len(rows)} labels vs {cycle_tensor.shape[0]} cycles."
        ), diagnostics

    labels: list[CycleRecord] = []
    healthy_records: list[CycleRecord] = []
    eval_records: list[CycleRecord] = []
    for row_index, row in enumerate(rows):
        cycle_id = int(row["cycle_id"])
        sample_id = int(row["sample_id"])
        fault_subsystem = str(row["fault_subsystem"]).strip()
        is_healthy = bool(int(row["is_healthy"]))
        is_single_component_fault = bool(int(row["is_single_component_fault"]))
        condition_type = str(row["condition_type"]).strip()
        record = CycleRecord(
            row_index=row_index,
            cycle_id=cycle_id,
            sample_id=sample_id,
            fault_subsystem=fault_subsystem,
            is_healthy=is_healthy,
            is_single_component_fault=is_single_component_fault,
            condition_type=condition_type,
            raw=dict(row),
        )
        labels.append(record)
        if is_healthy:
            healthy_records.append(record)
        if is_single_component_fault and fault_subsystem in subsystem_order:
            eval_records.append(record)

    diagnostics["healthy_count"] = len(healthy_records)
    diagnostics["single_component_fault_count"] = len(eval_records)
    diagnostics["single_fault_subsystem_counts"] = dict(
        sorted(Counter(record.fault_subsystem for record in eval_records).items())
    )
    diagnostics["cycle_tensor_shape"] = list(map(int, cycle_tensor.shape))
    diagnostics["channel_names"] = channel_names

    if not healthy_records:
        return None, "Hydraulic data is blocked: no healthy cycles are available.", diagnostics
    if not eval_records:
        return None, "Hydraulic data is blocked: no single-component degraded cycles are available.", diagnostics

    return (
        DataBundle(
            cycles=np.asarray(cycle_tensor, dtype=np.float32),
            labels=labels,
            healthy_records=healthy_records,
            eval_records=eval_records,
            channel_names=channel_names,
            subsystem_channels=subsystem_channels,
            subsystem_indices=subsystem_indices,
            representation_name=representation,
            processed_manifest_path=manifest_path,
            cycle_tensor_path=cycle_tensor_path,
            label_table_path=label_table_path,
        ),
        None,
        diagnostics,
    )


def _split_indices(
    num_items: int,
    *,
    fractions: Sequence[float],
    shuffle: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_items < 3:
        raise ValueError(f"Need at least 3 items to create train/val/test splits, got {num_items}.")
    train_frac, val_frac, test_frac = (float(value) for value in fractions)
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}.")

    order = np.arange(num_items, dtype=np.int64)
    if shuffle:
        rng.shuffle(order)

    train_count = max(1, int(round(num_items * train_frac)))
    val_count = max(1, int(round(num_items * val_frac)))
    test_count = num_items - train_count - val_count
    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            raise ValueError("Unable to allocate a non-empty test split.")

    while train_count + val_count + test_count > num_items:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Split counts are inconsistent.")

    while train_count + val_count + test_count < num_items:
        train_count += 1

    train_idx = order[:train_count]
    val_idx = order[train_count : train_count + val_count]
    test_idx = order[train_count + val_count : train_count + val_count + test_count]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Split allocation produced an empty split.")
    return train_idx, val_idx, test_idx


def _build_windows_for_cycle(cycle: np.ndarray, *, lag_window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    if cycle.ndim != 2:
        raise ValueError(f"Expected a single cycle matrix [time, channels], got {cycle.shape}.")
    total_steps, input_dim = cycle.shape
    max_start = total_steps - lag_window - horizon + 1
    if max_start <= 0:
        raise ValueError(
            f"Cycle length {total_steps} is too short for lag_window={lag_window}, horizon={horizon}."
        )
    x_windows = np.empty((max_start, lag_window, input_dim), dtype=np.float32)
    y_windows = np.empty((max_start, horizon, input_dim), dtype=np.float32)
    for start in range(max_start):
        stop = start + lag_window
        x_windows[start] = cycle[start:stop]
        y_windows[start] = cycle[stop : stop + horizon]
    return x_windows, y_windows


def _stack_cycle_windows(
    cycles: np.ndarray,
    *,
    lag_window: int,
    horizon: int,
    sample_prefix: str,
) -> dict[str, np.ndarray]:
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_sample_ids: list[str] = []
    cycle_local_index: list[int] = []
    for cycle_position, cycle in enumerate(cycles):
        x_windows, y_windows = _build_windows_for_cycle(cycle, lag_window=lag_window, horizon=horizon)
        all_x.append(x_windows)
        all_y.append(y_windows)
        all_sample_ids.extend(f"{sample_prefix}_c{cycle_position}_w{window_index}" for window_index in range(x_windows.shape[0]))
        cycle_local_index.extend([cycle_position] * x_windows.shape[0])
    x_array = np.concatenate(all_x, axis=0)
    y_array = np.concatenate(all_y, axis=0)
    return {
        "X": x_array,
        "Y": y_array,
        "sample_id": np.asarray(all_sample_ids, dtype=object),
        "cycle_local_index": np.asarray(cycle_local_index, dtype=np.int64),
    }


def _build_healthy_reference_bundle(
    cycles: np.ndarray,
    *,
    lag_window: int,
    horizon: int,
    split_fractions: Sequence[float],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    train_cycle_idx, val_cycle_idx, test_cycle_idx = _split_indices(
        cycles.shape[0],
        fractions=split_fractions,
        shuffle=True,
        rng=rng,
    )
    train_cycles = cycles[train_cycle_idx]
    val_cycles = cycles[val_cycle_idx]
    test_cycles = cycles[test_cycle_idx]

    channel_mean = train_cycles.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    channel_std = train_cycles.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    channel_std = np.where(channel_std < EPS, 1.0, channel_std).astype(np.float32)

    def _normalize(batch: np.ndarray) -> np.ndarray:
        return ((batch - channel_mean[None, None, :]) / channel_std[None, None, :]).astype(np.float32)

    normalized_train = _normalize(train_cycles)
    normalized_val = _normalize(val_cycles)
    normalized_test = _normalize(test_cycles)

    train_windows = _stack_cycle_windows(
        normalized_train,
        lag_window=lag_window,
        horizon=horizon,
        sample_prefix="healthy_train",
    )
    val_windows = _stack_cycle_windows(
        normalized_val,
        lag_window=lag_window,
        horizon=horizon,
        sample_prefix="healthy_val",
    )
    test_windows = _stack_cycle_windows(
        normalized_test,
        lag_window=lag_window,
        horizon=horizon,
        sample_prefix="healthy_test",
    )

    bundle = {
        "train": {
            "X": train_windows["X"],
            "Y": train_windows["Y"],
            "sample_id": train_windows["sample_id"],
        },
        "val": {
            "X": val_windows["X"],
            "Y": val_windows["Y"],
            "sample_id": val_windows["sample_id"],
        },
        "test": {
            "X": test_windows["X"],
            "Y": test_windows["Y"],
            "sample_id": test_windows["sample_id"],
        },
        "meta": {
            "dataset_name": "hydraulic_healthy_reference_autoregressive",
            "task_family": "hydraulic",
            "input_dim": int(cycles.shape[2]),
            "output_dim": int(cycles.shape[2]),
            "window_length": int(lag_window),
            "horizon": int(horizon),
            "split_protocol": "hydraulic_unsupervised_reference_split_v1",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        },
        "artifacts": {},
    }
    stats = {
        "channel_mean": channel_mean,
        "channel_std": channel_std,
    }
    summary = {
        "train_cycles": int(train_cycles.shape[0]),
        "val_cycles": int(val_cycles.shape[0]),
        "test_cycles": int(test_cycles.shape[0]),
        "train_windows": int(train_windows["X"].shape[0]),
        "val_windows": int(val_windows["X"].shape[0]),
        "test_windows": int(test_windows["X"].shape[0]),
    }
    return bundle, stats, summary


def _build_single_cycle_bundle(
    cycle: np.ndarray,
    *,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    lag_window: int,
    horizon: int,
    split_fractions: Sequence[float],
) -> tuple[dict[str, Any], np.ndarray]:
    normalized_cycle = ((cycle - channel_mean[None, :]) / channel_std[None, :]).astype(np.float32)
    x_windows, y_windows = _build_windows_for_cycle(normalized_cycle, lag_window=lag_window, horizon=horizon)

    num_windows = x_windows.shape[0]
    train_frac, val_frac, test_frac = (float(value) for value in split_fractions)
    train_end = max(1, int(math.floor(num_windows * train_frac)))
    val_end = max(train_end + 1, int(math.floor(num_windows * (train_frac + val_frac))))
    if val_end >= num_windows:
        val_end = num_windows - 1
    if train_end >= val_end:
        train_end = max(1, val_end - 1)
    if train_end <= 0 or val_end <= train_end or val_end >= num_windows:
        raise ValueError(
            "Single-cycle split failed. "
            f"Need train/val/test windows from {num_windows} windows, got train_end={train_end}, val_end={val_end}."
        )

    bundle = {
        "train": {
            "X": x_windows[:train_end],
            "Y": y_windows[:train_end],
            "sample_id": np.asarray([f"cycle_train_w{i}" for i in range(train_end)], dtype=object),
        },
        "val": {
            "X": x_windows[train_end:val_end],
            "Y": y_windows[train_end:val_end],
            "sample_id": np.asarray(
                [f"cycle_val_w{i}" for i in range(train_end, val_end)],
                dtype=object,
            ),
        },
        "test": {
            "X": x_windows[val_end:],
            "Y": y_windows[val_end:],
            "sample_id": np.asarray(
                [f"cycle_test_w{i}" for i in range(val_end, num_windows)],
                dtype=object,
            ),
        },
        "meta": {
            "dataset_name": "hydraulic_single_cycle_autoregressive",
            "task_family": "hydraulic",
            "input_dim": int(cycle.shape[1]),
            "output_dim": int(cycle.shape[1]),
            "window_length": int(lag_window),
            "horizon": int(horizon),
            "split_protocol": "hydraulic_single_cycle_local_split_v1",
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
        },
        "artifacts": {},
    }
    return bundle, x_windows


def _instantiate_method(method_name: str, method_config: Mapping[str, Any]) -> Any:
    config_payload = dict(method_config)
    if method_name == "arx":
        return ARXMethod(config=config_payload)
    if method_name == "var":
        return VARMethod(config=config_payload)
    if method_name == "narmax":
        return NARMAXMethod(config=config_payload)
    if method_name == "lstm":
        return LSTMMethod(config=config_payload)
    if method_name == "tcn":
        return TCNMethod(config=config_payload)
    if method_name == "devo":
        devo_config = DeVoConfig(**config_payload)
        return DeVoMethod(config=devo_config)
    return create_method(method_name, config=config_payload)


def _fit_method(method_name: str, method_config: Mapping[str, Any], bundle: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    method = _instantiate_method(method_name, method_config)
    result = method.fit(bundle)
    return method, result.to_dict() if hasattr(result, "to_dict") else {}


def _channel_scores_from_linear_methods(healthy_method: Any, anomaly_method: Any, *, input_dim: int) -> np.ndarray:
    healthy = np.asarray(healthy_method.coefficient_tensor_, dtype=np.float64)
    anomaly = np.asarray(anomaly_method.coefficient_tensor_, dtype=np.float64)
    delta = np.abs(anomaly - healthy)
    output_scores = delta.mean(axis=(0, 2, 3))
    input_scores = delta.mean(axis=(0, 1, 2))
    return (0.5 * (output_scores + input_scores)).astype(np.float64).reshape(input_dim)


def _term_signature(term_spec: Mapping[str, Any]) -> str:
    payload = {
        "group": str(term_spec.get("group", "")),
        "degree": int(term_spec.get("degree", 0)),
        "factors": [
            {
                "source_kind": str(factor.get("source_kind", "")),
                "channel": int(factor.get("channel", -1)),
                "lag": int(factor.get("lag", -1)),
                "power": int(factor.get("power", 1)),
            }
            for factor in term_spec.get("factors", [])
        ],
    }
    return json.dumps(payload, sort_keys=True)


def _channel_scores_from_narmax(healthy_method: NARMAXMethod, anomaly_method: NARMAXMethod, *, input_dim: int) -> np.ndarray:
    if healthy_method.coefficients_ is None or anomaly_method.coefficients_ is None:
        raise RuntimeError("NARMAX coefficients are unavailable after fit().")

    healthy_terms = {_term_signature(spec): (spec, index) for index, spec in enumerate(healthy_method.term_specs_)}
    anomaly_terms = {_term_signature(spec): (spec, index) for index, spec in enumerate(anomaly_method.term_specs_)}
    union_keys = sorted(set(healthy_terms) | set(anomaly_terms))

    output_scores = np.zeros((input_dim,), dtype=np.float64)
    input_scores = np.zeros((input_dim,), dtype=np.float64)
    term_counter = 0

    for key in union_keys:
        healthy_spec, healthy_index = healthy_terms.get(key, (None, None))
        anomaly_spec, anomaly_index = anomaly_terms.get(key, (None, None))
        spec = healthy_spec or anomaly_spec
        if spec is None:
            continue
        if str(spec.get("group", "")) == "bias":
            continue

        healthy_vector = (
            np.asarray(healthy_method.coefficients_[healthy_index], dtype=np.float64)
            if healthy_index is not None
            else np.zeros_like(anomaly_method.coefficients_[anomaly_index], dtype=np.float64)
        )
        anomaly_vector = (
            np.asarray(anomaly_method.coefficients_[anomaly_index], dtype=np.float64)
            if anomaly_index is not None
            else np.zeros_like(healthy_method.coefficients_[healthy_index], dtype=np.float64)
        )
        delta = np.abs(anomaly_vector - healthy_vector)
        output_scores += delta.reshape(-1)[:input_dim]

        factors = [dict(factor) for factor in spec.get("factors", [])]
        if not factors:
            continue
        counts: dict[int, float] = {}
        total_weight = 0.0
        for factor in factors:
            if str(factor.get("source_kind", "")) not in {"input", "output", "residual"}:
                continue
            channel_index = int(factor.get("channel", -1))
            power = max(1, int(factor.get("power", 1)))
            if channel_index < 0 or channel_index >= input_dim:
                continue
            counts[channel_index] = counts.get(channel_index, 0.0) + float(power)
            total_weight += float(power)
        if total_weight <= 0.0:
            continue
        term_weight = float(delta.mean())
        for channel_index, channel_weight in counts.items():
            input_scores[channel_index] += term_weight * (channel_weight / total_weight)
        term_counter += 1

    if term_counter > 0:
        input_scores /= float(term_counter)
        output_scores /= float(term_counter)
    return (0.5 * (input_scores + output_scores)).astype(np.float64)


def _compute_lstm_gradients(method: LSTMMethod, probe_x: np.ndarray, *, output_dim: int) -> np.ndarray:
    gradients: list[np.ndarray] = []
    for output_index in range(output_dim):
        inputs = method.prepare_inputs(probe_x, requires_grad=True)
        outputs = method.forward_tensor(inputs, reshape_output=False)
        method.model.zero_grad(set_to_none=True)
        scalar = outputs[:, output_index].sum()
        scalar.backward()
        gradients.append(inputs.grad.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.stack(gradients, axis=0)


def _compute_tcn_gradients(method: TCNMethod, probe_x: np.ndarray, *, output_dim: int) -> np.ndarray:
    gradients: list[np.ndarray] = []
    for output_index in range(output_dim):
        gradients.append(
            method.compute_input_gradients(
                probe_x,
                target_index=output_index,
                batch_size=min(int(method.config["batch_size"]), max(1, int(probe_x.shape[0]))),
            )
        )
    return np.stack(gradients, axis=0)


def _channel_scores_from_gradient_shift(
    healthy_gradients: np.ndarray,
    anomaly_gradients: np.ndarray,
) -> np.ndarray:
    delta = np.abs(anomaly_gradients.astype(np.float64) - healthy_gradients.astype(np.float64))
    output_scores = delta.mean(axis=(1, 2, 3))
    input_scores = delta.mean(axis=(0, 1, 2))
    return (0.5 * (output_scores + input_scores)).astype(np.float64)


def _channel_scores_from_devo(healthy_method: DeVoMethod, anomaly_method: DeVoMethod, *, input_dim: int) -> np.ndarray:
    healthy_recovery = healthy_method.recover_kernels(materialize_full=False)
    anomaly_recovery = anomaly_method.recover_kernels(materialize_full=False)
    healthy_orders = healthy_recovery.kernels.get("orders", {})
    anomaly_orders = anomaly_recovery.kernels.get("orders", {})
    order_keys = sorted(set(healthy_orders) & set(anomaly_orders), key=int)
    if not order_keys:
        raise RuntimeError("DeVo recovery returned no overlapping orders.")

    channel_scores = np.zeros((input_dim,), dtype=np.float64)
    for order_key in order_keys:
        healthy_payload = healthy_orders[order_key]
        anomaly_payload = anomaly_orders[order_key]
        healthy_coeff = np.asarray(healthy_payload["symmetric_coefficients"], dtype=np.float64)
        anomaly_coeff = np.asarray(anomaly_payload["symmetric_coefficients"], dtype=np.float64)
        lag_input_indices = np.asarray(healthy_payload["lag_input_indices"], dtype=np.int64)
        if healthy_coeff.shape != anomaly_coeff.shape:
            raise RuntimeError(
                f"DeVo recovered coefficient shapes do not match for order {order_key}: "
                f"{healthy_coeff.shape} vs {anomaly_coeff.shape}."
            )

        delta = np.abs(anomaly_coeff - healthy_coeff)
        order_output_scores = delta.mean(axis=(0, 2))
        order_input_scores = np.zeros((input_dim,), dtype=np.float64)
        feature_weights = delta.mean(axis=(0, 1))
        monomial_order = max(1, int(lag_input_indices.shape[1]))
        for feature_index, weight in enumerate(feature_weights):
            channel_indices = lag_input_indices[feature_index, :, 1]
            unique_channels, counts = np.unique(channel_indices, return_counts=True)
            for channel_index, count in zip(unique_channels, counts):
                if 0 <= int(channel_index) < input_dim:
                    order_input_scores[int(channel_index)] += float(weight) * (float(count) / float(monomial_order))
        channel_scores += 0.5 * (order_output_scores + order_input_scores)
    return channel_scores / float(len(order_keys))


def _aggregate_subsystems(channel_scores: np.ndarray, subsystem_indices: Mapping[str, Sequence[int]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for subsystem_name, indices in subsystem_indices.items():
        values = np.asarray([channel_scores[index] for index in indices], dtype=np.float64)
        scores[subsystem_name] = float(values.mean()) if values.size else 0.0
    return scores


def _rank_subsystems(scores: Mapping[str, float], subsystem_order: Sequence[str]) -> list[str]:
    return sorted(
        subsystem_order,
        key=lambda subsystem_name: (-float(scores[subsystem_name]), subsystem_order.index(subsystem_name)),
    )


def _evaluate_sample_metrics(
    subsystem_scores: Mapping[str, float],
    *,
    true_subsystem: str,
    subsystem_order: Sequence[str],
) -> dict[str, Any]:
    ranking = _rank_subsystems(subsystem_scores, subsystem_order)
    predicted_subsystem = ranking[0]
    true_rank = ranking.index(true_subsystem) + 1
    top_2 = ranking[:2]
    true_score = float(subsystem_scores[true_subsystem])
    competitor_scores = [float(subsystem_scores[name]) for name in subsystem_order if name != true_subsystem]
    best_other = max(competitor_scores) if competitor_scores else 0.0
    prediction_margin = float(subsystem_scores[ranking[0]] - subsystem_scores[ranking[1]]) if len(ranking) > 1 else 0.0
    return {
        "predicted_subsystem": predicted_subsystem,
        "ranking": ranking,
        "true_rank": int(true_rank),
        "top1_hit": int(predicted_subsystem == true_subsystem),
        "top2_hit": int(true_subsystem in top_2),
        "winning_margin": float(true_score - best_other),
        "prediction_margin": float(prediction_margin),
    }


def _summarize_method_metrics(
    sample_results: Sequence[Mapping[str, Any]],
    *,
    subsystem_order: Sequence[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not sample_results:
        return (
            {
                "num_eval_samples": 0,
                "top1_isolation_accuracy": None,
                "top2_coverage": None,
                "mean_rank_true_subsystem": None,
                "winning_margin": None,
            },
            {
                subsystem_name: {
                    "num_samples": 0,
                    "num_top1_hits": 0,
                    "top1_hit_rate": None,
                }
                for subsystem_name in subsystem_order
            },
        )

    num_eval_samples = len(sample_results)
    top1 = float(np.mean([int(row["top1_hit"]) for row in sample_results]))
    top2 = float(np.mean([int(row["top2_hit"]) for row in sample_results]))
    mean_rank = float(np.mean([float(row["true_rank"]) for row in sample_results]))
    mean_margin = float(np.mean([float(row["winning_margin"]) for row in sample_results]))

    by_subsystem: dict[str, list[Mapping[str, Any]]] = {subsystem_name: [] for subsystem_name in subsystem_order}
    for row in sample_results:
        by_subsystem[str(row["true_subsystem"])].append(row)

    subsystem_table: dict[str, dict[str, Any]] = {}
    for subsystem_name in subsystem_order:
        rows = by_subsystem[subsystem_name]
        hits = sum(int(row["top1_hit"]) for row in rows)
        subsystem_table[subsystem_name] = {
            "num_samples": len(rows),
            "num_top1_hits": hits,
            "top1_hit_rate": None if not rows else float(hits / len(rows)),
        }

    return (
        {
            "num_eval_samples": int(num_eval_samples),
            "top1_isolation_accuracy": top1,
            "top2_coverage": top2,
            "mean_rank_true_subsystem": mean_rank,
            "winning_margin": mean_margin,
        },
        subsystem_table,
    )


def _run_single_method(
    *,
    method_name: str,
    method_config: Mapping[str, Any],
    data_bundle: DataBundle,
    run_dir: Path,
    global_config: Mapping[str, Any],
    max_eval_cycles: Optional[int],
) -> dict[str, Any]:
    evaluation_cfg = dict(global_config.get("evaluation", {}))
    subsystem_order = list(evaluation_cfg.get("subsystem_order", DEFAULT_SUBSYSTEM_ORDER))
    lag_window = int(evaluation_cfg.get("lag_window", 8))
    horizon = int(evaluation_cfg.get("horizon", 1))
    healthy_split = tuple(float(value) for value in evaluation_cfg.get("healthy_reference_split", (0.7, 0.15, 0.15)))
    anomaly_split = tuple(float(value) for value in evaluation_cfg.get("single_cycle_split", (0.6, 0.2, 0.2)))
    seed = int(global_config.get("experiment", {}).get("seed", 7))
    rng = np.random.default_rng(seed)

    method_dir = run_dir / "methods" / method_name
    _ensure_dir(method_dir)

    start_time = time.time()
    healthy_cycles = np.asarray(
        [data_bundle.cycles[record.row_index] for record in data_bundle.healthy_records],
        dtype=np.float32,
    )
    healthy_bundle, normalization_stats, healthy_summary = _build_healthy_reference_bundle(
        healthy_cycles,
        lag_window=lag_window,
        horizon=horizon,
        split_fractions=healthy_split,
        rng=rng,
    )

    healthy_model, healthy_fit = _fit_method(method_name, method_config, healthy_bundle)
    healthy_state_path = None
    if method_name in {"arx", "var", "narmax", "lstm", "tcn"}:
        try:
            healthy_state_path = str(healthy_model.save(method_dir / "healthy_reference_state.json"))
        except Exception:
            healthy_state_path = None

    eval_records = list(data_bundle.eval_records)
    if max_eval_cycles is not None:
        eval_records = eval_records[: max(0, int(max_eval_cycles))]

    sample_results: list[dict[str, Any]] = []
    sample_errors: list[dict[str, Any]] = []

    for position, record in enumerate(eval_records, start=1):
        try:
            cycle = np.asarray(data_bundle.cycles[record.row_index], dtype=np.float32)
            anomaly_bundle, probe_x = _build_single_cycle_bundle(
                cycle,
                channel_mean=normalization_stats["channel_mean"],
                channel_std=normalization_stats["channel_std"],
                lag_window=lag_window,
                horizon=horizon,
                split_fractions=anomaly_split,
            )
            anomaly_model, anomaly_fit = _fit_method(method_name, method_config, anomaly_bundle)

            if method_name in {"arx", "var"}:
                channel_scores = _channel_scores_from_linear_methods(
                    healthy_model,
                    anomaly_model,
                    input_dim=len(data_bundle.channel_names),
                )
            elif method_name == "narmax":
                channel_scores = _channel_scores_from_narmax(
                    healthy_model,
                    anomaly_model,
                    input_dim=len(data_bundle.channel_names),
                )
            elif method_name == "lstm":
                healthy_gradients = _compute_lstm_gradients(
                    healthy_model,
                    probe_x,
                    output_dim=len(data_bundle.channel_names),
                )
                anomaly_gradients = _compute_lstm_gradients(
                    anomaly_model,
                    probe_x,
                    output_dim=len(data_bundle.channel_names),
                )
                channel_scores = _channel_scores_from_gradient_shift(healthy_gradients, anomaly_gradients)
            elif method_name == "tcn":
                healthy_gradients = _compute_tcn_gradients(
                    healthy_model,
                    probe_x,
                    output_dim=len(data_bundle.channel_names),
                )
                anomaly_gradients = _compute_tcn_gradients(
                    anomaly_model,
                    probe_x,
                    output_dim=len(data_bundle.channel_names),
                )
                channel_scores = _channel_scores_from_gradient_shift(healthy_gradients, anomaly_gradients)
            elif method_name == "devo":
                channel_scores = _channel_scores_from_devo(
                    healthy_model,
                    anomaly_model,
                    input_dim=len(data_bundle.channel_names),
                )
            else:
                raise KeyError(f"Unsupported method: {method_name}")

            subsystem_scores = _aggregate_subsystems(channel_scores, data_bundle.subsystem_indices)
            sample_metrics = _evaluate_sample_metrics(
                subsystem_scores,
                true_subsystem=record.fault_subsystem,
                subsystem_order=subsystem_order,
            )
            row = {
                "row_index": int(record.row_index),
                "cycle_id": int(record.cycle_id),
                "sample_id": int(record.sample_id),
                "true_subsystem": record.fault_subsystem,
                "predicted_subsystem": sample_metrics["predicted_subsystem"],
                "true_rank": int(sample_metrics["true_rank"]),
                "top1_hit": int(sample_metrics["top1_hit"]),
                "top2_hit": int(sample_metrics["top2_hit"]),
                "winning_margin": float(sample_metrics["winning_margin"]),
                "prediction_margin": float(sample_metrics["prediction_margin"]),
                "ranking": list(sample_metrics["ranking"]),
                "probe_windows": int(probe_x.shape[0]),
                "healthy_fit_summary": healthy_fit.get("training_summary", {}),
                "anomaly_fit_summary": anomaly_fit.get("training_summary", {}),
            }
            for subsystem_name in subsystem_order:
                row[f"score_{subsystem_name.lower()}"] = float(subsystem_scores[subsystem_name])
            for channel_name, value in zip(data_bundle.channel_names, channel_scores):
                row[f"channel_{channel_name}"] = float(value)
            sample_results.append(row)
            print(
                f"[{method_name}] {position:04d}/{len(eval_records):04d} "
                f"cycle={record.cycle_id} true={record.fault_subsystem} "
                f"pred={sample_metrics['predicted_subsystem']}"
            )
        except Exception as exc:
            sample_errors.append(
                {
                    "row_index": int(record.row_index),
                    "cycle_id": int(record.cycle_id),
                    "sample_id": int(record.sample_id),
                    "true_subsystem": record.fault_subsystem,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    metrics, subsystem_table = _summarize_method_metrics(sample_results, subsystem_order=subsystem_order)
    runtime_seconds = float(time.time() - start_time)

    sample_csv_fields = [
        "row_index",
        "cycle_id",
        "sample_id",
        "true_subsystem",
        "predicted_subsystem",
        "true_rank",
        "top1_hit",
        "top2_hit",
        "winning_margin",
        "prediction_margin",
    ] + [f"score_{name.lower()}" for name in subsystem_order] + [f"channel_{name}" for name in data_bundle.channel_names]

    subsystem_rows = [
        {
            "method": method_name,
            "subsystem": subsystem_name,
            **stats,
        }
        for subsystem_name, stats in subsystem_table.items()
    ]

    if sample_results:
        _write_csv(method_dir / "sample_results.csv", sample_results, fieldnames=sample_csv_fields)
    if subsystem_rows:
        _write_csv(
            method_dir / "subsystem_hit_rates.csv",
            subsystem_rows,
            fieldnames=("method", "subsystem", "num_samples", "num_top1_hits", "top1_hit_rate"),
        )
    if sample_errors:
        _write_json(method_dir / "sample_errors.json", {"errors": sample_errors})

    status = STATUS_COMPLETED if sample_results else STATUS_FAILED
    result_payload = {
        "status": status,
        "method": method_name,
        "reason": None if sample_results else "No evaluation samples completed successfully.",
        "runtime_seconds": runtime_seconds,
        "healthy_reference": {
            "healthy_cycle_count": len(data_bundle.healthy_records),
            "summary": healthy_summary,
            "state_path": healthy_state_path,
            "fit_result": healthy_fit,
            "normalization": {
                "channel_mean": normalization_stats["channel_mean"].tolist(),
                "channel_std": normalization_stats["channel_std"].tolist(),
            },
        },
        "evaluation": {
            "requested_eval_cycles": len(eval_records),
            "successful_eval_cycles": len(sample_results),
            "failed_eval_cycles": len(sample_errors),
            "lag_window": lag_window,
            "horizon": horizon,
            "subsystem_order": subsystem_order,
            "aggregation_rule": (
                "Per-channel shift = 0.5 * (output-side mean absolute shift + input-side mean absolute shift). "
                "Subsystem score = mean channel shift over subsystem channels."
            ),
        },
        "metrics": metrics,
        "subsystem_hit_rates": subsystem_table,
        "sample_result_file": str((method_dir / "sample_results.csv").resolve()) if sample_results else None,
        "sample_error_file": str((method_dir / "sample_errors.json").resolve()) if sample_errors else None,
        "result_file_version": 1,
    }
    _write_json(method_dir / "result.json", result_payload)
    return result_payload


def _write_placeholder_method_results(
    *,
    run_dir: Path,
    methods: Sequence[str],
    reason: str,
    diagnostics: Mapping[str, Any],
) -> None:
    for method_name in methods:
        method_dir = run_dir / "methods" / method_name
        _ensure_dir(method_dir)
        _write_json(
            method_dir / "result.json",
            {
                "status": STATUS_SKIPPED,
                "method": method_name,
                "reason": reason,
                "diagnostics": dict(diagnostics),
                "metrics": {
                    "num_eval_samples": 0,
                    "top1_isolation_accuracy": None,
                    "top2_coverage": None,
                    "mean_rank_true_subsystem": None,
                    "winning_margin": None,
                },
                "subsystem_hit_rates": {
                    subsystem_name: {
                        "num_samples": 0,
                        "num_top1_hits": 0,
                        "top1_hit_rate": None,
                    }
                    for subsystem_name in DEFAULT_SUBSYSTEM_ORDER
                },
            },
        )


def main() -> int:
    args = parse_args()
    base_config = _load_yaml(args.config)
    if args.smoke:
        smoke_override = dict(base_config.get("smoke_overrides", {}))
        base_config = _deep_update(base_config, smoke_override)

    configured_methods = list(base_config.get("experiment", {}).get("methods", SUPPORTED_METHODS))
    methods = [str(name).lower() for name in (args.methods if args.methods else configured_methods)]
    unknown_methods = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown_methods:
        raise KeyError(f"Unknown methods requested: {unknown_methods}. Supported methods: {SUPPORTED_METHODS}")

    max_eval_cycles = args.max_eval_cycles
    if max_eval_cycles is None:
        max_eval_cycles = base_config.get("experiment", {}).get("max_eval_cycles")
    max_eval_cycles = None if max_eval_cycles in (None, "", "null") else int(max_eval_cycles)

    run_name = args.run_name or _now_stamp()
    output_root = _resolve_path(
        base_config.get("experiment", {}).get(
            "output_root",
            "experiments/hydraulic/exp01_unsupervised_subsystem_fault_isolation/outputs",
        ),
        base_dir=REPO_ROOT,
    )
    run_dir = output_root / "runs" / run_name
    _ensure_dir(run_dir)

    resolved_config = deepcopy(base_config)
    resolved_config.setdefault("experiment", {})
    resolved_config["experiment"]["resolved_output_root"] = str(output_root)
    resolved_config["experiment"]["resolved_run_dir"] = str(run_dir)
    resolved_config["experiment"]["resolved_methods"] = methods
    resolved_config["experiment"]["resolved_max_eval_cycles"] = max_eval_cycles
    _write_json(run_dir / "config.snapshot.json", resolved_config)

    data_bundle, data_error, diagnostics = _load_hydraulic_data(base_config)
    run_manifest: dict[str, Any] = {
        "status": STATUS_COMPLETED,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "methods": methods,
        "max_eval_cycles": max_eval_cycles,
        "data_diagnostics": diagnostics,
        "method_results": {},
    }

    if data_bundle is None:
        run_manifest["status"] = STATUS_SKIPPED
        run_manifest["reason"] = data_error
        _write_placeholder_method_results(run_dir=run_dir, methods=methods, reason=str(data_error), diagnostics=diagnostics)
        _write_json(run_dir / "run_manifest.json", run_manifest)
        print(f"Hydraulic experiment skipped: {data_error}")
        return 0

    method_configs = dict(base_config.get("methods", {}))
    for method_name in methods:
        method_dir = run_dir / "methods" / method_name
        result_path = method_dir / "result.json"
        if result_path.exists() and not args.force:
            payload = _load_json(result_path)
            run_manifest["method_results"][method_name] = {
                "status": payload.get("status", STATUS_COMPLETED),
                "result_path": str(result_path),
                "reason": payload.get("reason"),
            }
            print(f"[{method_name}] reused existing result at {result_path}")
            continue

        try:
            print(f"[{method_name}] fitting healthy reference and evaluating single-fault cycles")
            payload = _run_single_method(
                method_name=method_name,
                method_config=dict(method_configs.get(method_name, {})),
                data_bundle=data_bundle,
                run_dir=run_dir,
                global_config=base_config,
                max_eval_cycles=max_eval_cycles,
            )
            run_manifest["method_results"][method_name] = {
                "status": payload.get("status", STATUS_COMPLETED),
                "result_path": str(result_path),
                "reason": payload.get("reason"),
            }
        except (ModuleNotFoundError, ImportError) as exc:
            _write_json(
                result_path,
                {
                    "status": STATUS_SKIPPED,
                    "method": method_name,
                    "reason": f"Method dependency is unavailable: {exc}",
                    "metrics": {
                        "num_eval_samples": 0,
                        "top1_isolation_accuracy": None,
                        "top2_coverage": None,
                        "mean_rank_true_subsystem": None,
                        "winning_margin": None,
                    },
                    "subsystem_hit_rates": {
                        subsystem_name: {
                            "num_samples": 0,
                            "num_top1_hits": 0,
                            "top1_hit_rate": None,
                        }
                        for subsystem_name in DEFAULT_SUBSYSTEM_ORDER
                    },
                },
            )
            run_manifest["method_results"][method_name] = {
                "status": STATUS_SKIPPED,
                "result_path": str(result_path),
                "reason": str(exc),
            }
        except Exception as exc:
            _write_json(
                result_path,
                {
                    "status": STATUS_FAILED,
                    "method": method_name,
                    "reason": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                    "metrics": {
                        "num_eval_samples": 0,
                        "top1_isolation_accuracy": None,
                        "top2_coverage": None,
                        "mean_rank_true_subsystem": None,
                        "winning_margin": None,
                    },
                    "subsystem_hit_rates": {
                        subsystem_name: {
                            "num_samples": 0,
                            "num_top1_hits": 0,
                            "top1_hit_rate": None,
                        }
                        for subsystem_name in DEFAULT_SUBSYSTEM_ORDER
                    },
                },
            )
            run_manifest["method_results"][method_name] = {
                "status": STATUS_FAILED,
                "result_path": str(result_path),
                "reason": str(exc),
            }

    if any(entry["status"] == STATUS_FAILED for entry in run_manifest["method_results"].values()):
        run_manifest["status"] = STATUS_FAILED
    elif all(entry["status"] == STATUS_SKIPPED for entry in run_manifest["method_results"].values()):
        run_manifest["status"] = STATUS_SKIPPED

    _write_json(run_dir / "run_manifest.json", run_manifest)
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
