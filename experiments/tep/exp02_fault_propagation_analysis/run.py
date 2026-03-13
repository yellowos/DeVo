from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(DEFAULT_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(DEFAULT_PROJECT_ROOT))

from methods.base import BaseMethod, create_method, load_dataset_bundle
from methods.devo import DeVoConfig


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    raw = load_yaml(config_path)
    paths = raw.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    project_root = resolve_path(paths.get("project_root", "../../.."), base_dir=SCRIPT_DIR)
    assert project_root is not None
    paths["project_root"] = str(project_root)
    for key in (
        "processed_root",
        "bundle_path",
        "five_unit_definition",
        "fault_truth_table",
        "propagation_subset",
        "output_root",
    ):
        if key not in paths:
            raise ValueError(f"config.paths.{key} is required")
        resolved = resolve_path(paths[key], base_dir=project_root)
        assert resolved is not None
        paths[key] = str(resolved)
    method_cfg = raw.setdefault("method", {})
    if not isinstance(method_cfg, dict):
        raise ValueError("config.method must be a mapping")
    state_path = resolve_path(method_cfg.get("state_path"), base_dir=project_root)
    method_cfg["state_path"] = None if state_path is None else str(state_path)
    return raw


def window_count(n_steps: int, window_length: int, horizon: int) -> int:
    return max(0, n_steps - window_length - horizon + 1)


def truth_rows_by_scenario(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("scenario"), str):
            result[str(row["scenario"])] = dict(row)
    return result


def build_unit_index_map(
    five_unit_definition: Mapping[str, Any],
    model_input_columns: Sequence[str],
) -> tuple[list[str], dict[str, np.ndarray]]:
    five_units = five_unit_definition.get("five_units", {})
    if not isinstance(five_units, Mapping):
        raise ValueError("five_unit_definition.five_units must be a mapping")
    name_to_index = {name: idx for idx, name in enumerate(model_input_columns)}
    unit_order: list[str] = []
    unit_indices: dict[str, np.ndarray] = {}
    for unit_name, payload in five_units.items():
        if not isinstance(payload, Mapping):
            continue
        variables = payload.get("variables", [])
        if not isinstance(variables, list):
            continue
        indices = [name_to_index[var] for var in variables if var in name_to_index]
        if not indices:
            raise ValueError(f"Unit {unit_name} has no overlap with model input columns")
        unit_order.append(str(unit_name))
        unit_indices[str(unit_name)] = np.asarray(indices, dtype=np.int64)
    if len(unit_order) != 5:
        raise ValueError(f"Expected 5 units, got {len(unit_order)}")
    return unit_order, unit_indices


def effective_run_scope(
    *,
    count: int,
    min_windows_to_process: int,
    short_run_policy: str,
    early_window_cap: int,
    full_window_count: int,
) -> tuple[str, np.ndarray | None, str | None]:
    if count <= 0:
        return "skipped", None, "no_windows"
    if count < min_windows_to_process and short_run_policy == "skip":
        return "skipped", None, "too_short_for_policy"
    if count < min_windows_to_process and short_run_policy == "early_only":
        return "early_only", np.arange(min(count, early_window_cap), dtype=np.int64), "short_run_early_only"
    if count < min_windows_to_process:
        return "truncated", np.arange(count, dtype=np.int64), "short_run_truncated"
    if short_run_policy == "early_only" and count < full_window_count:
        return "early_only", np.arange(min(count, early_window_cap), dtype=np.int64), "truncated_run_early_only"
    scope = "full" if count >= full_window_count else "truncated"
    return scope, np.arange(count, dtype=np.int64), None


def aggregate_attribution_scores(
    attribution: np.ndarray,
    *,
    unit_order: Sequence[str],
    unit_indices: Mapping[str, np.ndarray],
    time_reduction: str,
    unit_reduction: str,
    normalize_scores: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if time_reduction != "abs_mean":
        raise ValueError(f"Unsupported time_reduction={time_reduction}")
    if unit_reduction != "sum":
        raise ValueError(f"Unsupported unit_reduction={unit_reduction}")
    variable_scores = np.abs(attribution).mean(axis=1)
    raw = np.zeros((variable_scores.shape[0], len(unit_order)), dtype=np.float64)
    for idx, unit_name in enumerate(unit_order):
        raw[:, idx] = variable_scores[:, unit_indices[unit_name]].sum(axis=1)
    if not normalize_scores:
        return raw, raw.copy()
    denom = raw.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    normalized = raw / denom
    return raw, normalized


def build_run_views(
    observable: np.ndarray,
    *,
    window_length: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = window_count(int(observable.shape[0]), window_length, horizon)
    x_view = np.moveaxis(
        sliding_window_view(observable, window_shape=window_length, axis=0)[:count],
        -1,
        1,
    )
    y_view = np.moveaxis(
        sliding_window_view(observable[window_length:], window_shape=horizon, axis=0)[:count],
        -1,
        1,
    )
    return x_view, y_view


def sort_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    mode_preference: Sequence[str],
    min_windows_to_process: int,
) -> list[dict[str, Any]]:
    mode_rank = {mode: idx for idx, mode in enumerate(mode_preference)}
    normalized = [dict(item) for item in candidates]
    return sorted(
        normalized,
        key=lambda item: (
            0 if int(item.get("window_count", 0)) >= min_windows_to_process else 1,
            mode_rank.get(str(item.get("mode")), len(mode_rank)),
            -int(item.get("window_count", 0)),
            str(item.get("run_key")),
        ),
    )


def select_runs(
    *,
    subset_runs: Sequence[Mapping[str, Any]],
    representative_cfg: Mapping[str, Any],
    min_windows_to_process: int,
) -> list[dict[str, Any]]:
    explicit_run_keys = [str(item) for item in representative_cfg.get("run_keys", []) or []]
    if explicit_run_keys:
        index = {str(run["run_key"]): dict(run) for run in subset_runs}
        selected: list[dict[str, Any]] = []
        for run_key in explicit_run_keys:
            if run_key not in index:
                raise KeyError(f"Requested run_key not found in propagation subset: {run_key}")
            selected.append(index[run_key])
        return selected

    scenario_order = [str(item) for item in representative_cfg.get("scenarios", []) or []]
    allowed_modes = {str(item) for item in representative_cfg.get("modes", []) or []}
    mode_preference = [str(item) for item in representative_cfg.get("mode_preference", []) or []]
    max_runs_per_scenario = int(representative_cfg.get("max_runs_per_scenario", 1))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in subset_runs:
        scenario = str(run["scenario"])
        if scenario_order and scenario not in scenario_order:
            continue
        if allowed_modes and str(run["mode"]) not in allowed_modes:
            continue
        grouped.setdefault(scenario, []).append(dict(run))

    selected = []
    ordered_scenarios = scenario_order or sorted(grouped)
    for scenario in ordered_scenarios:
        candidates = grouped.get(scenario, [])
        if not candidates:
            continue
        selected.extend(
            sort_candidates(
                candidates,
                mode_preference=mode_preference,
                min_windows_to_process=min_windows_to_process,
            )[: max_runs_per_scenario]
        )
    return selected


def maybe_train_method(config: Mapping[str, Any], bundle_path: Path, output_root: Path) -> tuple[BaseMethod, dict[str, Any]]:
    method_cfg = config["method"]
    state_path = method_cfg.get("state_path")
    if isinstance(state_path, str) and Path(state_path).exists():
        method = BaseMethod.load(state_path)
        return method, {"source": "checkpoint", "state_path": str(Path(state_path).resolve())}

    if not bool(method_cfg.get("train_if_missing", False)):
        raise FileNotFoundError(f"Method state not found: {state_path}")

    method_name = str(method_cfg.get("name", "devo"))
    fit_cfg = dict(method_cfg.get("fit", {}))
    if method_name != "devo":
        raise ValueError("On-demand training is only implemented for devo in this experiment.")

    devo_config = DeVoConfig(**fit_cfg)
    method = create_method(method_name, config=devo_config)
    bundle = load_dataset_bundle(bundle_path)
    result = method.fit(bundle)
    saved_state = None
    if bool(method_cfg.get("save_trained_state", True)):
        method_dir = output_root / "method"
        method_dir.mkdir(parents=True, exist_ok=True)
        saved_state = method.save(method_dir / "method_state.json")
    return method, {
        "source": "trained_on_demand",
        "state_path": None if saved_state is None else str(saved_state.resolve()),
        "training_summary": dict(result.training_summary),
    }


def ensure_bundle_meta(method: BaseMethod, bundle_path: Path) -> None:
    if method.bundle_meta is not None:
        return
    bundle = load_dataset_bundle(bundle_path)
    method.bundle_meta = bundle.meta


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_analysis(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    scenario_override: Sequence[str],
    mode_override: Sequence[str],
    run_key_override: Sequence[str],
) -> dict[str, Any]:
    paths = config["paths"]
    project_root = Path(paths["project_root"])
    processed_root = Path(paths["processed_root"])
    bundle_path = Path(paths["bundle_path"])
    output_root = Path(paths["output_root"])
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    processed_manifest = load_json(bundle_path)
    bundle_meta = processed_manifest.get("bundle_meta", {})
    if not isinstance(bundle_meta, Mapping):
        raise ValueError("Processed manifest missing bundle_meta")
    bundle_extras = bundle_meta.get("extras", {}) if isinstance(bundle_meta.get("extras"), Mapping) else {}
    model_input_columns = bundle_extras.get("model_input_columns", [])
    if not isinstance(model_input_columns, list):
        raise ValueError("bundle_meta.extras.model_input_columns must be a list")

    five_unit_definition = load_json(Path(paths["five_unit_definition"]))
    truth_table = load_json(Path(paths["fault_truth_table"]))
    propagation_subset = load_json(Path(paths["propagation_subset"]))
    truth_by_scenario = truth_rows_by_scenario(truth_table)
    unit_order, unit_indices = build_unit_index_map(five_unit_definition, model_input_columns)

    representative_cfg = dict(config["experiment"]["representative_subset"])
    if scenario_override:
        representative_cfg["scenarios"] = list(scenario_override)
    if mode_override:
        representative_cfg["modes"] = list(mode_override)
    if run_key_override:
        representative_cfg["run_keys"] = list(run_key_override)

    variable_cfg = config["experiment"]["variable_length"]
    min_windows_to_process = int(variable_cfg["min_windows_to_process"])
    short_run_policy = str(variable_cfg["short_run_policy"]).strip().lower()
    early_window_cap = int(variable_cfg["early_window_cap"])
    window_length = int(processed_manifest["window_length"])
    horizon = int(processed_manifest["horizon"])
    full_window_count = window_count(int(variable_cfg["full_run_steps"]), window_length, horizon)

    subset_runs = propagation_subset.get("runs", [])
    if not isinstance(subset_runs, list):
        raise ValueError("Propagation subset manifest missing runs")
    selected_runs = select_runs(
        subset_runs=subset_runs,
        representative_cfg=representative_cfg,
        min_windows_to_process=min_windows_to_process,
    )
    if not selected_runs:
        raise ValueError("No runs selected for propagation analysis.")

    method, method_info = maybe_train_method(config, bundle_path, output_root)
    ensure_bundle_meta(method, bundle_path)

    attribution_cfg = config["attribution"] if "attribution" in config else config["experiment"]["attribution"]
    batch_size = int(attribution_cfg["batch_size"])
    attribute_mode = str(attribution_cfg["mode"])
    window_stride = int(attribution_cfg.get("window_stride", 1))
    normalize_scores = bool(attribution_cfg.get("normalize_scores", True))
    time_reduction = str(attribution_cfg.get("time_reduction", "abs_mean"))
    unit_reduction = str(attribution_cfg.get("unit_reduction", "sum"))

    timeline_rows: list[dict[str, Any]] = []
    run_status: list[dict[str, Any]] = []

    for run in selected_runs:
        run_key = str(run["run_key"])
        scenario = str(run["scenario"])
        mode = str(run["mode"])
        observable_path = processed_root / str(run["observable_file"])
        if not observable_path.exists():
            raise FileNotFoundError(f"Missing observable run file: {observable_path}")
        observable = np.load(observable_path, mmap_mode="r")
        count = window_count(int(observable.shape[0]), window_length, horizon)
        scope, indices, policy_note = effective_run_scope(
            count=count,
            min_windows_to_process=min_windows_to_process,
            short_run_policy=short_run_policy,
            early_window_cap=early_window_cap,
            full_window_count=full_window_count,
        )
        if indices is not None and window_stride > 1:
            indices = indices[::window_stride]
        truth = truth_by_scenario.get(scenario, {})
        status_row = {
            "run_key": run_key,
            "scenario": scenario,
            "mode": mode,
            "n_steps": int(observable.shape[0]),
            "available_window_count": count,
            "processed_window_count": 0 if indices is None else int(indices.size),
            "analysis_scope": scope,
            "skip_reason": "" if scope != "skipped" else str(policy_note or ""),
            "policy_note": "" if policy_note is None else str(policy_note),
            "truth_primary_unit": truth.get("primary_unit"),
            "truth_expected_units": "|".join(str(item) for item in truth.get("expected_units", []) or []),
            "truth_note": truth.get("note"),
        }
        if indices is None or indices.size == 0:
            run_status.append(status_row)
            continue

        x_view, y_view = build_run_views(observable, window_length=window_length, horizon=horizon)
        for batch_start in range(0, indices.size, batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            x_batch = np.asarray(x_view[batch_indices], dtype=np.float32)
            y_batch = np.asarray(y_view[batch_indices], dtype=np.float32)
            attribution = method.attribute(x_batch, y_batch, batch_size=batch_size, mode=attribute_mode)
            raw_scores, normalized_scores = aggregate_attribution_scores(
                attribution,
                unit_order=unit_order,
                unit_indices=unit_indices,
                time_reduction=time_reduction,
                unit_reduction=unit_reduction,
                normalize_scores=normalize_scores,
            )
            for local_idx, window_idx in enumerate(batch_indices.tolist()):
                dominant_idx = int(np.argmax(normalized_scores[local_idx]))
                sorted_idx = np.argsort(normalized_scores[local_idx])[::-1]
                dominant_score = float(normalized_scores[local_idx, dominant_idx])
                second_score = float(normalized_scores[local_idx, int(sorted_idx[1])]) if len(unit_order) > 1 else 0.0
                row = {
                    "run_key": run_key,
                    "scenario": scenario,
                    "mode": mode,
                    "window_index": int(window_idx),
                    "time_index": int(window_idx + window_length - 1),
                    "window_start": int(window_idx),
                    "window_end": int(window_idx + window_length - 1),
                    "target_time_index": int(window_idx + window_length + horizon - 1),
                    "n_steps": int(observable.shape[0]),
                    "available_window_count": count,
                    "processed_window_count": int(indices.size),
                    "analysis_scope": scope,
                    "dominant_unit": unit_order[dominant_idx],
                    "dominant_score": dominant_score,
                    "dominant_margin": dominant_score - second_score,
                }
                for unit_pos, unit_name in enumerate(unit_order):
                    row[f"{unit_name}_raw"] = float(raw_scores[local_idx, unit_pos])
                    row[f"{unit_name}_score"] = float(normalized_scores[local_idx, unit_pos])
                timeline_rows.append(row)
        run_status.append(status_row)

    timeline_fieldnames = [
        "run_key",
        "scenario",
        "mode",
        "window_index",
        "time_index",
        "window_start",
        "window_end",
        "target_time_index",
        "n_steps",
        "available_window_count",
        "processed_window_count",
        "analysis_scope",
        *(f"{unit_name}_raw" for unit_name in unit_order),
        *(f"{unit_name}_score" for unit_name in unit_order),
        "dominant_unit",
        "dominant_score",
        "dominant_margin",
    ]
    status_fieldnames = [
        "run_key",
        "scenario",
        "mode",
        "n_steps",
        "available_window_count",
        "processed_window_count",
        "analysis_scope",
        "skip_reason",
        "policy_note",
        "truth_primary_unit",
        "truth_expected_units",
        "truth_note",
    ]
    write_csv(raw_root / "unit_attribution_timeline.csv", timeline_rows, timeline_fieldnames)
    write_csv(raw_root / "run_status.csv", run_status, status_fieldnames)

    selected_runs_payload = {
        "generated_at": utc_now(),
        "selected_runs": selected_runs,
    }
    with (raw_root / "selected_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(selected_runs_payload), handle, ensure_ascii=False, indent=2)

    manifest = {
        "experiment_name": config["experiment"]["name"],
        "generated_at": utc_now(),
        "config_path": str(config_path.resolve()),
        "project_root": str(project_root),
        "processed_root": str(processed_root),
        "bundle_path": str(bundle_path),
        "unit_order": unit_order,
        "window_length": window_length,
        "horizon": horizon,
        "full_window_count_reference": full_window_count,
        "selected_run_count": len(selected_runs),
        "timeline_row_count": len(timeline_rows),
        "method": method_info,
        "inputs": {
            "five_unit_definition": str(Path(paths["five_unit_definition"]).resolve()),
            "fault_truth_table": str(Path(paths["fault_truth_table"]).resolve()),
            "propagation_subset": str(Path(paths["propagation_subset"]).resolve()),
        },
        "outputs": {
            "timeline_csv": str((raw_root / "unit_attribution_timeline.csv").resolve()),
            "run_status_csv": str((raw_root / "run_status.csv").resolve()),
            "selected_runs_json": str((raw_root / "selected_runs.json").resolve()),
        },
    }
    with (output_root / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(manifest), handle, ensure_ascii=False, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TEP fault propagation attribution analysis.")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.yaml"),
        type=Path,
        help="Path to experiment config YAML.",
    )
    parser.add_argument("--scenario", action="append", default=[], help="Override representative scenarios.")
    parser.add_argument("--mode", action="append", default=[], help="Restrict selected runs to specific modes.")
    parser.add_argument("--run-key", action="append", default=[], help="Use explicit run keys.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    manifest = run_analysis(
        config=config,
        config_path=args.config.resolve(),
        scenario_override=args.scenario,
        mode_override=args.mode,
        run_key_override=args.run_key,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
