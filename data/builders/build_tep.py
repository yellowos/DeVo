"""Build Tennessee Eastman Process (TEP) multimode dataset artifacts.

This builder operates on the multimode directory layout:
- data/raw/tep/M1 ... data/raw/tep/M6
- each mode contains mXd00.mat and mXd01.mat ... mXd28.mat

The data layer keeps run-level canonical arrays for every run and materializes
normal mode-holdout windows for train/val/test. Fault runs remain variable-length
run-level artifacts plus lightweight window-count manifests to avoid fabricating
fixed-length fault tensors.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap
from numpy.lib.stride_tricks import sliding_window_view

from data.adapters.base import DataProtocolError, DatasetBundle
from data.adapters.tep_adapter import TEPAdapter
from data.checks.bundle_checks import check_dataset_bundle


class TEPBuilderError(DataProtocolError):
    """Raised when TEP build cannot satisfy protocol constraints."""


@dataclass(frozen=True)
class TEPBuilderConfig:
    dataset_name: str = "tep"
    window_length: int = 128
    horizon: int = 1
    split_protocol: str = "tep_multimode_mode_holdout_v2"
    expected_modes: tuple[str, ...] = ("M1", "M2", "M3", "M4", "M5", "M6")

    def validate(self) -> None:
        if self.window_length <= 0:
            raise TEPBuilderError("window_length must be > 0")
        if self.horizon <= 0:
            raise TEPBuilderError("horizon must be > 0")


RUN_PATTERN = re.compile(r"^m(?P<mode>[1-6])(?P<scenario>d\d{2})$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _window_count(n_steps: int, window_length: int, horizon: int) -> int:
    return max(0, n_steps - window_length - horizon + 1)


def _parse_run_stem(stem: str) -> Dict[str, Any]:
    match = RUN_PATTERN.match(stem)
    if match is None:
        raise TEPBuilderError(f"invalid TEP run stem `{stem}`")
    mode_idx = int(match.group("mode"))
    scenario = match.group("scenario")
    mode = f"M{mode_idx}"
    return {
        "mode": mode,
        "mode_index": mode_idx,
        "scenario": scenario,
        "run_key": f"{mode}_{scenario}",
        "is_fault": scenario != "d00",
    }


def _load_core_matrix(path: Path) -> tuple[str, np.ndarray, List[str]]:
    try:
        from scipy.io import loadmat
    except Exception as exc:  # pragma: no cover
        raise TEPBuilderError("scipy is required to read TEP .mat files") from exc

    stem = path.stem
    payload = loadmat(path)
    available_vars = [key for key in payload.keys() if not key.startswith("__")]
    if stem not in payload:
        raise TEPBuilderError(
            f"{path.name}: core variable `{stem}` missing; available variables={available_vars}"
        )
    core = np.asarray(payload[stem])
    if core.ndim != 2:
        raise TEPBuilderError(f"{path.name}: core variable `{stem}` must be 2D, got shape={core.shape}.")
    if core.shape[1] != 81:
        raise TEPBuilderError(f"{path.name}: expected 81 columns, got {core.shape[1]}.")
    return stem, np.asarray(core, dtype=np.float32), available_vars


def discover_raw(raw_root: Path, cfg: TEPBuilderConfig) -> Dict[str, Any]:
    if not raw_root.exists():
        raise TEPBuilderError(f"TEP raw directory missing: {raw_root}")

    discovery_modes: List[Dict[str, Any]] = []
    for mode in cfg.expected_modes:
        mode_root = raw_root / mode
        if not mode_root.exists():
            raise TEPBuilderError(f"missing mode directory: {mode_root}")
        expected_stems = {f"{mode.lower()}d{idx:02d}" for idx in range(29)}
        actual_files = sorted([path for path in mode_root.glob("*.mat") if path.is_file()])
        actual_stems = {path.stem for path in actual_files}
        missing = sorted(expected_stems - actual_stems)
        if missing:
            raise TEPBuilderError(f"{mode}: missing required runs: {missing}")
        unexpected = sorted(actual_stems - expected_stems)
        discovery_modes.append(
            {
                "mode": mode,
                "mode_dir": str(mode_root),
                "d00_exists": f"{mode.lower()}d00" in actual_stems,
                "fault_count": sum(1 for stem in actual_stems if stem.endswith(tuple(f"{i:02d}" for i in range(1, 29)))),
                "missing_runs": missing,
                "unexpected_runs": unexpected,
                "files": [path.name for path in actual_files if path.stem in expected_stems],
            }
        )

    actual_run_count = int(sum(len(entry["files"]) for entry in discovery_modes))
    return {
        "raw_root": str(raw_root),
        "modes": discovery_modes,
        "mode_count": len(discovery_modes),
        "expected_run_count": len(cfg.expected_modes) * 29,
        "run_count": actual_run_count,
    }


def _save_run_level_arrays(
    *,
    runs_root: Path,
    run_key: str,
    x_obs: np.ndarray,
    x_idv: np.ndarray,
    time_index: np.ndarray,
) -> Dict[str, str]:
    runs_root.mkdir(parents=True, exist_ok=True)
    obs_name = f"{run_key}_X_obs.npy"
    idv_name = f"{run_key}_X_idv.npy"
    time_name = f"{run_key}_time_index.npy"
    np.save(runs_root / obs_name, np.asarray(x_obs, dtype=np.float32))
    np.save(runs_root / idv_name, np.asarray(x_idv, dtype=np.float32))
    np.save(runs_root / time_name, np.asarray(time_index, dtype=np.int64))
    return {
        "observable_file": f"runs/{obs_name}",
        "idv_file": f"runs/{idv_name}",
        "time_index_file": f"runs/{time_name}",
    }


def load_runs(
    *,
    raw_root: Path,
    processed_root: Path,
    cfg: TEPBuilderConfig,
    scenario_to_idv_map: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    runs_root = processed_root / "runs"
    mapping = scenario_to_idv_map.get("mapping", {})
    healthy = scenario_to_idv_map.get("healthy", {})
    if not isinstance(mapping, Mapping):
        raise TEPBuilderError("scenario_to_idv_map.mapping must be a mapping.")

    runs: List[Dict[str, Any]] = []
    for mode in cfg.expected_modes:
        mode_root = raw_root / mode
        for idx in range(29):
            scenario = f"d{idx:02d}"
            filename = f"{mode.lower()}{scenario}.mat"
            path = mode_root / filename
            stem, core, available_vars = _load_core_matrix(path)
            parsed = _parse_run_stem(stem)
            if not parsed["is_fault"] and int(core.shape[0]) != 7201:
                raise TEPBuilderError(
                    f"{path.name}: normal run expected 7201 rows, got {core.shape[0]}."
                )
            x_obs = np.asarray(core[:, :53], dtype=np.float32)
            x_idv = np.asarray(core[:, 53:], dtype=np.float32)
            time_index = np.arange(core.shape[0], dtype=np.int64)
            saved = _save_run_level_arrays(
                runs_root=runs_root,
                run_key=parsed["run_key"],
                x_obs=x_obs,
                x_idv=x_idv,
                time_index=time_index,
            )

            if parsed["is_fault"]:
                scenario_info = mapping.get(parsed["scenario"])
                if not isinstance(scenario_info, Mapping):
                    raise TEPBuilderError(
                        f"scenario_to_idv_map missing fault scenario `{parsed['scenario']}`"
                    )
                idv = str(scenario_info["idv"])
                fault_id = parsed["scenario"]
            else:
                idv = str(healthy.get("idv", "healthy"))
                fault_id = "healthy"

            runs.append(
                {
                    **parsed,
                    "file": str(path),
                    "file_name": path.name,
                    "stem": stem,
                    "core_var": stem,
                    "available_vars": available_vars,
                    "n_steps": int(core.shape[0]),
                    "n_obs": int(x_obs.shape[1]),
                    "n_idv": int(x_idv.shape[1]),
                    "idv": idv,
                    "fault_id": fault_id,
                    **saved,
                }
            )

    return runs


def _build_split_arrays(
    *,
    processed_root: Path,
    split_name: str,
    run_entries: Sequence[Mapping[str, Any]],
    window_length: int,
    horizon: int,
) -> tuple[Dict[str, str], Dict[str, Any]]:
    total = sum(_window_count(int(run["n_steps"]), window_length, horizon) for run in run_entries)
    if total <= 0:
        raise TEPBuilderError(f"{split_name}: no windows available.")

    windows_root = processed_root / "windows"
    windows_root.mkdir(parents=True, exist_ok=True)

    x_file = f"windows/{split_name}_X.npy"
    y_file = f"windows/{split_name}_Y.npy"
    idv_file = f"windows/{split_name}_idv_aux.npy"
    x_mem = open_memmap(processed_root / x_file, mode="w+", dtype=np.float32, shape=(total, window_length, 53))
    y_mem = open_memmap(processed_root / y_file, mode="w+", dtype=np.float32, shape=(total, horizon, 53))
    idv_mem = open_memmap(processed_root / idv_file, mode="w+", dtype=np.float32, shape=(total, 28))

    timestamp = np.empty(total, dtype=np.float64)
    window_idx = np.empty(total, dtype=np.int64)
    window_start = np.empty(total, dtype=np.int64)
    window_end = np.empty(total, dtype=np.int64)
    sample_id: List[str] = []
    run_id: List[str] = []
    mode: List[str] = []
    fault_id: List[str] = []
    scenario_id: List[str] = []
    idv_label: List[str] = []

    offset = 0
    per_run_counts: Dict[str, int] = {}
    for run in run_entries:
        obs = np.load(processed_root / str(run["observable_file"]), mmap_mode="r")
        idv = np.load(processed_root / str(run["idv_file"]), mmap_mode="r")
        count = _window_count(int(run["n_steps"]), window_length, horizon)
        per_run_counts[str(run["run_key"])] = count
        if count == 0:
            continue

        x_view = np.moveaxis(
            sliding_window_view(obs, window_shape=window_length, axis=0)[:count],
            -1,
            1,
        )
        y_view = np.moveaxis(
            sliding_window_view(obs[window_length:], window_shape=horizon, axis=0)[:count],
            -1,
            1,
        )
        x_mem[offset : offset + count] = x_view
        y_mem[offset : offset + count] = y_view
        idv_mem[offset : offset + count] = idv[window_length - 1 : window_length - 1 + count]

        starts = np.arange(count, dtype=np.int64)
        ends = starts + window_length - 1
        timestamp[offset : offset + count] = ends.astype(np.float64, copy=False)
        window_idx[offset : offset + count] = starts
        window_start[offset : offset + count] = starts
        window_end[offset : offset + count] = ends

        sample_id.extend(f"{run['run_key']}:{idx}" for idx in range(count))
        run_id.extend([str(run["run_key"])] * count)
        mode.extend([str(run["mode"])] * count)
        fault_id.extend([str(run["fault_id"])] * count)
        scenario_id.extend([str(run["scenario"])] * count)
        idv_label.extend([str(run["idv"])] * count)
        offset += count

    del x_mem
    del y_mem
    del idv_mem

    processed_files = {
        f"{split_name}_X": x_file,
        f"{split_name}_Y": y_file,
        f"{split_name}_idv_aux": idv_file,
    }

    extra_arrays = {
        "sample_id": np.asarray(sample_id, dtype=object),
        "run_id": np.asarray(run_id, dtype=object),
        "timestamp": timestamp,
        "mode": np.asarray(mode, dtype=object),
        "fault_id": np.asarray(fault_id, dtype=object),
        "scenario_id": np.asarray(scenario_id, dtype=object),
        "window_idx": window_idx,
        "window_start": window_start,
        "window_end": window_end,
        "idv_label": np.asarray(idv_label, dtype=object),
    }
    for key, array in extra_arrays.items():
        fname = f"windows/{split_name}_{key}.npy"
        np.save(processed_root / fname, array)
        processed_files[f"{split_name}_{key}"] = fname

    return processed_files, {
        "window_count": total,
        "run_keys": [str(run["run_key"]) for run in run_entries],
        "per_run_window_counts": per_run_counts,
    }


def _build_fault_eval_manifest(
    *,
    processed_root: Path,
    fault_runs: Sequence[Mapping[str, Any]],
    truth_rows: Mapping[str, Mapping[str, Any]],
    window_length: int,
    horizon: int,
) -> tuple[Path, Dict[str, Any], Path]:
    entries: List[Dict[str, Any]] = []
    for run in fault_runs:
        truth = truth_rows.get(str(run["scenario"]), {})
        entries.append(
            {
                "run_key": str(run["run_key"]),
                "mode": str(run["mode"]),
                "scenario": str(run["scenario"]),
                "idv": str(run["idv"]),
                "n_steps": int(run["n_steps"]),
                "window_count": _window_count(int(run["n_steps"]), window_length, horizon),
                "observable_file": str(run["observable_file"]),
                "idv_file": str(run["idv_file"]),
                "included_in_main_eval": bool(truth.get("included_in_main_eval", False)),
                "primary_unit": truth.get("primary_unit"),
                "expected_units": list(truth.get("expected_units", [])) if isinstance(truth.get("expected_units", []), list) else [],
            }
        )

    fault_manifest_path = processed_root / "tep_fault_eval_manifest.json"
    payload = {
        "dataset_name": "tep",
        "generated_at": _utc_now(),
        "window_length": window_length,
        "horizon": horizon,
        "run_count": len(entries),
        "total_window_count": int(sum(entry["window_count"] for entry in entries)),
        "runs": entries,
        "notes": "Fault runs are stored at run level plus per-run window counts. Dense fault windows are not materialized because fault runs are variable-length and large.",
    }
    _write_json(fault_manifest_path, payload)

    main_eval_path = processed_root / "tep_main_eval_subset_manifest.json"
    main_eval_runs = [entry for entry in entries if entry["included_in_main_eval"]]
    _write_json(
        main_eval_path,
        {
            "dataset_name": "tep",
            "generated_at": _utc_now(),
            "source": "fault_truth_table.included_in_main_eval",
            "run_count": len(main_eval_runs),
            "runs": main_eval_runs,
            "notes": "This subset depends entirely on curated truth metadata.",
        },
    )

    propagation_path = processed_root / "tep_propagation_subset_manifest.json"
    _write_json(
        propagation_path,
        {
            "dataset_name": "tep",
            "generated_at": _utc_now(),
            "source": "all_fault_runs_default",
            "run_count": len(entries),
            "runs": entries,
            "notes": "Default propagation subset keeps all fault runs until a narrower curated subset is provided.",
        },
    )
    return fault_manifest_path, payload, propagation_path


def _write_run_manifest(
    *,
    processed_root: Path,
    discovery: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> Path:
    path = processed_root / "tep_run_manifest.json"
    _write_json(
        path,
        {
            "dataset_name": "tep",
            "generated_at": _utc_now(),
            "discovery": discovery,
            "run_count": len(runs),
            "normal_run_count": sum(1 for run in runs if not bool(run["is_fault"])),
            "fault_run_count": sum(1 for run in runs if bool(run["is_fault"])),
            "runs": list(runs),
        },
    )
    return path


def _write_split_manifest(
    *,
    split_root: Path,
    protocol: Mapping[str, Any],
    split_counts: Mapping[str, int],
    split_summaries: Mapping[str, Mapping[str, Any]],
    fault_eval_payload: Mapping[str, Any],
) -> Path:
    split_root.mkdir(parents=True, exist_ok=True)
    train_count = int(split_counts["train"])
    val_count = int(split_counts["val"])
    test_count = int(split_counts["test"])
    split_path = split_root / "tep_mode_holdout_split_manifest.json"
    _write_json(
        split_path,
        {
            "dataset_name": "tep",
            "protocol_name": protocol["protocol_name"],
            "split_protocol": protocol["split_protocol"],
            "generated_at": _utc_now(),
            "split_indices": {
                "train": list(range(0, train_count)),
                "val": list(range(train_count, train_count + val_count)),
                "test": list(range(train_count + val_count, train_count + val_count + test_count)),
            },
            "counts": {
                "train": train_count,
                "val": val_count,
                "test": test_count,
                "fault_eval": int(fault_eval_payload["total_window_count"]),
            },
            "run_sets": protocol["run_sets"],
            "modes": protocol["modes"],
            "windowing": protocol["windowing"],
            "split_summary": split_summaries,
            "fault_eval_run_count": int(fault_eval_payload["run_count"]),
        },
    )
    return split_path


def export_bundle(
    *,
    cfg: TEPBuilderConfig,
    project_root: Path,
    processed_root: Path,
    split_root: Path,
    metadata_root: Path,
    channel_map: Mapping[str, Any],
    five_unit: Mapping[str, Any],
    feed_side: Mapping[str, Any],
    truth_table: Mapping[str, Any],
    protocol: Mapping[str, Any],
    scenario_map: Mapping[str, Any],
    discovery: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
) -> DatasetBundle:
    processed_root.mkdir(parents=True, exist_ok=True)
    truth_rows = {
        str(row["scenario"]): row
        for row in truth_table.get("rows", [])
        if isinstance(row, Mapping) and row.get("scenario")
    }

    train_keys = set(protocol["run_sets"]["train_normal_run_keys"])
    val_keys = set(protocol["run_sets"]["val_normal_run_keys"])
    test_keys = set(protocol["run_sets"]["normal_test_run_keys"])
    train_runs = [run for run in runs if str(run["run_key"]) in train_keys]
    val_runs = [run for run in runs if str(run["run_key"]) in val_keys]
    test_runs = [run for run in runs if str(run["run_key"]) in test_keys]
    fault_runs = [run for run in runs if bool(run["is_fault"])]

    processed_files: Dict[str, str] = {}
    split_summaries: Dict[str, Mapping[str, Any]] = {}
    split_counts: Dict[str, int] = {}
    for split_name, split_runs in (("train", train_runs), ("val", val_runs), ("test", test_runs)):
        files, summary = _build_split_arrays(
            processed_root=processed_root,
            split_name=split_name,
            run_entries=split_runs,
            window_length=cfg.window_length,
            horizon=cfg.horizon,
        )
        processed_files.update(files)
        split_summaries[split_name] = summary
        split_counts[split_name] = int(summary["window_count"])

    run_manifest_path = _write_run_manifest(processed_root=processed_root, discovery=discovery, runs=runs)
    fault_manifest_path, fault_eval_payload, propagation_subset_path = _build_fault_eval_manifest(
        processed_root=processed_root,
        fault_runs=fault_runs,
        truth_rows=truth_rows,
        window_length=cfg.window_length,
        horizon=cfg.horizon,
    )
    split_path = _write_split_manifest(
        split_root=split_root,
        protocol=protocol,
        split_counts=split_counts,
        split_summaries=split_summaries,
        fault_eval_payload=fault_eval_payload,
    )

    main_eval_subset_path = processed_root / "tep_main_eval_subset_manifest.json"
    window_manifest_path = processed_root / "tep_window_manifest.json"
    _write_json(
        window_manifest_path,
        {
            "dataset_name": "tep",
            "generated_at": _utc_now(),
            "window_length": cfg.window_length,
            "horizon": cfg.horizon,
            "normal_splits": split_summaries,
            "fault_eval": {
                "manifest_file": str(fault_manifest_path),
                "run_count": int(fault_eval_payload["run_count"]),
                "window_count": int(fault_eval_payload["total_window_count"]),
            },
            "main_eval_subset_manifest_file": str(main_eval_subset_path),
            "propagation_subset_manifest_file": str(propagation_subset_path),
        },
    )

    bundle = TEPAdapter.build_bundle(
        dataset_name=cfg.dataset_name,
        train={
            "X": np.load(processed_root / processed_files["train_X"], allow_pickle=True),
            "Y": np.load(processed_root / processed_files["train_Y"], allow_pickle=True),
            "sample_id": np.load(processed_root / processed_files["train_sample_id"], allow_pickle=True),
            "run_id": np.load(processed_root / processed_files["train_run_id"], allow_pickle=True),
            "timestamp": np.load(processed_root / processed_files["train_timestamp"], allow_pickle=True),
        },
        val={
            "X": np.load(processed_root / processed_files["val_X"], allow_pickle=True),
            "Y": np.load(processed_root / processed_files["val_Y"], allow_pickle=True),
            "sample_id": np.load(processed_root / processed_files["val_sample_id"], allow_pickle=True),
            "run_id": np.load(processed_root / processed_files["val_run_id"], allow_pickle=True),
            "timestamp": np.load(processed_root / processed_files["val_timestamp"], allow_pickle=True),
        },
        test={
            "X": np.load(processed_root / processed_files["test_X"], allow_pickle=True),
            "Y": np.load(processed_root / processed_files["test_Y"], allow_pickle=True),
            "sample_id": np.load(processed_root / processed_files["test_sample_id"], allow_pickle=True),
            "run_id": np.load(processed_root / processed_files["test_run_id"], allow_pickle=True),
            "timestamp": np.load(processed_root / processed_files["test_timestamp"], allow_pickle=True),
        },
        meta={
            "dataset_name": cfg.dataset_name,
            "task_family": "tep",
            "input_dim": 53,
            "output_dim": 53,
            "window_length": cfg.window_length,
            "horizon": cfg.horizon,
            "split_protocol": cfg.split_protocol,
            "has_ground_truth_kernel": False,
            "has_ground_truth_gfrf": False,
            "extras": {
                "model_input_columns": list(channel_map["observable_variables"]["names"]),
                "metadata_only_columns": list(channel_map["idv_variables"]["names"]),
                "channel_map": channel_map,
                "five_unit_definition": five_unit,
                "feed_side_definition": feed_side,
                "mode_holdout_protocol": protocol,
                "window_manifest_file": str(window_manifest_path),
                "run_manifest_file": str(run_manifest_path),
                "fault_eval_manifest_file": str(fault_manifest_path),
                "main_eval_subset_manifest_file": str(main_eval_subset_path),
                "propagation_subset_manifest_file": str(propagation_subset_path),
                "scenario_to_idv_map_file": str(metadata_root / "scenario_to_idv_map.json"),
                "variable_length_fault_runs": True,
                "normal_run_n_steps": 7201,
            },
        },
        artifacts={
            "truth_file": str(metadata_root / "fault_truth_table.json"),
            "grouping_file": str(metadata_root / "five_unit_definition.json"),
            "protocol_file": str(metadata_root / "mode_holdout_protocol.json"),
            "extra": {
                "feed_side_definition_file": str(metadata_root / "feed_side_definition.json"),
                "scenario_to_idv_map_file": str(metadata_root / "scenario_to_idv_map.json"),
                "split_file": str(split_path),
                "run_manifest_file": str(run_manifest_path),
                "window_manifest_file": str(window_manifest_path),
                "fault_eval_manifest_file": str(fault_manifest_path),
                "main_eval_subset_manifest_file": str(main_eval_subset_path),
                "propagation_subset_manifest_file": str(propagation_subset_path),
            },
        },
    )
    check_dataset_bundle(bundle)

    _write_json(
        processed_root / "tep_processed_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": _utc_now(),
            "split_file": str(split_path),
            "window_length": cfg.window_length,
            "horizon": cfg.horizon,
            "raw_discovery": discovery,
            "run_manifest_file": str(run_manifest_path),
            "window_manifest_file": str(window_manifest_path),
            "fault_eval_manifest_file": str(fault_manifest_path),
            "sample_counts": split_counts,
            "processed_files": processed_files,
            "bundle_meta": bundle.meta.to_dict(),
            "bundle_artifacts": bundle.artifacts.to_dict(),
        },
    )

    _write_json(
        metadata_root / "tep_builder_manifest.json",
        {
            "dataset_name": cfg.dataset_name,
            "generated_at": _utc_now(),
            "mode_holdout_protocol_file": str(metadata_root / "mode_holdout_protocol.json"),
            "five_unit_definition_file": str(metadata_root / "five_unit_definition.json"),
            "feed_side_definition_file": str(metadata_root / "feed_side_definition.json"),
            "fault_truth_table_file": str(metadata_root / "fault_truth_table.json"),
            "scenario_to_idv_map_file": str(metadata_root / "scenario_to_idv_map.json"),
            "channel_map_file": str(metadata_root / "channel_map.json"),
            "processed_manifest_file": str(processed_root / "tep_processed_manifest.json"),
            "split_file": str(split_path),
            "run_manifest_file": str(run_manifest_path),
            "window_manifest_file": str(window_manifest_path),
        },
    )

    return bundle


def run_build(project_root: Path) -> DatasetBundle:
    cfg = TEPBuilderConfig()
    cfg.validate()

    project_root = project_root.expanduser().resolve()
    metadata_root = project_root / "data" / "metadata" / "tep"
    raw_root = project_root / "data" / "raw" / "tep"
    processed_root = project_root / "data" / "processed" / "tep"
    split_root = project_root / "data" / "splits" / "tep"

    channel_map = _load_json(metadata_root / "channel_map.json")
    five_unit = _load_json(metadata_root / "five_unit_definition.json")
    feed_side = _load_json(metadata_root / "feed_side_definition.json")
    truth_table = _load_json(metadata_root / "fault_truth_table.json")
    protocol = _load_json(metadata_root / "mode_holdout_protocol.json")
    scenario_map = _load_json(metadata_root / "scenario_to_idv_map.json")

    total_vars = int(channel_map["observable_variables"].get("count", 0)) + int(
        channel_map["idv_variables"].get("count", 0)
    )
    if total_vars != 81:
        raise TEPBuilderError("channel_map must define exactly 81 variables (v01-v81).")

    discovery = discover_raw(raw_root, cfg)
    runs = load_runs(
        raw_root=raw_root,
        processed_root=processed_root,
        cfg=cfg,
        scenario_to_idv_map=scenario_map,
    )

    return export_bundle(
        cfg=cfg,
        project_root=project_root,
        processed_root=processed_root,
        split_root=split_root,
        metadata_root=metadata_root,
        channel_map=channel_map,
        five_unit=five_unit,
        feed_side=feed_side,
        truth_table=truth_table,
        protocol=protocol,
        scenario_map=scenario_map,
        discovery=discovery,
        runs=runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TEP multimode dataset bundle.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_build(args.project_root.expanduser().resolve())


if __name__ == "__main__":
    main()
