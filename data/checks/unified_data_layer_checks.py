"""Unified data-layer checks for Nonlinear / Hydraulic / TEP outputs.

This module is intentionally protocol-first and does not depend on methods,
model, training logic, or experiment runners.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from data.adapters.base import (
    DataProtocolError,
    DatasetArtifacts,
    DatasetBundle,
    DatasetMeta,
    DatasetSplit,
    TaskFamily,
)
from data.checks.bundle_checks import check_dataset_bundle
from data.checks.nonlinear_metadata_checks import (
    validate_benchmark_entry,
    validate_benchmark_manifest,
    validate_cross_manifest_consistency,
    validate_truth_manifest,
)


@dataclass
class CheckReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_set(values: Any) -> set[Any]:
    if values is None:
        return set()
    out: set[Any] = set()
    arr = np.asarray(values, dtype=object).reshape(-1)
    for v in arr:
        if isinstance(v, (bytes, bytearray, memoryview)):
            try:
                v = v.decode("utf-8")
            except Exception:
                v = str(v)
        out.add(v)
    return out


def _to_list(values: Any) -> List[Any]:
    return list(np.asarray(values, dtype=object).reshape(-1))


def _resolve_path(base: Path, value: Optional[Any]) -> Optional[Path]:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    if "://" in value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path


def _pick_first_existing(path_root: Path, candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_processed_manifest(processed_root: Path, dataset_name: str) -> Optional[Path]:
    candidates = [
        processed_root / f"{dataset_name}_processed_manifest.json",
        processed_root / "processed_manifest.json",
        processed_root / "manifest.json",
    ]
    manifest = _pick_first_existing(processed_root, candidates)
    if manifest is not None:
        return manifest
    by_glob = sorted(processed_root.glob("*processed_manifest.json"))
    if by_glob:
        return by_glob[0]
    by_any = sorted(processed_root.glob("*.json"))
    return by_any[0] if by_any else None


def _get_bundle_payload(manifest: Mapping[str, Any], processed_root: Path) -> Tuple[MutableMapping[str, Any], MutableMapping[str, Any]]:
    payload = manifest.get("bundle_meta")
    if payload is None:
        payload = manifest.get("meta", {})
    artifacts = manifest.get("bundle_artifacts")
    if artifacts is None:
        artifacts = manifest.get("artifacts", {})
    if not isinstance(payload, Mapping):
        payload = {}
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    return dict(payload), dict(artifacts)


def _processed_file_map(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    files = manifest.get("processed_files")
    if isinstance(files, Mapping):
        return files
    files = manifest.get("files")
    if isinstance(files, Mapping):
        return files
    return {}


def _load_split_arrays(
    processed_root: Path,
    processed_files: Mapping[str, Any],
    split_name: str,
) -> Dict[str, np.ndarray]:
    payload: Dict[str, np.ndarray] = {}
    required = ["X", "Y", "sample_id", "run_id", "timestamp", "mode", "fault_id", "scenario_id", "window_idx", "window_start", "window_end", "idv_aux", "idv_label", "cycle_id", "fault_label", "subsystem_label", "condition_type"]
    for key in required:
        candidate_names = [
            f"{split_name}_{key}",
            f"{split_name}_{key.lower()}",
            f"{split_name}_{key.upper()}",
        ]
        fname: Optional[str] = None
        for name in candidate_names:
            if name in processed_files:
                fname = str(processed_files[name])
                break
        if fname is None:
            direct = processed_root / f"{split_name}_{key}.npy"
            if direct.exists():
                fname = str(direct)
        if fname is not None:
            path = processed_root / fname
            if path.suffix.lower() != ".npy":
                continue
            if path.exists():
                payload[key] = np.load(path, allow_pickle=True)
            else:
                raise FileNotFoundError(f"missing processed file: {path}")
    return payload


def _load_split_payloads(processed_root: Path, split_manifest: Mapping[str, Any], dataset: str) -> Dict[str, Dict[str, np.ndarray]]:
    payload: Dict[str, Dict[str, np.ndarray]] = {}
    processed_files = _processed_file_map(split_manifest)
    for split_name in ("train", "val", "test"):
        payload[split_name] = _load_split_arrays(processed_root, processed_files, split_name)
        if "X" not in payload[split_name] or "Y" not in payload[split_name]:
            raise RuntimeError(f"{dataset} {split_name} split missing X or Y in processed payload.")
    return payload


def _build_dataset_bundle(
    dataset_name: str,
    meta_payload: Mapping[str, Any],
    artifacts_payload: Mapping[str, Any],
    split_payloads: Mapping[str, Mapping[str, Any]],
) -> DatasetBundle:
    return DatasetBundle(
        train=DatasetSplit.from_mapping(split_payloads["train"], split_name="train"),
        val=DatasetSplit.from_mapping(split_payloads["val"], split_name="val"),
        test=DatasetSplit.from_mapping(split_payloads["test"], split_name="test"),
        meta=DatasetMeta.from_mapping(meta_payload),
        artifacts=DatasetArtifacts.from_mapping(artifacts_payload),
    )


def _check_shapes(report: CheckReport, split_payloads: Mapping[str, Mapping[str, np.ndarray]], meta_payload: Mapping[str, Any], dataset: str) -> None:
    for split_name, payload in split_payloads.items():
        x = np.asarray(payload["X"])
        y = np.asarray(payload["Y"])
        if x.ndim < 2:
            report.error(f"{dataset}:{split_name}.X must be 2D or higher, got shape={x.shape}")
            continue
        if y.ndim < 2:
            report.error(f"{dataset}:{split_name}.Y must be 2D or higher, got shape={y.shape}")
            continue
        if len(x) != len(y):
            report.error(f"{dataset}:{split_name} X/Y length mismatch: {len(x)} vs {len(y)}")
        x_len = len(x)
        for key in ("sample_id", "run_id", "timestamp"):
            if key in payload and payload[key] is not None:
                if len(payload[key]) != x_len:
                    report.error(f"{dataset}:{split_name}.{key} length {len(payload[key])} != X length {x_len}")

    input_dim = meta_payload.get("input_dim")
    output_dim = meta_payload.get("output_dim")
    if isinstance(input_dim, int) and input_dim > 0:
        train_x = np.asarray(split_payloads["train"]["X"])
        if train_x.size == 0 or train_x.shape[-1] != input_dim:
            report.error(f"{dataset}: meta.input_dim={input_dim}, but train X last dim is {train_x.shape[-1] if train_x.size else 0}")
    if isinstance(output_dim, int) and output_dim > 0:
        train_y = np.asarray(split_payloads["train"]["Y"])
        if train_y.size == 0 or train_y.shape[-1] != output_dim:
            report.error(f"{dataset}: meta.output_dim={output_dim}, but train Y last dim is {train_y.shape[-1] if train_y.size else 0}")


def _check_split_disjointness(report: CheckReport, split_payloads: Mapping[str, Mapping[str, np.ndarray]], split_indices_payload: Mapping[str, Any], dataset: str) -> None:
    indices = split_indices_payload.get("split_indices", {})
    split_names = ("train", "val", "test")
    if isinstance(indices, Mapping):
        parsed = {name: list(_to_list(indices.get(name, []))) for name in split_names}
        sets = {name: set(int(x) for x in parsed[name]) for name in split_names}
        for name in split_names:
            if len(parsed[name]) != len(sets[name]):
                report.warning(f"{dataset}: {name} split indices contain duplicates; deduplicated in checks.")
        for i, a in enumerate(split_names):
            for b in split_names[i + 1 :]:
                overlap = sets[a] & sets[b]
                if overlap:
                    report.error(f"{dataset}: split leakage by sample index in {a}/{b}, overlap size={len(overlap)}")

    for key in ("sample_id", "run_id", "window_idx", "cycle_id"):
        if all(key in split_payloads[s] for s in split_names):
            sets = {split_name: _to_set(split_payloads[split_name][key]) for split_name in split_names}
            for i, a in enumerate(split_names):
                for b in split_names[i + 1 :]:
                    overlap = sets[a] & sets[b]
                    if overlap:
                        report.error(f"{dataset}: split leakage by `{key}` in {a}/{b}, overlap size={len(overlap)}")
            break


def _get_text_map(data_root: Path, relative: str) -> Optional[Mapping[str, Any]]:
    path = data_root / relative
    if not path.exists():
        return None
    return _read_json(path)


def _artifact_path_ok(report: CheckReport, dataset: str, value: Optional[Any], what: str, data_root: Path) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        report.error(f"{dataset}: {what} must be str or null if provided (got {type(value).__name__})")
        return
    if value == "":
        report.error(f"{dataset}: {what} is empty string")
        return
    if "://" in value:
        return
    path = Path(value)
    if not path.is_absolute():
        path = data_root / path
    if not path.exists():
        report.warning(f"{dataset}: {what} path not found locally: {path}")


def validate_generic_bundle(
    report: CheckReport,
    dataset: str,
    data_root: Path,
    family: TaskFamily,
    processed_root: Path,
    split_root: Path,
) -> bool:
    manifest = _find_processed_manifest(processed_root, dataset)
    if manifest is None:
        report.error(f"{dataset}: cannot find processed manifest under {processed_root}")
        return False
    manifest_payload = _read_json(manifest)
    split_file = manifest_payload.get("split_file")
    if isinstance(split_file, str):
        split_path = (data_root / split_file) if not Path(split_file).is_absolute() else Path(split_file)
    else:
        split_path = split_root / f"{dataset}_split_manifest.json"
    if not split_path.exists():
        report.error(f"{dataset}: split manifest missing at {split_path}")
        return False
    split_payload = _read_json(split_path)

    meta_payload, artifact_payload = _get_bundle_payload(manifest_payload, processed_root)
    if not meta_payload:
        report.error(f"{dataset}: processed manifest missing bundle_meta/meta.")
        return False

    # Build split payloads from npy files.
    try:
        split_payloads = _load_split_payloads(processed_root, manifest_payload, dataset)
    except Exception as exc:
        report.error(f"{dataset}: failed to load processed split payloads: {exc}")
        return False

    # Generic schema and shape checks.
    _check_shapes(report, split_payloads, meta_payload, dataset)
    _check_split_disjointness(report, split_payloads, split_payload, dataset)

    try:
        bundle = _build_dataset_bundle(dataset, meta_payload, artifact_payload, split_payloads)
        if bundle.meta.task_family != family:
            report.error(
                f"{dataset}: expected task_family={family.value}, got {bundle.meta.task_family.value}."
            )
        check_dataset_bundle(bundle)
    except DataProtocolError as exc:
        report.error(f"{dataset}: protocol validation failed: {exc}")
        return False
    except Exception as exc:
        report.error(f"{dataset}: cannot construct DatasetBundle for checks: {exc}")
        return False

    return report.ok


def validate_nonlinear_dataset(
    report: CheckReport,
    data_root: Path,
    dataset: str,
    benchmark_manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    gfrf_manifest: Mapping[str, Any],
) -> bool:
    meta_root = data_root / "metadata" / "nonlinear"
    processed_root = data_root / "processed" / "nonlinear" / dataset
    split_root = data_root / "splits" / "nonlinear"

    if not validate_generic_bundle(
        report=report,
        dataset=dataset,
        data_root=data_root,
        family=TaskFamily.NONLINEAR,
        processed_root=processed_root,
        split_root=split_root,
    ):
        return False

    manifest_payload = _read_json(_find_processed_manifest(processed_root, dataset))
    meta_payload, artifact_payload = _get_bundle_payload(manifest_payload, processed_root)
    try:
        meta = DatasetMeta.from_mapping(meta_payload)
    except Exception as exc:
        report.error(f"{dataset}: invalid bundle meta: {exc}")
        return False

    # manifest consistency between benchmark and builder outputs
    benchmark_entry = None
    for item in benchmark_manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset:
            benchmark_entry = item
            break
    if not isinstance(benchmark_entry, Mapping):
        report.error(f"{dataset}: benchmark_manifest missing entry.")
        return False

    try:
        validate_benchmark_entry(benchmark_entry)
    except Exception as exc:
        report.error(f"{dataset}: invalid benchmark manifest entry: {exc}")

    if meta.has_ground_truth_kernel != bool(benchmark_entry.get("has_ground_truth_kernel", False)):
        report.error(
            f"{dataset}: meta.has_ground_truth_kernel {meta.has_ground_truth_kernel} != benchmark manifest {benchmark_entry.get('has_ground_truth_kernel')}"
        )
    if meta.has_ground_truth_gfrf != bool(benchmark_entry.get("has_ground_truth_gfrf", False)):
        report.error(
            f"{dataset}: meta.has_ground_truth_gfrf {meta.has_ground_truth_gfrf} != benchmark manifest {benchmark_entry.get('has_ground_truth_gfrf')}"
        )

    # manifest consistency checks
    try:
        validate_truth_manifest(kernel_manifest, "kernel")
        validate_truth_manifest(gfrf_manifest, "gfrf")
        validate_cross_manifest_consistency(benchmark_manifest, kernel_manifest, gfrf_manifest)
    except Exception as exc:
        report.error(f"{dataset}: manifest consistency check failed: {exc}")

    kernel_entry = None
    gfrf_entry = None
    for item in kernel_manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset:
            kernel_entry = item
            break
    for item in gfrf_manifest.get("benchmarks", []):
        if isinstance(item, Mapping) and item.get("benchmark_name") == dataset:
            gfrf_entry = item
            break

    if kernel_entry is None:
        report.error(f"{dataset}: kernel truth manifest missing entry.")
    if gfrf_entry is None:
        report.error(f"{dataset}: gfrf truth manifest missing entry.")

    artifacts = DatasetArtifacts.from_mapping(artifact_payload)
    if meta.has_ground_truth_kernel:
        if not artifacts.truth_file:
            report.error(f"{dataset}: has_ground_truth_kernel=True but artifacts.truth_file is empty.")
        else:
            _artifact_path_ok(report, dataset, artifacts.truth_file, "artifacts.truth_file", data_root)

    if meta.has_ground_truth_gfrf:
        if not artifacts.extra.get("gfrf_reference_file"):
            maybe_gfrf = artifacts.extra.get("gfrf_reference")
            if not maybe_gfrf:
                report.warning(f"{dataset}: gfrf flag true, but truth gfrf artifact path is not set in bundle artifacts.extra.")
        else:
            _artifact_path_ok(report, dataset, artifacts.extra.get("gfrf_reference_file"), "artifacts.extra.gfrf_reference_file", data_root)

    if kernel_entry:
        if bool(kernel_entry.get("has_ground_truth_kernel", False)) != meta.has_ground_truth_kernel:
            report.error(f"{dataset}: kernel manifest flag mismatch with bundle meta.")
    if gfrf_entry:
        if bool(gfrf_entry.get("has_ground_truth_gfrf", False)) != meta.has_ground_truth_gfrf:
            report.error(f"{dataset}: gfrf manifest flag mismatch with bundle meta.")

    if dataset == "cascaded_tanks":
        benchmark_extra = benchmark_entry.get("artifacts", {}).get("extra", {})
        raw_sources = benchmark_extra.get("raw_sources")
        if not isinstance(raw_sources, list) or not any(
            isinstance(item, Mapping) and "dataBenchmark" in str(item.get("path", ""))
            for item in raw_sources
        ):
            report.error("cascaded_tanks: benchmark_manifest is missing dataBenchmark raw source traceability.")

        raw_field_mapping = benchmark_extra.get("raw_field_mapping")
        if not isinstance(raw_field_mapping, Mapping):
            report.error("cascaded_tanks: benchmark_manifest is missing raw_field_mapping.")

        raw_trace = manifest_payload.get("raw_source")
        if not isinstance(raw_trace, Mapping):
            meta_raw_trace = meta.extras.get("raw_source") if isinstance(meta.extras, Mapping) else None
            artifact_raw_trace = artifacts.extra.get("raw_source") if isinstance(artifacts.extra, Mapping) else None
            raw_trace = meta_raw_trace if isinstance(meta_raw_trace, Mapping) else artifact_raw_trace

        if not isinstance(raw_trace, Mapping):
            report.error("cascaded_tanks: processed manifest or bundle metadata is missing raw_source.")
        else:
            selected_path = raw_trace.get("selected_path") or raw_trace.get("path")
            if not isinstance(selected_path, str) or "dataBenchmark" not in Path(selected_path).name:
                report.error("cascaded_tanks: raw_source.selected_path must reference dataBenchmark.mat/csv.")
            else:
                _artifact_path_ok(
                    report,
                    dataset,
                    selected_path,
                    "raw_source.selected_path",
                    data_root,
                )
            if raw_trace.get("raw_format") not in {"mat", "csv"}:
                report.error("cascaded_tanks: raw_source.raw_format must be 'mat' or 'csv'.")
            if raw_trace.get("discovery_method") not in {"whitelist", "fallback"}:
                report.warning("cascaded_tanks: raw_source.discovery_method should be whitelist or fallback.")

        processed_segment_roles = manifest_payload.get("segment_roles")
        if not isinstance(processed_segment_roles, Mapping):
            report.error("cascaded_tanks: processed manifest is missing segment_roles.")
        else:
            for segment_name in ("estimation", "validation"):
                segment_spec = processed_segment_roles.get(segment_name)
                if not isinstance(segment_spec, Mapping):
                    report.error(f"cascaded_tanks: processed manifest missing segment role '{segment_name}'.")

        split_file = manifest_payload.get("split_file")
        split_path = (data_root / split_file) if isinstance(split_file, str) and not Path(split_file).is_absolute() else Path(split_file) if isinstance(split_file, str) else split_root / f"{dataset}_split_manifest.json"
        split_payload = _read_json(split_path)
        split_sources = split_payload.get("split_sources")
        expected_split_sources = {"train": "estimation", "val": "validation", "test": "validation"}
        if split_sources != expected_split_sources:
            report.error(
                "cascaded_tanks: split_sources must map train->estimation and val/test->validation."
            )

        split_segment_roles = split_payload.get("segment_roles")
        if not isinstance(split_segment_roles, Mapping):
            report.error("cascaded_tanks: split manifest is missing segment_roles.")

    if dataset == "coupled_duffing":
        benchmark_extra = benchmark_entry.get("artifacts", {}).get("extra", {})
        raw_sources = benchmark_extra.get("raw_sources")
        if not isinstance(raw_sources, list):
            report.error("coupled_duffing: benchmark_manifest is missing raw_sources.")
        else:
            raw_paths = {str(item.get("path")) for item in raw_sources if isinstance(item, Mapping)}
            if "DATAUNIF.MAT" not in raw_paths or "DATAPRBS.MAT" not in raw_paths:
                report.error("coupled_duffing: benchmark_manifest raw_sources must include DATAUNIF.MAT and DATAPRBS.MAT.")

        trajectory_mapping = benchmark_extra.get("trajectory_field_mapping")
        expected_trajectories = {"uniform_1", "uniform_2", "prbs_1", "prbs_2", "prbs_3"}
        if not isinstance(trajectory_mapping, Mapping):
            report.error("coupled_duffing: benchmark_manifest is missing trajectory_field_mapping.")
        else:
            missing = expected_trajectories.difference(set(trajectory_mapping.keys()))
            if missing:
                report.error(f"coupled_duffing: benchmark_manifest trajectory_field_mapping missing {sorted(missing)}.")

        raw_trace = manifest_payload.get("raw_sources")
        if not isinstance(raw_trace, Mapping):
            meta_raw = meta.extras.get("raw_sources") if isinstance(meta.extras, Mapping) else None
            artifact_raw = artifacts.extra.get("raw_sources") if isinstance(artifacts.extra, Mapping) else None
            raw_trace = meta_raw if isinstance(meta_raw, Mapping) else artifact_raw

        if not isinstance(raw_trace, Mapping):
            report.error("coupled_duffing: processed manifest or bundle metadata is missing raw_sources.")
        else:
            for key, expected_name in (("uniform_source", "DATAUNIF"), ("prbs_source", "DATAPRBS")):
                source = raw_trace.get(key)
                if not isinstance(source, Mapping):
                    report.error(f"coupled_duffing: raw_sources missing '{key}'.")
                    continue
                selected_path = source.get("selected_path")
                if not isinstance(selected_path, str) or expected_name not in Path(selected_path).name.upper():
                    report.error(f"coupled_duffing: raw_sources.{key}.selected_path must reference {expected_name}.")
                else:
                    _artifact_path_ok(
                        report,
                        dataset,
                        selected_path,
                        f"raw_sources.{key}.selected_path",
                        data_root,
                    )
                if source.get("raw_format") not in {"mat", "csv"}:
                    report.error(f"coupled_duffing: raw_sources.{key}.raw_format must be 'mat' or 'csv'.")

        split_file = manifest_payload.get("split_file")
        split_path = (data_root / split_file) if isinstance(split_file, str) and not Path(split_file).is_absolute() else Path(split_file) if isinstance(split_file, str) else split_root / f"{dataset}_split_manifest.json"
        split_payload = _read_json(split_path)

        expected_split_sources = {
            "train": ["uniform_1", "uniform_2"],
            "val": ["prbs_1"],
            "test": ["prbs_2", "prbs_3"],
        }
        if split_payload.get("split_sources") != expected_split_sources:
            report.error(
                "coupled_duffing: split_sources must map train->[uniform_1,uniform_2], val->[prbs_1], test->[prbs_2,prbs_3]."
            )

        trajectory_roles = split_payload.get("trajectory_roles")
        if not isinstance(trajectory_roles, Mapping):
            report.error("coupled_duffing: split manifest is missing trajectory_roles.")
        else:
            missing = expected_trajectories.difference(set(trajectory_roles.keys()))
            if missing:
                report.error(f"coupled_duffing: split manifest trajectory_roles missing {sorted(missing)}.")

        split_arrays = _load_split_payloads(processed_root, manifest_payload, dataset)
        expected_run_ids = {
            "train": {"uniform_1", "uniform_2"},
            "val": {"prbs_1"},
            "test": {"prbs_2", "prbs_3"},
        }
        for split_name, allowed in expected_run_ids.items():
            payload = split_arrays.get(split_name, {})
            if "run_id" not in payload:
                report.error(f"coupled_duffing: {split_name} split is missing run_id.")
                continue
            run_ids = set(_to_set(payload["run_id"]))
            if not run_ids.issubset(allowed):
                diff = sorted(run_ids - allowed)
                report.error(f"coupled_duffing: {split_name} split contains unexpected run_id values: {diff}")

    return report.ok


def _load_json_or_warning(report: CheckReport, data_root: Path, rel: str, dataset: str) -> Optional[Mapping[str, Any]]:
    payload = _get_text_map(data_root, rel)
    if payload is None:
        report.error(f"{dataset}: required metadata file missing: {rel}")
    return payload


def validate_hydraulic_dataset(report: CheckReport, data_root: Path) -> bool:
    dataset = "hydraulic"
    processed_root = data_root / "processed" / "hydraulic"
    split_root = data_root / "splits" / "hydraulic"

    if not validate_generic_bundle(
        report=report,
        dataset=dataset,
        data_root=data_root,
        family=TaskFamily.HYDRAULIC,
        processed_root=processed_root,
        split_root=split_root,
    ):
        return False

    metadata_root = data_root / "metadata" / dataset
    channel_map = _load_json_or_warning(report, data_root, f"metadata/{dataset}/channel_map.json", dataset)
    subsystem_groups = _load_json_or_warning(report, data_root, f"metadata/{dataset}/subsystem_groups.json", dataset)
    single_fault_protocol = _load_json_or_warning(report, data_root, f"metadata/{dataset}/single_fault_protocol.json", dataset)
    if channel_map is None or subsystem_groups is None or single_fault_protocol is None:
        return False

    channels = channel_map.get("channels")
    if not isinstance(channels, list) or len(channels) != 17:
        report.error("hydraulic: channel_map.channels must be a list of exactly 17 entries.")
    else:
        expected_channel_order = [
            "PS1",
            "PS2",
            "PS3",
            "PS4",
            "PS5",
            "PS6",
            "EPS1",
            "FS1",
            "FS2",
            "TS1",
            "TS2",
            "TS3",
            "TS4",
            "VS1",
            "CE",
            "CP",
            "SE",
        ]
        if channels != expected_channel_order:
            report.error("hydraulic: channel_map.channels does not match the canonical 17-channel order.")

    subsystem_map = subsystem_groups.get("subsystems")
    if not isinstance(subsystem_map, Mapping):
        report.error("hydraulic: subsystem_groups.subsystems must be a mapping.")
    else:
        required = {"cooler", "valve", "pump", "accumulator"}
        missing = required.difference(set(subsystem_map.keys()))
        if missing:
            report.error(f"hydraulic: subsystem_groups missing required groups: {sorted(missing)}")
        for name, spec in subsystem_map.items():
            chans = spec.get("channels") if isinstance(spec, Mapping) else None
            if not isinstance(chans, list) or len(chans) == 0:
                report.warning(f"hydraulic: subsystem '{name}' has empty/invalid channel list.")
            if isinstance(chans, list):
                absent = [c for c in chans if c not in channels]
                if absent:
                    report.error(f"hydraulic: subsystem '{name}' references unknown channels: {absent}")
        expected_groups = {
            "cooler": ["TS1", "CE", "CP"],
            "valve": ["FS1", "PS2"],
            "pump": ["PS1", "PS3", "PS4", "PS5", "PS6", "EPS1", "VS1", "SE"],
            "accumulator": ["TS2", "TS3", "TS4", "FS2"],
        }
        for name, expected_channels in expected_groups.items():
            spec = subsystem_map.get(name)
            actual = spec.get("channels") if isinstance(spec, Mapping) else None
            if actual != expected_channels:
                report.error(f"hydraulic: subsystem '{name}' channels do not match the paper-defined grouping.")

    split_protocol = single_fault_protocol.get("split_protocol")
    if split_protocol is None:
        report.error("hydraulic: single_fault_protocol.split_protocol missing.")

    if single_fault_protocol.get("labeling", {}).get("subsystem_class_order") is None:
        report.error("hydraulic: single_fault_protocol.labeling.subsystem_class_order missing.")
    nominal = single_fault_protocol.get("nominal_conditions")
    if nominal != {
        "cooler_condition": 100,
        "valve_condition": 100,
        "pump_leakage": 0,
        "accumulator_pressure": 130,
    }:
        report.error("hydraulic: nominal_conditions must match the paper-defined healthy state.")

    stable_flag_policy = single_fault_protocol.get("stable_flag_policy")
    if not isinstance(stable_flag_policy, Mapping):
        report.error("hydraulic: stable_flag_policy missing from single_fault_protocol.")
    else:
        if stable_flag_policy.get("builder_behavior") != "retain_all_cycles":
            report.error("hydraulic: stable_flag_policy.builder_behavior must be `retain_all_cycles`.")

    # load built split payloads to verify dimensional consistency and required fields
    manifest = _read_json(_find_processed_manifest(processed_root, dataset))
    available_representations = manifest.get("available_representations")
    if not isinstance(available_representations, Mapping):
        report.error("hydraulic: processed manifest missing available_representations.")
    else:
        expected_reps = {"cycle_60", "cycle_600"}
        if set(available_representations.keys()) != expected_reps:
            report.error("hydraulic: available_representations must expose cycle_60 and cycle_600.")
    if manifest.get("representation_name") != "cycle_60":
        report.error("hydraulic: top-level processed manifest must default to representation cycle_60.")

    split_payload = _read_json((data_root / manifest.get("split_file") if isinstance(manifest.get("split_file"), str) else split_root / "hydraulic_split_manifest.json"))
    processed_files = _processed_file_map(manifest)
    split_arrays = _load_split_payloads(processed_root, manifest, dataset)

    for split_name in ("train", "val", "test"):
        payload = split_arrays[split_name]
        required_fields = {"fault_label", "subsystem_label", "condition_type", "cycle_id", "sample_id"}
        missing = [f for f in required_fields if f not in payload]
        if missing:
            report.warning(f"hydraulic {split_name} split missing expected fields: {missing}")

    train_x = np.asarray(split_arrays["train"]["X"])
    train_y = np.asarray(split_arrays["train"]["Y"])
    if channels and train_x.size:
        if train_x.shape[-1] != len(channels):
            report.error(
                f"hydraulic: input feature count mismatch, expected {len(channels)} from metadata channel_map but train X has {train_x.shape[-1]}"
            )

    if single_fault_protocol.get("labeling", {}).get("subsystem_class_order") and train_y.size:
        expected = len(single_fault_protocol["labeling"]["subsystem_class_order"])
        if train_y.shape[-1] != expected:
            report.error(
                f"hydraulic: output dim mismatch, expected {expected} classes, train Y has {train_y.shape[-1]}"
            )

    # cycle-wise leakage check
    grouping = single_fault_protocol.get("grouping", {}) if isinstance(single_fault_protocol, Mapping) else {}
    if grouping.get("level") == "cycle":
        group_key_present = all("cycle_id" in split_arrays[s] for s in ("train", "val", "test"))
        if not group_key_present:
            report.warning("hydraulic: grouping.level is cycle, but cycle_id is unavailable in processed split payloads.")

    return report.ok


def validate_tep_dataset(report: CheckReport, data_root: Path) -> bool:
    dataset = "tep"
    processed_root = data_root / "processed" / "tep"
    split_root = data_root / "splits" / "tep"

    if not validate_generic_bundle(
        report=report,
        dataset=dataset,
        data_root=data_root,
        family=TaskFamily.TEP,
        processed_root=processed_root,
        split_root=split_root,
    ):
        return False

    metadata_root = data_root / "metadata" / dataset
    channel_map = _load_json_or_warning(report, data_root, f"metadata/{dataset}/channel_map.json", dataset)
    five_unit = _load_json_or_warning(report, data_root, f"metadata/{dataset}/five_unit_definition.json", dataset)
    feed_side = _load_json_or_warning(report, data_root, f"metadata/{dataset}/feed_side_definition.json", dataset)
    truth_table = _load_json_or_warning(report, data_root, f"metadata/{dataset}/fault_truth_table.json", dataset)
    protocol = _load_json_or_warning(report, data_root, f"metadata/{dataset}/mode_holdout_protocol.json", dataset)
    scenario_map = _load_json_or_warning(report, data_root, f"metadata/{dataset}/scenario_to_idv_map.json", dataset)
    if channel_map is None or five_unit is None or feed_side is None or truth_table is None or protocol is None or scenario_map is None:
        return False

    obs = channel_map.get("observable_variables", {})
    idv = channel_map.get("idv_variables", {})
    if not isinstance(obs, Mapping) or int(obs.get("count", 0)) != 53:
        report.error("tep: observable_variables.count must be 53.")
    if not isinstance(idv, Mapping) or int(idv.get("count", 0)) != 28:
        report.error("tep: idv_variables.count must be 28.")

    manifest = _read_json(_find_processed_manifest(processed_root, dataset))
    meta_payload, artifact_payload = _get_bundle_payload(manifest, processed_root)
    split_payload = _read_json((data_root / manifest.get("split_file") if isinstance(manifest.get("split_file"), str) else split_root / "tep_mode_holdout_split_manifest.json"))

    manifest_meta = DatasetMeta.from_mapping(meta_payload)
    if manifest_meta.input_dim != 53:
        report.error("tep: bundle meta.input_dim must be 53 (v01-v53 model inputs).")
    if manifest_meta.has_ground_truth_kernel or manifest_meta.has_ground_truth_gfrf:
        report.warning("tep: ground-truth kernel/GFRF flags expected false by protocol.")

    if not obs.get("names"):
        report.error("tep: channel_map.observable_variables.names missing.")
    if not idv.get("names"):
        report.error("tep: channel_map.idv_variables.names missing.")
    if list(channel_map.get("model_input_columns", [])) != list(obs.get("names", [])):
        report.error("tep: channel_map.model_input_columns must exactly match v01-v53 observable columns.")
    if list(channel_map.get("metadata_only_columns", [])) != list(idv.get("names", [])):
        report.error("tep: channel_map.metadata_only_columns must exactly match v54-v81 IDV columns.")

    five_units = five_unit.get("five_units")
    if not isinstance(five_units, Mapping) or len(five_units) != 5:
        report.error("tep: five_unit_definition.five_units should define 5 units.")
    eval_cfg = five_unit.get("evaluation", {}) if isinstance(five_unit, Mapping) else {}
    if not bool(eval_cfg.get("exclude_feed_side_unit", False)):
        report.warning("tep: five_unit_definition.evaluation.exclude_feed_side_unit should be true.")
    expected_five_units = {
        "Reactor": {"v05", "v06", "v07", "v08", "v09", "v20", "v23", "v24", "v25", "v26", "v27", "v28", "v46", "v47"},
        "Condenser": {"v32", "v33", "v34", "v48", "v49"},
        "Separator": {"v10", "v11", "v12", "v13", "v14", "v22", "v29", "v30", "v31", "v35", "v36"},
        "Compressor": {"v21", "v42", "v43", "v44", "v45"},
        "Stripper": {"v15", "v16", "v17", "v18", "v19", "v37", "v38", "v39", "v40", "v41"},
    }
    if isinstance(five_units, Mapping):
        for unit_name, expected_vars in expected_five_units.items():
            actual = five_units.get(unit_name, {})
            actual_vars = set(actual.get("variables", [])) if isinstance(actual, Mapping) else set()
            if actual_vars != expected_vars:
                report.error(f"tep: five_unit_definition for {unit_name} does not match required variable set.")

    expected_feed_side = {"v01", "v02", "v03", "v04", "v50", "v51", "v52", "v53"}
    actual_feed_side = set(feed_side.get("feed_side_variables", [])) if isinstance(feed_side, Mapping) else set()
    if actual_feed_side != expected_feed_side:
        report.error("tep: feed_side_definition must match v01-v04 and v50-v53.")
    if bool(feed_side.get("included_in_unit_scoring", True)):
        report.error("tep: feed-side variables must not be included in unit scoring.")
    if actual_feed_side & {var for vars_ in expected_five_units.values() for var in vars_}:
        report.error("tep: feed-side variables must not overlap any five-unit variable set.")

    # split-level mode protocol checks
    mode_names = protocol.get("modes") if isinstance(protocol, Mapping) else {}
    train_modes = set(mode_names.get("train_modes", [])) if isinstance(mode_names, Mapping) else set()
    val_modes = set(mode_names.get("val_modes", [])) if isinstance(mode_names, Mapping) else set()
    test_normal_modes = set(mode_names.get("test_normal_modes", [])) if isinstance(mode_names, Mapping) else set()
    if not train_modes or not val_modes or not test_normal_modes:
        report.warning("tep: mode-holdout protocol mode lists (train/val/test) incomplete.")
    if train_modes != {"M1", "M2", "M3", "M4"}:
        report.error("tep: mode-holdout train_modes must be M1-M4.")
    if val_modes != {"M5"}:
        report.error("tep: mode-holdout val_modes must be M5.")
    if test_normal_modes != {"M6"}:
        report.error("tep: mode-holdout test_normal_modes must be M6.")
    run_sets = protocol.get("run_sets", {}) if isinstance(protocol, Mapping) else {}
    if set(run_sets.get("train_normal_run_keys", [])) != {"M1_d00", "M2_d00", "M3_d00", "M4_d00"}:
        report.error("tep: mode-holdout train_normal_run_keys must be M1_d00..M4_d00.")
    if set(run_sets.get("val_normal_run_keys", [])) != {"M5_d00"}:
        report.error("tep: mode-holdout val_normal_run_keys must be M5_d00.")
    if set(run_sets.get("normal_test_run_keys", [])) != {"M6_d00"}:
        report.error("tep: mode-holdout normal_test_run_keys must be M6_d00.")
    if set(run_sets.get("fault_eval_scenarios", [])) != {f"d{i:02d}" for i in range(1, 29)}:
        report.error("tep: mode-holdout fault_eval_scenarios must cover d01-d28.")
    windowing = protocol.get("windowing", {}) if isinstance(protocol, Mapping) else {}
    if not bool(windowing.get("window_within_run_only", False)):
        report.error("tep: windowing.window_within_run_only must be true.")
    if bool(windowing.get("cross_run_windows", True)):
        report.error("tep: windowing.cross_run_windows must be false.")

    # load actual split arrays and check mode constraints
    processed_files = _processed_file_map(manifest)
    split_arrays = _load_split_payloads(processed_root, manifest, dataset)
    for split_name in ("train", "val", "test"):
        payload = split_arrays[split_name]
        if "mode" not in payload:
            report.warning(f"tep: {split_name} split is missing mode field.")
            continue
        split_modes = set(_to_set(payload["mode"]))
        if split_name == "train" and train_modes and not split_modes.issubset(train_modes):
            diff = sorted(split_modes - train_modes)
            report.error(f"tep: train split contains unexpected modes not in train_modes: {diff}")
        if split_name == "val" and val_modes and not split_modes.issubset(val_modes):
            diff = sorted(split_modes - val_modes)
            report.error(f"tep: val split contains unexpected modes not in val_modes: {diff}")
        if "scenario_id" in payload:
            scenarios = set(_to_set(payload["scenario_id"]))
            if scenarios != {"d00"}:
                report.error(f"tep: {split_name} split must contain only d00 windows, got {sorted(scenarios)}")
        if "fault_id" in payload:
            fault_values = set(_to_set(payload["fault_id"]))
            if fault_values != {"healthy"}:
                report.error(f"tep: {split_name} split must contain only healthy labels, got {sorted(fault_values)}")

    if "test" in split_arrays and "mode" in split_arrays["test"]:
        test_mode_set = set(_to_set(split_arrays["test"]["mode"]))
        normal_ok = bool(test_normal_modes)
        if normal_ok and not test_mode_set.issubset(test_normal_modes):
            diff = sorted(test_mode_set - test_normal_modes)
            report.error(f"tep: test split contains unexpected modes: {diff}")

    # five-unit and truth-table consistency
    rows = truth_table.get("rows") if isinstance(truth_table, Mapping) else []
    if not rows:
        report.error("tep: fault_truth_table.rows missing.")
    scenario_rows: Dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, list):
        known_units = set(five_units.keys()) if isinstance(five_units, Mapping) else set()
        for row in rows:
            if not isinstance(row, Mapping):
                report.warning("tep: fault_truth_table row is not a mapping.")
                continue
            scenario = row.get("scenario")
            if not isinstance(scenario, str) or not scenario.startswith("d"):
                report.error("tep: fault_truth_table rows must be keyed by raw dxx scenario.")
                continue
            scenario_rows[scenario] = row
            pu = row.get("primary_unit")
            exp = row.get("expected_units", [])
            if pu is not None and pu not in known_units:
                report.error(f"tep: fault_truth_table primary_unit '{pu}' is not defined in five_unit_definition.five_units.")
            if not isinstance(exp, list) or any(u not in known_units for u in exp):
                report.error(f"tep: fault_truth_table expected_units invalid for scenario={row.get('scenario')}.")
        if set(scenario_rows.keys()) != {f"d{i:02d}" for i in range(1, 29)}:
            report.error("tep: fault_truth_table must contain rows for d01-d28.")
        if not any(bool(row.get("included_in_main_eval", False)) for row in scenario_rows.values()):
            report.warning("tep: fault_truth_table has no included_in_main_eval=true rows; main five-unit subset will be empty until curated truth is provided.")

    scenario_mapping = scenario_map.get("mapping", {}) if isinstance(scenario_map, Mapping) else {}
    if not isinstance(scenario_mapping, Mapping):
        report.error("tep: scenario_to_idv_map.mapping missing.")
    else:
        expected_map = {f"d{i:02d}": f"idv{29 - i:02d}" for i in range(1, 29)}
        actual_map = {
            key: str(value.get("idv"))
            for key, value in scenario_mapping.items()
            if isinstance(value, Mapping)
        }
        if actual_map != expected_map:
            report.error("tep: scenario_to_idv_map must implement reverse mapping d01->idv28 ... d28->idv01.")
    healthy = scenario_map.get("healthy", {}) if isinstance(scenario_map, Mapping) else {}
    if str(healthy.get("scenario")) != "d00" or str(healthy.get("idv")) != "healthy":
        report.error("tep: scenario_to_idv_map.healthy must describe d00 -> healthy.")

    # leakage check: v54-v81 must remain auxiliary metadata
    manifest_files = _processed_file_map(manifest)
    train_payload = _load_split_arrays(processed_root, manifest_files, "train")
    if "X" in train_payload and train_payload["X"].ndim >= 2:
        if train_payload["X"].shape[-1] != 53:
            report.error(f"tep: model input X has dim {train_payload['X'].shape[-1]}, expected 53 (v01-v53 only).")
    if "idv_aux" not in train_payload:
        report.warning("tep: idv_aux missing from train split; v54-v81 should be preserved as metadata/aux.")
    else:
        if train_payload["idv_aux"].ndim >= 2 and train_payload["idv_aux"].shape[-1] != 28:
            report.error(f"tep: idv_aux must have 28 IDV channels, got {train_payload['idv_aux'].shape[-1]}.")
    if "idv_label" not in train_payload:
        report.warning("tep: idv_label missing from train split.")

    # protocol/truth references in manifest for reproducibility
    _artifact_path_ok(report, dataset, artifact_payload.get("protocol_file"), "artifacts.protocol_file", data_root)
    _artifact_path_ok(report, dataset, artifact_payload.get("grouping_file"), "artifacts.grouping_file", data_root)
    if isinstance(artifact_payload.get("extra"), Mapping):
        _artifact_path_ok(report, dataset, artifact_payload["extra"].get("split_file"), "artifacts.extra.split_file", data_root)
        _artifact_path_ok(report, dataset, artifact_payload["extra"].get("scenario_to_idv_map_file"), "artifacts.extra.scenario_to_idv_map_file", data_root)
        _artifact_path_ok(report, dataset, artifact_payload["extra"].get("feed_side_definition_file"), "artifacts.extra.feed_side_definition_file", data_root)
        _artifact_path_ok(report, dataset, artifact_payload["extra"].get("run_manifest_file"), "artifacts.extra.run_manifest_file", data_root)
        _artifact_path_ok(report, dataset, artifact_payload["extra"].get("fault_eval_manifest_file"), "artifacts.extra.fault_eval_manifest_file", data_root)

    extras = manifest_meta.extras if isinstance(manifest_meta.extras, Mapping) else {}
    if extras.get("variable_length_fault_runs") is not True:
        report.error("tep: bundle meta.extras.variable_length_fault_runs must be true.")
    if int(extras.get("normal_run_n_steps", 0)) != 7201:
        report.error("tep: bundle meta.extras.normal_run_n_steps must be 7201.")

    run_manifest_path = _resolve_path(data_root, manifest.get("run_manifest_file")) or _resolve_path(data_root, extras.get("run_manifest_file"))
    if run_manifest_path is None or not run_manifest_path.exists():
        report.error("tep: run_manifest_file missing from processed outputs.")
    else:
        run_manifest = _read_json(run_manifest_path)
        runs = run_manifest.get("runs", [])
        if len(runs) != 174:
            report.error(f"tep: run manifest should contain 174 runs, got {len(runs)}.")
        normal_steps = {int(run.get("n_steps", -1)) for run in runs if isinstance(run, Mapping) and not bool(run.get("is_fault"))}
        if normal_steps != {7201}:
            report.error(f"tep: all normal d00 runs must have 7201 steps, got {sorted(normal_steps)}.")
        fault_steps = {int(run.get("n_steps", -1)) for run in runs if isinstance(run, Mapping) and bool(run.get("is_fault"))}
        if fault_steps == {7201}:
            report.error("tep: run manifest incorrectly implies all fault runs are equal length.")

    fault_manifest_path = _resolve_path(data_root, manifest.get("fault_eval_manifest_file")) or _resolve_path(data_root, extras.get("fault_eval_manifest_file"))
    if fault_manifest_path is None or not fault_manifest_path.exists():
        report.error("tep: fault_eval_manifest_file missing from processed outputs.")
    else:
        fault_manifest = _read_json(fault_manifest_path)
        fault_runs = fault_manifest.get("runs", [])
        if len(fault_runs) != 168:
            report.error(f"tep: fault eval manifest should contain 168 fault runs, got {len(fault_runs)}.")

    # fault ids in split should be covered by truth table (when fault mode)
    truth_faults = {str(row["fault_id"]) for row in rows if isinstance(row, Mapping) and row.get("fault_id")}
    for split_name in ("train", "val", "test"):
        payload = split_arrays[split_name]
        if "fault_id" not in payload:
            continue
        for raw in _to_set(payload["fault_id"]):
            fid = str(raw)
            if fid in {"healthy", "NORMAL", "", "None", "NA", "N/A"}:
                continue
            if truth_faults and fid not in truth_faults:
                report.error(f"tep: split {split_name} contains fault_id '{fid}' not defined in fault_truth_table.")

    return report.ok


def validate_data_layer(project_root: Path, task_family: str = "all", datasets: Optional[Sequence[str]] = None) -> CheckReport:
    project_root = project_root.expanduser().resolve()
    report = CheckReport()

    data_root = project_root / "data"
    if not data_root.exists():
        report.error(f"data directory missing at {data_root}")
        return report

    # load nonlinear manifests once
    nonlinear_meta_root = data_root / "metadata" / "nonlinear"
    benchmark_manifest = _read_json(nonlinear_meta_root / "benchmark_manifest.json")
    kernel_manifest = _read_json(nonlinear_meta_root / "kernel_truth_manifest.json")
    gfrf_manifest = _read_json(nonlinear_meta_root / "gfrf_truth_manifest.json")

    if task_family in ("all", "nonlinear"):
        try:
            validate_benchmark_manifest(benchmark_manifest)
        except Exception as exc:
            report.error(f"nonlinear: benchmark manifest invalid: {exc}")
        # names from benchmark manifest drive the validation targets.
        nonlinear_datasets = [
            item.get("benchmark_name")
            for item in benchmark_manifest.get("benchmarks", [])
            if isinstance(item, Mapping) and item.get("benchmark_name")
        ]
        for name in nonlinear_datasets:
            if datasets and name not in datasets:
                continue
            validate_nonlinear_dataset(
                report=report,
                data_root=data_root,
                dataset=name,
                benchmark_manifest=benchmark_manifest,
                kernel_manifest=kernel_manifest,
                gfrf_manifest=gfrf_manifest,
            )

    if task_family in ("all", "hydraulic"):
        if datasets is None or "hydraulic" in datasets:
            validate_hydraulic_dataset(report, data_root)

    if task_family in ("all", "tep"):
        if datasets is None or "tep" in datasets:
            validate_tep_dataset(report, data_root)

    return report


def print_report(report: CheckReport) -> None:
    if report.ok:
        print("data-layer checks passed")
    else:
        print("data-layer checks failed")
    for message in report.errors:
        print(f"[ERROR] {message}")
    for message in report.warnings:
        print(f"[WARN] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate unified data-layer outputs.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--task-family",
        default="all",
        choices=("all", "nonlinear", "hydraulic", "tep"),
        help="Limit checks to one task family.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Optional dataset name filter. Can be used multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_data_layer(
        project_root=args.project_root,
        task_family=args.task_family,
        datasets=args.dataset,
    )
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
