"""Metadata checks for nonlinear benchmark manifest family."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from data.adapters.base import DataProtocolError

BENCHMARK_REQUIRED_KEYS = {
    "benchmark_name",
    "system_type",
    "task_usage",
    "input_channels",
    "output_channels",
    "default_window_length",
    "default_horizon",
    "has_ground_truth_kernel",
    "has_ground_truth_gfrf",
    "recommended_split_protocol",
}


def _validate_boolean(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise DataProtocolError(f"{name} must be boolean for nonlinear manifest entry.")


def _validate_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value <= 0:
        raise DataProtocolError(f"{name} must be a positive integer.")
    return value


def validate_benchmark_entry(entry: Mapping[str, Any]) -> None:
    missing = BENCHMARK_REQUIRED_KEYS.difference(entry)
    if missing:
        raise DataProtocolError(
            f"nonlinear benchmark entry '{entry.get('benchmark_name', 'unknown')}' "
            f"missing required keys: {', '.join(sorted(missing))}"
        )

    if not isinstance(entry["benchmark_name"], str) or not entry["benchmark_name"].strip():
        raise DataProtocolError("benchmark_name must be a non-empty string.")
    if not isinstance(entry["system_type"], str) or not entry["system_type"].strip():
        raise DataProtocolError("system_type must be a non-empty string.")
    if not isinstance(entry["task_usage"], list) or not entry["task_usage"]:
        raise DataProtocolError("task_usage must be a non-empty list.")
    if not isinstance(entry["input_channels"], list) or not entry["input_channels"]:
        raise DataProtocolError("input_channels must be a non-empty list.")
    if not isinstance(entry["output_channels"], list) or not entry["output_channels"]:
        raise DataProtocolError("output_channels must be a non-empty list.")

    _validate_int("default_window_length", entry["default_window_length"])
    _validate_int("default_horizon", entry["default_horizon"])
    _validate_boolean("has_ground_truth_kernel", entry["has_ground_truth_kernel"])
    _validate_boolean("has_ground_truth_gfrf", entry["has_ground_truth_gfrf"])

    if not isinstance(entry["recommended_split_protocol"], str) or not entry["recommended_split_protocol"].strip():
        raise DataProtocolError("recommended_split_protocol must be non-empty string.")


def validate_truth_entry(entry: Mapping[str, Any], truth_key: str) -> None:
    benchmark_name = entry.get("benchmark_name", "unknown")
    has_key = "has_ground_truth_kernel" if truth_key == "kernel" else "has_ground_truth_gfrf"
    if has_key not in entry:
        raise DataProtocolError(
            f"truth manifest entry '{benchmark_name}' missing '{has_key}'."
        )

    if not isinstance(entry[has_key], bool):
        raise DataProtocolError(f"{has_key} must be boolean for '{benchmark_name}'.")

    reference_key = "kernel_reference" if truth_key == "kernel" else "gfrf_reference"
    if entry.get(reference_key) is not None and not isinstance(entry.get(reference_key), str):
        raise DataProtocolError(f"{reference_key} must be string when provided for '{benchmark_name}'.")


def validate_benchmark_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise DataProtocolError("benchmark manifest must be a mapping.")
    items = manifest.get("benchmarks")
    if not isinstance(items, list):
        raise DataProtocolError("benchmark manifest must contain list field 'benchmarks'.")

    for item in items:
        if not isinstance(item, Mapping):
            raise DataProtocolError("Each benchmark manifest entry must be a mapping.")
        validate_benchmark_entry(item)


def validate_truth_manifest(manifest: Mapping[str, Any], truth_key: str) -> None:
    if not isinstance(manifest, Mapping):
        raise DataProtocolError(f"{truth_key} manifest must be a mapping.")
    items = manifest.get("benchmarks")
    if not isinstance(items, list):
        raise DataProtocolError(f"{truth_key} manifest must contain list field 'benchmarks'.")

    for item in items:
        if not isinstance(item, Mapping):
            raise DataProtocolError("Each truth manifest entry must be a mapping.")
        validate_truth_entry(item, truth_key=truth_key)


def validate_cross_manifest_consistency(
    benchmark_manifest: Mapping[str, Any],
    kernel_manifest: Mapping[str, Any],
    gfrf_manifest: Mapping[str, Any],
    *,
    require_kernel_entry: bool = True,
    require_gfrf_entry: bool = True,
) -> None:
    b = {
        item.get("benchmark_name"): item
        for item in benchmark_manifest.get("benchmarks", [])
        if isinstance(item, Mapping)
    }
    k = {
        item.get("benchmark_name"): item
        for item in kernel_manifest.get("benchmarks", [])
        if isinstance(item, Mapping)
    }
    g = {
        item.get("benchmark_name"): item
        for item in gfrf_manifest.get("benchmarks", [])
        if isinstance(item, Mapping)
    }

    for benchmark_name, bench_item in b.items():
        kernel_item = k.get(benchmark_name)
        gfrf_item = g.get(benchmark_name)

        if require_kernel_entry and kernel_item is None:
            raise DataProtocolError(
                f"kernel manifest missing benchmark '{benchmark_name}'."
            )
        if require_gfrf_entry and gfrf_item is None:
            raise DataProtocolError(
                f"gfrf manifest missing benchmark '{benchmark_name}'."
            )

        if kernel_item is not None:
            if bool(bench_item.get("has_ground_truth_kernel", False)) != bool(
                kernel_item.get("has_ground_truth_kernel", False)
            ):
                raise DataProtocolError(
                    f"Kernel availability mismatch for '{benchmark_name}' between manifests."
                )

        if gfrf_item is not None:
            if bool(bench_item.get("has_ground_truth_gfrf", False)) != bool(
                gfrf_item.get("has_ground_truth_gfrf", False)
            ):
                raise DataProtocolError(
                    f"GFRF availability mismatch for '{benchmark_name}' between manifests."
                )


def validate_nonlin_truth_fields(
    truth_manifest: Mapping[str, Any],
    truth_key: str,
    dataset_names: Iterable[str],
) -> None:
    lookup = {
        item.get("benchmark_name"): item
        for item in truth_manifest.get("benchmarks", [])
        if isinstance(item, Mapping)
    }
    for name in dataset_names:
        entry = lookup.get(name)
        if entry is None:
            raise DataProtocolError(f"{truth_key} manifest missing required dataset '{name}'.")
        validate_truth_entry(entry, truth_key=truth_key)
