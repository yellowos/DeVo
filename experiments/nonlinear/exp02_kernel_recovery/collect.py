#!/usr/bin/env python3
"""Collect exp02 kernel recovery results into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .run import (
        DATASET_DISPLAY_NAMES,
        DATASET_ORDER,
        EXPERIMENT_NAME,
        METHOD_DISPLAY_NAMES,
        METHOD_ORDER,
        METRIC_BY_DATASET,
        _latest_run_dir,
        _resolve_path,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from run import (  # type: ignore[no-redef]
        DATASET_DISPLAY_NAMES,
        DATASET_ORDER,
        EXPERIMENT_NAME,
        METHOD_DISPLAY_NAMES,
        METHOD_ORDER,
        METRIC_BY_DATASET,
        _latest_run_dir,
        _resolve_path,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _result_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/*/result.json"))


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    files = _result_files(run_dir)
    if not files:
        results_json = run_dir / "results.json"
        if results_json.exists():
            payload = _read_json(results_json)
            return [dict(item) for item in payload.get("records", [])]
        raise FileNotFoundError(f"No result.json files found under {run_dir}")
    return [_read_json(path) for path in files]


def _metric_value(record: dict[str, Any], dataset_name: str) -> Any:
    metric_name = METRIC_BY_DATASET[dataset_name]
    metric = dict(record.get("metrics", {}).get(metric_name, {}))
    return metric.get("value")


def _metric_status(record: dict[str, Any], dataset_name: str) -> str:
    metric_name = METRIC_BY_DATASET[dataset_name]
    metric = dict(record.get("metrics", {}).get(metric_name, {}))
    return str(metric.get("status", record.get("metric_status", "unknown")))


def _metric_message(record: dict[str, Any], dataset_name: str) -> str:
    metric_name = METRIC_BY_DATASET[dataset_name]
    metric = dict(record.get("metrics", {}).get(metric_name, {}))
    return str(metric.get("message", ""))


def _format_value(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def collect(run_dir: Path, *, output_dir: Path) -> dict[str, Any]:
    records = _load_records(run_dir)
    by_dataset_method = {
        (record["dataset_name"], record["method_name"]): record
        for record in records
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    long_csv = output_dir / "kernel_recovery_long.csv"
    with long_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_name",
                "dataset_display_name",
                "method_name",
                "method_display_name",
                "status",
                "recovery_status",
                "metric_name",
                "metric_status",
                "metric_value",
                "metric_message",
                "recovered_kernel_manifest",
                "truth_reference_path",
            ],
        )
        writer.writeheader()
        for dataset_name in DATASET_ORDER:
            for method_name in METHOD_ORDER:
                record = by_dataset_method.get((dataset_name, method_name))
                if record is None:
                    continue
                writer.writerow(
                    {
                        "dataset_name": dataset_name,
                        "dataset_display_name": DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name),
                        "method_name": method_name,
                        "method_display_name": METHOD_DISPLAY_NAMES.get(method_name, method_name),
                        "status": record.get("status"),
                        "recovery_status": record.get("recovery_status"),
                        "metric_name": METRIC_BY_DATASET[dataset_name],
                        "metric_status": _metric_status(record, dataset_name),
                        "metric_value": _metric_value(record, dataset_name),
                        "metric_message": _metric_message(record, dataset_name),
                        "recovered_kernel_manifest": record.get("artifacts", {}).get("recovered_kernel_manifest"),
                        "truth_reference_path": record.get("artifacts", {}).get("truth_reference_path"),
                    }
                )

    table_csv = output_dir / "paper_table_5_1_7.csv"
    table_md = output_dir / "paper_table_5_1_7.md"
    rows: list[dict[str, Any]] = []
    for method_name in METHOD_ORDER:
        volterra = by_dataset_method.get(("volterra_wiener", method_name), {})
        duffing = by_dataset_method.get(("duffing", method_name), {})
        rows.append(
            {
                "Method": METHOD_DISPLAY_NAMES.get(method_name, method_name),
                "Volterra-Wiener KNMSE": _format_value(_metric_value(volterra, "volterra_wiener")) if volterra else "NA",
                "Volterra-Wiener status": volterra.get("status", "missing"),
                "Duffing GFRF-RE": _format_value(_metric_value(duffing, "duffing")) if duffing else "NA",
                "Duffing status": duffing.get("status", "missing"),
            }
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "Method",
        "Volterra-Wiener KNMSE",
        "Volterra-Wiener status",
        "Duffing GFRF-RE",
        "Duffing status",
    ]
    with table_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with table_md.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[name]) for name in fieldnames) + " |\n")

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_dir": str(run_dir),
        "long_csv": str(long_csv),
        "paper_table_csv": str(table_csv),
        "paper_table_md": str(table_md),
        "record_count": len(records),
        "rows": rows,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect exp02 kernel recovery results.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Root directory containing experiment run subdirectories.",
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory. Defaults to latest run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for collected tables. Defaults to <run-dir>/collected.",
    )
    parser.add_argument("--print-output-dir", action="store_true", help="Print the collection output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = _resolve_path(args.results_root)
    run_dir = _resolve_path(args.run_dir) if args.run_dir else _latest_run_dir(results_root)
    output_dir = _resolve_path(args.output_dir) if args.output_dir else (run_dir / "collected")
    summary = collect(run_dir, output_dir=output_dir)
    if args.print_output_dir:
        print(summary["paper_table_csv"])


if __name__ == "__main__":
    main()
