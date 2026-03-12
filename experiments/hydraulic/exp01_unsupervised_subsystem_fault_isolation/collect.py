#!/usr/bin/env python3
"""Collect tables for the hydraulic unsupervised isolation experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional


DEFAULT_SUBSYSTEM_ORDER = ("Cooler", "Valve", "Pump", "Accumulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Hydraulic exp01 result tables.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory under outputs/runs/. Defaults to the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit collection output directory. Defaults to <run-dir>/collected.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}.")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _format_metric(value: Any) -> Any:
    if value is None:
        return None
    return float(value)


def _default_run_dir() -> Path:
    root = Path(__file__).with_name("outputs") / "runs"
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _result_paths(run_dir: Path) -> list[Path]:
    methods_dir = run_dir / "methods"
    if not methods_dir.exists():
        return []
    return sorted(path / "result.json" for path in methods_dir.iterdir() if (path / "result.json").exists())


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else _default_run_dir().resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "collected").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows: list[dict[str, Any]] = []
    subsystem_rows: list[dict[str, Any]] = []
    subsystem_wide_rows: list[dict[str, Any]] = []

    for result_path in _result_paths(run_dir):
        payload = _load_json(result_path)
        method = str(payload.get("method", result_path.parent.name))
        status = str(payload.get("status", "unknown"))
        metrics = dict(payload.get("metrics", {}))
        row = {
            "method": method,
            "status": status,
            "reason": payload.get("reason"),
            "num_eval_samples": metrics.get("num_eval_samples", 0),
            "top1_isolation_accuracy": _format_metric(metrics.get("top1_isolation_accuracy")),
            "top2_coverage": _format_metric(metrics.get("top2_coverage")),
            "mean_rank_true_subsystem": _format_metric(metrics.get("mean_rank_true_subsystem")),
            "winning_margin": _format_metric(metrics.get("winning_margin")),
            "result_path": str(result_path.resolve()),
        }
        main_rows.append(row)

        subsystem_payload = dict(payload.get("subsystem_hit_rates", {}))
        wide_row = {
            "method": method,
            "status": status,
            "reason": payload.get("reason"),
        }
        for subsystem_name in DEFAULT_SUBSYSTEM_ORDER:
            stats = dict(subsystem_payload.get(subsystem_name, {}))
            subsystem_rows.append(
                {
                    "method": method,
                    "status": status,
                    "reason": payload.get("reason"),
                    "subsystem": subsystem_name,
                    "num_samples": stats.get("num_samples", 0),
                    "num_top1_hits": stats.get("num_top1_hits", 0),
                    "top1_hit_rate": _format_metric(stats.get("top1_hit_rate")),
                }
            )
            wide_row[f"{subsystem_name.lower()}_num_samples"] = stats.get("num_samples", 0)
            wide_row[f"{subsystem_name.lower()}_num_top1_hits"] = stats.get("num_top1_hits", 0)
            wide_row[f"{subsystem_name.lower()}_top1_hit_rate"] = _format_metric(stats.get("top1_hit_rate"))
        subsystem_wide_rows.append(wide_row)

    main_fieldnames = [
        "method",
        "status",
        "reason",
        "num_eval_samples",
        "top1_isolation_accuracy",
        "top2_coverage",
        "mean_rank_true_subsystem",
        "winning_margin",
        "result_path",
    ]
    subsystem_long_fieldnames = [
        "method",
        "status",
        "reason",
        "subsystem",
        "num_samples",
        "num_top1_hits",
        "top1_hit_rate",
    ]
    subsystem_wide_fieldnames = [
        "method",
        "status",
        "reason",
        *[
            field
            for subsystem_name in DEFAULT_SUBSYSTEM_ORDER
            for field in (
                f"{subsystem_name.lower()}_num_samples",
                f"{subsystem_name.lower()}_num_top1_hits",
                f"{subsystem_name.lower()}_top1_hit_rate",
            )
        ],
    ]

    _write_csv(output_dir / "table_5_2_5_main.csv", main_rows, main_fieldnames)
    _write_csv(output_dir / "table_5_2_6_subsystems_long.csv", subsystem_rows, subsystem_long_fieldnames)
    _write_csv(output_dir / "table_5_2_6_subsystems_wide.csv", subsystem_wide_rows, subsystem_wide_fieldnames)
    _write_json(
        output_dir / "collection_manifest.json",
        {
            "run_dir": str(run_dir),
            "main_table_rows": main_rows,
            "subsystem_long_rows": subsystem_rows,
            "subsystem_wide_rows": subsystem_wide_rows,
        },
    )
    print(f"Collected tables written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
