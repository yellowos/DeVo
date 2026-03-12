#!/usr/bin/env python3
"""Plot summaries for the hydraulic unsupervised isolation experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUBSYSTEM_ORDER = ("Cooler", "Valve", "Pump", "Accumulator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Hydraulic exp01 result summaries.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory under outputs/runs/. Defaults to the latest run.",
    )
    parser.add_argument(
        "--collected-dir",
        type=Path,
        default=None,
        help="Directory produced by collect.py. Defaults to <run-dir>/collected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output directory. Defaults to <run-dir>/plots.",
    )
    return parser.parse_args()


def _default_run_dir() -> Path:
    root = Path(__file__).with_name("outputs") / "runs"
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float:
    if value in ("", None, "None"):
        return float("nan")
    return float(value)


def _filter_completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("status", "")).lower() == "completed"]


def _plot_top1_top2(main_rows: list[dict[str, Any]], output_path: Path) -> None:
    methods = [row["method"] for row in main_rows]
    top1 = np.asarray([_as_float(row["top1_isolation_accuracy"]) for row in main_rows], dtype=np.float64)
    top2 = np.asarray([_as_float(row["top2_coverage"]) for row in main_rows], dtype=np.float64)
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2.0, top1, width=width, label="Top-1", color="#264653")
    ax.bar(x + width / 2.0, top2, width=width, label="Top-2", color="#f4a261")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Hydraulic Isolation: Top-1 vs Top-2")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_subsystem_hits(subsystem_rows: list[dict[str, Any]], output_path: Path) -> None:
    methods = sorted({row["method"] for row in subsystem_rows})
    x = np.arange(len(DEFAULT_SUBSYSTEM_ORDER), dtype=np.float64)
    width = 0.82 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(11, 5))
    for method_index, method in enumerate(methods):
        method_rows = {row["subsystem"]: row for row in subsystem_rows if row["method"] == method}
        y = np.asarray(
            [_as_float(method_rows.get(subsystem_name, {}).get("top1_hit_rate")) for subsystem_name in DEFAULT_SUBSYSTEM_ORDER],
            dtype=np.float64,
        )
        offset = (method_index - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, y, width=width, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(DEFAULT_SUBSYSTEM_ORDER)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Top-1 Hit Rate")
    ax.set_title("Hydraulic Isolation: Subsystem Top-1 Hit Rates")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncols=max(1, min(3, len(methods))))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else _default_run_dir().resolve()
    collected_dir = args.collected_dir.resolve() if args.collected_dir else (run_dir / "collected").resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = _filter_completed(_read_csv(collected_dir / "table_5_2_5_main.csv"))
    subsystem_rows = _filter_completed(_read_csv(collected_dir / "table_5_2_6_subsystems_long.csv"))
    if not main_rows:
        raise RuntimeError("No completed methods found in the collected main table.")
    if not subsystem_rows:
        raise RuntimeError("No completed methods found in the collected subsystem table.")

    _plot_top1_top2(main_rows, output_dir / "top1_top2_comparison.png")
    _plot_subsystem_hits(subsystem_rows, output_dir / "subsystem_top1_hit_rates.png")
    print(f"Plots written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
