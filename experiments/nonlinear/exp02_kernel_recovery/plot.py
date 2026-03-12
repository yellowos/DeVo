#!/usr/bin/env python3
"""Plot exp02 kernel recovery comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .run import (
        DATASET_DISPLAY_NAMES,
        METHOD_DISPLAY_NAMES,
        METHOD_ORDER,
        METRIC_BY_DATASET,
        _latest_run_dir,
        _resolve_path,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from run import (  # type: ignore[no-redef]
        DATASET_DISPLAY_NAMES,
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


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    files = sorted(run_dir.glob("*/*/result.json"))
    if files:
        return [_read_json(path) for path in files]
    payload = _read_json(run_dir / "results.json")
    return [dict(item) for item in payload.get("records", [])]


def _metric_entry(record: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    metric_name = METRIC_BY_DATASET[dataset_name]
    return dict(record.get("metrics", {}).get(metric_name, {}))


def _build_metric_plot(records: dict[tuple[str, str], dict[str, Any]], *, dataset_name: str, output_path: Path) -> None:
    metric_name = METRIC_BY_DATASET[dataset_name]
    labels = [METHOD_DISPLAY_NAMES.get(name, name) for name in METHOD_ORDER]
    values: list[float] = []
    status_lines: list[str] = []
    available = False
    for method_name in METHOD_ORDER:
        record = records.get((dataset_name, method_name), {})
        metric = _metric_entry(record, dataset_name)
        value = metric.get("value")
        if value is None:
            values.append(np.nan)
            status_lines.append(f"{METHOD_DISPLAY_NAMES.get(method_name, method_name)}: {metric.get('status', record.get('status', 'missing'))}")
        else:
            available = True
            values.append(float(value))
            status_lines.append(f"{METHOD_DISPLAY_NAMES.get(method_name, method_name)}: {value:.6f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    if available:
        bar_values = np.asarray(values, dtype=np.float64)
        x = np.arange(len(labels))
        bars = ax.bar(x, np.nan_to_num(bar_values, nan=0.0), color="#4C6EF5")
        for index, value in enumerate(bar_values):
            if np.isnan(value):
                bars[index].set_color("#ADB5BD")
                ax.text(index, 0.0, "NA", ha="center", va="bottom", fontsize=9)
            else:
                ax.text(index, value, f"{value:.3g}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"{DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)} {metric_name.upper()} Comparison")
        if np.all(bar_values[np.isfinite(bar_values)] > 0):
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            f"No numeric {metric_name.upper()} values available for {DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)}.\n\n"
            + "\n".join(status_lines),
            va="top",
            ha="left",
            fontsize=11,
            family="monospace",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _load_npz_orders(path: Path) -> dict[int, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return {
        int(key.split("_", 1)[1]): np.asarray(payload[key])
        for key in payload.files
        if key.startswith("order_")
    }


def _plot_duffing_gfrf(records: dict[tuple[str, str], dict[str, Any]], *, output_path: Path) -> bool:
    truth_npz: Optional[Path] = None
    recovered: dict[str, Path] = {}
    for method_name in METHOD_ORDER:
        record = records.get(("duffing", method_name))
        if not record:
            continue
        artifact_path = record.get("artifacts", {}).get("recovered_gfrf_npz")
        if artifact_path:
            path = Path(artifact_path)
            if path.exists():
                recovered[method_name] = path
        truth_payload_path = record.get("artifacts", {}).get("truth_payload_path")
        if truth_payload_path:
            candidate = Path(truth_payload_path)
            if candidate.exists() and candidate.suffix.lower() == ".npz":
                truth_npz = candidate

    if truth_npz is None or not recovered:
        return False

    truth_orders = _load_npz_orders(truth_npz)
    truth_order_1 = truth_orders.get(1)
    if truth_order_1 is None:
        return False

    freq = np.arange(truth_order_1.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freq, np.abs(truth_order_1), color="black", linewidth=2.0, label="Truth")
    for method_name, npz_path in recovered.items():
        order_1 = _load_npz_orders(npz_path).get(1)
        if order_1 is None or order_1.shape != truth_order_1.shape:
            continue
        ax.plot(freq, np.abs(order_1), linewidth=1.2, label=METHOD_DISPLAY_NAMES.get(method_name, method_name))
    ax.set_title("Duffing First-Order GFRF Magnitude")
    ax.set_xlabel("Frequency Bin")
    ax.set_ylabel("|GFRF|")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def plot(run_dir: Path, *, output_dir: Path) -> dict[str, Any]:
    records_list = _load_records(run_dir)
    records = {(record["dataset_name"], record["method_name"]): record for record in records_list}
    output_dir.mkdir(parents=True, exist_ok=True)

    knmse_plot = output_dir / "knmse_comparison.png"
    gfrf_plot = output_dir / "gfrf_re_comparison.png"
    _build_metric_plot(records, dataset_name="volterra_wiener", output_path=knmse_plot)
    _build_metric_plot(records, dataset_name="duffing", output_path=gfrf_plot)

    duffing_gfrf_plot = output_dir / "duffing_gfrf_comparison.png"
    duffing_gfrf_available = _plot_duffing_gfrf(records, output_path=duffing_gfrf_plot)
    if not duffing_gfrf_available and duffing_gfrf_plot.exists():
        duffing_gfrf_plot.unlink()

    summary = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "plots": {
            "knmse_comparison": str(knmse_plot),
            "gfrf_re_comparison": str(gfrf_plot),
            "duffing_gfrf_comparison": str(duffing_gfrf_plot) if duffing_gfrf_available else None,
        },
    }
    with (output_dir / "plot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exp02 kernel recovery results.")
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
        help="Output directory for plots. Defaults to <run-dir>/plots.",
    )
    parser.add_argument("--print-output-dir", action="store_true", help="Print the plot output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = _resolve_path(args.results_root)
    run_dir = _resolve_path(args.run_dir) if args.run_dir else _latest_run_dir(results_root)
    output_dir = _resolve_path(args.output_dir) if args.output_dir else (run_dir / "plots")
    summary = plot(run_dir, output_dir=output_dir)
    if args.print_output_dir:
        print(summary["output_dir"])


if __name__ == "__main__":
    main()
