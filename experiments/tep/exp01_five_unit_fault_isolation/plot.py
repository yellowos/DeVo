"""Plot method comparisons for TEP experiment 01 five-unit fault isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - runtime only
    raise RuntimeError("PyYAML is required to plot experiment results.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
METRICS = [
    ("top1", "Top-1"),
    ("top3", "Top-3"),
    ("soft_precision_at_3", "Soft P@3"),
    ("early_hit", "Early Hit"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot five-unit fault isolation comparisons.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--profile", type=str, default="full")
    parser.add_argument("--results-root", type=Path, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping at {path}")
    return payload


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_results_root(config: Mapping[str, Any], *, config_path: Path, profile: str, override: Optional[Path]) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    output_root = resolve_path(config_path.parent.resolve(), config["paths"]["output_root"])
    return output_root / profile


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(results_root: Path) -> List[Dict[str, Any]]:
    collected = results_root / "collected" / "paper_table_5_3_3.json"
    if collected.exists():
        payload = load_json(collected)
        return list(payload.get("rows", []))

    rows: list[Dict[str, Any]] = []
    for result_path in sorted(results_root.glob("h*/**/result.json")):
        payload = load_json(result_path)
        aggregate = payload.get("aggregate_metrics", {})
        rows.append(
            {
                "method": payload.get("display_name", payload.get("method_id")),
                "method_id": payload.get("method_id"),
                "horizon": payload.get("horizon"),
                "status": payload.get("status"),
                "top1": (aggregate.get("top1") or {}).get("mean"),
                "top3": (aggregate.get("top3") or {}).get("mean"),
                "soft_precision_at_3": (aggregate.get("soft_precision_at_3") or {}).get("mean"),
                "early_hit": (aggregate.get("early_hit") or {}).get("mean"),
                "evaluable_runs": aggregate.get("evaluable_runs", 0),
            }
        )
    rows.sort(key=lambda row: (int(row["horizon"]), str(row["method"])))
    return rows


def finite_metric_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        if not any(row.get(metric_key) is not None for metric_key, _ in METRICS):
            continue
        kept.append(row)
    return kept


def placeholder_figure(output_path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    kept = finite_metric_rows(rows)
    output_path = output_dir / "method_comparison_by_metric.png"
    if not kept:
        placeholder_figure(output_path, "Method Comparison by Metric", "No evaluable metric rows available.")
        return

    horizons = sorted({int(row["horizon"]) for row in kept})
    methods = sorted({str(row["method"]) for row in kept})
    method_positions = np.arange(len(methods))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.9, len(horizons)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()
    bar_width = 0.8 / max(len(horizons), 1)

    for axis, (metric_key, metric_label) in zip(axes, METRICS):
        for horizon_idx, horizon in enumerate(horizons):
            values = []
            for method in methods:
                match = next((row for row in kept if int(row["horizon"]) == horizon and str(row["method"]) == method), None)
                value = np.nan if match is None or match.get(metric_key) is None else float(match[metric_key])
                values.append(value)
            offset = (horizon_idx - (len(horizons) - 1) / 2.0) * bar_width
            axis.bar(method_positions + offset, values, width=bar_width, color=colors[horizon_idx], label=f"h={horizon}")
        axis.set_title(metric_label)
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(method_positions)
        axis.set_xticklabels(methods, rotation=20, ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(horizons)))
    fig.suptitle("Five-Unit Fault Isolation: Metric Comparison")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_horizon_grouped(rows: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    kept = finite_metric_rows(rows)
    output_path = output_dir / "horizon_grouped_metrics.png"
    if not kept:
        placeholder_figure(output_path, "Horizon Grouped Metrics", "No evaluable metric rows available.")
        return

    horizons = sorted({int(row["horizon"]) for row in kept})
    methods = sorted({str(row["method"]) for row in kept})
    colors = plt.get_cmap("Set2")(np.linspace(0.0, 0.9, len(methods)))

    fig, axes = plt.subplots(1, len(horizons), figsize=(7 * max(1, len(horizons)), 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    metric_positions = np.arange(len(METRICS))
    bar_width = 0.8 / max(len(methods), 1)

    for axis, horizon in zip(axes, horizons):
        for method_idx, method in enumerate(methods):
            values = []
            match = next((row for row in kept if int(row["horizon"]) == horizon and str(row["method"]) == method), None)
            for metric_key, _metric_label in METRICS:
                value = np.nan if match is None or match.get(metric_key) is None else float(match[metric_key])
                values.append(value)
            offset = (method_idx - (len(methods) - 1) / 2.0) * bar_width
            axis.bar(metric_positions + offset, values, width=bar_width, color=colors[method_idx], label=method)
        axis.set_title(f"h = {horizon}")
        axis.set_xticks(metric_positions)
        axis.set_xticklabels([label for _, label in METRICS], rotation=20, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)))
    fig.suptitle("Five-Unit Fault Isolation: Metrics Grouped by Horizon")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    results_root = resolve_results_root(config, config_path=args.config.resolve(), profile=args.profile, override=args.results_root)
    rows = load_rows(results_root)
    plot_dir = results_root / "plots"
    plot_metric_comparison(rows, plot_dir)
    plot_horizon_grouped(rows, plot_dir)
    print(f"[exp01] plots written to: {plot_dir}")


if __name__ == "__main__":
    main()
