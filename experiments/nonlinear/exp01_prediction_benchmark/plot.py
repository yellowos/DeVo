from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


def _resolve_path(base_dir: Path, raw: str | None, *, fallback: str) -> Path:
    target = Path(raw or fallback)
    if not target.is_absolute():
        target = (base_dir / target).resolve()
    return target


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a mapping.")
    return payload


def _resolve_profile(config_path: Path, profile: str) -> dict[str, Any]:
    raw = _load_yaml(config_path)
    profiles = dict(raw.get("profiles", {}) or {})
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown profile '{profile}'. Available: {available}")
    profile_payload = dict(profiles[profile] or {})
    config_dir = config_path.parent.resolve()
    results_dir = _resolve_path(
        config_dir,
        str(profile_payload.get("output_dir", "")) or None,
        fallback=f"./outputs/{profile}",
    )
    return {
        "datasets": list(profile_payload.get("datasets", [])),
        "methods": list(profile_payload.get("methods", [])),
        "results_dir": results_dir,
    }


def _load_summary_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"Summary CSV is empty: {path}")
    return rows


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _draw_grouped_bar(
    rows: list[dict[str, Any]],
    *,
    datasets: list[str],
    methods: list[str],
    metric: str,
    title: str,
    output_path: Path,
) -> None:
    row_lookup = {(row["dataset"], row["method"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.8), 6))

    x_positions = list(range(len(datasets)))
    num_methods = max(len(methods), 1)
    bar_width = 0.8 / num_methods
    offsets = [(-0.4 + bar_width / 2) + index * bar_width for index in range(num_methods)]
    colors = plt.cm.tab20.colors

    for method_index, method in enumerate(methods):
        values: list[float] = []
        errors: list[float] = []
        for dataset in datasets:
            row = row_lookup.get((dataset, method))
            value = _safe_float(None if row is None else row.get(f"{metric}_mean"))
            error = _safe_float(None if row is None else row.get(f"{metric}_std"))
            values.append(float("nan") if value is None else value)
            errors.append(0.0 if error is None else error)

        positions = [x + offsets[method_index] for x in x_positions]
        ax.bar(
            positions,
            np.asarray(values, dtype=float),
            yerr=errors,
            width=bar_width,
            label=method,
            color=colors[method_index % len(colors)],
            alpha=0.9,
            capsize=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=min(4, len(methods)))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot nonlinear benchmark experiment 01 collected results.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
        help="Path to the experiment YAML config.",
    )
    parser.add_argument("--profile", default="full", help="Config profile to read.")
    parser.add_argument("--summary-csv", type=Path, help="Path to collect.py long summary csv.")
    parser.add_argument("--output-dir", type=Path, help="Directory for generated figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = _resolve_profile(args.config.resolve(), args.profile)
    results_dir = profile["results_dir"]
    summary_csv = (args.summary_csv or results_dir / "summary" / "benchmark_summary_long.csv").expanduser().resolve()
    figure_dir = (args.output_dir or results_dir / "plots").expanduser().resolve()

    rows = _load_summary_rows(summary_csv)
    _draw_grouped_bar(
        rows,
        datasets=profile["datasets"],
        methods=profile["methods"],
        metric="nmse",
        title="Nonlinear Benchmark Prediction NMSE",
        output_path=figure_dir / "nmse_by_dataset.png",
    )
    _draw_grouped_bar(
        rows,
        datasets=profile["datasets"],
        methods=profile["methods"],
        metric="rmse",
        title="Nonlinear Benchmark Prediction RMSE",
        output_path=figure_dir / "rmse_by_dataset.png",
    )

    manifest = {
        "summary_csv": str(summary_csv),
        "figure_dir": str(figure_dir),
        "figures": [
            str((figure_dir / "nmse_by_dataset.png").resolve()),
            str((figure_dir / "rmse_by_dataset.png").resolve()),
        ],
    }
    (figure_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
