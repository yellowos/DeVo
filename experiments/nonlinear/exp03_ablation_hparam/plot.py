"""Plot experiment 03 summaries without retraining."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - script dependency guard.
    raise RuntimeError("matplotlib is required to generate plots.") from exc

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - script dependency guard.
    raise RuntimeError("PyYAML is required to run experiment scripts.") from exc


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Config at {path} must be a mapping.")
    return dict(payload)


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _output_root(config: Mapping[str, Any], *, base_dir: Path) -> Path:
    experiment = config.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment config must be a mapping.")
    return _resolve_path(str(experiment.get("output_root", "results/nonlinear/exp03_ablation_hparam")), base_dir=base_dir)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: Any) -> float | None:
    if value in {None, "", "None"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_sort_key(value: Any) -> tuple[int, float | str]:
    numeric = _to_float(value)
    if numeric is not None:
        return (0, numeric)
    return (1, str(value))


def _ablation_metric_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    preferred = [
        "silverbox_nmse",
        "volterra_wiener_knmse",
        "duffing_gfrf_re",
        "volterra_wiener_nmse",
        "duffing_nmse",
    ]
    available = [key for key in rows[0].keys() if key.endswith(("nmse", "knmse", "gfrf_re")) and not key.endswith("_status")]
    ordered = [key for key in preferred if key in available]
    for key in available:
        if key not in ordered:
            ordered.append(key)
    return ordered


def _plot_ablation(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [row.get("label") or row.get("variant") or "item" for row in rows]
    metric_columns = _ablation_metric_columns(rows)
    if not metric_columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No ablation metrics available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, len(metric_columns), figsize=(5 * len(metric_columns), 4), squeeze=False)
    for axis, metric_column in zip(axes[0], metric_columns):
        values = [_to_float(row.get(metric_column)) for row in rows]
        numeric_values = [0.0 if value is None else value for value in values]
        axis.bar(range(len(labels)), numeric_values, color="#2E6F95")
        axis.set_title(metric_column.replace("_", " "))
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_ylabel("value")
        if all(value is None for value in values):
            axis.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=axis.transAxes)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_hyperparameter(rows: list[dict[str, str]], output_path: Path) -> None:
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No hyperparameter summary rows available.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    hyperparameters = sorted({str(row.get("hyperparameter") or "") for row in rows if row.get("hyperparameter")}, key=str)
    fig, axes = plt.subplots(1, max(1, len(hyperparameters)), figsize=(6 * max(1, len(hyperparameters)), 4), squeeze=False)
    for axis, hyperparameter in zip(axes[0], hyperparameters):
        subset = [row for row in rows if str(row.get("hyperparameter")) == hyperparameter]
        benchmark_labels = sorted({str(row.get("benchmark") or "") for row in subset if row.get("benchmark")})
        plotted = False
        for benchmark in benchmark_labels:
            benchmark_rows = [row for row in subset if str(row.get("benchmark")) == benchmark]
            benchmark_rows.sort(key=lambda row: _numeric_sort_key(row.get("hyperparameter_value")))
            x_values = [row.get("hyperparameter_value") for row in benchmark_rows]
            for metric_name in ("nmse", "knmse", "gfrf_re"):
                y_values = [_to_float(row.get(f"{metric_name}_mean")) for row in benchmark_rows]
                if all(value is None for value in y_values):
                    continue
                plotted = True
                axis.plot(
                    range(len(x_values)),
                    [float("nan") if value is None else value for value in y_values],
                    marker="o",
                    label=f"{benchmark} {metric_name}",
                )
            axis.set_xticks(range(len(x_values)))
            axis.set_xticklabels([str(value) for value in x_values])
        axis.set_title(hyperparameter.replace("_", " "))
        axis.set_xlabel("setting")
        axis.set_ylabel("value")
        if plotted:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No numeric metrics", ha="center", va="center", transform=axis.transAxes)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_yaml(config_path)
    base_dir = config_path.parent
    root = _output_root(config, base_dir=base_dir) / args.suite
    summary_root = root / "summaries"
    plot_root = root / "plots"

    ablation_table = _read_csv(summary_root / "table_5_1_8.csv")
    hyper_summary = _read_csv(summary_root / "hyperparameter_summary.csv")

    _plot_ablation(ablation_table, plot_root / "ablation_comparison.png")
    _plot_hyperparameter(hyper_summary, plot_root / "hyperparameter_sensitivity.png")
    print(f"[exp03-plot] wrote plots under {plot_root}")


if __name__ == "__main__":
    main()
