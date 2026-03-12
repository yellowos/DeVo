from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def resolve_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_config(config_path: Path) -> dict[str, Any]:
    payload = load_yaml(config_path)
    paths = payload.setdefault("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    project_root = resolve_path(paths.get("project_root", "../../.."), base_dir=SCRIPT_DIR)
    assert project_root is not None
    paths["project_root"] = str(project_root)
    output_root = resolve_path(paths["output_root"], base_dir=project_root)
    assert output_root is not None
    paths["output_root"] = str(output_root)
    return payload


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_rows(rows: Sequence[Mapping[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["run_key"])].append(dict(row))
    for items in grouped.values():
        items.sort(key=lambda item: int(item["window_index"]))
    return grouped


def score_matrix(rows: Sequence[Mapping[str, str]], units: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    time_axis = np.asarray([int(row["time_index"]) for row in rows], dtype=np.int64)
    matrix = np.asarray(
        [[float(row[f"{unit}_score"]) for row in rows] for unit in units],
        dtype=np.float64,
    )
    return time_axis, matrix


def run_status_map(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["run_key"]): dict(row) for row in rows}


def prepare_clean_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def plot_timeline(
    *,
    target_path: Path,
    rows: Sequence[Mapping[str, str]],
    status: Mapping[str, str],
    units: Sequence[str],
    figsize: Sequence[float],
    dpi: int,
) -> None:
    colors = {
        "Reactor": "#b3362d",
        "Condenser": "#2d6a4f",
        "Separator": "#1d4e89",
        "Compressor": "#8a5a00",
        "Stripper": "#6b2f6b",
    }
    fig, ax = plt.subplots(figsize=tuple(figsize), dpi=dpi)
    time_axis = np.asarray([int(row["time_index"]) for row in rows], dtype=np.int64)
    for unit in units:
        ax.plot(
            time_axis,
            [float(row[f"{unit}_score"]) for row in rows],
            linewidth=1.5,
            label=unit,
            color=colors.get(unit),
        )
    ax.set_xlabel("Time index")
    ax.set_ylabel("Normalized attribution")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.2, linewidth=0.6)
    title = f"{rows[0]['scenario']} {rows[0]['mode']} unit-level attribution timeline"
    subtitle = f"{rows[0]['run_key']} | scope={status.get('analysis_scope', '')} | observed {time_axis[0]}-{time_axis[-1]}"
    ax.set_title(title)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.legend(ncol=min(5, len(units)), frameon=False, loc="upper right")
    fig.tight_layout()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    *,
    target_path: Path,
    rows: Sequence[Mapping[str, str]],
    status: Mapping[str, str],
    units: Sequence[str],
    figsize: Sequence[float],
    dpi: int,
    cmap: str,
) -> None:
    time_axis, matrix = score_matrix(rows, units)
    fig, ax = plt.subplots(figsize=tuple(figsize), dpi=dpi)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0,
        vmax=max(0.2, float(np.max(matrix))),
        extent=[int(time_axis[0]), int(time_axis[-1]), len(units) - 0.5, -0.5],
    )
    ax.set_yticks(np.arange(len(units)))
    ax.set_yticklabels(units)
    ax.set_xlabel("Time index")
    ax.set_title(f"{rows[0]['scenario']} {rows[0]['mode']} five-unit attribution heatmap")
    ax.text(
        0.0,
        1.02,
        f"{rows[0]['run_key']} | scope={status.get('analysis_scope', '')} | available only, no padded tail",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("Normalized attribution")
    fig.tight_layout()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TEP propagation analysis figures.")
    parser.add_argument(
        "--config",
        default=Path(__file__).with_name("config.yaml"),
        type=Path,
        help="Path to experiment config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    output_root = Path(config["paths"]["output_root"])
    manifest = load_json(output_root / "analysis_manifest.json")
    timeline_rows = read_csv(Path(manifest["outputs"]["timeline_csv"]))
    run_status_rows = read_csv(Path(manifest["outputs"]["run_status_csv"]))
    units = list(manifest["unit_order"])
    grouped = grouped_rows(timeline_rows)
    status_by_run = run_status_map(run_status_rows)
    plot_cfg = config["plot"]
    timeline_cfg = plot_cfg["timeline"]
    heatmap_cfg = plot_cfg["heatmap"]

    figure_root = output_root / "figures"
    timeline_root = figure_root / "figure_5_3_1_timeline"
    heatmap_root = figure_root / "figure_5_3_2_heatmap"
    prepare_clean_directory(timeline_root)
    prepare_clean_directory(heatmap_root)
    figure_manifest: dict[str, Any] = {
        "generated_at": manifest["generated_at"],
        "timeline_figures": {},
        "heatmap_figures": {},
    }
    for run_key, rows in grouped.items():
        status = status_by_run.get(run_key, {})
        timeline_path = timeline_root / f"{run_key}_timeline.png"
        heatmap_path = heatmap_root / f"{run_key}_heatmap.png"
        plot_timeline(
            target_path=timeline_path,
            rows=rows,
            status=status,
            units=units,
            figsize=timeline_cfg["figsize"],
            dpi=int(timeline_cfg["dpi"]),
        )
        plot_heatmap(
            target_path=heatmap_path,
            rows=rows,
            status=status,
            units=units,
            figsize=heatmap_cfg["figsize"],
            dpi=int(heatmap_cfg["dpi"]),
            cmap=str(heatmap_cfg["cmap"]),
        )
        figure_manifest["timeline_figures"][run_key] = str(timeline_path.resolve())
        figure_manifest["heatmap_figures"][run_key] = str(heatmap_path.resolve())
    with (figure_root / "plot_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(figure_manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"timeline_count": len(figure_manifest["timeline_figures"]), "heatmap_count": len(figure_manifest["heatmap_figures"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
