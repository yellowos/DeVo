from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    for key in ("fault_truth_table", "output_root"):
        resolved = resolve_path(paths[key], base_dir=project_root)
        assert resolved is not None
        paths[key] = str(resolved)
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def truth_rows_by_scenario(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("rows", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(rows, list):
        return result
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("scenario"), str):
            result[str(row["scenario"])] = dict(row)
    return result


def float_row(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    return 0.0 if value == "" else float(value)


def stage_slices(length: int, early_end_fraction: float, mid_end_fraction: float) -> dict[str, slice]:
    if length <= 0:
        return {"early": slice(0, 0), "mid": slice(0, 0), "late": slice(0, 0)}
    early_end = max(1, min(length, int(math.ceil(length * early_end_fraction))))
    mid_end = max(early_end + 1, min(length, int(math.ceil(length * mid_end_fraction))))
    if mid_end > length:
        mid_end = length
    late_start = min(mid_end, max(0, length - 1))
    return {
        "early": slice(0, early_end),
        "mid": slice(early_end, mid_end),
        "late": slice(late_start, length),
    }


def stage_mean(rows: Sequence[Mapping[str, str]], units: Sequence[str], subset: slice) -> dict[str, float]:
    selected = list(rows[subset])
    if not selected:
        selected = list(rows)
    if not selected:
        return {unit: 0.0 for unit in units}
    return {
        unit: sum(float_row(row, f"{unit}_score") for row in selected) / float(len(selected))
        for unit in units
    }


def ranked_units(score_map: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(((str(unit), float(score)) for unit, score in score_map.items()), key=lambda item: (-item[1], item[0]))


def summarize_pattern(
    *,
    early_scores: Mapping[str, float],
    mid_scores: Mapping[str, float],
    late_scores: Mapping[str, float],
    diffusion_threshold: float,
    secondary_unit_threshold: float,
    truncated: bool,
) -> tuple[str, str, str, str]:
    early_ranked = ranked_units(early_scores)
    mid_ranked = ranked_units(mid_scores)
    late_ranked = ranked_units(late_scores)
    early_dom = early_ranked[0][0]
    late_dom = late_ranked[0][0]
    diffusion_units = [
        unit
        for unit, score in mid_ranked
        if score >= diffusion_threshold or (score >= secondary_unit_threshold and unit != early_dom)
    ]
    if not diffusion_units:
        diffusion_units = [mid_ranked[0][0]]
    notes: list[str] = []
    if early_dom == late_dom:
        notes.append(f"starts in {early_dom} and remains dominant")
    else:
        notes.append(f"shifts from {early_dom} to {late_dom}")
    if len(diffusion_units) > 1:
        notes.append(f"mid-stage diffusion across {', '.join(diffusion_units)}")
    else:
        notes.append(f"mid-stage remains concentrated on {diffusion_units[0]}")
    if truncated:
        notes.append("observed on truncated horizon")
    return early_dom, late_dom, "|".join(diffusion_units), "; ".join(notes)


def summarize_runs(
    *,
    timeline_rows: Sequence[Mapping[str, str]],
    run_status_rows: Sequence[Mapping[str, str]],
    truth_by_scenario: Mapping[str, Mapping[str, Any]],
    units: Sequence[str],
    collect_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status_by_run = {str(row["run_key"]): dict(row) for row in run_status_rows}
    grouped_by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in timeline_rows:
        grouped_by_run[str(row["run_key"])].append(dict(row))
    for rows in grouped_by_run.values():
        rows.sort(key=lambda item: int(item["window_index"]))

    early_end_fraction = float(collect_cfg["stage_boundaries"]["early_end_fraction"])
    mid_end_fraction = float(collect_cfg["stage_boundaries"]["mid_end_fraction"])
    diffusion_threshold = float(collect_cfg["diffusion_threshold"])
    secondary_unit_threshold = float(collect_cfg["secondary_unit_threshold"])

    run_summaries: list[dict[str, Any]] = []
    scenario_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_key, rows in grouped_by_run.items():
        if not rows:
            continue
        stage_map = stage_slices(len(rows), early_end_fraction, mid_end_fraction)
        early_scores = stage_mean(rows, units, stage_map["early"])
        mid_scores = stage_mean(rows, units, stage_map["mid"])
        late_scores = stage_mean(rows, units, stage_map["late"])
        status = status_by_run.get(run_key, {})
        truncated = str(status.get("analysis_scope", "")) != "full"
        early_dom, late_dom, diffusion_units, pattern = summarize_pattern(
            early_scores=early_scores,
            mid_scores=mid_scores,
            late_scores=late_scores,
            diffusion_threshold=diffusion_threshold,
            secondary_unit_threshold=secondary_unit_threshold,
            truncated=truncated,
        )
        scenario = str(rows[0]["scenario"])
        truth = truth_by_scenario.get(scenario, {})
        summary = {
            "scenario": scenario,
            "run_key": run_key,
            "mode": str(rows[0]["mode"]),
            "analysis_scope": status.get("analysis_scope", ""),
            "processed_window_count": int(status.get("processed_window_count", len(rows)) or len(rows)),
            "available_window_count": int(status.get("available_window_count", len(rows)) or len(rows)),
            "early_dominant_unit": early_dom,
            "early_dominant_share": early_scores[early_dom],
            "mid_diffusion_units": diffusion_units,
            "late_dominant_unit": late_dom,
            "late_dominant_share": late_scores[late_dom],
            "observation_pattern": pattern,
            "truth_primary_unit": truth.get("primary_unit"),
            "truth_expected_units": "|".join(str(item) for item in truth.get("expected_units", []) or []),
        }
        run_summaries.append(summary)
        scenario_groups[scenario].append(summary)

    scenario_summaries: list[dict[str, Any]] = []
    for scenario, rows in sorted(scenario_groups.items()):
        if len(rows) == 1:
            base = dict(rows[0])
            base["run_count"] = 1
            base["run_keys"] = base["run_key"]
            base["modes"] = base["mode"]
            scenario_summaries.append(base)
            continue
        early_counts: dict[str, int] = defaultdict(int)
        late_counts: dict[str, int] = defaultdict(int)
        diffusion_counts: dict[str, int] = defaultdict(int)
        early_share: dict[str, float] = defaultdict(float)
        late_share: dict[str, float] = defaultdict(float)
        for row in rows:
            early_counts[str(row["early_dominant_unit"])] += 1
            late_counts[str(row["late_dominant_unit"])] += 1
            early_share[str(row["early_dominant_unit"])] += float(row["early_dominant_share"])
            late_share[str(row["late_dominant_unit"])] += float(row["late_dominant_share"])
            for unit in str(row["mid_diffusion_units"]).split("|"):
                if unit:
                    diffusion_counts[unit] += 1
        early_dom = max(sorted(early_counts), key=lambda unit: (early_counts[unit], early_share[unit]))
        late_dom = max(sorted(late_counts), key=lambda unit: (late_counts[unit], late_share[unit]))
        diffusion_units = [
            unit
            for unit, count in sorted(diffusion_counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= max(1, math.ceil(len(rows) / 2))
        ]
        if not diffusion_units:
            diffusion_units = [early_dom]
        scenario_summaries.append(
            {
                "scenario": scenario,
                "run_key": "",
                "mode": "",
                "analysis_scope": "mixed" if len({row["analysis_scope"] for row in rows}) > 1 else rows[0]["analysis_scope"],
                "processed_window_count": int(sum(int(row["processed_window_count"]) for row in rows) / len(rows)),
                "available_window_count": int(sum(int(row["available_window_count"]) for row in rows) / len(rows)),
                "early_dominant_unit": early_dom,
                "early_dominant_share": early_share[early_dom] / float(early_counts[early_dom]),
                "mid_diffusion_units": "|".join(diffusion_units),
                "late_dominant_unit": late_dom,
                "late_dominant_share": late_share[late_dom] / float(late_counts[late_dom]),
                "observation_pattern": f"aggregated across {len(rows)} runs",
                "truth_primary_unit": rows[0]["truth_primary_unit"],
                "truth_expected_units": rows[0]["truth_expected_units"],
                "run_count": len(rows),
                "run_keys": "|".join(str(row["run_key"]) for row in rows),
                "modes": "|".join(str(row["mode"]) for row in rows),
            }
        )
    return run_summaries, scenario_summaries


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "scenario",
        "modes",
        "early_dominant_unit",
        "mid_diffusion_units",
        "late_dominant_unit",
        "observation_pattern",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            values = [str(row.get(header, "")) for header in headers]
            handle.write("| " + " | ".join(values) + " |\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TEP propagation summaries.")
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
    truth_by_scenario = truth_rows_by_scenario(load_json(Path(config["paths"]["fault_truth_table"])))
    units = list(manifest["unit_order"])
    run_summaries, scenario_summaries = summarize_runs(
        timeline_rows=timeline_rows,
        run_status_rows=run_status_rows,
        truth_by_scenario=truth_by_scenario,
        units=units,
        collect_cfg=config["collect"],
    )

    summary_root = output_root / "summaries"
    run_fields = [
        "scenario",
        "run_key",
        "mode",
        "analysis_scope",
        "processed_window_count",
        "available_window_count",
        "early_dominant_unit",
        "early_dominant_share",
        "mid_diffusion_units",
        "late_dominant_unit",
        "late_dominant_share",
        "observation_pattern",
        "truth_primary_unit",
        "truth_expected_units",
    ]
    scenario_fields = [
        "scenario",
        "run_count",
        "run_keys",
        "modes",
        "analysis_scope",
        "processed_window_count",
        "available_window_count",
        "early_dominant_unit",
        "early_dominant_share",
        "mid_diffusion_units",
        "late_dominant_unit",
        "late_dominant_share",
        "observation_pattern",
        "truth_primary_unit",
        "truth_expected_units",
    ]
    write_csv(summary_root / "propagation_run_summary.csv", run_summaries, run_fields)
    write_csv(summary_root / "propagation_scenario_summary.csv", scenario_summaries, scenario_fields)
    write_markdown(summary_root / "propagation_scenario_summary.md", scenario_summaries)

    with (summary_root / "collect_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "generated_at": utc_now(),
                "timeline_csv": manifest["outputs"]["timeline_csv"],
                "run_status_csv": manifest["outputs"]["run_status_csv"],
                "run_summary_csv": str((summary_root / "propagation_run_summary.csv").resolve()),
                "scenario_summary_csv": str((summary_root / "propagation_scenario_summary.csv").resolve()),
                "scenario_summary_md": str((summary_root / "propagation_scenario_summary.md").resolve()),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps({"run_summary_count": len(run_summaries), "scenario_summary_count": len(scenario_summaries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
