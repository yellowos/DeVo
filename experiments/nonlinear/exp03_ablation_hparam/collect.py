"""Collect experiment 03 results into summaries and paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON mapping at {path}.")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _result_paths(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("runs/*/*/result.json"))


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _std(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return 0.0
    mean = sum(valid) / len(valid)
    return (sum((value - mean) ** 2 for value in valid) / (len(valid) - 1)) ** 0.5


def _status_summary(values: Iterable[str]) -> str:
    counts = Counter(value for value in values if value)
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts)) if counts else ""


def _stability_summary(rows: list[Mapping[str, Any]]) -> str:
    total = len(rows)
    success = sum(1 for row in rows if row.get("success"))
    converged = sum(1 for row in rows if row.get("converged"))
    nonfinite = sum(1 for row in rows if row.get("had_nonfinite"))
    early_stop = sum(1 for row in rows if row.get("early_stopped"))
    return (
        f"success={success}/{total}; converged={converged}/{total}; "
        f"nonfinite={nonfinite}/{total}; early_stop={early_stop}/{total}"
    )


def _flatten_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    stability = payload.get("stability", {})
    row = {
        "run_id": payload.get("run_id"),
        "experiment_kind": payload.get("experiment_kind"),
        "benchmark": payload.get("benchmark"),
        "label": payload.get("label"),
        "variant": payload.get("variant"),
        "hyperparameter": payload.get("hyperparameter"),
        "hyperparameter_value": payload.get("hyperparameter_value"),
        "sweep_label": payload.get("sweep_label"),
        "seed": payload.get("seed"),
        "success": bool(stability.get("success", False)),
        "converged": bool(stability.get("converged", False)),
        "had_nonfinite": bool(stability.get("had_nonfinite", False)),
        "early_stopped": bool(stability.get("early_stopped", False)),
        "fit_seconds": _float_or_none(stability.get("fit_seconds")),
        "exception": stability.get("exception"),
    }
    for metric_name in ("nmse", "knmse", "gfrf_re"):
        metric_payload = metrics.get(metric_name, {})
        if not isinstance(metric_payload, Mapping):
            metric_payload = {}
        row[f"{metric_name}_value"] = _float_or_none(metric_payload.get("value"))
        row[f"{metric_name}_status"] = metric_payload.get("status")
        row[f"{metric_name}_notes"] = metric_payload.get("notes")
    return row


def _group_rows(rows: list[Mapping[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in key_fields)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple("" if value is None else str(value) for value in item)):
        group = grouped[key]
        summary = {field: value for field, value in zip(key_fields, key)}
        summary["run_count"] = len(group)
        summary["nmse_mean"] = _mean(row.get("nmse_value") for row in group)
        summary["nmse_std"] = _std(row.get("nmse_value") for row in group)
        summary["knmse_mean"] = _mean(row.get("knmse_value") for row in group)
        summary["knmse_std"] = _std(row.get("knmse_value") for row in group)
        summary["gfrf_re_mean"] = _mean(row.get("gfrf_re_value") for row in group)
        summary["gfrf_re_std"] = _std(row.get("gfrf_re_value") for row in group)
        summary["success_rate"] = _mean(float(bool(row.get("success"))) for row in group)
        summary["converged_rate"] = _mean(float(bool(row.get("converged"))) for row in group)
        summary["nonfinite_rate"] = _mean(float(bool(row.get("had_nonfinite"))) for row in group)
        summary["early_stop_rate"] = _mean(float(bool(row.get("early_stopped"))) for row in group)
        summary["fit_seconds_mean"] = _mean(row.get("fit_seconds") for row in group)
        summary["nmse_status"] = _status_summary(str(row.get("nmse_status") or "") for row in group)
        summary["knmse_status"] = _status_summary(str(row.get("knmse_status") or "") for row in group)
        summary["gfrf_re_status"] = _status_summary(str(row.get("gfrf_re_status") or "") for row in group)
        summary["stability_summary"] = _stability_summary(group)
        summary_rows.append(summary)
    return summary_rows


def _wide_table(
    *,
    rows: list[Mapping[str, Any]],
    summary_rows: list[Mapping[str, Any]],
    row_key_fields: list[str],
) -> list[dict[str, Any]]:
    stability_grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        stability_grouped[tuple(row.get(field) for field in row_key_fields)].append(row)

    table_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in stability_grouped.items():
        table_rows[key] = {field: value for field, value in zip(row_key_fields, key)}
        table_rows[key]["stability_summary"] = _stability_summary(group)
        table_rows[key]["success_rate"] = _mean(float(bool(row.get("success"))) for row in group)
        table_rows[key]["converged_rate"] = _mean(float(bool(row.get("converged"))) for row in group)
        table_rows[key]["nonfinite_rate"] = _mean(float(bool(row.get("had_nonfinite"))) for row in group)
        table_rows[key]["early_stop_rate"] = _mean(float(bool(row.get("early_stopped"))) for row in group)

    for summary in summary_rows:
        key = tuple(summary.get(field) for field in row_key_fields)
        benchmark = str(summary.get("benchmark"))
        if key not in table_rows:
            table_rows[key] = {field: value for field, value in zip(row_key_fields, key)}
        target = table_rows[key]
        for metric_name in ("nmse", "knmse", "gfrf_re"):
            target[f"{benchmark}_{metric_name}"] = summary.get(f"{metric_name}_mean")
            target[f"{benchmark}_{metric_name}_status"] = summary.get(f"{metric_name}_status")
    return [
        table_rows[key]
        for key in sorted(table_rows, key=lambda item: tuple("" if value is None else str(value) for value in item))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_yaml(config_path)
    base_dir = config_path.parent
    results_root = _output_root(config, base_dir=base_dir) / args.suite
    summary_root = results_root / "summaries"

    rows = [_flatten_result(_read_json(path)) for path in _result_paths(results_root)]
    ablation_rows = [row for row in rows if row.get("experiment_kind") == "ablation"]
    hyper_rows = [row for row in rows if row.get("experiment_kind") == "hyperparameter"]

    ablation_summary = _group_rows(ablation_rows, ["variant", "label", "benchmark"])
    hyper_summary = _group_rows(hyper_rows, ["hyperparameter", "hyperparameter_value", "sweep_label", "benchmark"])

    ablation_table = _wide_table(
        rows=ablation_rows,
        summary_rows=ablation_summary,
        row_key_fields=["variant", "label"],
    )
    hyper_table = _wide_table(
        rows=hyper_rows,
        summary_rows=hyper_summary,
        row_key_fields=["hyperparameter", "hyperparameter_value", "sweep_label"],
    )

    _write_csv(summary_root / "ablation_long.csv", ablation_rows)
    _write_csv(summary_root / "hyperparameter_long.csv", hyper_rows)
    _write_csv(summary_root / "ablation_summary.csv", ablation_summary)
    _write_csv(summary_root / "hyperparameter_summary.csv", hyper_summary)
    _write_csv(summary_root / "table_5_1_8.csv", ablation_table)
    _write_csv(summary_root / "table_5_1_9.csv", hyper_table)

    _write_json(summary_root / "ablation_summary.json", ablation_summary)
    _write_json(summary_root / "hyperparameter_summary.json", hyper_summary)
    _write_json(summary_root / "table_5_1_8.json", ablation_table)
    _write_json(summary_root / "table_5_1_9.json", hyper_table)

    print(f"[exp03-collect] wrote summaries under {summary_root}")


if __name__ == "__main__":
    main()
