"""Experiments-layer result summarization utilities."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .run import RESULT_FILENAME, iter_valid_run_dirs, read_run_result
from .schema import ExperimentRunResult
from .utils import flatten_mapping, to_csv_scalar


RESULT_STATUS_ORDER: dict[str, int] = {
    "failed": 0,
    "skipped": 1,
    "running": 2,
    "ready": 3,
    "partial": 4,
    "not_ready": 5,
    "missing": 6,
}

TRACKED_EXTENSIONS = {
    ".csv",
    ".jpeg",
    ".jpg",
    ".json",
    ".log",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".tex",
    ".tsv",
    ".txt",
}

FIGURE_EXTENSIONS = {".jpeg", ".jpg", ".pdf", ".png", ".svg"}
TABLE_EXTENSIONS = {".tex"}
CSV_EXTENSIONS = {".csv", ".tsv"}
TEXT_EXTENSIONS = {".log", ".md", ".txt"}
RESULT_FILE_KEYWORDS = (
    "benchmark",
    "figure",
    "index",
    "leaderboard",
    "manifest",
    "metric",
    "result",
    "summary",
    "table",
)
SKIP_TOKENS = ("skip", "skipped")
FAIL_TOKENS = ("error", "fail", "failed", "traceback")
STATUS_KEYS = ("status", "state", "result_status", "outcome", "phase")
MAX_JSON_BYTES = 5_000_000
MAX_TEXT_BYTES = 24_000

_BASE_SUMMARY_COLUMNS = [
    "schema_version",
    "experiment_name",
    "dataset",
    "method",
    "seed",
    "run_id",
    "status.state",
    "status.message",
    "status.error_type",
    "start_time",
    "end_time",
    "duration_seconds",
    "result_path",
    "run_dir",
]


def scan_run_results(results_root: str | Path) -> list[ExperimentRunResult]:
    """Recursively load every experiments-layer `result.json` file under a root."""

    records: list[ExperimentRunResult] = []
    for run_dir in iter_valid_run_dirs(results_root):
        records.append(read_run_result(run_dir / RESULT_FILENAME))
    return records


def build_run_summary_rows(
    records: Sequence[ExperimentRunResult],
) -> list[dict[str, Any]]:
    """Flatten per-run results into one-row-per-run dictionaries."""

    rows: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.experiment_name,
            item.dataset,
            item.method,
            -1 if item.seed is None else item.seed,
            item.run_id,
        ),
    ):
        row: dict[str, Any] = {
            "schema_version": record.schema_version,
            "experiment_name": record.experiment_name,
            "dataset": record.dataset,
            "method": record.method,
            "seed": record.seed,
            "run_id": record.run_id,
            "status.state": record.status.state,
            "status.message": record.status.message,
            "status.error_type": record.status.error_type,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "duration_seconds": record.duration_seconds,
        }
        row.update(
            flatten_mapping(
                {name: metric.to_dict() for name, metric in record.metrics.items()},
                prefix="metrics",
            )
        )
        row.update(
            flatten_mapping(
                {name: artifact.to_dict() for name, artifact in record.artifact_paths.items()},
                prefix="artifact_paths",
            )
        )
        row.update(flatten_mapping({"config": record.config}))
        row.update(flatten_mapping({"status": record.status.to_dict()}))
        row.update(flatten_mapping({"metadata": record.metadata}))
        rows.append(row)
    return rows


def _result_paths_for_rows(
    results_root: str | Path,
) -> dict[tuple[str, str, str, Any, str], tuple[str, str]]:
    path_index: dict[tuple[str, str, str, Any, str], tuple[str, str]] = {}
    for run_dir in iter_valid_run_dirs(results_root):
        result_path = run_dir / RESULT_FILENAME
        record = read_run_result(result_path)
        key = (
            record.experiment_name,
            record.dataset,
            record.method,
            record.seed,
            record.run_id,
        )
        path_index[key] = (str(result_path), str(run_dir))
    return path_index


def summarize_to_csv(
    records: Sequence[ExperimentRunResult] | Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    results_root: str | Path | None = None,
) -> Path:
    """Write a flat summary CSV that downstream collect/plot scripts can reuse."""

    if isinstance(records, Sequence) and records and isinstance(records[0], ExperimentRunResult):
        normalized_rows = build_run_summary_rows(records)
        if results_root is not None:
            path_index = _result_paths_for_rows(results_root)
            for row in normalized_rows:
                key = (
                    row["experiment_name"],
                    row["dataset"],
                    row["method"],
                    row["seed"],
                    row["run_id"],
                )
                result_path, run_dir = path_index.get(key, (None, None))
                row["result_path"] = result_path
                row["run_dir"] = run_dir
    elif isinstance(records, Sequence):
        normalized_rows = [dict(row) for row in records]
    else:
        normalized_rows = [dict(row) for row in records]

    for row in normalized_rows:
        row.setdefault("result_path", None)
        row.setdefault("run_dir", None)

    fieldnames = list(_BASE_SUMMARY_COLUMNS)
    extra_columns = sorted(
        {key for row in normalized_rows for key in row.keys() if key not in fieldnames}
    )
    fieldnames.extend(extra_columns)

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow({name: to_csv_scalar(row.get(name)) for name in fieldnames})
    return target


@dataclass(frozen=True)
class PaperItemSpec:
    item_id: str
    title: str
    item_type: str
    keywords: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    family: str
    title: str
    relative_dir: str
    dependency_key: str
    paper_items: tuple[PaperItemSpec, ...] = ()


@dataclass
class DependencyAssessment:
    status: str
    notes: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)


@dataclass
class ArtifactRecord:
    experiment_id: str
    artifact_kind: str
    path: str
    rel_path: str
    matched_paper_items: list[str] = field(default_factory=list)
    source: str = "scan"


@dataclass
class PaperItemRecord:
    experiment_id: str
    item_id: str
    title: str
    item_type: str
    status: str
    description: str
    artifact_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    experiment_id: str
    family: str
    title: str
    relative_dir: str
    experiment_dir: str
    dir_exists: bool
    overall_status: str
    result_status: str
    dependency_status: str
    dependency_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    result_file_count: int = 0
    artifact_counts: dict[str, int] = field(default_factory=dict)
    paper_items_ready: int = 0
    paper_items_total: int = 0
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    paper_items: list[PaperItemRecord] = field(default_factory=list)
    result_files: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)


@dataclass
class SummaryReport:
    generated_at: str
    project_root: str
    experiments_root: str
    output_dir: str
    experiments: list[ExperimentRecord]
    status_counts: dict[str, int]


DEFAULT_EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        experiment_id="nonlinear/exp01_prediction_benchmark",
        family="nonlinear",
        title="Prediction benchmark",
        relative_dir="nonlinear/exp01_prediction_benchmark",
        dependency_key="nonlinear_prediction",
        paper_items=(
            PaperItemSpec(
                item_id="tab:nonlinear_prediction_benchmark",
                title="Nonlinear prediction benchmark",
                item_type="table",
                keywords=("prediction", "benchmark", "leaderboard"),
                description="Main benchmark table for nonlinear prediction experiments.",
            ),
        ),
    ),
    ExperimentSpec(
        experiment_id="nonlinear/exp02_kernel_recovery",
        family="nonlinear",
        title="Kernel recovery",
        relative_dir="nonlinear/exp02_kernel_recovery",
        dependency_key="nonlinear_kernel",
        paper_items=(
            PaperItemSpec(
                item_id="tab:nonlinear_kernel_recovery",
                title="Nonlinear kernel recovery",
                item_type="table",
                keywords=("kernel", "recovery"),
                description="Main kernel recovery table for truth-backed nonlinear systems.",
            ),
        ),
    ),
    ExperimentSpec(
        experiment_id="nonlinear/exp03_ablation_hparam",
        family="nonlinear",
        title="Hyperparameter ablation",
        relative_dir="nonlinear/exp03_ablation_hparam",
        dependency_key="nonlinear_prediction",
        paper_items=(
            PaperItemSpec(
                item_id="tab:nonlinear_ablation_hparam",
                title="Nonlinear hyperparameter ablation",
                item_type="table",
                keywords=("ablation", "hparam", "hyperparameter"),
                description="Main ablation table for nonlinear hyperparameter sensitivity.",
            ),
        ),
    ),
    ExperimentSpec(
        experiment_id="hydraulic/exp01_unsupervised_subsystem_fault_isolation",
        family="hydraulic",
        title="Unsupervised subsystem fault isolation",
        relative_dir="hydraulic/exp01_unsupervised_subsystem_fault_isolation",
        dependency_key="hydraulic_unsupervised",
        paper_items=(
            PaperItemSpec(
                item_id="tab:hydraulic_subsystem_fault_isolation",
                title="Hydraulic subsystem fault isolation",
                item_type="table",
                keywords=("hydraulic", "subsystem", "fault", "isolation"),
                description="Main table for hydraulic subsystem fault isolation.",
            ),
        ),
    ),
    ExperimentSpec(
        experiment_id="tep/exp01_five_unit_fault_isolation",
        family="tep",
        title="Five-unit fault isolation",
        relative_dir="tep/exp01_five_unit_fault_isolation",
        dependency_key="tep_five_unit",
        paper_items=(
            PaperItemSpec(
                item_id="tab:tep_five_unit_fault_isolation",
                title="TEP five-unit fault isolation",
                item_type="table",
                keywords=("tep", "five_unit", "five-unit", "fault", "isolation"),
                description="Main table for TEP five-unit fault isolation.",
            ),
        ),
    ),
    ExperimentSpec(
        experiment_id="tep/exp02_fault_propagation_analysis",
        family="tep",
        title="Fault propagation analysis",
        relative_dir="tep/exp02_fault_propagation_analysis",
        dependency_key="tep_fault_propagation",
        paper_items=(
            PaperItemSpec(
                item_id="fig:tep_fault_propagation",
                title="TEP fault propagation analysis",
                item_type="figure",
                keywords=("propagation", "fault", "tep"),
                description="Main figure set for TEP fault propagation analysis.",
            ),
        ),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _relative_path(path: Path, root: Path) -> str:
    try:
        return _to_posix(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return _to_posix(path.resolve())


def _safe_read_text(path: Path, *, max_bytes: int = MAX_TEXT_BYTES) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.read(max_bytes)


def _load_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size > MAX_JSON_BYTES:
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _latex_escape(value: str) -> str:
    escaped = value
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        escaped = escaped.replace(source, target)
    return escaped


def _status_rank(status: str) -> int:
    return RESULT_STATUS_ORDER.get(status, 999)


def _path_tokens(path: Path) -> set[str]:
    parts = [_normalize_token(part) for part in path.parts]
    stem = _normalize_token(path.stem)
    if stem:
        parts.append(stem)
    return {part for part in parts if part}


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _classify_artifact(path: Path) -> str:
    ext = path.suffix.lower()
    parts = {_normalize_token(part) for part in path.parts}
    if ext in FIGURE_EXTENSIONS or {"figure", "figures", "plot", "plots"} & parts:
        return "figure"
    if ext in TABLE_EXTENSIONS or {"table", "tables"} & parts:
        return "table"
    if ext in CSV_EXTENSIONS:
        return "csv"
    if ext == ".json":
        return "json"
    if ext == ".md":
        return "markdown"
    if ext in TEXT_EXTENSIONS:
        return "log"
    return "other"


def _is_candidate_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    ext = path.suffix.lower()
    if ext in TRACKED_EXTENSIONS:
        return True
    lowered = path.name.lower()
    return any(keyword in lowered for keyword in RESULT_FILE_KEYWORDS)


def _should_ignore_discovery_path(path: Path) -> bool:
    ignored_tokens = {"tmp", "temp", "draft", "drafts", "scratch", "_tmp", "__pycache__"}
    return any(_normalize_token(part) in ignored_tokens for part in path.parts)


def _scan_status_markers(path: Path) -> tuple[set[str], list[str]]:
    lowered_name = path.name.lower()
    markers: set[str] = set()
    notes: list[str] = []

    if any(token in lowered_name for token in SKIP_TOKENS):
        markers.add("skipped")
        notes.append(f"skip marker file: {path.name}")
    if any(token in lowered_name for token in FAIL_TOKENS):
        markers.add("failed")
        notes.append(f"failure marker file: {path.name}")

    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        if isinstance(payload, Mapping):
            json_markers = _collect_json_status_markers(payload)
            markers.update(json_markers)
            if "running" in json_markers:
                notes.append(f"status payload marks running: {path.name}")
            if "skipped" in json_markers:
                notes.append(f"status payload marks skipped: {path.name}")
            if "failed" in json_markers:
                notes.append(f"status payload marks failed: {path.name}")
    elif path.suffix.lower() in TEXT_EXTENSIONS:
        text = _safe_read_text(path).lower()
        if any(token in text for token in ("status: skipped", "status=skipped", '"status": "skipped"', " skipped ")):
            markers.add("skipped")
            notes.append(f"text payload marks skipped: {path.name}")
        if any(token in text for token in ("traceback", "status: failed", "status=failed", '"status": "failed"')):
            markers.add("failed")
            notes.append(f"text payload marks failed: {path.name}")

    return markers, notes


def _collect_json_status_markers(payload: Mapping[str, Any], *, depth: int = 0) -> set[str]:
    if depth > 3:
        return set()

    markers: set[str] = set()
    for key in STATUS_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            normalized = _normalize_token(value)
            if normalized in {"running", "inprogress", "in_progress"}:
                markers.add("running")
            if normalized in {"skip", "skipped"}:
                markers.add("skipped")
            if normalized in {"error", "fail", "failed"}:
                markers.add("failed")
    success = payload.get("success")
    if success is False:
        markers.add("failed")
    skipped = payload.get("skipped")
    if skipped is True:
        markers.add("skipped")
    failed = payload.get("failed")
    if failed is True:
        markers.add("failed")

    for value in payload.values():
        if isinstance(value, Mapping):
            markers.update(_collect_json_status_markers(value, depth=depth + 1))
        elif isinstance(value, list) and depth < 2:
            for item in value[:10]:
                if isinstance(item, Mapping):
                    markers.update(_collect_json_status_markers(item, depth=depth + 1))
    return markers


def _is_result_file(path: Path) -> bool:
    artifact_kind = _classify_artifact(path)
    lowered = path.name.lower()
    if artifact_kind in {"csv", "figure", "table"}:
        return True
    if artifact_kind in {"json", "markdown", "log"}:
        return any(keyword in lowered for keyword in RESULT_FILE_KEYWORDS)
    return False


def _match_paper_items(
    spec: ExperimentSpec,
    artifacts: list[ArtifactRecord],
    *,
    result_status: str,
    dependency_status: str,
) -> list[PaperItemRecord]:
    records: list[PaperItemRecord] = []

    for item in spec.paper_items:
        if item.item_type == "figure":
            eligible = [artifact for artifact in artifacts if artifact.artifact_kind == "figure"]
        else:
            eligible = [artifact for artifact in artifacts if artifact.artifact_kind in {"table", "csv"}]

        matched = [
            artifact
            for artifact in eligible
            if _matches_keywords(Path(artifact.rel_path), item.keywords)
        ]
        if not matched and len(eligible) == 1:
            matched = list(eligible)

        if matched:
            status = "ready"
        elif result_status == "skipped":
            status = "skipped"
        elif result_status == "failed":
            status = "failed"
        elif dependency_status == "not_ready":
            status = "not_ready"
        else:
            status = "missing"

        notes: list[str] = []
        if matched and dependency_status in {"not_ready", "partial"}:
            notes.append(f"dependency status is {dependency_status}; treat artifact as provisional")

        record = PaperItemRecord(
            experiment_id=spec.experiment_id,
            item_id=item.item_id,
            title=item.title,
            item_type=item.item_type,
            status=status,
            description=item.description,
            artifact_paths=[artifact.path for artifact in matched],
            notes=notes,
        )
        records.append(record)

        for artifact in matched:
            if item.item_id not in artifact.matched_paper_items:
                artifact.matched_paper_items.append(item.item_id)

    return records


def _matches_keywords(path: Path, keywords: Sequence[str]) -> bool:
    if not keywords:
        return False
    tokens = _path_tokens(path)
    for keyword in keywords:
        if _normalize_token(keyword) in tokens:
            return True
    return False


def _discover_files(experiment_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not experiment_dir.exists():
        return files
    for path in sorted(experiment_dir.rglob("*")):
        if _should_ignore_discovery_path(path):
            continue
        if _is_candidate_file(path):
            files.append(path)
    return files


def _evaluate_nonlinear_prediction(project_root: Path) -> DependencyAssessment:
    metadata_path = project_root / "data" / "metadata" / "nonlinear" / "benchmark_manifest.json"
    payload = _load_json(metadata_path)
    if not isinstance(payload, Mapping):
        return DependencyAssessment(status="missing", notes=["missing nonlinear benchmark manifest"])

    benchmarks = payload.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        return DependencyAssessment(status="missing", notes=["invalid nonlinear benchmark manifest"])

    missing_paths: list[str] = []
    evidence = [_relative_path(metadata_path, project_root)]
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping):
            continue
        dataset_name = benchmark.get("benchmark_name")
        if not isinstance(dataset_name, str) or not dataset_name:
            continue
        manifest_path = (
            project_root
            / "data"
            / "processed"
            / "nonlinear"
            / dataset_name
            / f"{dataset_name}_processed_manifest.json"
        )
        evidence.append(_relative_path(manifest_path, project_root))
        if not manifest_path.exists():
            missing_paths.append(_relative_path(manifest_path, project_root))

    if missing_paths:
        return DependencyAssessment(
            status="missing",
            notes=["missing processed manifests: " + ", ".join(missing_paths)],
            evidence_paths=_unique_strings(evidence),
        )
    return DependencyAssessment(
        status="ready",
        notes=["all nonlinear benchmark processed manifests are available"],
        evidence_paths=_unique_strings(evidence),
    )


def _evaluate_nonlinear_kernel(project_root: Path) -> DependencyAssessment:
    metadata_path = project_root / "data" / "metadata" / "nonlinear" / "benchmark_manifest.json"
    kernel_truth_path = project_root / "data" / "metadata" / "nonlinear" / "kernel_truth_manifest.json"
    payload = _load_json(metadata_path)
    if not isinstance(payload, Mapping):
        return DependencyAssessment(status="missing", notes=["missing nonlinear benchmark manifest"])
    if not kernel_truth_path.exists():
        return DependencyAssessment(
            status="missing",
            notes=["missing nonlinear kernel truth manifest"],
            evidence_paths=[_relative_path(metadata_path, project_root)],
        )

    missing_paths: list[str] = []
    evidence = [
        _relative_path(metadata_path, project_root),
        _relative_path(kernel_truth_path, project_root),
    ]
    for benchmark in payload.get("benchmarks", []):
        if not isinstance(benchmark, Mapping):
            continue
        if not benchmark.get("has_ground_truth_kernel"):
            continue
        dataset_name = benchmark.get("benchmark_name")
        if not isinstance(dataset_name, str) or not dataset_name:
            continue
        manifest_path = (
            project_root
            / "data"
            / "processed"
            / "nonlinear"
            / dataset_name
            / f"{dataset_name}_processed_manifest.json"
        )
        evidence.append(_relative_path(manifest_path, project_root))
        if not manifest_path.exists():
            missing_paths.append(_relative_path(manifest_path, project_root))

    if missing_paths:
        return DependencyAssessment(
            status="missing",
            notes=["missing kernel-ready processed manifests: " + ", ".join(missing_paths)],
            evidence_paths=_unique_strings(evidence),
        )
    return DependencyAssessment(
        status="ready",
        notes=["kernel truth metadata and processed manifests are available"],
        evidence_paths=_unique_strings(evidence),
    )


def _evaluate_hydraulic_unsupervised(project_root: Path) -> DependencyAssessment:
    processed_path = project_root / "data" / "processed" / "hydraulic" / "hydraulic_processed_manifest.json"
    protocol_path = project_root / "data" / "metadata" / "hydraulic" / "single_fault_protocol.json"
    missing_paths = [path for path in (processed_path, protocol_path) if not path.exists()]
    if missing_paths:
        return DependencyAssessment(
            status="missing",
            notes=["missing hydraulic dependency files"],
            evidence_paths=[_relative_path(path, project_root) for path in missing_paths],
        )

    notes: list[str] = []
    processed_payload = _load_json(processed_path)
    protocol_payload = _load_json(protocol_path)
    if isinstance(protocol_payload, Mapping):
        if protocol_payload.get("split_protocol_kind") == "bundle_scaffold_only":
            notes.append("hydraulic split protocol is scaffold-only, not the paper-final protocol")
        grouping = protocol_payload.get("grouping")
        if isinstance(grouping, Mapping):
            grouping_notes = str(grouping.get("notes", ""))
            if "not the paper's final experiment protocol" in grouping_notes:
                notes.append(grouping_notes)
    if isinstance(processed_payload, Mapping):
        task_note = str(processed_payload.get("meta", {}).get("extras", {}).get("task_note", ""))
        if task_note:
            notes.append(task_note)
        if not task_note:
            task_note = str(processed_payload.get("task_note", ""))
            if task_note:
                notes.append(task_note)
    status = "not_ready" if notes else "ready"
    if not notes:
        notes.append("hydraulic processed manifest exists")
    return DependencyAssessment(
        status=status,
        notes=_unique_strings(notes),
        evidence_paths=[
            _relative_path(processed_path, project_root),
            _relative_path(protocol_path, project_root),
        ],
    )


def _evaluate_tep_five_unit(project_root: Path) -> DependencyAssessment:
    processed_path = project_root / "data" / "processed" / "tep" / "tep_processed_manifest.json"
    truth_path = project_root / "data" / "metadata" / "tep" / "fault_truth_table.json"
    definition_path = project_root / "data" / "metadata" / "tep" / "five_unit_definition.json"
    main_eval_subset_path = project_root / "data" / "processed" / "tep" / "tep_main_eval_subset_manifest.json"
    missing_paths = [path for path in (processed_path, truth_path, definition_path, main_eval_subset_path) if not path.exists()]
    if missing_paths:
        return DependencyAssessment(
            status="missing",
            notes=["missing TEP five-unit dependency files"],
            evidence_paths=[_relative_path(path, project_root) for path in missing_paths],
        )

    notes: list[str] = []
    truth_payload = _load_json(truth_path)
    if isinstance(truth_payload, Mapping):
        truth_notes = str(truth_payload.get("notes", ""))
        if truth_notes:
            notes.append(truth_notes)
        rows = truth_payload.get("rows", [])
        included = 0
        if isinstance(rows, list):
            included = sum(1 for row in rows if isinstance(row, Mapping) and row.get("included_in_main_eval") is True)
        if included == 0:
            notes.append("TEP five-unit truth has zero scenarios included in main evaluation")
    main_eval_payload = _load_json(main_eval_subset_path)
    if isinstance(main_eval_payload, Mapping):
        run_count = main_eval_payload.get("run_count")
        if isinstance(run_count, int) and run_count <= 0:
            notes.append("TEP main evaluation subset currently has zero runs")
        subset_notes = str(main_eval_payload.get("notes", ""))
        if subset_notes:
            notes.append(subset_notes)
    status = "not_ready" if notes else "ready"
    if not notes:
        notes.append("TEP five-unit truth and processed manifests are available")
    return DependencyAssessment(
        status=status,
        notes=_unique_strings(notes),
        evidence_paths=[
            _relative_path(processed_path, project_root),
            _relative_path(truth_path, project_root),
            _relative_path(definition_path, project_root),
            _relative_path(main_eval_subset_path, project_root),
        ],
    )


def _evaluate_tep_fault_propagation(project_root: Path) -> DependencyAssessment:
    processed_path = project_root / "data" / "processed" / "tep" / "tep_processed_manifest.json"
    propagation_path = project_root / "data" / "processed" / "tep" / "tep_propagation_subset_manifest.json"
    truth_path = project_root / "data" / "metadata" / "tep" / "fault_truth_table.json"
    missing_paths = [path for path in (processed_path, propagation_path, truth_path) if not path.exists()]
    if missing_paths:
        return DependencyAssessment(
            status="missing",
            notes=["missing TEP propagation dependency files"],
            evidence_paths=[_relative_path(path, project_root) for path in missing_paths],
        )

    notes: list[str] = []
    propagation_payload = _load_json(propagation_path)
    if isinstance(propagation_payload, Mapping):
        run_count = propagation_payload.get("run_count")
        if isinstance(run_count, int) and run_count <= 0:
            notes.append("TEP propagation subset has zero runs")
        propagation_notes = str(propagation_payload.get("notes", ""))
        if propagation_notes:
            notes.append(propagation_notes)
    truth_payload = _load_json(truth_path)
    if isinstance(truth_payload, Mapping):
        truth_notes = str(truth_payload.get("notes", ""))
        if truth_notes:
            notes.append(truth_notes)

    if any("curated" in note.lower() or "until" in note.lower() for note in notes):
        status = "not_ready"
    else:
        status = "ready"
    if not notes:
        notes.append("TEP propagation subset manifest exists")
    return DependencyAssessment(
        status=status,
        notes=_unique_strings(notes),
        evidence_paths=[
            _relative_path(processed_path, project_root),
            _relative_path(propagation_path, project_root),
            _relative_path(truth_path, project_root),
        ],
    )


DEPENDENCY_CHECKERS = {
    "nonlinear_prediction": _evaluate_nonlinear_prediction,
    "nonlinear_kernel": _evaluate_nonlinear_kernel,
    "hydraulic_unsupervised": _evaluate_hydraulic_unsupervised,
    "tep_five_unit": _evaluate_tep_five_unit,
    "tep_fault_propagation": _evaluate_tep_fault_propagation,
}


def _evaluate_dependency(spec: ExperimentSpec, project_root: Path) -> DependencyAssessment:
    checker = DEPENDENCY_CHECKERS[spec.dependency_key]
    return checker(project_root)


def _compute_result_status(
    *,
    dir_exists: bool,
    result_files: list[str],
    artifacts: list[ArtifactRecord],
    status_markers: set[str],
    paper_items_ready: int,
) -> str:
    if "failed" in status_markers:
        return "failed"
    if "skipped" in status_markers:
        return "skipped"
    if not dir_exists:
        return "missing"
    if not result_files and not artifacts:
        return "missing"
    if paper_items_ready > 0 and result_files:
        return "ready"
    return "partial"


def _compute_overall_status(result_status: str, dependency_status: str) -> str:
    if result_status in {"failed", "skipped"}:
        return result_status
    if result_status == "ready" and dependency_status == "ready":
        return "ready"
    if result_status == "ready" and dependency_status in {"partial", "not_ready"}:
        return "partial"
    if result_status == "partial":
        return "partial"
    if result_status == "missing" and dependency_status == "not_ready":
        return "not_ready"
    if result_status == "missing" and dependency_status == "partial":
        return "partial"
    return result_status


def collect_experiment_summary(
    *,
    project_root: str | Path,
    experiments_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    specs: Sequence[ExperimentSpec] = DEFAULT_EXPERIMENT_SPECS,
) -> SummaryReport:
    root = Path(project_root).expanduser().resolve()
    exp_root = Path(experiments_root).expanduser().resolve() if experiments_root else root / "experiments"
    target_output_dir = Path(output_dir).expanduser().resolve() if output_dir else exp_root / "_summary"

    experiments: list[ExperimentRecord] = []
    for spec in specs:
        experiment_dir = exp_root / spec.relative_dir
        files = _discover_files(experiment_dir)
        artifacts: list[ArtifactRecord] = []
        result_files: list[str] = []
        notes: list[str] = []
        status_markers: set[str] = set()

        for file_path in files:
            markers, marker_notes = _scan_status_markers(file_path)
            status_markers.update(markers)
            notes.extend(marker_notes)

            artifact_kind = _classify_artifact(file_path)
            if artifact_kind != "other":
                artifacts.append(
                    ArtifactRecord(
                        experiment_id=spec.experiment_id,
                        artifact_kind=artifact_kind,
                        path=_to_posix(file_path.resolve()),
                        rel_path=_relative_path(file_path, exp_root),
                    )
                )

            if _is_result_file(file_path):
                result_files.append(_to_posix(file_path.resolve()))

        dependency = _evaluate_dependency(spec, root)
        provisional_paper_items = _match_paper_items(
            spec,
            artifacts,
            result_status="missing",
            dependency_status=dependency.status,
        )
        paper_items_ready = sum(1 for item in provisional_paper_items if item.status == "ready")
        result_status = _compute_result_status(
            dir_exists=experiment_dir.exists(),
            result_files=result_files,
            artifacts=artifacts,
            status_markers=status_markers,
            paper_items_ready=paper_items_ready,
        )
        paper_items = _match_paper_items(
            spec,
            artifacts,
            result_status=result_status,
            dependency_status=dependency.status,
        )
        paper_items_ready = sum(1 for item in paper_items if item.status == "ready")
        if result_status == "partial" and paper_items_ready > 0 and result_files:
            result_status = "ready"
        overall_status = _compute_overall_status(result_status, dependency.status)

        if not experiment_dir.exists():
            notes.append(f"expected experiment directory missing: {spec.relative_dir}")
        if experiment_dir.exists() and not files:
            notes.append("experiment directory exists but no tracked result artifacts were found")

        record = ExperimentRecord(
            experiment_id=spec.experiment_id,
            family=spec.family,
            title=spec.title,
            relative_dir=spec.relative_dir,
            experiment_dir=_to_posix(experiment_dir.resolve()),
            dir_exists=experiment_dir.exists(),
            overall_status=overall_status,
            result_status=result_status,
            dependency_status=dependency.status,
            dependency_notes=_unique_strings(dependency.notes),
            notes=_unique_strings([*dependency.notes, *notes]),
            result_file_count=len(_unique_strings(result_files)),
            artifact_counts=dict(Counter(artifact.artifact_kind for artifact in artifacts)),
            paper_items_ready=paper_items_ready,
            paper_items_total=len(paper_items),
            artifacts=artifacts,
            paper_items=paper_items,
            result_files=_unique_strings(result_files),
            evidence_paths=_unique_strings(dependency.evidence_paths),
        )
        experiments.append(record)

    experiments.sort(key=lambda record: (record.family, record.experiment_id))
    status_counts = dict(Counter(record.overall_status for record in experiments))
    return SummaryReport(
        generated_at=_utc_now(),
        project_root=_to_posix(root),
        experiments_root=_to_posix(exp_root),
        output_dir=_to_posix(target_output_dir),
        experiments=experiments,
        status_counts=status_counts,
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_summary_markdown(report: SummaryReport) -> str:
    lines = [
        "# Experiments Summary",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Experiments root: `{report.experiments_root}`",
        "",
        "## Overview",
        "",
        "| Experiment | Overall | Results | Dependency | Paper items | Tables | Figures | CSV |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for experiment in report.experiments:
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment.experiment_id,
                    experiment.overall_status,
                    experiment.result_status,
                    experiment.dependency_status,
                    f"{experiment.paper_items_ready}/{experiment.paper_items_total}",
                    str(experiment.artifact_counts.get("table", 0)),
                    str(experiment.artifact_counts.get("figure", 0)),
                    str(experiment.artifact_counts.get("csv", 0)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Ready Paper Items", ""])
    ready_items = [
        item
        for experiment in report.experiments
        for item in experiment.paper_items
        if item.status == "ready"
    ]
    if ready_items:
        lines.append("| Item | Experiment | Type | Artifact paths |")
        lines.append("| --- | --- | --- | --- |")
        for item in ready_items:
            lines.append(
                f"| {item.item_id} | {item.experiment_id} | {item.item_type} | "
                + "<br>".join(f"`{path}`" for path in item.artifact_paths)
                + " |"
            )
    else:
        lines.append("- No paper-ready tables or figures detected yet.")

    lines.extend(["", "## Blockers", ""])
    blocked = [
        experiment
        for experiment in report.experiments
        if experiment.overall_status in {"failed", "missing", "not_ready", "skipped"}
    ]
    if blocked:
        for experiment in blocked:
            lines.append(f"- `{experiment.experiment_id}`: {experiment.overall_status}")
            for note in experiment.notes[:4]:
                lines.append(f"  - {note}")
    else:
        lines.append("- No blocked experiments.")

    return "\n".join(lines) + "\n"


def _render_latex_tables(report: SummaryReport) -> str:
    lines = [
        "% Auto-generated by experiments/collect_summary.py",
        r"\begin{tabular}{llll}",
        r"\toprule",
        "Item & Experiment & Status & Artifact \\\\",
        r"\midrule",
    ]
    for experiment in report.experiments:
        for item in experiment.paper_items:
            artifact = item.artifact_paths[0] if item.artifact_paths else "--"
            lines.append(
                f"{_latex_escape(item.item_id)} & "
                f"{_latex_escape(item.experiment_id)} & "
                f"{_latex_escape(item.status)} & "
                f"{_latex_escape(artifact)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_summary_outputs(report: SummaryReport, *, output_dir: str | Path | None = None) -> dict[str, Path]:
    target_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(report.output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "summary_manifest.json"
    summary_md_path = target_dir / "summary.md"
    result_index_path = target_dir / "result_index.csv"
    paper_tables_path = target_dir / "paper_tables.csv"
    missing_skipped_path = target_dir / "missing_skipped_failed.csv"
    artifact_manifest_path = target_dir / "artifact_manifest.csv"
    figure_manifest_path = target_dir / "figure_manifest.csv"
    latex_tables_path = target_dir / "paper_tables_index.tex"

    payload = {
        "generated_at": report.generated_at,
        "project_root": report.project_root,
        "experiments_root": report.experiments_root,
        "output_dir": _to_posix(target_dir),
        "status_counts": dict(report.status_counts),
        "experiments": [asdict(experiment) for experiment in report.experiments],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(_render_summary_markdown(report), encoding="utf-8")
    latex_tables_path.write_text(_render_latex_tables(report), encoding="utf-8")

    result_index_rows = [
        {
            "experiment_id": experiment.experiment_id,
            "family": experiment.family,
            "title": experiment.title,
            "overall_status": experiment.overall_status,
            "result_status": experiment.result_status,
            "dependency_status": experiment.dependency_status,
            "paper_items_ready": experiment.paper_items_ready,
            "paper_items_total": experiment.paper_items_total,
            "result_file_count": experiment.result_file_count,
            "table_count": experiment.artifact_counts.get("table", 0),
            "figure_count": experiment.artifact_counts.get("figure", 0),
            "csv_count": experiment.artifact_counts.get("csv", 0),
            "dir_exists": experiment.dir_exists,
            "relative_dir": experiment.relative_dir,
            "experiment_dir": experiment.experiment_dir,
            "notes": " | ".join(experiment.notes),
        }
        for experiment in report.experiments
    ]
    _write_csv(
        result_index_path,
        fieldnames=[
            "experiment_id",
            "family",
            "title",
            "overall_status",
            "result_status",
            "dependency_status",
            "paper_items_ready",
            "paper_items_total",
            "result_file_count",
            "table_count",
            "figure_count",
            "csv_count",
            "dir_exists",
            "relative_dir",
            "experiment_dir",
            "notes",
        ],
        rows=result_index_rows,
    )

    paper_rows = [
        {
            "experiment_id": item.experiment_id,
            "item_id": item.item_id,
            "title": item.title,
            "item_type": item.item_type,
            "status": item.status,
            "artifact_paths": " | ".join(item.artifact_paths),
            "notes": " | ".join(item.notes),
            "description": item.description,
        }
        for experiment in report.experiments
        for item in experiment.paper_items
    ]
    _write_csv(
        paper_tables_path,
        fieldnames=[
            "experiment_id",
            "item_id",
            "title",
            "item_type",
            "status",
            "artifact_paths",
            "notes",
            "description",
        ],
        rows=paper_rows,
    )

    blocked_rows = [
        {
            "experiment_id": experiment.experiment_id,
            "overall_status": experiment.overall_status,
            "result_status": experiment.result_status,
            "dependency_status": experiment.dependency_status,
            "reason": " | ".join(experiment.notes),
            "evidence_paths": " | ".join(experiment.evidence_paths),
        }
        for experiment in report.experiments
        if experiment.overall_status in {"failed", "missing", "not_ready", "skipped"}
    ]
    _write_csv(
        missing_skipped_path,
        fieldnames=[
            "experiment_id",
            "overall_status",
            "result_status",
            "dependency_status",
            "reason",
            "evidence_paths",
        ],
        rows=blocked_rows,
    )

    artifact_rows = [
        {
            "experiment_id": artifact.experiment_id,
            "artifact_kind": artifact.artifact_kind,
            "path": artifact.path,
            "relative_path": artifact.rel_path,
            "matched_paper_items": " | ".join(artifact.matched_paper_items),
            "source": artifact.source,
        }
        for experiment in report.experiments
        for artifact in experiment.artifacts
    ]
    _write_csv(
        artifact_manifest_path,
        fieldnames=[
            "experiment_id",
            "artifact_kind",
            "path",
            "relative_path",
            "matched_paper_items",
            "source",
        ],
        rows=artifact_rows,
    )

    figure_rows = [row for row in artifact_rows if row["artifact_kind"] == "figure"]
    _write_csv(
        figure_manifest_path,
        fieldnames=[
            "experiment_id",
            "artifact_kind",
            "path",
            "relative_path",
            "matched_paper_items",
            "source",
        ],
        rows=figure_rows,
    )

    return {
        "manifest": manifest_path,
        "markdown": summary_md_path,
        "result_index": result_index_path,
        "paper_tables": paper_tables_path,
        "missing_skipped_failed": missing_skipped_path,
        "artifact_manifest": artifact_manifest_path,
        "figure_manifest": figure_manifest_path,
        "latex_tables_index": latex_tables_path,
    }
