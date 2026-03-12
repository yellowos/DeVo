"""Stable result schema for the experiments layer."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, TypeAlias

from .utils import to_jsonable


RunState: TypeAlias = Literal["success", "skipped", "failed"]
EXPERIMENT_RESULT_SCHEMA_VERSION = "experiments.run_result.v1"


def _coerce_path(value: Optional[str | Path]) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value))


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


@dataclass(frozen=True)
class MetricRecord:
    """One metric entry attached to an experiment run."""

    name: str
    value: Any
    split: Optional[str] = None
    higher_is_better: Optional[bool] = None
    step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, name: str, value: Any) -> "MetricRecord":
        if isinstance(value, MetricRecord):
            return value
        if isinstance(value, Mapping):
            return cls(
                name=str(value.get("name", name)),
                value=value.get("value"),
                split=value.get("split"),
                higher_is_better=value.get("higher_is_better"),
                step=_coerce_optional_int(value.get("step")),
                metadata=dict(value.get("metadata", {})),
            )
        return cls(name=name, value=value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": to_jsonable(self.value),
            "split": self.split,
            "higher_is_better": self.higher_is_better,
            "step": self.step,
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class ArtifactPathRef:
    """Reference to an existing experiment or method artifact."""

    kind: str = "generic"
    path: Optional[str] = None
    uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any, *, default_kind: str = "generic") -> "ArtifactPathRef":
        if isinstance(value, ArtifactPathRef):
            return value
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls.from_value(value.to_dict(), default_kind=default_kind)
        if isinstance(value, Mapping):
            return cls(
                kind=str(value.get("kind", default_kind)),
                path=_coerce_path(value.get("path")),
                uri=value.get("uri"),
                metadata=dict(value.get("metadata", {})),
            )
        return cls(kind=default_kind, path=_coerce_path(value))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": _coerce_path(self.path),
            "uri": self.uri,
            "metadata": to_jsonable(self.metadata),
        }


@dataclass(frozen=True)
class RunStatusRecord:
    """Normalized terminal state for one experiment run."""

    state: RunState
    message: Optional[str] = None
    error_type: Optional[str] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, message: Optional[str] = "completed") -> "RunStatusRecord":
        return cls(state="success", message=message)

    @classmethod
    def skipped(
        cls,
        message: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RunStatusRecord":
        return cls(state="skipped", message=message, metadata=dict(metadata or {}))

    @classmethod
    def failed(
        cls,
        *,
        message: Optional[str] = None,
        exc: BaseException | None = None,
        traceback_text: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RunStatusRecord":
        error_type = type(exc).__name__ if exc is not None else None
        if traceback_text is None and exc is not None:
            traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            state="failed",
            message=message or (str(exc) if exc is not None else "failed"),
            error_type=error_type,
            traceback=traceback_text,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_value(cls, value: Any) -> "RunStatusRecord":
        if isinstance(value, RunStatusRecord):
            return value
        if isinstance(value, Mapping):
            raw_state = str(value.get("state", "failed"))
            state: RunState
            if raw_state not in {"success", "skipped", "failed"}:
                raise ValueError(f"Unsupported run state: {raw_state}")
            state = raw_state
            return cls(
                state=state,
                message=value.get("message"),
                error_type=value.get("error_type"),
                traceback=value.get("traceback"),
                metadata=dict(value.get("metadata", {})),
            )
        raise TypeError("status must be a RunStatusRecord or mapping.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "error_type": self.error_type,
            "traceback": self.traceback,
            "metadata": to_jsonable(self.metadata),
        }


def coerce_metrics(value: Optional[Mapping[str, Any]]) -> Dict[str, MetricRecord]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metrics must be a mapping.")
    return {str(name): MetricRecord.from_value(str(name), item) for name, item in value.items()}


def coerce_artifact_paths(value: Optional[Mapping[str, Any]]) -> Dict[str, ArtifactPathRef]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("artifact_paths must be a mapping.")
    return {
        str(name): ArtifactPathRef.from_value(item, default_kind=str(name))
        for name, item in value.items()
    }


def artifact_paths_from_method_result(result: Any) -> Dict[str, ArtifactPathRef]:
    """Extract artifact references from a methods-layer result without copying files."""

    if result is None:
        return {}
    if hasattr(result, "to_dict") and callable(result.to_dict):
        payload = result.to_dict()
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        raise TypeError("method_result must be a mapping or expose to_dict().")

    artifacts: dict[str, ArtifactPathRef] = {}
    model_state_path = payload.get("model_state_path")
    if model_state_path:
        artifacts["model_state"] = ArtifactPathRef(kind="model_state", path=_coerce_path(model_state_path))

    raw_artifacts = payload.get("artifacts", {})
    if isinstance(raw_artifacts, Mapping):
        for name, item in raw_artifacts.items():
            artifacts[str(name)] = ArtifactPathRef.from_value(item, default_kind=str(name))

    kernel_payload = payload.get("kernel_recovery")
    if isinstance(kernel_payload, Mapping):
        kernel_artifacts = kernel_payload.get("artifacts", {})
        if isinstance(kernel_artifacts, Mapping):
            for name, item in kernel_artifacts.items():
                artifact_name = f"kernel_recovery.{name}"
                artifacts[artifact_name] = ArtifactPathRef.from_value(item, default_kind=artifact_name)

    return artifacts


@dataclass(frozen=True)
class ExperimentRunResult:
    """Stable experiments-layer result payload saved per run."""

    experiment_name: str
    dataset: str
    method: str
    seed: Optional[int]
    config: Dict[str, Any]
    metrics: Dict[str, MetricRecord]
    artifact_paths: Dict[str, ArtifactPathRef]
    status: RunStatusRecord
    run_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = EXPERIMENT_RESULT_SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentRunResult":
        return cls(
            schema_version=str(value.get("schema_version", EXPERIMENT_RESULT_SCHEMA_VERSION)),
            run_id=str(value["run_id"]),
            experiment_name=str(value["experiment_name"]),
            dataset=str(value["dataset"]),
            method=str(value["method"]),
            seed=_coerce_optional_int(value.get("seed")),
            config=dict(value.get("config", {})),
            metrics=coerce_metrics(value.get("metrics")),
            artifact_paths=coerce_artifact_paths(value.get("artifact_paths")),
            status=RunStatusRecord.from_value(value["status"]),
            start_time=value.get("start_time"),
            end_time=value.get("end_time"),
            duration_seconds=_coerce_optional_float(value.get("duration_seconds")),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "config": to_jsonable(self.config),
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "artifact_paths": {
                name: artifact.to_dict() for name, artifact in self.artifact_paths.items()
            },
            "status": self.status.to_dict(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "metadata": to_jsonable(self.metadata),
        }
