"""Stable output schema for the methods layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _coerce_path(value: Optional[str | Path]) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value))


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a reproducible artifact emitted by a method."""

    kind: str = "generic"
    path: Optional[str] = None
    uri: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "path": _coerce_path(self.path),
            "uri": self.uri,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KernelRecoveryResult:
    """Optional kernel-recovery output for methods that support it."""

    kernels: Any
    summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernels": self.kernels,
            "summary": dict(self.summary),
            "artifacts": {name: artifact.to_dict() for name, artifact in self.artifacts.items()},
        }


@dataclass(frozen=True)
class MethodResult:
    """Unified methods-layer output structure."""

    predictions: Any = None
    model_state_path: Optional[str] = None
    training_summary: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)
    kernel_recovery: Optional[KernelRecoveryResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MethodResult":
        artifacts_payload = value.get("artifacts", {})
        artifacts: Dict[str, ArtifactRef] = {}
        if isinstance(artifacts_payload, Mapping):
            for name, payload in artifacts_payload.items():
                if isinstance(payload, ArtifactRef):
                    artifacts[name] = payload
                elif isinstance(payload, Mapping):
                    artifacts[name] = ArtifactRef(
                        kind=str(payload.get("kind", "generic")),
                        path=_coerce_path(payload.get("path")),
                        uri=payload.get("uri"),
                        metadata=dict(payload.get("metadata", {})),
                    )
                else:
                    artifacts[name] = ArtifactRef(path=_coerce_path(payload))

        kernel_payload = value.get("kernel_recovery")
        kernel_recovery: Optional[KernelRecoveryResult] = None
        if isinstance(kernel_payload, KernelRecoveryResult):
            kernel_recovery = kernel_payload
        elif isinstance(kernel_payload, Mapping):
            kernel_artifacts = kernel_payload.get("artifacts", {})
            parsed_artifacts: Dict[str, ArtifactRef] = {}
            if isinstance(kernel_artifacts, Mapping):
                for name, payload in kernel_artifacts.items():
                    if isinstance(payload, ArtifactRef):
                        parsed_artifacts[name] = payload
                    elif isinstance(payload, Mapping):
                        parsed_artifacts[name] = ArtifactRef(
                            kind=str(payload.get("kind", "generic")),
                            path=_coerce_path(payload.get("path")),
                            uri=payload.get("uri"),
                            metadata=dict(payload.get("metadata", {})),
                        )
                    else:
                        parsed_artifacts[name] = ArtifactRef(path=_coerce_path(payload))
            kernel_recovery = KernelRecoveryResult(
                kernels=kernel_payload.get("kernels"),
                summary=dict(kernel_payload.get("summary", {})),
                artifacts=parsed_artifacts,
            )

        return cls(
            predictions=value.get("predictions"),
            model_state_path=_coerce_path(value.get("model_state_path")),
            training_summary=dict(value.get("training_summary", {})),
            artifacts=artifacts,
            kernel_recovery=kernel_recovery,
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predictions": self.predictions,
            "model_state_path": _coerce_path(self.model_state_path),
            "training_summary": dict(self.training_summary),
            "artifacts": {name: artifact.to_dict() for name, artifact in self.artifacts.items()},
            "kernel_recovery": self.kernel_recovery.to_dict() if self.kernel_recovery else None,
            "metadata": dict(self.metadata),
        }
