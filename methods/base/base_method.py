"""Abstract base class shared by all methods-layer implementations."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Mapping, Optional, Self

from methods.utils.device import DeviceContext, select_device

from .io_schema import DatasetBundleSource, MethodDatasetBundle, coerce_dataset_bundle
from .result_schema import KernelRecoveryResult, MethodResult


PathLike = str | Path


class KernelRecoveryNotSupportedError(NotImplementedError):
    """Raised when a method does not implement kernel recovery."""


class BaseMethod(ABC):
    """Stable methods-layer protocol.

    Subclasses are expected to implement `fit`, `predict`, and optionally override
    persistence/artifact hooks if JSON persistence is not sufficient.
    """

    METHOD_NAME: ClassVar[Optional[str]] = None
    SUPPORTS_KERNEL_RECOVERY: ClassVar[bool] = False

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        self.config: dict[str, Any] = dict(config or {})
        self.runtime: DeviceContext = select_device(preferred_device=device, preferred_dtype=dtype)
        self.training_summary: dict[str, Any] = {}
        self.model_state_path: Optional[str] = None
        self.is_fitted: bool = False

    @staticmethod
    def _normalize_method_name(name: str) -> str:
        return name.strip().lower().replace("-", "_").replace(" ", "_")

    @property
    def method_name(self) -> str:
        return self.METHOD_NAME or self.__class__.__name__.lower()

    def normalize_dataset_bundle(
        self,
        dataset_bundle: DatasetBundleSource,
        *,
        dataset_name: Optional[str] = None,
    ) -> MethodDatasetBundle:
        return coerce_dataset_bundle(dataset_bundle, dataset_name=dataset_name)

    @abstractmethod
    def fit(self, dataset_bundle: DatasetBundleSource, **kwargs: Any) -> MethodResult:
        """Train or calibrate the method on a data-layer dataset bundle."""

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Run inference on already prepared input features."""

    def save(self, path: PathLike) -> Path:
        """Persist the method using JSON state.

        Subclasses can override this when they need custom torch checkpoints or
        multi-file artifact layouts.
        """

        target = Path(path).expanduser()
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
            target = target / "method_state.json"

        payload = {
            "method_name": self.method_name,
            "class_name": self.__class__.__name__,
            "module_name": self.__class__.__module__,
            "config": dict(self.config),
            "runtime": self.runtime.to_dict(),
            "training_summary": dict(self.training_summary),
            "is_fitted": self.is_fitted,
            "state": dict(self.get_state()),
        }
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self._save_additional_state(target=target, payload=payload)
        self.model_state_path = str(target.resolve())
        return target

    @classmethod
    def _from_serialized_payload(cls, payload: Mapping[str, Any]) -> Self:
        runtime_payload = payload.get("runtime", {})
        device = runtime_payload.get("device_type") if isinstance(runtime_payload, Mapping) else None
        dtype = runtime_payload.get("dtype") if isinstance(runtime_payload, Mapping) else None
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("Serialized method config must be a mapping.")
        return cls(config=config, device=device, dtype=dtype)

    @classmethod
    def load(cls, path: PathLike) -> Self:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Serialized method payload must be a JSON object: {source}")

        method_cls: type[BaseMethod]
        serialized_method_name = payload.get("method_name")
        if cls is BaseMethod:
            from .registry import get_method_class

            if not isinstance(serialized_method_name, str) or not serialized_method_name.strip():
                raise ValueError("Serialized payload missing method_name.")
            try:
                method_cls = get_method_class(serialized_method_name)
            except KeyError:
                module_name = payload.get("module_name")
                if isinstance(module_name, str) and module_name.strip():
                    importlib.import_module(module_name)
                    method_cls = get_method_class(serialized_method_name)
                else:
                    raise
            base_loader = getattr(BaseMethod.load, "__func__", BaseMethod.load)
            method_loader = getattr(method_cls.load, "__func__", method_cls.load)
            if method_cls is not BaseMethod and method_loader is not base_loader:
                return method_cls.load(source)
        else:
            if isinstance(serialized_method_name, str) and serialized_method_name.strip():
                expected_name = cls.METHOD_NAME or cls.__name__.lower()
                if cls._normalize_method_name(serialized_method_name) != cls._normalize_method_name(expected_name):
                    raise ValueError(
                        f"Serialized state belongs to '{serialized_method_name}', not '{expected_name}'."
                    )
            method_cls = cls

        instance = method_cls._from_serialized_payload(payload)
        runtime_payload = payload.get("runtime", {})
        if isinstance(runtime_payload, Mapping):
            instance.runtime = select_device(
                preferred_device=runtime_payload.get("device_type"),
                preferred_dtype=runtime_payload.get("dtype"),
            )
        summary = payload.get("training_summary", {})
        if isinstance(summary, Mapping):
            instance.training_summary = dict(summary)
        instance.is_fitted = bool(payload.get("is_fitted", False))
        instance.model_state_path = str(source)
        state = payload.get("state", {})
        if state is None:
            state = {}
        if not isinstance(state, Mapping):
            raise ValueError("Serialized method state must be a mapping.")
        instance.set_state(state)
        instance._load_additional_state(source=source, payload=payload)
        return instance  # type: ignore[return-value]

    @abstractmethod
    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        """Export auxiliary artifacts produced by the method."""

    def supports_kernel_recovery(self) -> bool:
        return bool(self.SUPPORTS_KERNEL_RECOVERY)

    def recover_kernels(self, **kwargs: Any) -> KernelRecoveryResult:
        raise KernelRecoveryNotSupportedError(
            f"{self.method_name} does not support kernel recovery."
        )

    def get_state(self) -> Mapping[str, Any]:
        """Return JSON-serializable internal state for default persistence."""

        return {}

    def set_state(self, state: Mapping[str, Any]) -> None:
        """Restore internal state emitted by `get_state`."""

        del state

    def _save_additional_state(self, target: Path, payload: Mapping[str, Any]) -> None:
        """Persist subclass-specific state alongside JSON serialization."""

        del target, payload

    def _load_additional_state(self, source: Path, payload: Mapping[str, Any]) -> None:
        """Restore subclass-specific state after JSON deserialization."""

        del source, payload
