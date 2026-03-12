"""Central registry for methods-layer implementations."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, TypeVar

from .base_method import BaseMethod

MethodT = TypeVar("MethodT", bound=BaseMethod)


def _normalize_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        raise ValueError("Method name must be a non-empty string.")
    return normalized


class MethodRegistry:
    """Central registry used by all method implementations."""

    def __init__(self) -> None:
        self._registry: Dict[str, type[BaseMethod]] = {}

    def register(
        self,
        name: str,
        method_cls: type[MethodT],
        *,
        overwrite: bool = False,
    ) -> type[MethodT]:
        if not issubclass(method_cls, BaseMethod):
            raise TypeError("Registered class must inherit from BaseMethod.")
        key = _normalize_name(name)
        existing = self._registry.get(key)
        if existing is not None and existing is not method_cls and not overwrite:
            raise KeyError(f"Method '{key}' is already registered.")
        self._registry[key] = method_cls
        method_cls.METHOD_NAME = key
        return method_cls

    def decorator(self, name: str, *, overwrite: bool = False) -> Callable[[type[MethodT]], type[MethodT]]:
        def _wrap(method_cls: type[MethodT]) -> type[MethodT]:
            return self.register(name, method_cls, overwrite=overwrite)

        return _wrap

    def get(self, name: str) -> type[BaseMethod]:
        key = _normalize_name(name)
        if key not in self._registry:
            raise KeyError(f"Unknown method: {name}")
        return self._registry[key]

    def create(self, name: str, **kwargs: Any) -> BaseMethod:
        method_cls = self.get(name)
        return method_cls(**kwargs)

    def list(self) -> list[str]:
        return sorted(self._registry)


METHOD_REGISTRY = MethodRegistry()
_BUILTIN_METHODS_LOADED = False


def _ensure_builtin_methods_loaded() -> None:
    global _BUILTIN_METHODS_LOADED
    if _BUILTIN_METHODS_LOADED:
        return
    importlib.import_module("methods.baselines")
    _BUILTIN_METHODS_LOADED = True


def register_method(
    name: str,
    method_cls: type[MethodT] | None = None,
    *,
    overwrite: bool = False,
) -> type[MethodT] | Callable[[type[MethodT]], type[MethodT]]:
    if method_cls is not None:
        return METHOD_REGISTRY.register(name, method_cls, overwrite=overwrite)
    return METHOD_REGISTRY.decorator(name, overwrite=overwrite)


def get_method_class(name: str) -> type[BaseMethod]:
    _ensure_builtin_methods_loaded()
    return METHOD_REGISTRY.get(name)


def create_method(name: str, **kwargs: Any) -> BaseMethod:
    _ensure_builtin_methods_loaded()
    return METHOD_REGISTRY.create(name, **kwargs)


def list_registered_methods() -> list[str]:
    _ensure_builtin_methods_loaded()
    return METHOD_REGISTRY.list()
