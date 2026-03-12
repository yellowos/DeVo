"""Shared method interface for the project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from methods.utils.device import select_default_device


class BaseMethod(ABC):
    """Minimal method contract shared across structured models.

    Methods are expected to accept the unified dataset bundle produced by the
    data layer and expose a stable `fit` / `predict` interface. Kernel recovery
    is optional and defaults to unsupported.
    """

    method_name = "base_method"

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = select_default_device(device)
        self.dtype = dtype
        self.is_fitted = False
        self.bundle_meta = None

    @abstractmethod
    def fit(self, dataset_bundle: Any, **kwargs: Any) -> "BaseMethod":
        """Train the method against the provided dataset bundle."""

    @abstractmethod
    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Run prediction for raw input windows."""

    def supports_kernel_recovery(self) -> bool:
        """Whether the method can recover interpretable kernels."""

        return False

    def recover_kernels(self, **kwargs: Any) -> Any:
        """Recover kernel objects when supported."""

        raise NotImplementedError(f"{self.method_name} does not expose kernel recovery.")

    def export_parameters(self) -> Any:
        """Export learned parameters in a method-specific structured format."""

        raise NotImplementedError(f"{self.method_name} does not expose parameter export.")
