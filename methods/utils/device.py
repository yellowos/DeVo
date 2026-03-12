"""Device helpers with MPS-first selection."""

from __future__ import annotations

from typing import Optional

import torch


def select_default_device(preferred: Optional[str] = None) -> torch.device:
    """Pick the default execution device.

    MPS is preferred on macOS when available, otherwise CPU is used. CUDA is
    intentionally not assumed anywhere in the project.
    """

    if preferred:
        return torch.device(preferred)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
