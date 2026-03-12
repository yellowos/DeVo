"""Small runtime helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Set Python / NumPy / PyTorch seeds for reproducible smoke tests."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
