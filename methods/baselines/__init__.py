"""Baseline method implementations.

Importing this package eagerly loads built-in baseline modules so their
registration decorators populate the central method registry.
"""

from .arx_var import ARXMethod, VARMethod
from .cp_volterra import CPVolterraMethod
from .laguerre_volterra import LaguerreVolterraMethod
from .lstm import LSTMMethod
from .mlp import MLPMethod
from .narmax import NARMAXMethod
from .tcn import TCNMethod
from .tt_volterra import TTVolterraMethod

__all__ = [
    "ARXMethod",
    "CPVolterraMethod",
    "LSTMMethod",
    "LaguerreVolterraMethod",
    "MLPMethod",
    "NARMAXMethod",
    "TCNMethod",
    "TTVolterraMethod",
    "VARMethod",
]
