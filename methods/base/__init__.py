"""Base protocol exports for the methods layer."""

from .base_method import BaseMethod, KernelRecoveryNotSupportedError
from .io_schema import MethodDatasetBundle, MethodDatasetSplit, coerce_dataset_bundle, load_dataset_bundle
from .registry import MethodRegistry, create_method, get_method_class, list_registered_methods, register_method
from .result_schema import ArtifactRef, KernelRecoveryResult, MethodResult

__all__ = [
    "ArtifactRef",
    "BaseMethod",
    "KernelRecoveryNotSupportedError",
    "KernelRecoveryResult",
    "MethodDatasetBundle",
    "MethodDatasetSplit",
    "MethodRegistry",
    "MethodResult",
    "coerce_dataset_bundle",
    "create_method",
    "get_method_class",
    "list_registered_methods",
    "load_dataset_bundle",
    "register_method",
]
