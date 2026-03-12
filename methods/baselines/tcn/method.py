"""Methods-layer TCN baseline compatible with BaseMethod."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from methods.base.base_method import BaseMethod, PathLike
from methods.base.registry import register_method
from methods.base.result_schema import ArtifactRef, KernelRecoveryResult, MethodResult

from .model import TCNRegressor

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("TCN baseline requires PyTorch to be installed.")


class _SequenceRegressionDataset(Dataset):
    """Lightweight numpy-backed dataset to keep memory usage bounded."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        return torch.from_numpy(self.X[index]), torch.from_numpy(self.Y[index])


def _as_float32_array(value: Any, *, name: str) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return array
    return np.ascontiguousarray(array)


def _normalize_num_channels(value: Any) -> list[int]:
    if isinstance(value, int):
        channels = [int(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        channels = [int(item) for item in value]
    else:
        raise ValueError("num_channels must be an int or a sequence of ints.")
    if not channels or any(item <= 0 for item in channels):
        raise ValueError("num_channels must contain positive integers.")
    return channels


def _normalize_dilation_schedule(value: Any, *, depth: int) -> list[int]:
    if value is None:
        schedule = [2**idx for idx in range(depth)]
    elif isinstance(value, int):
        schedule = [int(value)]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        schedule = [int(item) for item in value]
    else:
        raise ValueError("dilation_schedule must be None, an int, or a sequence of ints.")
    if len(schedule) != depth:
        raise ValueError("dilation_schedule must match the number of TCN layers.")
    if any(item <= 0 for item in schedule):
        raise ValueError("dilation_schedule must contain positive integers.")
    return schedule


def _resolve_target_shape(Y: np.ndarray) -> tuple[tuple[int, ...], int]:
    if Y.ndim == 1:
        return (), 1
    target_shape = tuple(int(dim) for dim in Y.shape[1:])
    target_dim = int(np.prod(target_shape, dtype=np.int64))
    return target_shape, target_dim


@register_method("tcn")
class TCNMethod(BaseMethod):
    """Black-box TCN baseline for sequence prediction."""

    SUPPORTS_KERNEL_RECOVERY = False
    _DEFAULT_CONFIG = {
        "num_channels": [32, 32, 32],
        "kernel_size": 3,
        "dilation_schedule": None,
        "dropout": 0.1,
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "shuffle": True,
        "grad_clip_norm": None,
    }

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        resolved_config = self._resolve_config(config)
        super().__init__(config=resolved_config, device=device, dtype=dtype)
        self.model: Optional[TCNRegressor] = None
        self.input_dim: Optional[int] = None
        self.target_shape: Optional[tuple[int, ...]] = None
        self.target_dim: Optional[int] = None
        self.checkpoint_path: Optional[str] = None

    @classmethod
    def _resolve_config(cls, config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        merged = dict(cls._DEFAULT_CONFIG)
        if config:
            merged.update(dict(config))

        num_channels = _normalize_num_channels(merged["num_channels"])
        dilation_source = merged.get("dilation_schedule", merged.get("dilations"))
        dilation_schedule = _normalize_dilation_schedule(dilation_source, depth=len(num_channels))
        kernel_size = int(merged["kernel_size"])
        dropout = float(merged["dropout"])
        batch_size = int(merged["batch_size"])
        epochs = int(merged["epochs"])
        learning_rate = float(merged["learning_rate"])
        weight_decay = float(merged["weight_decay"])
        grad_clip_norm = merged.get("grad_clip_norm")

        if kernel_size <= 1:
            raise ValueError("kernel_size must be greater than 1 for a meaningful TCN receptive field.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if grad_clip_norm is not None:
            grad_clip_norm = float(grad_clip_norm)
            if grad_clip_norm <= 0.0:
                raise ValueError("grad_clip_norm must be positive when provided.")

        merged["num_channels"] = num_channels
        merged["dilation_schedule"] = dilation_schedule
        merged.pop("dilations", None)
        merged["kernel_size"] = kernel_size
        merged["dropout"] = dropout
        merged["batch_size"] = batch_size
        merged["epochs"] = epochs
        merged["learning_rate"] = learning_rate
        merged["weight_decay"] = weight_decay
        merged["shuffle"] = bool(merged["shuffle"])
        merged["grad_clip_norm"] = grad_clip_norm
        return merged

    def supports_kernel_recovery(self) -> bool:
        return False

    def recover_kernels(self, **kwargs: Any) -> KernelRecoveryResult:
        del kwargs
        return KernelRecoveryResult(
            kernels=None,
            summary={
                "supported": False,
                "reason": "TCN is a black-box baseline and does not implement kernel recovery.",
            },
            artifacts={},
        )

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        _require_torch()
        bundle = self.normalize_dataset_bundle(dataset_bundle)
        X_train = self._coerce_inputs(bundle.train.X, name="train.X")
        Y_train, target_shape = self._coerce_targets(bundle.train.Y, name="train.Y")
        X_val = self._coerce_inputs(bundle.val.X, name="val.X")
        Y_val, _ = self._coerce_targets(bundle.val.Y, name="val.Y", expected_shape=target_shape)

        if X_train.shape[0] == 0:
            raise ValueError("TCN baseline requires at least one training sample.")
        if X_val.shape[-1] != X_train.shape[-1]:
            raise ValueError("Validation split must use the same input_dim as training.")

        train_config = dict(self.config)
        train_config.update({key: value for key, value in kwargs.items() if key in train_config})
        batch_size = min(int(train_config["batch_size"]), int(X_train.shape[0]))

        self.input_dim = int(X_train.shape[-1])
        self.target_shape = target_shape
        self.target_dim = int(Y_train.shape[-1])
        self._build_model(input_dim=self.input_dim, target_dim=self.target_dim)

        train_loader = DataLoader(
            _SequenceRegressionDataset(X_train, Y_train),
            batch_size=batch_size,
            shuffle=bool(train_config["shuffle"]),
            num_workers=0,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        loss_fn = nn.MSELoss()
        grad_clip_norm = train_config.get("grad_clip_norm")
        best_val_loss = math.inf
        last_train_loss = math.inf

        for epoch in range(int(train_config["epochs"])):
            self.model.train()
            cumulative_loss = 0.0
            total_samples = 0

            for batch_X, batch_Y in train_loader:
                batch_X = batch_X.to(device=self.runtime.device, dtype=self.runtime.dtype)
                batch_Y = batch_Y.to(device=self.runtime.device, dtype=self.runtime.dtype)

                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch_X)
                loss = loss_fn(predictions, batch_Y)
                loss.backward()
                if grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

                batch_size_actual = int(batch_X.shape[0])
                cumulative_loss += float(loss.detach().cpu().item()) * batch_size_actual
                total_samples += batch_size_actual

            last_train_loss = cumulative_loss / max(total_samples, 1)
            val_loss = self._evaluate_loss(X_val, Y_val, batch_size=batch_size, loss_fn=loss_fn)
            if math.isfinite(val_loss):
                best_val_loss = min(best_val_loss, val_loss)

        self.is_fitted = True
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "task_family": bundle.meta.task_family.value,
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "window_length": int(X_train.shape[1]),
            "input_dim": int(self.input_dim),
            "target_shape": list(self.target_shape or []),
            "epochs": int(train_config["epochs"]),
            "batch_size": batch_size,
            "learning_rate": float(train_config["learning_rate"]),
            "train_loss": float(last_train_loss),
            "best_val_loss": None if math.isinf(best_val_loss) else float(best_val_loss),
            "device_type": self.runtime.device_type,
            "dtype": self.runtime.dtype_name,
            "input_layout": "[N, M, D]",
            "internal_layout": "[N, D, M]",
            "supports_local_input_gradients": True,
            "supports_kernel_recovery": False,
        }
        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={
                "supports_local_input_gradients": True,
                "supports_kernel_recovery": False,
            },
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        predictions = self._predict_flat(
            X,
            batch_size=int(kwargs.get("batch_size", self.config["batch_size"])),
        )
        return self._restore_output_shape(predictions)

    def predict_tensor(self, X: Any) -> Any:
        _require_torch()
        self._ensure_model_ready(require_fitted=True)

        if isinstance(X, torch.Tensor):
            tensor = X.to(device=self.runtime.device, dtype=self.runtime.dtype)
        else:
            array = self._coerce_inputs(X, name="X")
            tensor = torch.as_tensor(array, device=self.runtime.device, dtype=self.runtime.dtype)

        self.model.eval()
        return self.model(tensor)

    def compute_input_gradients(
        self,
        X: Any,
        *,
        target_index: Optional[int | tuple[int, ...]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        _require_torch()
        self._ensure_model_ready(require_fitted=True)
        array = self._coerce_inputs(X, name="X")
        effective_batch_size = max(1, int(batch_size or self.config["batch_size"]))
        flat_target_index = self._resolve_target_index(target_index)
        gradients: list[np.ndarray] = []

        self.model.eval()
        for start in range(0, int(array.shape[0]), effective_batch_size):
            stop = min(start + effective_batch_size, int(array.shape[0]))
            batch = torch.as_tensor(
                array[start:stop],
                device=self.runtime.device,
                dtype=self.runtime.dtype,
            )
            batch.requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            predictions = self.model(batch)
            if flat_target_index is None:
                objective = predictions.sum()
            else:
                objective = predictions[:, flat_target_index].sum()
            objective.backward()
            gradients.append(batch.grad.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.concatenate(gradients, axis=0)

    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        target = Path(output_dir).expanduser()
        target.mkdir(parents=True, exist_ok=True)

        config_path = target / "tcn_config.json"
        summary_path = target / "tcn_training_summary.json"
        capabilities_path = target / "tcn_capabilities.json"

        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(self.config), handle, indent=2)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(self.training_summary), handle, indent=2)
        with capabilities_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "supports_kernel_recovery": False,
                    "supports_local_input_gradients": True,
                    "input_layout": "[N, M, D]",
                    "internal_layout": "[N, D, M]",
                    "device_type": self.runtime.device_type,
                    "dtype": self.runtime.dtype_name,
                },
                handle,
                indent=2,
            )

        return {
            "config": ArtifactRef(kind="json", path=str(config_path)),
            "training_summary": ArtifactRef(kind="json", path=str(summary_path)),
            "capabilities": ArtifactRef(kind="json", path=str(capabilities_path)),
        }

    def save(self, path: PathLike) -> Path:
        _require_torch()
        target = Path(path).expanduser()
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            manifest_path = target
            checkpoint_path = target.with_suffix(".pt")
        else:
            target.mkdir(parents=True, exist_ok=True)
            manifest_path = target / "method_state.json"
            checkpoint_path = target / "model_state.pt"

        state_payload = dict(self.get_state())
        if self.model is not None:
            checkpoint = {
                "model_state_dict": {name: tensor.detach().cpu() for name, tensor in self.model.state_dict().items()},
                "input_dim": self.input_dim,
                "target_shape": list(self.target_shape or []),
                "target_dim": self.target_dim,
            }
            torch.save(checkpoint, checkpoint_path)
            state_payload["checkpoint_path"] = os.path.relpath(checkpoint_path, manifest_path.parent)
            self.checkpoint_path = str(checkpoint_path.resolve())
        else:
            state_payload["checkpoint_path"] = None
            self.checkpoint_path = None

        payload = {
            "method_name": self.method_name,
            "class_name": self.__class__.__name__,
            "config": dict(self.config),
            "runtime": self.runtime.to_dict(),
            "training_summary": dict(self.training_summary),
            "is_fitted": self.is_fitted,
            "state": state_payload,
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.model_state_path = str(manifest_path.resolve())
        return manifest_path

    def get_state(self) -> Mapping[str, Any]:
        return {
            "input_dim": self.input_dim,
            "target_shape": list(self.target_shape) if self.target_shape is not None else None,
            "target_dim": self.target_dim,
            "checkpoint_path": None,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        _require_torch()
        input_dim_raw = state.get("input_dim")
        target_dim_raw = state.get("target_dim")
        target_shape_raw = state.get("target_shape")

        self.input_dim = int(input_dim_raw) if input_dim_raw is not None else None
        self.target_dim = int(target_dim_raw) if target_dim_raw is not None else None
        if target_shape_raw is None:
            self.target_shape = None
        else:
            self.target_shape = tuple(int(item) for item in target_shape_raw)

        if self.input_dim is not None and self.target_dim is not None:
            self._build_model(input_dim=self.input_dim, target_dim=self.target_dim)

        checkpoint_ref = state.get("checkpoint_path")
        if checkpoint_ref and self.model is not None:
            if self.model_state_path is None:
                raise ValueError("Cannot restore TCN checkpoint without model_state_path.")
            manifest_path = Path(self.model_state_path).expanduser().resolve()
            checkpoint_path = (manifest_path.parent / str(checkpoint_ref)).resolve()
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_state_dict = checkpoint.get("model_state_dict")
            if not isinstance(model_state_dict, Mapping):
                raise ValueError("Serialized TCN checkpoint missing model_state_dict.")
            self.model.load_state_dict(model_state_dict)
            self.model.to(device=self.runtime.device, dtype=self.runtime.dtype)
            self.model.eval()
            self.checkpoint_path = str(checkpoint_path)
        else:
            self.checkpoint_path = None

    def _build_model(self, *, input_dim: int, target_dim: int) -> None:
        _require_torch()
        self.model = TCNRegressor(
            input_dim=input_dim,
            output_dim=target_dim,
            num_channels=self.config["num_channels"],
            kernel_size=int(self.config["kernel_size"]),
            dilation_schedule=self.config["dilation_schedule"],
            dropout=float(self.config["dropout"]),
        )
        self.model.to(device=self.runtime.device, dtype=self.runtime.dtype)

    def _ensure_model_ready(self, *, require_fitted: bool) -> None:
        if self.model is None:
            raise RuntimeError("TCN model is uninitialized; call fit(...) or load(...) first.")
        if require_fitted and not self.is_fitted:
            raise RuntimeError("TCN model must be fitted or loaded before inference.")

    def _coerce_inputs(self, X: Any, *, name: str) -> np.ndarray:
        array = _as_float32_array(X, name=name)
        if array.ndim != 3:
            raise ValueError(f"{name} must have shape [N, M, D], received {array.shape}.")
        return array

    def _coerce_targets(
        self,
        Y: Any,
        *,
        name: str,
        expected_shape: Optional[tuple[int, ...]] = None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        array = _as_float32_array(Y, name=name)
        if array.ndim == 0:
            raise ValueError(f"{name} must contain at least one target dimension.")
        target_shape, target_dim = _resolve_target_shape(array)
        if expected_shape is not None and target_shape != expected_shape:
            raise ValueError(f"{name} target shape {target_shape} does not match expected {expected_shape}.")
        reshaped = array.reshape(int(array.shape[0]), target_dim)
        return reshaped, target_shape

    def _predict_flat(self, X: Any, *, batch_size: int) -> np.ndarray:
        _require_torch()
        self._ensure_model_ready(require_fitted=True)
        array = self._coerce_inputs(X, name="X")
        if array.shape[0] == 0:
            return np.empty((0, int(self.target_dim or 1)), dtype=np.float32)

        effective_batch_size = max(1, int(batch_size))
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, int(array.shape[0]), effective_batch_size):
                stop = min(start + effective_batch_size, int(array.shape[0]))
                batch = torch.as_tensor(
                    array[start:stop],
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                batch_output = self.model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
                outputs.append(batch_output)
        return np.concatenate(outputs, axis=0)

    def _restore_output_shape(self, outputs: np.ndarray) -> np.ndarray:
        if self.target_shape is None:
            return outputs
        if not self.target_shape:
            return outputs.reshape(int(outputs.shape[0]))
        return outputs.reshape(int(outputs.shape[0]), *self.target_shape)

    def _evaluate_loss(self, X: np.ndarray, Y: np.ndarray, *, batch_size: int, loss_fn: Any) -> float:
        if X.shape[0] == 0:
            return math.nan
        self._ensure_model_ready(require_fitted=False)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, int(X.shape[0]), max(1, int(batch_size))):
                stop = min(start + max(1, int(batch_size)), int(X.shape[0]))
                batch = torch.as_tensor(
                    X[start:stop],
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                batch_output = self.model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
                predictions.append(batch_output)
        predictions = np.concatenate(predictions, axis=0)
        predictions_tensor = torch.from_numpy(predictions)
        targets_tensor = torch.from_numpy(Y)
        return float(loss_fn(predictions_tensor, targets_tensor).item())

    def _resolve_target_index(self, target_index: Optional[int | tuple[int, ...]]) -> Optional[int]:
        if target_index is None:
            return None
        if self.target_dim is None:
            raise RuntimeError("Target dimensionality is unknown; fit or load the model first.")
        if isinstance(target_index, tuple):
            if self.target_shape is None:
                raise ValueError("Tuple target_index requires a known target_shape.")
            if not self.target_shape:
                raise ValueError("Scalar targets do not support tuple target_index.")
            flat_index = int(np.ravel_multi_index(target_index, self.target_shape))
        else:
            flat_index = int(target_index)
        if flat_index < 0 or flat_index >= int(self.target_dim):
            raise IndexError(f"target_index={flat_index} is out of bounds for target_dim={self.target_dim}.")
        return flat_index
