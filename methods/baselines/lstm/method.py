"""Methods-layer LSTM baseline implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from methods.base import ArtifactRef, BaseMethod, KernelRecoveryNotSupportedError, MethodResult, register_method

from .model import LSTMRegressor


class _SequenceRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-conscious sequence dataset backed by float32 CPU tensors."""

    def __init__(self, X: Any, Y: Any) -> None:
        features = np.asarray(X, dtype=np.float32)
        if features.ndim != 3:
            raise ValueError(
                "LSTM baseline expects sequence inputs with shape [N, M, D]. "
                f"Received shape {features.shape}."
            )
        targets = np.asarray(Y, dtype=np.float32)
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                "Sequence inputs and targets must have the same batch dimension. "
                f"Received {features.shape[0]} and {targets.shape[0]}."
            )
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets.reshape(targets.shape[0], -1))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@register_method("lstm")
class LSTMMethod(BaseMethod):
    """PyTorch LSTM baseline compatible with the shared methods protocol."""

    DEFAULT_CONFIG = {
        "hidden_size": 64,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "batch_size": 128,
        "max_epochs": 20,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 1e-6,
        "seed": 42,
    }
    SUPPORTS_KERNEL_RECOVERY = False
    _WEIGHTS_FILENAME = "model_weights.pt"

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        merged_config = dict(self.DEFAULT_CONFIG)
        if config:
            merged_config.update(dict(config))
        super().__init__(config=merged_config, device=device, dtype=dtype)
        self.model: Optional[LSTMRegressor] = None
        self.input_dim: Optional[int] = None
        self.window_length: Optional[int] = None
        self.target_shape: tuple[int, ...] = ()
        self.flat_output_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        bundle = self.normalize_dataset_bundle(dataset_bundle)
        fit_config = dict(self.config)
        fit_config.update(kwargs)
        self.config = fit_config

        seed = int(fit_config["seed"])
        torch.manual_seed(seed)

        train_X = np.asarray(bundle.train.X, dtype=np.float32)
        train_Y = np.asarray(bundle.train.Y, dtype=np.float32)
        val_X = np.asarray(bundle.val.X, dtype=np.float32)
        val_Y = np.asarray(bundle.val.Y, dtype=np.float32)

        self._validate_sequence_inputs(train_X, train_Y, bundle.meta.input_dim, bundle.meta.window_length)
        self._validate_sequence_inputs(val_X, val_Y, bundle.meta.input_dim, bundle.meta.window_length)

        self.input_dim = int(train_X.shape[-1])
        self.window_length = int(train_X.shape[1])
        self.target_shape = tuple(int(dim) for dim in train_Y.shape[1:])
        self.flat_output_dim = int(np.prod(self.target_shape)) if self.target_shape else 1
        self.output_dim = int(bundle.meta.output_dim)
        if train_Y.ndim > 1 and train_Y.shape[-1] != self.output_dim:
            raise ValueError(
                "Target last dimension must match dataset meta output_dim. "
                f"Received target shape {train_Y.shape} and output_dim={self.output_dim}."
            )

        self._build_model()
        train_loader = self._make_loader(train_X, train_Y, batch_size=int(fit_config["batch_size"]), shuffle=True)
        val_loader = self._make_loader(val_X, val_Y, batch_size=int(fit_config["batch_size"]), shuffle=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(fit_config["learning_rate"]),
            weight_decay=float(fit_config["weight_decay"]),
        )
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        best_epoch = 0
        best_state_dict: Optional[dict[str, torch.Tensor]] = None
        epochs_without_improvement = 0
        epochs_trained = 0
        last_train_loss = float("inf")
        last_val_loss = float("inf")

        for epoch in range(int(fit_config["max_epochs"])):
            last_train_loss = self._train_one_epoch(
                train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                gradient_clip_norm=float(fit_config["gradient_clip_norm"]),
            )
            last_val_loss = self._evaluate(val_loader, loss_fn=loss_fn)
            epochs_trained = epoch + 1

            if last_val_loss + float(fit_config["early_stopping_min_delta"]) < best_val_loss:
                best_val_loss = last_val_loss
                best_epoch = epochs_trained
                best_state_dict = self._state_dict_to_cpu()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(fit_config["early_stopping_patience"]):
                    break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.is_fitted = True
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "task_family": bundle.meta.task_family.value,
            "device_type": self.runtime.device_type,
            "dtype": self.runtime.dtype_name,
            "train_samples": int(train_X.shape[0]),
            "val_samples": int(val_X.shape[0]),
            "window_length": self.window_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "target_shape": list(self.target_shape),
            "flat_output_dim": self.flat_output_dim,
            "hidden_size": int(fit_config["hidden_size"]),
            "num_layers": int(fit_config["num_layers"]),
            "dropout": float(fit_config["dropout"]),
            "bidirectional": bool(fit_config["bidirectional"]),
            "batch_size": int(fit_config["batch_size"]),
            "learning_rate": float(fit_config["learning_rate"]),
            "weight_decay": float(fit_config["weight_decay"]),
            "max_epochs": int(fit_config["max_epochs"]),
            "epochs_trained": epochs_trained,
            "best_epoch": best_epoch,
            "train_loss": float(last_train_loss),
            "best_val_loss": float(best_val_loss),
            "last_val_loss": float(last_val_loss),
            "parameter_count": self.parameter_count(),
            "kernel_recovery_supported": False,
            "gradient_ready_forward": True,
        }
        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={
                "supports_kernel_recovery": False,
                "supports_input_gradients": True,
            },
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        self._ensure_model_ready()
        batch_size = int(kwargs.get("batch_size", self.config["batch_size"]))
        outputs: list[torch.Tensor] = []
        self.model.eval()
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                features = self._coerce_batchable_tensor(X)
                for start in range(0, int(features.shape[0]), batch_size):
                    batch = features[start : start + batch_size].to(
                        device=self.runtime.device,
                        dtype=self.runtime.dtype,
                    )
                    outputs.append(self.model(batch))
            else:
                features = self._coerce_feature_array(X)
                for start in range(0, int(features.shape[0]), batch_size):
                    batch_array = features[start : start + batch_size]
                    batch = torch.as_tensor(
                        batch_array,
                        dtype=self.runtime.dtype,
                        device=self.runtime.device,
                    )
                    outputs.append(self.model(batch))
        predictions = torch.cat(outputs, dim=0)
        reshaped = self._reshape_output(predictions)
        return reshaped.detach().cpu().numpy().astype(np.float32, copy=False)

    def prepare_inputs(self, X: Any, *, requires_grad: bool = False) -> torch.Tensor:
        tensor = self._coerce_inputs(X)
        if requires_grad:
            tensor = tensor.detach().clone().requires_grad_(True)
        return tensor

    def forward_tensor(self, X: Any, *, reshape_output: bool = True) -> torch.Tensor:
        self._ensure_model_ready()
        inputs = self._coerce_inputs(X) if not isinstance(X, torch.Tensor) else self._coerce_inputs(X)
        self.model.eval()
        outputs = self.model(inputs)
        if reshape_output:
            return self._reshape_output(outputs)
        return outputs

    def export_artifacts(self, output_dir: str | Path) -> Mapping[str, Any]:
        self._ensure_model_ready()
        target_dir = Path(output_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.save(target_dir / "lstm_method.json")
        summary_path = target_dir / "training_summary.json"
        config_path = target_dir / "model_config.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(self.training_summary, handle, indent=2)
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(self.config), handle, indent=2)
        return {
            "state": ArtifactRef(kind="method_state", path=str(state_path)).to_dict(),
            "summary": ArtifactRef(kind="training_summary", path=str(summary_path)).to_dict(),
            "config": ArtifactRef(kind="model_config", path=str(config_path)).to_dict(),
        }

    def supports_kernel_recovery(self) -> bool:
        return False

    def recover_kernels(self, **kwargs: Any) -> Any:
        del kwargs
        raise KernelRecoveryNotSupportedError(
            "LSTM baseline does not support kernel recovery. "
            "Use forward_tensor(...) / prepare_inputs(...) for local gradient analysis instead."
        )

    def get_state(self) -> Mapping[str, Any]:
        return {
            "input_dim": self.input_dim,
            "window_length": self.window_length,
            "output_dim": self.output_dim,
            "target_shape": list(self.target_shape),
            "flat_output_dim": self.flat_output_dim,
            "weights_file": self._WEIGHTS_FILENAME,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        self.input_dim = self._optional_int(state.get("input_dim"))
        self.window_length = self._optional_int(state.get("window_length"))
        self.output_dim = self._optional_int(state.get("output_dim"))
        target_shape = state.get("target_shape", [])
        if isinstance(target_shape, Sequence) and not isinstance(target_shape, (str, bytes)):
            self.target_shape = tuple(int(dim) for dim in target_shape)
        else:
            self.target_shape = ()
        self.flat_output_dim = self._optional_int(state.get("flat_output_dim"))
        if self.input_dim and self.flat_output_dim:
            self._build_model()

    def _save_additional_state(self, target: Path, payload: Mapping[str, Any]) -> None:
        del payload
        self._ensure_model_ready()
        weights_path = target.parent / self._WEIGHTS_FILENAME
        torch.save(self._state_dict_to_cpu(), weights_path)

    def _load_additional_state(self, source: Path, payload: Mapping[str, Any]) -> None:
        state_payload = payload.get("state", {})
        if not isinstance(state_payload, Mapping):
            raise ValueError("Serialized LSTM state must be a mapping.")
        weights_file = state_payload.get("weights_file", self._WEIGHTS_FILENAME)
        weights_path = source.parent / str(weights_file)
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing LSTM weights checkpoint: {weights_path}")
        self._ensure_model_ready()
        checkpoint = torch.load(weights_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Invalid LSTM checkpoint payload: {weights_path}")
        self.model.load_state_dict(dict(checkpoint))
        self.model.to(device=self.runtime.device, dtype=self.runtime.dtype)

    def _validate_sequence_inputs(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        expected_input_dim: int,
        expected_window_length: int,
    ) -> None:
        if X.ndim != 3:
            raise ValueError(
                "LSTM baseline expects sequence inputs with shape [N, M, D]. "
                f"Received shape {X.shape}."
            )
        if Y.shape[0] != X.shape[0]:
            raise ValueError(
                "Sequence inputs and targets must have the same batch dimension. "
                f"Received {X.shape[0]} and {Y.shape[0]}."
            )
        if int(X.shape[-1]) != int(expected_input_dim):
            raise ValueError(
                f"Dataset meta input_dim={expected_input_dim} does not match input shape {X.shape}."
            )
        if int(X.shape[1]) != int(expected_window_length):
            raise ValueError(
                f"Dataset meta window_length={expected_window_length} does not match input shape {X.shape}."
            )

    def _build_model(self) -> None:
        if self.input_dim is None or self.flat_output_dim is None:
            raise ValueError("Cannot build LSTM model before input and target shapes are known.")
        self.model = LSTMRegressor(
            input_dim=self.input_dim,
            hidden_size=int(self.config["hidden_size"]),
            num_layers=int(self.config["num_layers"]),
            dropout=float(self.config["dropout"]),
            bidirectional=bool(self.config["bidirectional"]),
            output_dim=self.flat_output_dim,
        ).to(device=self.runtime.device, dtype=self.runtime.dtype)

    def _make_loader(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        return DataLoader(
            _SequenceRegressionDataset(X, Y),
            batch_size=max(1, int(batch_size)),
            shuffle=shuffle,
            drop_last=False,
        )

    def _train_one_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        gradient_clip_norm: float,
    ) -> float:
        self.model.train()
        loss_total = 0.0
        sample_count = 0
        for features, targets in loader:
            features = features.to(device=self.runtime.device, dtype=self.runtime.dtype)
            targets = targets.to(device=self.runtime.device, dtype=self.runtime.dtype)
            optimizer.zero_grad(set_to_none=True)
            predictions = self.model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            if gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()
            batch_size = int(features.shape[0])
            loss_total += float(loss.item()) * batch_size
            sample_count += batch_size
        return loss_total / max(sample_count, 1)

    def _evaluate(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        loss_fn: nn.Module,
    ) -> float:
        self.model.eval()
        loss_total = 0.0
        sample_count = 0
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(device=self.runtime.device, dtype=self.runtime.dtype)
                targets = targets.to(device=self.runtime.device, dtype=self.runtime.dtype)
                predictions = self.model(features)
                loss = loss_fn(predictions, targets)
                batch_size = int(features.shape[0])
                loss_total += float(loss.item()) * batch_size
                sample_count += batch_size
        return loss_total / max(sample_count, 1)

    def _coerce_inputs(self, X: Any) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            tensor = self._coerce_batchable_tensor(X).to(device=self.runtime.device, dtype=self.runtime.dtype)
        else:
            array = self._coerce_feature_array(X)
            tensor = torch.as_tensor(array, dtype=self.runtime.dtype, device=self.runtime.device)
        return tensor.contiguous()

    def _coerce_feature_array(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        if array.ndim != 3:
            raise ValueError(
                "LSTM baseline expects inputs shaped as [N, M, D] or a single sample [M, D]. "
                f"Received shape {array.shape}."
            )
        return np.ascontiguousarray(array)

    def _coerce_batchable_tensor(self, X: torch.Tensor) -> torch.Tensor:
        tensor = X
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(
                "LSTM baseline expects inputs shaped as [N, M, D] or a single sample [M, D]. "
                f"Received shape {tuple(tensor.shape)}."
            )
        return tensor.contiguous()

    def _reshape_output(self, outputs: torch.Tensor) -> torch.Tensor:
        if not self.target_shape:
            if outputs.shape[-1] == 1:
                return outputs.reshape(outputs.shape[0])
            return outputs
        return outputs.reshape(outputs.shape[0], *self.target_shape)

    def _state_dict_to_cpu(self) -> dict[str, torch.Tensor]:
        self._ensure_model_ready()
        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def _ensure_model_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("LSTM model is not initialized. Call fit(...) or load(...) first.")

    def parameter_count(self) -> int:
        self._ensure_model_ready()
        return int(sum(parameter.numel() for parameter in self.model.parameters()))

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return int(value)
