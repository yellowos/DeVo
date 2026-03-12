"""PyTorch MLP baseline for methods-layer nonlinear prediction benchmarks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from methods.base import BaseMethod, KernelRecoveryNotSupportedError, MethodResult, register_method


PathLike = str | Path

_DEFAULT_CONFIG = {
    "hidden_dim": 128,
    "num_hidden_layers": 2,
    "hidden_sizes": None,
    "dropout": 0.1,
    "batch_size": 256,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "shuffle": True,
    "log_every": 10,
    "verbose": True,
    "loss": "mse",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class _WindowRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """CPU-backed dataset that keeps arrays in float32 and flattens windows lazily."""

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        self._x = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self._y = torch.from_numpy(np.asarray(Y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._x[index].reshape(-1)
        y = self._y[index]
        return x, y


class _MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = int(input_dim)
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_features = int(hidden_size)
        layers.append(nn.Linear(in_features, int(output_dim)))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


@register_method("mlp")
class MLPMethod(BaseMethod):
    """Task-agnostic MLP baseline for windowed regression problems."""

    SUPPORTS_KERNEL_RECOVERY = False

    def __init__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ) -> None:
        merged_config = dict(_DEFAULT_CONFIG)
        if config:
            merged_config.update(dict(config))
        super().__init__(config=merged_config, device=device, dtype=dtype)
        self.model: Optional[_MLPRegressor] = None
        self.input_shape: Optional[tuple[int, ...]] = None
        self.target_shape: Optional[tuple[int, ...]] = None
        self.output_dim_flat: Optional[int] = None

    def supports_kernel_recovery(self) -> bool:
        return False

    def recover_kernels(self, **kwargs: Any) -> Any:
        del kwargs
        raise KernelRecoveryNotSupportedError(
            f"{self.method_name} is a black-box prediction baseline and does not support kernel recovery."
        )

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> MethodResult:
        bundle = self.normalize_dataset_bundle(dataset_bundle)
        train_cfg = self._resolve_training_config(kwargs)
        self.model_state_path = None

        train_x = np.asarray(bundle.train.X, dtype=np.float32)
        train_y = np.asarray(bundle.train.Y, dtype=np.float32)
        val_x = np.asarray(bundle.val.X, dtype=np.float32)
        val_y = np.asarray(bundle.val.Y, dtype=np.float32)

        train_x_flat, input_shape = self._flatten_batch_inputs(train_x)
        train_y_flat, target_shape = self._flatten_batch_targets(train_y)
        val_x_flat, _ = self._flatten_batch_inputs(val_x)
        val_y_flat, _ = self._flatten_batch_targets(val_y)

        self.input_shape = input_shape
        self.target_shape = target_shape
        self.output_dim_flat = int(train_y_flat.shape[1])
        self.model = self._build_model(input_dim=int(train_x_flat.shape[1]), output_dim=self.output_dim_flat)

        train_loader = self._build_dataloader(
            X=train_x_flat,
            Y=train_y_flat,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=bool(train_cfg["shuffle"]),
        )
        val_loader = self._build_dataloader(
            X=val_x_flat,
            Y=val_y_flat,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        criterion = nn.MSELoss()
        best_state = self._snapshot_state()
        best_metric = float("inf")
        best_epoch = 0
        epochs = int(train_cfg["epochs"])

        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer=optimizer, criterion=criterion)
            val_loss = self._evaluate(val_loader, criterion=criterion)

            metric = val_loss if val_loss is not None else train_loss
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = self._snapshot_state()

            if self._should_log(epoch=epoch, epochs=epochs, log_every=int(train_cfg["log_every"])) and bool(
                train_cfg["verbose"]
            ):
                val_loss_text = "n/a" if val_loss is None else f"{val_loss:.6f}"
                print(
                    f"[mlp] epoch {epoch}/{epochs} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss_text} device={self.runtime.device_type}"
                )

        self.model.load_state_dict(best_state)
        self.is_fitted = True
        self.training_summary = {
            "dataset_name": bundle.meta.dataset_name,
            "train_samples": int(bundle.train.num_samples),
            "val_samples": int(bundle.val.num_samples),
            "test_samples": int(bundle.test.num_samples),
            "input_shape": list(self.input_shape),
            "target_shape": list(self.target_shape),
            "flattened_input_dim": int(train_x_flat.shape[1]),
            "flattened_output_dim": int(self.output_dim_flat),
            "hidden_sizes": list(self._resolve_hidden_sizes(self.config)),
            "dropout": float(self.config["dropout"]),
            "epochs_requested": epochs,
            "best_epoch": int(best_epoch),
            "best_loss": float(best_metric),
            "loss_family": str(train_cfg["loss"]),
            "device": self.runtime.device_type,
            "dtype": self.runtime.dtype_name,
            "supports_kernel_recovery": False,
        }

        return MethodResult(
            predictions=None,
            model_state_path=self.model_state_path,
            training_summary=dict(self.training_summary),
            artifacts={},
            metadata={
                "method_name": self.method_name,
                "runtime": self.runtime.to_dict(),
            },
        )

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        del kwargs
        self._require_fitted_model()
        x_flat = self._prepare_prediction_inputs(X)
        if x_flat.shape[0] == 0:
            return np.empty((0, *self.target_shape), dtype=np.float32)
        dataset = torch.from_numpy(x_flat)
        batch_size = int(self.config["batch_size"])
        predictions: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start : start + batch_size].to(
                    device=self.runtime.device,
                    dtype=self.runtime.dtype,
                )
                outputs = self.model(batch)
                predictions.append(outputs.detach().cpu().numpy())

        stacked = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
        return stacked.reshape((stacked.shape[0], *self.target_shape))

    def save(self, path: PathLike) -> Path:
        self._require_fitted_model()
        target = Path(path).expanduser()
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
            target = target / "mlp_method.pt"

        payload = {
            "method_name": self.method_name,
            "class_name": self.__class__.__name__,
            "config": dict(self.config),
            "runtime": self.runtime.to_dict(),
            "training_summary": dict(self.training_summary),
            "is_fitted": self.is_fitted,
            "input_shape": list(self.input_shape or []),
            "target_shape": list(self.target_shape or []),
            "output_dim_flat": self.output_dim_flat,
            "state_dict": self._snapshot_state(),
        }
        torch.save(payload, target)
        self.model_state_path = str(target.resolve())
        return target

    @classmethod
    def load(cls, path: PathLike) -> "MLPMethod":
        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Serialized MLP payload must be a mapping: {source}")
        if str(payload.get("method_name", "")).strip().lower() != "mlp":
            raise ValueError(f"Serialized checkpoint belongs to '{payload.get('method_name')}', not 'mlp'.")

        runtime_payload = payload.get("runtime", {})
        config = payload.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("Serialized MLP config must be a mapping.")

        instance = cls(
            config=config,
            device=runtime_payload.get("device_type") if isinstance(runtime_payload, Mapping) else None,
            dtype=runtime_payload.get("dtype") if isinstance(runtime_payload, Mapping) else None,
        )
        instance.input_shape = tuple(int(dim) for dim in payload.get("input_shape", []))
        instance.target_shape = tuple(int(dim) for dim in payload.get("target_shape", []))
        instance.output_dim_flat = int(payload["output_dim_flat"])
        instance.model = instance._build_model(
            input_dim=int(np.prod(instance.input_shape)),
            output_dim=instance.output_dim_flat,
        )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Serialized MLP checkpoint missing state_dict.")
        instance.model.load_state_dict(state_dict)
        summary = payload.get("training_summary", {})
        if isinstance(summary, Mapping):
            instance.training_summary = dict(summary)
        instance.is_fitted = bool(payload.get("is_fitted", False))
        instance.model_state_path = str(source)
        return instance

    def export_artifacts(self, output_dir: PathLike) -> Mapping[str, Any]:
        self._require_fitted_model()
        target_dir = Path(output_dir).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)

        config_path = target_dir / "mlp_config.json"
        summary_path = target_dir / "mlp_training_summary.json"
        metadata_path = target_dir / "mlp_metadata.json"

        config_path.write_text(json.dumps(_json_ready(self.config), indent=2), encoding="utf-8")
        summary_path.write_text(json.dumps(_json_ready(self.training_summary), indent=2), encoding="utf-8")
        metadata_path.write_text(
            json.dumps(
                {
                    "method_name": self.method_name,
                    "supports_kernel_recovery": False,
                    "runtime": _json_ready(self.runtime.to_dict()),
                    "input_shape": list(self.input_shape or []),
                    "target_shape": list(self.target_shape or []),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        artifacts: dict[str, Any] = {
            "config": str(config_path),
            "training_summary": str(summary_path),
            "metadata": str(metadata_path),
        }

        if self.model_state_path:
            checkpoint_path = target_dir / "mlp_checkpoint.pt"
            source = Path(self.model_state_path).expanduser().resolve()
            if source != checkpoint_path.resolve():
                shutil.copy2(source, checkpoint_path)
            artifacts["checkpoint"] = str(checkpoint_path)

        return artifacts

    def _resolve_training_config(self, runtime_overrides: Mapping[str, Any]) -> dict[str, Any]:
        train_cfg = dict(self.config)
        for key in (
            "batch_size",
            "epochs",
            "learning_rate",
            "weight_decay",
            "shuffle",
            "log_every",
            "verbose",
            "loss",
        ):
            if key in runtime_overrides:
                train_cfg[key] = runtime_overrides[key]

        if str(train_cfg["loss"]).lower() not in {"mse", "mse_loss", "l2"}:
            raise ValueError("MLPMethod only supports the MSE loss family.")
        if int(train_cfg["batch_size"]) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(train_cfg["epochs"]) <= 0:
            raise ValueError("epochs must be positive.")
        if int(train_cfg["log_every"]) <= 0:
            raise ValueError("log_every must be positive.")
        if float(train_cfg["dropout"]) < 0.0 or float(train_cfg["dropout"]) >= 1.0:
            raise ValueError("dropout must be in [0, 1).")
        return train_cfg

    def _resolve_hidden_sizes(self, config: Mapping[str, Any]) -> list[int]:
        hidden_sizes = config.get("hidden_sizes")
        if hidden_sizes is not None:
            if not isinstance(hidden_sizes, Sequence) or isinstance(hidden_sizes, (str, bytes)):
                raise ValueError("hidden_sizes must be a sequence of positive integers.")
            sizes = [int(size) for size in hidden_sizes]
        else:
            hidden_dim = int(config["hidden_dim"])
            num_hidden_layers = int(config["num_hidden_layers"])
            sizes = [hidden_dim] * num_hidden_layers
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("MLP hidden sizes must contain positive integers.")
        return sizes

    def _build_model(self, *, input_dim: int, output_dim: int) -> _MLPRegressor:
        model = _MLPRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=self._resolve_hidden_sizes(self.config),
            dropout=float(self.config["dropout"]),
        )
        return model.to(device=self.runtime.device, dtype=self.runtime.dtype)

    def _build_dataloader(
        self,
        *,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        dataset = _WindowRegressionDataset(X, Y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _run_epoch(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        self._require_fitted_model(allow_unfitted=True)
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device=self.runtime.device, dtype=self.runtime.dtype)
            batch_y = batch_y.to(device=self.runtime.device, dtype=self.runtime.dtype)

            optimizer.zero_grad(set_to_none=True)
            predictions = self.model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = int(batch_x.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    def _evaluate(
        self,
        dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        criterion: nn.Module,
    ) -> Optional[float]:
        self._require_fitted_model(allow_unfitted=True)
        if len(dataloader.dataset) == 0:
            return None

        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device=self.runtime.device, dtype=self.runtime.dtype)
                batch_y = batch_y.to(device=self.runtime.device, dtype=self.runtime.dtype)
                loss = criterion(self.model(batch_x), batch_y)
                batch_size = int(batch_x.shape[0])
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_samples += batch_size
        return total_loss / max(total_samples, 1)

    def _snapshot_state(self) -> dict[str, torch.Tensor]:
        self._require_fitted_model(allow_unfitted=True)
        return {name: tensor.detach().cpu().clone() for name, tensor in self.model.state_dict().items()}

    def _prepare_prediction_inputs(self, X: Any) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if self.input_shape is None:
            raise RuntimeError("MLP input shape is unavailable before fit/load.")

        expected_rank = len(self.input_shape)
        if array.ndim == expected_rank:
            array = array.reshape((1, *self.input_shape))
        elif array.ndim != expected_rank + 1:
            raise ValueError(
                f"Expected inputs with rank {expected_rank + 1} for batched data or rank {expected_rank} "
                f"for a single sample, got shape {array.shape}."
            )

        flat = array.reshape(array.shape[0], -1).astype(np.float32, copy=False)
        expected_dim = int(np.prod(self.input_shape))
        if flat.shape[1] != expected_dim:
            raise ValueError(f"Expected flattened input dim {expected_dim}, got {flat.shape[1]}.")
        return flat

    def _flatten_batch_inputs(self, X: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        if X.ndim < 2:
            raise ValueError(f"Expected batched inputs with shape [N, ...], got {X.shape}.")
        feature_shape = tuple(int(dim) for dim in X.shape[1:])
        flattened = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)
        return flattened, feature_shape

    def _flatten_batch_targets(self, Y: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        if Y.ndim == 1:
            target_shape = (1,)
            flattened = Y.reshape(-1, 1).astype(np.float32, copy=False)
            return flattened, target_shape
        if Y.ndim < 2:
            raise ValueError(f"Expected batched targets with shape [N, ...], got {Y.shape}.")
        target_shape = tuple(int(dim) for dim in Y.shape[1:])
        flattened = Y.reshape(Y.shape[0], -1).astype(np.float32, copy=False)
        return flattened, target_shape

    def _require_fitted_model(self, *, allow_unfitted: bool = False) -> None:
        if self.model is None:
            raise RuntimeError("MLP model is not initialized. Call fit() or load() first.")
        if not allow_unfitted and not self.is_fitted:
            raise RuntimeError("MLP model is not fitted. Call fit() or load() first.")

    @staticmethod
    def _should_log(*, epoch: int, epochs: int, log_every: int) -> bool:
        return epoch == 1 or epoch == epochs or epoch % log_every == 0
