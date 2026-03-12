"""Training and method wrapper for DeVo."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.adapters.base import DatasetBundle
from methods.base import BaseMethod
from methods.utils import coerce_dataset_bundle, set_random_seed

from .attribution import prediction_error_gradient_attribution
from .model import DeVoConfig, DeVoModel
from .recovery import RecoveredKernelBundle, recover_devo_kernels


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prepare_input_array(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError("Expected X with shape [N, M, D] or [N, M].")
    return np.asarray(array, dtype=np.float32)


def _prepare_target_array(value: Any, *, horizon: int, output_dim: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 1:
        if horizon != 1 or output_dim != 1:
            raise ValueError("1D Y only supports horizon=1 and output_dim=1.")
        array = array[:, None, None]
    elif array.ndim == 2:
        if array.shape[1] == output_dim and horizon == 1:
            array = array[:, None, :]
        elif array.shape[1] == horizon and output_dim == 1:
            array = array[:, :, None]
        elif array.shape[1] == horizon * output_dim:
            array = array.reshape(array.shape[0], horizon, output_dim)
        else:
            raise ValueError(
                "2D Y must be [N, output_dim], [N, horizon], or [N, horizon * output_dim]."
            )
    elif array.ndim != 3:
        raise ValueError("Expected Y with shape [N, H, O], [N, O], [N, H], or [N].")

    if array.shape[1] != horizon or array.shape[2] != output_dim:
        raise ValueError(
            f"Expected Y shape [N, {horizon}, {output_dim}], got {tuple(array.shape)}."
        )
    return np.asarray(array, dtype=np.float32)


def _to_loader(X: np.ndarray, Y: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
    )


class DeVoMethod(BaseMethod):
    """High-level DeVo method wrapper.

    This class binds the shared dataset-bundle protocol to the structured DeVo
    model, training loop, kernel recovery, and gradient attribution utilities.
    """

    method_name = "devo"

    def __init__(self, config: Optional[DeVoConfig] = None, **config_overrides: Any) -> None:
        self.config = config or DeVoConfig(**config_overrides)
        super().__init__(device=self.config.device, dtype=self.config.dtype)
        self.model: Optional[DeVoModel] = None
        self.training_history: list[Dict[str, float]] = []

    def _build_model(self, bundle: DatasetBundle) -> DeVoModel:
        return DeVoModel(
            window_length=bundle.meta.window_length,
            input_dim=bundle.meta.input_dim,
            output_dim=bundle.meta.output_dim,
            horizon=bundle.meta.horizon,
            config=self.config,
        ).to(device=self.device, dtype=self.dtype)

    def _evaluate_loss(self, loader: DataLoader) -> float:
        assert self.model is not None
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)
                pred = self.model(x_batch)
                loss = torch.mean((pred - y_batch) ** 2)
                total += float(loss.item()) * x_batch.shape[0]
                count += x_batch.shape[0]
        return total / max(count, 1)

    def fit(self, dataset_bundle: Any, **kwargs: Any) -> "DeVoMethod":
        bundle = coerce_dataset_bundle(dataset_bundle)
        self.bundle_meta = bundle.meta
        set_random_seed(self.config.seed)

        X_train = _prepare_input_array(bundle.train.X)
        Y_train = _prepare_target_array(
            bundle.train.Y,
            horizon=bundle.meta.horizon,
            output_dim=bundle.meta.output_dim,
        )
        X_val = _prepare_input_array(bundle.val.X)
        Y_val = _prepare_target_array(
            bundle.val.Y,
            horizon=bundle.meta.horizon,
            output_dim=bundle.meta.output_dim,
        )

        self.model = self._build_model(bundle)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = _to_loader(
            X_train,
            Y_train,
            batch_size=int(kwargs.get("batch_size", self.config.batch_size)),
            shuffle=True,
            seed=self.config.seed,
        )
        val_loader = _to_loader(
            X_val,
            Y_val,
            batch_size=int(kwargs.get("eval_batch_size", self.config.eval_batch_size)),
            shuffle=False,
            seed=self.config.seed,
        )

        epochs = int(kwargs.get("epochs", self.config.epochs))
        verbose = bool(kwargs.get("verbose", self.config.verbose))
        self.training_history = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            count = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device, dtype=self.dtype)
                y_batch = y_batch.to(self.device, dtype=self.dtype)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(x_batch)
                loss = torch.mean((pred - y_batch) ** 2)
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                optimizer.step()
                total += float(loss.item()) * x_batch.shape[0]
                count += x_batch.shape[0]

            train_loss = total / max(count, 1)
            val_loss = self._evaluate_loss(val_loader)
            record = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            self.training_history.append(record)

            if verbose and (epoch % max(self.config.log_every, 1) == 0 or epoch == epochs):
                print(
                    f"[DeVo] epoch={epoch:03d} train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f} device={self.device.type}"
                )

        self.is_fitted = True
        return self

    def predict(self, X: Any, **kwargs: Any) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before predict().")

        array = _prepare_input_array(X)
        batch_size = int(kwargs.get("batch_size", self.config.eval_batch_size))
        self.model.eval()
        outputs = []

        with torch.no_grad():
            for start in range(0, array.shape[0], batch_size):
                stop = min(start + batch_size, array.shape[0])
                batch = torch.from_numpy(array[start:stop]).to(self.device, dtype=self.dtype)
                pred = self.model(batch).detach().cpu().numpy()
                outputs.append(pred)

        return np.concatenate(outputs, axis=0)

    def supports_kernel_recovery(self) -> bool:
        return True

    def recover_kernels(self, **kwargs: Any) -> RecoveredKernelBundle:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before recover_kernels().")
        return recover_devo_kernels(self.model, **kwargs)

    def export_parameters(self) -> Dict[str, object]:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before export_parameters().")
        return self.model.export_parameters()

    def attribute(self, X: Any, Y: Optional[Any] = None, **kwargs: Any) -> np.ndarray:
        """Return a generic `[sample, time, variable]` attribution tensor."""

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Call fit() before attribute().")

        inputs = torch.from_numpy(_prepare_input_array(X))
        targets = None
        if Y is not None:
            assert self.bundle_meta is not None
            targets = torch.from_numpy(
                _prepare_target_array(
                    Y,
                    horizon=self.bundle_meta.horizon,
                    output_dim=self.bundle_meta.output_dim,
                )
            )
        attribution = prediction_error_gradient_attribution(
            self.model,
            inputs,
            targets,
            batch_size=int(kwargs.get("batch_size", self.config.eval_batch_size)),
            mode=str(kwargs.get("mode", "gradient")),
        )
        return attribution.numpy()
