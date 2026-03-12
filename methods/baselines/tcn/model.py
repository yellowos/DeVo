"""TCN building blocks for sequence regression on [N, M, D] inputs."""

from __future__ import annotations

from typing import Sequence

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    F = None
    nn = None


if nn is not None:

    def _init_module_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


    class CausalConv1d(nn.Module):
        """1D convolution with left-only padding to preserve causality."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            *,
            dilation: int,
        ) -> None:
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            padded = F.pad(inputs, (self.left_padding, 0))
            return self.conv(padded)


    class TemporalBlock(nn.Module):
        """Residual TCN block with two causal convolutions."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.conv1 = CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.conv2 = CausalConv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            self.residual = nn.Identity()
            if in_channels != out_channels:
                self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            self.output_activation = nn.ReLU()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.dropout1(self.relu1(self.conv1(inputs)))
            outputs = self.dropout2(self.relu2(self.conv2(outputs)))
            return self.output_activation(outputs + self.residual(inputs))


    class TemporalConvEncoder(nn.Module):
        """Stacked residual TCN encoder over [N, C, L] inputs."""

        def __init__(
            self,
            input_dim: int,
            num_channels: Sequence[int],
            *,
            kernel_size: int,
            dilation_schedule: Sequence[int],
            dropout: float,
        ) -> None:
            super().__init__()
            layers = []
            in_channels = input_dim
            for out_channels, dilation in zip(num_channels, dilation_schedule, strict=True):
                layers.append(
                    TemporalBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
            self.network = nn.Sequential(*layers)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.network(inputs)


    class TCNRegressor(nn.Module):
        """Regression head on top of a temporal convolutional encoder."""

        def __init__(
            self,
            *,
            input_dim: int,
            output_dim: int,
            num_channels: Sequence[int],
            kernel_size: int,
            dilation_schedule: Sequence[int],
            dropout: float,
        ) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be positive.")
            if output_dim <= 0:
                raise ValueError("output_dim must be positive.")
            if not num_channels:
                raise ValueError("num_channels must contain at least one layer.")

            self.encoder = TemporalConvEncoder(
                input_dim=input_dim,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dilation_schedule=dilation_schedule,
                dropout=dropout,
            )
            self.readout = nn.Linear(num_channels[-1], output_dim)
            self.apply(_init_module_weights)

        def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
            if inputs.ndim != 3:
                raise ValueError("TCNRegressor expects inputs with shape [N, M, D].")
            sequence_first = inputs.transpose(1, 2)
            return self.encoder(sequence_first)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            encoded = self.forward_features(inputs)
            last_hidden = encoded[:, :, -1]
            return self.readout(last_hidden)

else:

    class TCNRegressor:  # pragma: no cover - exercised only when torch is absent.
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            raise RuntimeError("TCN baseline requires PyTorch to be installed.")
