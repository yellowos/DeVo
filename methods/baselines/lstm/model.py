"""Neural sequence model building blocks for the LSTM baseline."""

from __future__ import annotations

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Many-to-one LSTM regressor with optional bidirectionality."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        output_dim: int,
    ) -> None:
        super().__init__()
        effective_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.bidirectional = bool(bidirectional)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.output_dim = int(output_dim)
        self.encoder = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=effective_dropout,
            bidirectional=self.bidirectional,
        )
        direction_factor = 2 if self.bidirectional else 1
        self.readout = nn.Linear(self.hidden_size * direction_factor, self.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.encoder(inputs)
        if self.bidirectional:
            last_hidden = torch.cat((hidden_state[-2], hidden_state[-1]), dim=-1)
        else:
            last_hidden = hidden_state[-1]
        return self.readout(last_hidden)
