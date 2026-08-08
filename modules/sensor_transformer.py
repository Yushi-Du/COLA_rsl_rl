"""Transformer encoder for sequences of tactile sensor samples."""

from __future__ import annotations

import torch
from torch import nn


class SensorTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        seq_len: int = 5,
        num_sensors: int = 48,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        output_dim: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_sensors = num_sensors
        self.input_dim = input_dim
        self.d_model = d_model
        self.flatten_len = seq_len * num_sensors
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch_size, sequence_length, sensor_count, input_dim = x.shape
            x = x.reshape(
                batch_size, sequence_length * sensor_count, input_dim
            )
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.fc_out(x)
