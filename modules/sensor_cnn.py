"""Tactile sensor encoders used by the COLA actor and student networks."""

from __future__ import annotations

import torch
from torch import nn


class SensorCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        output_size: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(out_channels * 48, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TemporalSensorCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        output_size: int = 3,
        seq_len: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=1
        )
        self.relu = nn.ReLU()
        channels = out_channels * 48
        self.temporal_conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, sensor_count, channels = x.shape
        x = x.reshape(batch_size * sequence_length, sensor_count, channels)
        x = x.permute(0, 2, 1)
        x = self.relu(self.spatial_conv(x))
        x = x.reshape(batch_size, sequence_length, -1).permute(0, 2, 1)
        x = self.relu(self.temporal_conv1(x))
        x = self.relu(self.temporal_conv2(x)).mean(dim=2)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TemporalSensorCNN_Seqlen(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 3,
        hidden_size: int = 64,
        output_size: int = 3,
        seq_len: int = 6,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.spatial_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=1
        )
        self.relu = nn.ReLU()
        channels = out_channels * 48
        self.temporal_conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.temporal_conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, sensor_count, channels = x.shape
        x = x.reshape(batch_size * sequence_length, sensor_count, channels)
        x = x.permute(0, 2, 1)
        x = self.relu(self.spatial_conv(x))
        x = x.reshape(batch_size, sequence_length, -1).permute(0, 2, 1)
        x = self.relu(self.temporal_conv1(x))
        x = self.relu(self.temporal_conv2(x)).permute(0, 2, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class TemporalSensorCNN_OnlyCnn(nn.Module):
    def __init__(
        self,
        input_size: int = 48,
        spatial_hidden: int = 128,
        temporal_hidden: int = 128,
        output_size: int = 16,
        temporal_layers: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(input_size, spatial_hidden),
            nn.ReLU(),
            nn.Linear(spatial_hidden, spatial_hidden),
            nn.ReLU(),
        )
        temporal_convs: list[nn.Module] = []
        in_channels = spatial_hidden
        for _ in range(temporal_layers):
            temporal_convs.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        temporal_hidden,
                        kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.ReLU(),
                ]
            )
            in_channels = temporal_hidden
        self.temporal_conv = nn.Sequential(*temporal_convs)
        self.fc = nn.Linear(temporal_hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_mlp(x).permute(0, 2, 1)
        x = self.temporal_conv(x).permute(0, 2, 1)
        return self.fc(x)
