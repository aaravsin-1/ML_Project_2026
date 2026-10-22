"""Tier 4: Sequential Recurrent Neural Networks (LSTM & GRU) for LOB Sequences."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMLOBNet(nn.Module):
    """LSTM network modeling temporal trajectories across sequence length T."""

    def __init__(
        self,
        num_features: int = 40,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (Batch, 1, Seq_Len, Features) or (Batch, Seq_Len, Features)
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # (Batch, Seq_Len, Features)
            
        # Apply BatchNorm across features
        b, s, f = x.size()
        x_flat = x.reshape(b * s, f)
        x_norm = self.bn_input(x_flat).reshape(b, s, f)

        out_seq, (hn, cn) = self.lstm(x_norm)
        # Take the final temporal state
        last_step = out_seq[:, -1, :]
        logits = self.fc(last_step)
        return logits


class GRULOBNet(nn.Module):
    """GRU network for temporal sequence modeling."""

    def __init__(
        self,
        num_features: int = 40,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(num_features)
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        b, s, f = x.size()
        x_flat = x.reshape(b * s, f)
        x_norm = self.bn_input(x_flat).reshape(b, s, f)
        out_seq, hn = self.gru(x_norm)
        last_step = out_seq[:, -1, :]
        logits = self.fc(last_step)
        return logits
