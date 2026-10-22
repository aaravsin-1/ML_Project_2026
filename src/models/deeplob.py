"""Tier 5: DeepLOB Architecture (Zhang, Zohren, Roberts 2019).

Deep Convolutional Neural Networks for Limit Order Books:
- Spatial feature extraction across price levels via 2D causal convolutions
- Multi-scale temporal feature maps via Inception modules
- Temporal sequence modeling via LSTM
- Fully connected classification head
"""

import torch
import torch.nn as nn
from typing import Tuple


class InceptionModule(nn.Module):
    """Inception block with multi-scale parallel convolutional filters."""

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        # Path 1: 1x1 conv -> 3x1 conv
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # Path 2: 1x1 conv -> 5x1 conv
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # Path 3: MaxPool -> 1x1 conv
        self.p3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.p1(x)
        out2 = self.p2(x)
        out3 = self.p3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.bn(out)


class DeepLOB(nn.Module):
    """Canonical DeepLOB Spatio-Temporal Deep Learning Architecture."""

    def __init__(self, num_classes: int = 3, lstm_hidden: int = 64):
        super().__init__()

        # Conv Block 1: collapse price-volume pairs (40 -> 20)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        # Conv Block 2: collapse bid-ask pairs (20 -> 10)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        # Conv Block 3: collapse 10 depth levels (10 -> 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32)
        )

        # Inception Modules
        self.inception1 = InceptionModule(in_channels=32, out_channels=64)
        self.inception2 = InceptionModule(in_channels=192, out_channels=64)

        # Recurrent temporal layer (192 input features from concatenated inception outputs)
        self.lstm = nn.LSTM(input_size=192, hidden_size=lstm_hidden, num_layers=1, batch_first=True)

        # Classification Head
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (Batch, 1, Seq_Len=100, Features=40)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.inception1(x)
        x = self.inception2(x)

        # Shape at this point: (Batch, Channels=192, Seq_Len_reduced, Width=1)
        # Squeeze spatial width dimension and permute to (Batch, Seq_Len, Channels)
        x = x.squeeze(3).permute(0, 2, 1)

        # LSTM temporal pass
        lstm_out, _ = self.lstm(x)

        # Use final sequence output for classification
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits
