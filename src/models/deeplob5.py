"""5-Level DeepLOB variant (DeepLOB5) for 5-level Indian NSE order books (N, 1, T, 20).

Identical to DeepLOB (Zhang, Zohren, Roberts 2019) except Conv Block 3 collapses
5 depth levels with kernel (1, 5) instead of 10 depth levels with kernel (1, 10),
so zero synthetic padding of levels 6-10 is needed.
"""

import torch
import torch.nn as nn

from src.models.deeplob import InceptionModule


class DeepLOB5(nn.Module):
    """DeepLOB spatio-temporal CNN + Inception + LSTM for 5-level LOB tensors (B, 1, T, 20)."""

    def __init__(self, num_classes: int = 3, lstm_hidden: int = 64):
        super().__init__()
        # Conv Block 1: collapse price-volume pairs (20 -> 10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        # Conv Block 2: collapse bid-ask pairs (10 -> 5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        # Conv Block 3: collapse 5 depth levels (5 -> 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5), stride=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.inception1 = InceptionModule(32, out_channels=64)
        self.inception2 = InceptionModule(192, out_channels=64)
        self.lstm = nn.LSTM(input_size=192, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
