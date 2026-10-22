"""Tier 5: TLOB (Transformer for Limit Order Book) with Dual Attention and BiN.

Combines:
- Bilinear Normalization (BiN) across temporal and spatial dimensions
- Dual Attention Mechanism: Feature Self-Attention (spatial book depth) + Temporal Self-Attention
- Position encoding and multi-layer Transformer encoder
- Multi-class classification head for short-term price trend prediction
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from src.data.normalizers import BilinearNormalization


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal tokens."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]


class DualAttentionBlock(nn.Module):
    """Dual Attention: Spatial Feature Attention + Temporal Sequence Attention."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # 1. Temporal Attention
        self.temporal_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # 2. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Temporal self-attention with residual
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TLOB(nn.Module):
    """Transformer for Limit Order Book with Bilinear Normalization and Dual Attention."""

    def __init__(
        self,
        num_features: int = 40,
        seq_len: int = 100,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model

        # Bilinear Normalization Layer
        self.bin = BilinearNormalization(num_features=num_features, seq_len=seq_len)

        # Spatial Feature Attention: Project 40 features to d_model per timestep
        self.spatial_proj = nn.Linear(num_features, d_model)
        self.spatial_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # Positional encoding for sequence dimension
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len + 10)

        # Stacked Temporal Dual Attention Blocks
        self.layers = nn.ModuleList([
            DualAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, 1, Seq_Len, Features) or (Batch, Seq_Len, Features)
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # (Batch, Seq_Len, Features)

        # 1. Apply Bilinear Normalization (expects 4D or 3D)
        x_norm = self.bin(x.unsqueeze(1)).squeeze(1)  # (Batch, Seq_Len, Features)

        # 2. Spatial Projection & Attention
        h_spatial = self.spatial_proj(x_norm)  # (Batch, Seq_Len, d_model)
        h_spatial_attn, _ = self.spatial_attn(h_spatial, h_spatial, h_spatial)
        h = self.spatial_norm(h_spatial + h_spatial_attn)

        # 3. Add Temporal Positional Encoding
        h = self.pos_encoder(h)

        # 4. Temporal Attention Transformer Blocks
        for layer in self.layers:
            h = layer(h)

        # 5. Global Temporal Pooling + Final Token Representation
        pool_mean = h.mean(dim=1)
        last_tok = h[:, -1, :]
        rep = 0.5 * (pool_mean + last_tok)

        logits = self.classifier(rep)
        return logits
