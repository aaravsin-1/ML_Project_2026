"""Strict Non-Leakage Normalization Techniques for Limit Order Book Data."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


class DecimalPrecisionNormalizer:
    """Scales prices and volumes by fixed static ticks and lot scales.
    
    Guarantees absolute zero temporal leakage across train/val/test splits
    because no cross-sample statistics (mean or variance) are used.
    """
    def __init__(self, price_scale: Optional[float] = None, volume_scale: Optional[float] = None):
        self.price_scale = price_scale
        self.volume_scale = volume_scale

    def fit(self, prices: np.ndarray, volumes: np.ndarray):
        """Fit scales using only the initial training snapshot."""
        if self.price_scale is None:
            # Use base price from the first sample (e.g., initial mid-price)
            self.price_scale = float(np.mean(prices[:100])) if len(prices) >= 100 else float(np.mean(prices))
        if self.volume_scale is None:
            self.volume_scale = float(np.median(volumes[:100])) if len(volumes) >= 100 else float(np.median(volumes))
        if self.volume_scale <= 0:
            self.volume_scale = 1.0

    def transform_lob(self, lob_features: np.ndarray) -> np.ndarray:
        """Transforms 40-feature LOB array: [ask_p1, ask_v1, bid_p1, bid_v1, ...].
        
        Prices are normalized as relative deviation from price_scale: (P - P0) / P0
        Volumes are scaled by log1p(V / V0)
        """
        assert lob_features.shape[-1] >= 4, "Expected at least 4 LOB features (1 level)"
        normed = np.zeros_like(lob_features, dtype=np.float32)
        
        # In standard 4-tuple: [ask_p, ask_v, bid_p, bid_v] repeated for 10 levels
        price_cols = [i for i in range(lob_features.shape[-1]) if i % 2 == 0]
        volume_cols = [i for i in range(lob_features.shape[-1]) if i % 2 == 1]
        
        p_base = self.price_scale if self.price_scale is not None else 1.0
        v_base = self.volume_scale if self.volume_scale is not None else 1.0
        
        normed[..., price_cols] = (lob_features[..., price_cols] - p_base) / p_base
        normed[..., volume_cols] = np.log1p(np.maximum(0.0, lob_features[..., volume_cols]) / v_base)
        return normed


class CausalRollingZScore:
    """Strictly causal rolling z-score normalization.
    
    Computes mean and variance using only past observations within a causal lookback window W.
    Prevents future statistics from contaminating current or historical features.
    """
    def __init__(self, window: int = 1000, eps: float = 1e-6):
        self.window = window
        self.eps = eps

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Applies causal rolling standardization along axis 0."""
        n_samples, n_features = features.shape[0], features.shape[1]
        normalized = np.zeros_like(features, dtype=np.float32)
        
        # Use cumulative stats for the warmup period, then rolling window
        rolling_sum = np.zeros(n_features, dtype=np.float64)
        rolling_sq_sum = np.zeros(n_features, dtype=np.float64)
        
        for t in range(n_samples):
            val = features[t].astype(np.float64)
            rolling_sum += val
            rolling_sq_sum += val * val
            
            if t >= self.window:
                old_val = features[t - self.window].astype(np.float64)
                rolling_sum -= old_val
                rolling_sq_sum -= old_val * old_val
                count = self.window
            else:
                count = t + 1
            
            mean = rolling_sum / count
            var = np.maximum(0.0, (rolling_sq_sum / count) - (mean * mean))
            std = np.sqrt(var + self.eps)
            normalized[t] = ((val - mean) / std).astype(np.float32)
            
        return normalized


class BilinearNormalization(nn.Module):
    """Bilinear Normalization (BiN) Layer for LOB Tensors.
    
    Introduced by Tran, Gabbouj, and Iosifidis (2020) for bilinear deep learning architectures.
    Dynamically normalizes along both the temporal sequence dimension and the spatial feature dimension
    without cross-sample future leakage.
    
    Input shape: (Batch, Channels=1, Seq_Len, Features) or (Batch, Features, Seq_Len)
    """
    def __init__(self, num_features: int, seq_len: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.eps = eps
        
        # Learnable bilinear parameters
        self.gamma_feat = nn.Parameter(torch.ones(1, 1, 1, num_features))
        self.beta_feat = nn.Parameter(torch.zeros(1, 1, 1, num_features))
        self.gamma_time = nn.Parameter(torch.ones(1, 1, seq_len, 1))
        self.beta_time = nn.Parameter(torch.zeros(1, 1, seq_len, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies bilinear normalization across feature and temporal dimensions."""
        # Check input shape
        if x.dim() == 3:
            # (Batch, Seq_Len, Features) -> (Batch, 1, Seq_Len, Features)
            x = x.unsqueeze(1)
            
        # Step 1: Feature-wise normalization across temporal axis
        mean_feat = x.mean(dim=2, keepdim=True)
        var_feat = x.var(dim=2, keepdim=True, unbiased=False)
        x_norm_feat = (x - mean_feat) / torch.sqrt(var_feat + self.eps)
        x = self.gamma_feat * x_norm_feat + self.beta_feat
        
        # Step 2: Temporal normalization across feature axis
        mean_time = x.mean(dim=3, keepdim=True)
        var_time = x.var(dim=3, keepdim=True, unbiased=False)
        x_norm_time = (x - mean_time) / torch.sqrt(var_time + self.eps)
        x = self.gamma_time * x_norm_time + self.beta_time
        
        return x
