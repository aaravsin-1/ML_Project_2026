"""Global Configuration and Hyperparameters for High-Frequency LOB Research."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Directories. LOB_DATA_DIR / LOB_ARTIFACTS_DIR override the defaults, e.g. to keep
# results and checkpoints on Google Drive when running on Colab.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LOB_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACTS_DIR = Path(os.environ.get("LOB_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODELS_DIR = PROJECT_ROOT / "weights"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MicrostructureConfig:
    """Microstructure and feature engineering parameters."""
    num_levels: int = 10
    pca_components: int = 3
    # VPIN configuration
    vpin_bucket_volume: float = 1000.0  # shares or contracts per volume bucket
    vpin_window_buckets: int = 50       # rolling buckets to compute VPIN
    vpin_toxic_threshold: float = 0.65  # trigger for adverse selection warning
    # Micro-price parameters
    imbalance_decay: float = 1.0


@dataclass
class LabelConfig:
    """Label formulation parameters."""
    # Prediction horizons in events (e.g., FI-2010 standard) or milliseconds
    horizons: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    time_horizons_ms: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000, 10000, 30000])
    # Threshold alpha for discrete UP/FLAT/DOWN classification
    alpha_threshold: float = 0.0002  # 2 basis points
    smoothing_window: int = 5


@dataclass
class ModelConfig:
    """Model architectures and training hyperparameters."""
    sequence_length: int = 100
    num_features: int = 40  # 10 levels * (bid_p, bid_v, ask_p, ask_v)
    num_classes: int = 3   # 0: FLAT, 1: UP, 2: DOWN (or mapped)
    batch_size: int = 128
    learning_rate: float = 0.0005
    epochs: int = 15
    early_stopping_patience: int = 5
    device: str = "cuda"  # Auto-falls back to cpu if unavailable
    # LightGBM hyperparameters
    lgb_num_leaves: int = 63
    lgb_max_depth: int = 8
    lgb_n_estimators: int = 250
    lgb_learning_rate: float = 0.05
    # TLOB hyperparameters
    tlob_d_model: int = 64
    tlob_num_heads: int = 4
    tlob_num_layers: int = 2
    tlob_dropout: float = 0.1


@dataclass
class ExecutionConfig:
    """High-Fidelity Execution Simulator parameters."""
    initial_cash: float = 1000000.0   # $1,000,000 baseline capital
    lot_size: float = 100.0           # standard trade size
    maker_fee_bps: float = -0.5       # -0.5 bps rebate for passive liquidity
    taker_fee_bps: float = 1.5        # 1.5 bps fee for aggressive spread crossing
    system_latency_ms: float = 1.0    # end-to-end execution latency
    cpp_latency_us: float = 45.0      # sub-millisecond C++ pipeline latency (microseconds)
    deep_latency_ms: float = 25.0     # DeepLOB / TLOB inference latency (milliseconds)
    max_position: float = 500.0       # maximum allowable inventory units
    adverse_selection_penalty: float = 0.0003  # slippage on toxic fills


@dataclass
class GlobalConfig:
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)


# Default global instance
CONFIG = GlobalConfig()
