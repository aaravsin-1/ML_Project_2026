"""PyTorch Dataset and Chronological DataLoader utilities for Limit Order Book Sequences."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any


class LOBSequenceDataset(Dataset):
    """Sequence dataset for DeepLOB and TLOB architectures.
    
    Extracts sliding temporal windows of length T (e.g. 100 timesteps) across F features.
    Shape returned: (1, T, F) for DeepLOB (2D conv input) or (T, F) for sequence transformers.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, seq_len: int = 100, add_channel_dim: bool = True):
        assert len(features) == len(labels), f"Length mismatch: {len(features)} vs {len(labels)}"
        assert len(features) > seq_len, f"Data length {len(features)} must be greater than seq_len {seq_len}"
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len
        self.add_channel_dim = add_channel_dim

    def __len__(self) -> int:
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Window: [idx : idx + seq_len]
        x = self.features[idx : idx + self.seq_len]
        # Label: corresponding to the state at the end of the window (idx + seq_len - 1)
        y = self.labels[idx + self.seq_len - 1]
        
        if self.add_channel_dim:
            # (1, T, F) for 2D Conv
            x = x.unsqueeze(0)
            
        return x, y


def temporal_train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Strict chronological partitioning to prevent lookahead and temporal data leakage.
    
    Guarantees:
    Train: [0 : N_train]
    Val:   [N_train : N_train + N_val]
    Test:  [N_train + N_val : N]
    """
    total_len = len(features)
    assert total_len == len(labels), "Features and labels must have the same length"
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    splits_feat = {
        "train": features[:train_end],
        "val": features[train_end:val_end],
        "test": features[val_end:]
    }
    
    splits_labels = {
        "train": labels[:train_end],
        "val": labels[train_end:val_end],
        "test": labels[val_end:]
    }
    
    return splits_feat, splits_labels


def build_dataloaders(
    features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, np.ndarray],
    seq_len: int = 100,
    batch_size: int = 128,
    add_channel_dim: bool = True,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """Builds PyTorch DataLoaders for train, val, and test partitions."""
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = LOBSequenceDataset(
            features=features_dict[split],
            labels=labels_dict[split],
            seq_len=seq_len,
            add_channel_dim=add_channel_dim
        )
        # Never shuffle validation and test sets; shuffle train sequences
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    return loaders
