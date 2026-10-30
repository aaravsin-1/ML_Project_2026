"""PyTorch Dataset and Chronological DataLoader utilities for Limit Order Book Sequences."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, Any


class LOBSequenceDataset(Dataset):
    """Sequence dataset for DeepLOB and TLOB architectures.
    
    Extracts sliding temporal windows of length T (e.g. 100 timesteps) across F features.
    Shape returned: (1, T, F) for DeepLOB (2D conv input) or (T, F) for sequence transformers.
    If `target_indices` is given, sample i is the window ending at row target_indices[i]
    (rows before it supply causal context, even if they belong to an earlier split).
    This lets sequence models be scored on exactly the same rows as tabular models.
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 100,
        add_channel_dim: bool = True,
        target_indices: Optional[np.ndarray] = None
    ):
        assert len(features) == len(labels), f"Length mismatch: {len(features)} vs {len(labels)}"
        assert len(features) > seq_len, f"Data length {len(features)} must be greater than seq_len {seq_len}"
        
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len
        self.add_channel_dim = add_channel_dim
        if target_indices is None:
            target_indices = np.arange(seq_len - 1, len(features))
        assert np.min(target_indices) >= seq_len - 1, "Every target row needs seq_len - 1 rows of history"
        self.target_indices = torch.as_tensor(np.asarray(target_indices), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end = int(self.target_indices[idx])
        # Window: rows (end - seq_len, end], label belongs to the last row of the window
        x = self.features[end - self.seq_len + 1 : end + 1]
        y = self.labels[end]
        
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


def purged_split_indices(
    n_samples: int,
    horizon_k: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    min_start: int = 0
) -> Dict[str, np.ndarray]:
    """Chronological train/val/test row indices with a purge gap of `horizon_k` rows.

    The label at row t looks k events ahead, so the last k rows of each split have
    labels that overlap the next split. Those rows are dropped, as are the final k
    rows of the series whose labels are undefined (the generator marks them FLAT).
    `min_start` lets callers reserve warm-up history (e.g. for sequence windows).
    """
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    return {
        "train": np.arange(min_start, train_end - horizon_k),
        "val": np.arange(train_end, val_end - horizon_k),
        "test": np.arange(val_end, n_samples - horizon_k),
    }


def build_indexed_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    split_indices: Dict[str, np.ndarray],
    seq_len: int = 100,
    batch_size: int = 128,
    add_channel_dim: bool = True
) -> Dict[str, DataLoader]:
    """DataLoaders whose samples end exactly at the given split rows."""
    loaders = {}
    for split, idx in split_indices.items():
        dataset = LOBSequenceDataset(
            features, labels, seq_len=seq_len, add_channel_dim=add_channel_dim, target_indices=idx
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=torch.cuda.is_available()
        )
    return loaders


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
