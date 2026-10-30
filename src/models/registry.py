"""Unified Model Registry and Training / Inference Engine."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix

from src.models.baselines import NaiveMomentumBaseline, NaiveFlatBaseline, LogisticRegressionBaseline
from src.models.tabular_trees import RandomForestLOB, LightGBMLOB
from src.models.recurrent import LSTMLOBNet, GRULOBNet
from src.models.deeplob import DeepLOB
from src.models.tlob import TLOB


class ModelRegistry:
    """Factory and manager for all limit order book model architectures."""

    @staticmethod
    def create_model(model_name: str, **kwargs) -> Any:
        model_name = model_name.lower().replace("-", "_")
        if model_name in ["naive_momentum", "momentum"]:
            return NaiveMomentumBaseline(**kwargs)
        elif model_name in ["naive_flat", "flat"]:
            return NaiveFlatBaseline()
        elif model_name in ["logistic", "ridge", "linear"]:
            return LogisticRegressionBaseline(**kwargs)
        elif model_name in ["rf", "random_forest"]:
            return RandomForestLOB(**kwargs)
        elif model_name in ["lgb", "lightgbm"]:
            return LightGBMLOB(**kwargs)
        elif model_name == "lstm":
            return LSTMLOBNet(**kwargs)
        elif model_name == "gru":
            return GRULOBNet(**kwargs)
        elif model_name == "deeplob":
            return DeepLOB(**kwargs)
        elif model_name == "tlob":
            return TLOB(**kwargs)
        else:
            raise ValueError(f"Unknown model architecture: {model_name}")


def balanced_class_weights(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """sklearn-style 'balanced' weights: n_samples / (n_classes * count_c)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    return len(y) / (num_classes * np.maximum(counts, 1.0))


class DeepLearningTrainer:
    """Trainer for PyTorch sequential architectures (LSTM, GRU, DeepLOB, TLOB)."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.0005,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
        class_weights: Optional[np.ndarray] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        # Same "balanced" class weighting as the tree and linear models, so every tier
        # optimizes the same objective and macro-F1 comparisons are like for like.
        weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """Evaluates model on dataloader.
        
        Returns:
            loss, accuracy, macro_f1, all_preds, all_probs
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                total_loss += loss.item()
                n_batches += 1
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)

        avg_loss = total_loss / max(1, n_batches)
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        return avg_loss, float(acc), float(macro_f1), all_preds, all_probs

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 4
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, _, _ = self.evaluate(val_loader)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

        return history
