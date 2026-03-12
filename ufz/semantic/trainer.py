"""SemanticTrainer: Orchestrates Imputer/RefineNet training."""

import logging
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticTrainer:
    """Train Imputer or RefineNet models with early stopping."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward: assumes batch has x, edge_index, y, mask
            # For Imputer: batch = (x, edge_index, y, mask)
            # For RefineNet: batch = (x_phys, x_sem, edge_index, y, mask)
            if len(batch) == 4:
                x, edge_index, y, mask = batch
                pred = self.model(x.to(self.device), edge_index.to(self.device))
            else:
                x_phys, x_sem, edge_index, y, mask = batch
                pred = self.model(
                    x_phys.to(self.device),
                    x_sem.to(self.device),
                    edge_index.to(self.device),
                )

            y = y.to(self.device)
            mask = mask.to(self.device)

            pred_masked = pred[mask]
            y_masked = y[mask]

            loss = self.loss_fn(pred_masked, y_masked)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """Validation epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            if len(batch) == 4:
                x, edge_index, y, mask = batch
                pred = self.model(x.to(self.device), edge_index.to(self.device))
            else:
                x_phys, x_sem, edge_index, y, mask = batch
                pred = self.model(
                    x_phys.to(self.device),
                    x_sem.to(self.device),
                    edge_index.to(self.device),
                )

            y = y.to(self.device)
            mask = mask.to(self.device)

            pred_masked = pred[mask]
            y_masked = y[mask]

            loss = self.loss_fn(pred_masked, y_masked)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        patience: int = 30,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, list]:
        """Train with early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if checkpoint_dir:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        Path(checkpoint_dir) / "best_model.pt",
                    )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history

    def predict(self, loader: DataLoader) -> torch.Tensor:
        """Predict on full dataset."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 4:
                    x, edge_index, _, _ = batch
                    pred = self.model(x.to(self.device), edge_index.to(self.device))
                else:
                    x_phys, x_sem, edge_index, _, _ = batch
                    pred = self.model(
                        x_phys.to(self.device),
                        x_sem.to(self.device),
                        edge_index.to(self.device),
                    )
                predictions.append(pred.cpu())

        return torch.cat(predictions, dim=0)
