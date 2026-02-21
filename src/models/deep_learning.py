# -*- coding: utf-8 -*-
"""
Exploratory LSTM — Phase 4.2: Deep Learning (pedagogical).
============================================================

METHODOLOGICAL WARNING:
    This model is included for PEDAGOGICAL and exploratory purposes.
    With ~288 training rows (36 months x 8 departments),
    an LSTM is NOT suitable and is highly likely to overfit.
    Classical models (Ridge, LightGBM) are expected to perform
    better on this data volume.

Architecture:
    - 1-layer LSTM (32 units), dropout 0.3
    - Dense output layer (1 neuron, regression)
    - Sequences of length 3 months (minimal lookback)
    - Loss = HuberLoss (robust to outliers, delta=1.0)
    - Optimizer = Adam (lr=0.001)

Dependencies:
    pip install -r requirements-dl.txt  # torch>=2.1.0

Usage:
    >>> from src.models.deep_learning import LSTMModel
    >>> lstm = LSTMModel(config, target="nb_installations_pac")
    >>> result = lstm.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class LSTMModel:
    """Minimalist LSTM model for HVAC time series.

    Deliberately simple architecture to avoid overfitting
    on a small dataset.

    Attributes:
        config: Project configuration.
        target: Target variable.
        lookback: Number of input time steps (months).
        hidden_size: Number of LSTM units.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        batch_size: Batch size.
    """

    def __init__(
        self,
        config: ProjectConfig,
        target: str = "nb_installations_pac",
        lookback: int = 3,
        hidden_size: int = 32,
        epochs: int = 100,
        patience: int = 15,
        batch_size: int = 16,
    ) -> None:
        self.config = config
        self.target = target
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.logger = logging.getLogger("models.lstm")

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create temporal sequences for the LSTM.

        Transforms tabular data into sequences of length
        `lookback` for LSTM input.

        Args:
            X: Feature array (n_samples, n_features).
            y: Target array (n_samples,).

        Returns:
            Tuple (X_seq, y_seq):
                X_seq shape = (n_sequences, lookback, n_features)
                y_seq shape = (n_sequences,)
        """
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i - self.lookback : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Train the LSTM and evaluate on val and test.

        Args:
            X_train: Scaled training features.
            y_train: Training target.
            X_val: Scaled validation features.
            y_val: Validation target.
            X_test: Scaled test features.
            y_test: Test target.

        Returns:
            Dictionary with metrics and predictions.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from src.models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(self.config)

        self.logger.info("  LSTM : lookback=%d, hidden=%d, epochs=%d",
                         self.lookback, self.hidden_size, self.epochs)

        # Impute remaining NaN
        X_train_np = X_train.fillna(0).values.astype(np.float32)
        X_val_np = X_val.fillna(0).values.astype(np.float32)
        X_test_np = X_test.fillna(0).values.astype(np.float32)
        y_train_np = y_train.fillna(0).values.astype(np.float32)
        y_val_np = y_val.fillna(0).values.astype(np.float32)
        y_test_np = y_test.fillna(0).values.astype(np.float32)

        # Create sequences
        # For val and test, concatenate with the end of the previous set
        X_all = np.vstack([X_train_np, X_val_np, X_test_np])
        y_all = np.concatenate([y_train_np, y_val_np, y_test_np])

        n_train = len(X_train_np)
        n_val = len(X_val_np)

        X_seq_train, y_seq_train = self._create_sequences(
            X_train_np, y_train_np,
        )

        # For val: use the end of train as context
        X_for_val = np.vstack([X_train_np[-(self.lookback):], X_val_np])
        y_for_val = np.concatenate([y_train_np[-(self.lookback):], y_val_np])
        X_seq_val, y_seq_val = self._create_sequences(X_for_val, y_for_val)

        # For test: use the end of val as context
        X_for_test = np.vstack([X_val_np[-(self.lookback):], X_test_np])
        y_for_test = np.concatenate([y_val_np[-(self.lookback):], y_test_np])
        X_seq_test, y_seq_test = self._create_sequences(X_for_test, y_for_test)

        if len(X_seq_train) < 10:
            self.logger.warning(
                "  LSTM: too few sequences (%d). Results unreliable.",
                len(X_seq_train),
            )

        self.logger.info(
            "  Sequences : train=%d, val=%d, test=%d",
            len(X_seq_train), len(X_seq_val), len(X_seq_test),
        )

        n_features = X_seq_train.shape[2]

        # PyTorch tensors
        device = torch.device("cpu")

        train_ds = TensorDataset(
            torch.FloatTensor(X_seq_train),
            torch.FloatTensor(y_seq_train),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=False,
        )

        # Define the model
        model = _LSTMNet(n_features, self.hidden_size).to(device)
        # HuberLoss: combines MSE (small errors) and MAE (large errors)
        # Robust to outliers unlike MSE which squares them
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            epoch_loss /= len(train_ds)

            # Validation
            model.eval()
            with torch.no_grad():
                val_tensor = torch.FloatTensor(X_seq_val).to(device)
                val_pred = model(val_tensor).squeeze()
                val_loss = criterion(
                    val_pred, torch.FloatTensor(y_seq_val).to(device)
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                self.logger.info(
                    "  Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                    epoch + 1, self.epochs, epoch_loss, val_loss,
                )

            if patience_counter >= self.patience:
                self.logger.info(
                    "  Early stopping a l'epoch %d (patience=%d)",
                    epoch + 1, self.patience,
                )
                break

        # Load best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final predictions
        model.eval()
        with torch.no_grad():
            y_pred_val = model(
                torch.FloatTensor(X_seq_val).to(device)
            ).squeeze().numpy()
            y_pred_test = model(
                torch.FloatTensor(X_seq_test).to(device)
            ).squeeze().numpy()

        y_pred_val = np.clip(y_pred_val, 0, None)
        y_pred_test = np.clip(y_pred_test, 0, None)

        # Metrics
        metrics_val = evaluator.compute_metrics(y_seq_val, y_pred_val)
        metrics_test = evaluator.compute_metrics(y_seq_test, y_pred_test)

        self.logger.info(
            "  LSTM — Val RMSE=%.2f, Test RMSE=%.2f, R2_val=%.3f",
            metrics_val["rmse"], metrics_test["rmse"], metrics_val["r2"],
        )

        return {
            "model": model,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "predictions_val": y_pred_val,
            "predictions_test": y_pred_test,
            "lookback": self.lookback,
            "best_epoch": epoch + 1 - patience_counter,
        }


class _LSTMNet:
    """Minimalist LSTM network via PyTorch.

    Architecture:
        Input (lookback, n_features)
        -> LSTM (hidden_size=32, 1 layer)
        -> Dropout (0.3)
        -> Linear (hidden_size -> 1)

    Note: internal class, do not use directly.
    """

    def __new__(cls, n_features: int, hidden_size: int = 32):
        """Create the PyTorch module dynamically.

        This avoids importing torch at the module level,
        allowing the rest of the code to work without PyTorch.
        """
        import torch
        import torch.nn as nn

        class LSTMNet(nn.Module):
            def __init__(self, n_features: int, hidden_size: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                )
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take the last output of the sequence
                last_hidden = lstm_out[:, -1, :]
                out = self.dropout(last_hidden)
                out = self.fc(out)
                return out

        return LSTMNet(n_features, hidden_size)
