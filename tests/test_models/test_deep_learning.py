# -*- coding: utf-8 -*-
"""
Tests for the deep learning module (LSTMModel and _LSTMNet).

Tests sequence creation, model architecture (layer shapes),
training loop with tiny data, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.deep_learning import LSTMModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """Create a mock ProjectConfig for LSTMModel."""
    config = MagicMock()
    config.features_data_dir = "/tmp/test"
    return config


@pytest.fixture
def lstm_model(mock_config):
    """Create an LSTMModel with small parameters for testing."""
    return LSTMModel(
        config=mock_config,
        target="nb_installations_pac",
        lookback=2,
        hidden_size=8,
        epochs=5,
        patience=3,
        batch_size=4,
    )


@pytest.fixture
def tiny_data():
    """Create tiny datasets for LSTM testing (20 samples x 4 features)."""
    np.random.seed(42)
    n_train, n_val, n_test = 20, 8, 8
    n_features = 4

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_train = pd.Series(np.random.uniform(10, 80, n_train).astype(np.float32))

    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_val = pd.Series(np.random.uniform(10, 80, n_val).astype(np.float32))

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features).astype(np.float32),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_test = pd.Series(np.random.uniform(10, 80, n_test).astype(np.float32))

    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------

class TestCreateSequences:
    """Tests for LSTMModel._create_sequences()."""

    def test_sequence_shapes(self, lstm_model):
        """_create_sequences() produces correct output shapes."""
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randn(10).astype(np.float32)

        X_seq, y_seq = lstm_model._create_sequences(X, y)

        # lookback=2, so we get 10 - 2 = 8 sequences
        assert X_seq.shape == (8, 2, 4)
        assert y_seq.shape == (8,)

    def test_sequence_values_correct(self, lstm_model):
        """_create_sequences() creates correct sliding windows."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=np.float32)
        y = np.array([10, 20, 30, 40, 50], dtype=np.float32)

        X_seq, y_seq = lstm_model._create_sequences(X, y)

        # First sequence: X[0:2] -> y[2]
        np.testing.assert_array_equal(X_seq[0], [[1, 2], [3, 4]])
        assert y_seq[0] == 30

        # Second sequence: X[1:3] -> y[3]
        np.testing.assert_array_equal(X_seq[1], [[3, 4], [5, 6]])
        assert y_seq[1] == 40

    def test_sequence_with_lookback_1(self, mock_config):
        """Sequences with lookback=1 use single time step."""
        model = LSTMModel(mock_config, lookback=1, hidden_size=8)
        X = np.array([[1], [2], [3]], dtype=np.float32)
        y = np.array([10, 20, 30], dtype=np.float32)

        X_seq, y_seq = model._create_sequences(X, y)

        assert X_seq.shape == (2, 1, 1)
        assert y_seq.shape == (2,)

    def test_sequence_empty_when_too_few_samples(self, mock_config):
        """Returns empty arrays when data shorter than lookback."""
        model = LSTMModel(mock_config, lookback=5, hidden_size=8)
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([10, 20], dtype=np.float32)

        X_seq, y_seq = model._create_sequences(X, y)

        assert len(X_seq) == 0
        assert len(y_seq) == 0

    def test_sequence_lookback_equals_data_length(self, mock_config):
        """Edge case: lookback equals data length gives 0 sequences."""
        model = LSTMModel(mock_config, lookback=3, hidden_size=8)
        X = np.array([[1], [2], [3]], dtype=np.float32)
        y = np.array([10, 20, 30], dtype=np.float32)

        X_seq, y_seq = model._create_sequences(X, y)
        assert len(X_seq) == 0


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TestModelArchitecture:
    """Tests for the _LSTMNet architecture."""

    def test_lstm_net_creation(self):
        """_LSTMNet creates a valid PyTorch module."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=4, hidden_size=16)

        # Should be a PyTorch Module
        assert isinstance(net, torch.nn.Module)

    def test_lstm_net_has_expected_layers(self):
        """_LSTMNet contains LSTM, Dropout, and Linear layers."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=4, hidden_size=16)

        assert hasattr(net, "lstm")
        assert hasattr(net, "dropout")
        assert hasattr(net, "fc")

    def test_lstm_input_size(self):
        """LSTM layer has correct input_size."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=8, hidden_size=32)

        assert net.lstm.input_size == 8
        assert net.lstm.hidden_size == 32

    def test_lstm_output_size(self):
        """Linear layer outputs a single value (regression)."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=4, hidden_size=16)

        assert net.fc.out_features == 1

    def test_forward_pass_shape(self):
        """Forward pass produces correct output shape."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=4, hidden_size=16)

        # batch_size=3, lookback=5, n_features=4
        x = torch.randn(3, 5, 4)
        output = net(x)

        assert output.shape == (3, 1)

    def test_forward_pass_single_sample(self):
        """Forward pass works with single sample."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=2, hidden_size=8)

        x = torch.randn(1, 3, 2)
        output = net(x)

        assert output.shape == (1, 1)

    def test_dropout_rate(self):
        """Dropout rate is 0.3 as specified."""
        torch = pytest.importorskip("torch")
        from src.models.deep_learning import _LSTMNet

        net = _LSTMNet(n_features=4, hidden_size=16)

        assert net.dropout.p == 0.3


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

class TestTrainAndEvaluate:
    """Tests for the full train_and_evaluate() method."""

    def test_train_returns_results_dict(self, lstm_model, tiny_data):
        """train_and_evaluate() returns a results dictionary."""
        torch = pytest.importorskip("torch")
        X_train, y_train, X_val, y_val, X_test, y_test = tiny_data

        with patch("src.models.deep_learning.ModelEvaluator") as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.compute_metrics.return_value = {
                "rmse": 10.0, "mae": 8.0, "r2": 0.5, "mape": 15.0,
            }

            result = lstm_model.train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test,
            )

        assert "model" in result
        assert "metrics_val" in result
        assert "metrics_test" in result
        assert "predictions_val" in result
        assert "predictions_test" in result
        assert "lookback" in result
        assert "best_epoch" in result

    def test_predictions_are_non_negative(self, lstm_model, tiny_data):
        """Predictions are clipped to be non-negative."""
        torch = pytest.importorskip("torch")
        X_train, y_train, X_val, y_val, X_test, y_test = tiny_data

        with patch("src.models.deep_learning.ModelEvaluator") as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.compute_metrics.return_value = {
                "rmse": 10.0, "mae": 8.0, "r2": 0.5, "mape": 15.0,
            }

            result = lstm_model.train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test,
            )

        assert (result["predictions_val"] >= 0).all()
        assert (result["predictions_test"] >= 0).all()

    def test_train_handles_nan_in_data(self, lstm_model):
        """train_and_evaluate() fills NaN values with 0."""
        torch = pytest.importorskip("torch")
        np.random.seed(42)

        n_train, n_val, n_test = 20, 8, 8
        n_features = 4

        X_train = pd.DataFrame(np.random.randn(n_train, n_features))
        X_train.iloc[0, 0] = np.nan  # Inject NaN
        y_train = pd.Series(np.random.uniform(10, 80, n_train))

        X_val = pd.DataFrame(np.random.randn(n_val, n_features))
        y_val = pd.Series(np.random.uniform(10, 80, n_val))

        X_test = pd.DataFrame(np.random.randn(n_test, n_features))
        y_test = pd.Series(np.random.uniform(10, 80, n_test))

        with patch("src.models.deep_learning.ModelEvaluator") as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.compute_metrics.return_value = {
                "rmse": 10.0, "mae": 8.0, "r2": 0.5, "mape": 15.0,
            }

            # Should not raise
            result = lstm_model.train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test,
            )

        assert result is not None

    def test_early_stopping(self, mock_config, tiny_data):
        """Early stopping triggers before max epochs."""
        torch = pytest.importorskip("torch")
        X_train, y_train, X_val, y_val, X_test, y_test = tiny_data

        model = LSTMModel(
            config=mock_config,
            lookback=2,
            hidden_size=8,
            epochs=200,
            patience=2,  # Very low patience
            batch_size=4,
        )

        with patch("src.models.deep_learning.ModelEvaluator") as MockEval:
            mock_evaluator = MockEval.return_value
            mock_evaluator.compute_metrics.return_value = {
                "rmse": 10.0, "mae": 8.0, "r2": 0.5, "mape": 15.0,
            }

            result = model.train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test,
            )

        # Should stop well before 200 epochs
        assert result["best_epoch"] < 200


# ---------------------------------------------------------------------------
# LSTMModel initialization
# ---------------------------------------------------------------------------

class TestLSTMModelInit:
    """Tests for LSTMModel initialization parameters."""

    def test_default_parameters(self, mock_config):
        """Default parameters are set correctly."""
        model = LSTMModel(mock_config)

        assert model.target == "nb_installations_pac"
        assert model.lookback == 3
        assert model.hidden_size == 32
        assert model.epochs == 100
        assert model.patience == 15
        assert model.batch_size == 16

    def test_custom_parameters(self, mock_config):
        """Custom parameters override defaults."""
        model = LSTMModel(
            mock_config,
            target="nb_dpe_total",
            lookback=6,
            hidden_size=64,
            epochs=50,
            patience=10,
            batch_size=32,
        )

        assert model.target == "nb_dpe_total"
        assert model.lookback == 6
        assert model.hidden_size == 64
        assert model.epochs == 50
        assert model.patience == 10
        assert model.batch_size == 32
