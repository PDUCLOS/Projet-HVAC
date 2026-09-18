# -*- coding: utf-8 -*-
"""
Tests for HyperparameterTuner — Optuna-based hyperparameter optimization.
===========================================================================

Tests cover:
- Ridge tuning with Optuna (or fallback if not installed)
- LightGBM tuning
- Result structure validation
- Best params are valid and usable
- Small dataset handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.hyperparameter_tuning import HyperparameterTuner


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tuner() -> HyperparameterTuner:
    """Create a HyperparameterTuner with default settings."""
    return HyperparameterTuner(cv_splits=2)


@pytest.fixture
def small_dataset():
    """Create a small train/val dataset for tuning tests.

    Returns:
        Tuple (X_train, y_train, X_val, y_val).
    """
    np.random.seed(42)
    n_train, n_val = 80, 20
    n_features = 5

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    base = np.linspace(0, 10, n_train)
    y_train = pd.Series(
        base * 3 + X_train["feature_0"] * 2 + np.random.randn(n_train),
        name="target",
    )

    X_val = pd.DataFrame(
        np.random.randn(n_val, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    base_val = np.linspace(10, 12.5, n_val)
    y_val = pd.Series(
        base_val * 3 + X_val["feature_0"] * 2 + np.random.randn(n_val),
        name="target",
    )

    return X_train, y_train, X_val, y_val


@pytest.fixture
def tiny_dataset():
    """Create a very small dataset (edge case for CV).

    Returns:
        Tuple (X_train, y_train, X_val, y_val).
    """
    np.random.seed(42)
    X_train = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    y_train = pd.Series([3.0, 6.0, 9.0, 12.0, 15.0])

    X_val = pd.DataFrame({
        "a": [6.0, 7.0],
        "b": [12.0, 14.0],
    })
    y_val = pd.Series([18.0, 21.0])

    return X_train, y_train, X_val, y_val


# =============================================================================
# Tests: tune_ridge
# =============================================================================


class TestTuneRidge:
    """Tests for Ridge hyperparameter tuning."""

    def test_returns_best_params(self, tuner: HyperparameterTuner, small_dataset):
        """Tuning returns a best_params dict with alpha."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=5)

        assert "best_params" in result
        assert "alpha" in result["best_params"]
        assert result["best_params"]["alpha"] > 0

    def test_returns_best_score(self, tuner: HyperparameterTuner, small_dataset):
        """Tuning returns a non-negative best_score (RMSE)."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=5)

        assert "best_score" in result
        assert result["best_score"] >= 0

    def test_returns_val_rmse(self, tuner: HyperparameterTuner, small_dataset):
        """Tuning returns val_rmse computed on validation set."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=5)

        assert "val_rmse" in result
        assert result["val_rmse"] >= 0

    def test_best_params_usable(self, tuner: HyperparameterTuner, small_dataset):
        """Best params can be used to train a Ridge model."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=5)

        model = Ridge(**result["best_params"])
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert len(preds) == len(y_val)

    def test_n_trials_recorded(self, tuner: HyperparameterTuner, small_dataset):
        """Number of completed trials is recorded."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=5)

        assert "n_trials" in result
        assert result["n_trials"] >= 1

    def test_tiny_dataset_works(self, tuner: HyperparameterTuner, tiny_dataset):
        """Tuning works with very small datasets (falls back gracefully)."""
        X_train, y_train, X_val, y_val = tiny_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=3)

        assert "best_params" in result
        assert "alpha" in result["best_params"]

    def test_study_results_included(self, tuner: HyperparameterTuner, small_dataset):
        """Study results (trial history) are included."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=5)

        assert "study_results" in result
        assert isinstance(result["study_results"], list)


# =============================================================================
# Tests: tune_lightgbm
# =============================================================================


class TestTuneLightGBM:
    """Tests for LightGBM hyperparameter tuning."""

    def test_returns_best_params(self, tuner: HyperparameterTuner, small_dataset):
        """Tuning returns best_params with expected hyperparameters."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=3)

        assert "best_params" in result
        params = result["best_params"]
        # Check key hyperparameters are present
        expected_keys = {
            "max_depth", "num_leaves", "learning_rate",
            "n_estimators", "min_child_samples",
            "reg_alpha", "reg_lambda",
        }
        for key in expected_keys:
            assert key in params, f"'{key}' missing from best_params"

    def test_returns_best_score(self, tuner: HyperparameterTuner, small_dataset):
        """Tuning returns a non-negative best_score."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=3)

        assert "best_score" in result
        assert result["best_score"] >= 0

    def test_returns_val_rmse(self, tuner: HyperparameterTuner, small_dataset):
        """Tuning returns val_rmse from validation set evaluation."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=3)

        assert "val_rmse" in result
        assert result["val_rmse"] >= 0

    def test_best_params_usable(self, tuner: HyperparameterTuner, small_dataset):
        """Best params can be used to create a LightGBM model."""
        import lightgbm as lgb

        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=3)

        params = dict(result["best_params"])
        params["verbose"] = -1
        params["random_state"] = 42

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        assert len(preds) == len(y_val)

    def test_hyperparams_in_valid_ranges(
        self, tuner: HyperparameterTuner, small_dataset
    ):
        """Best hyperparameters are within the defined search space."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=3)

        params = result["best_params"]
        assert 2 <= params["max_depth"] <= 8
        assert 7 <= params["num_leaves"] <= 127
        assert 0.01 <= params["learning_rate"] <= 0.3
        assert 50 <= params["n_estimators"] <= 500
        assert 5 <= params["min_child_samples"] <= 50

    def test_n_trials_recorded(self, tuner: HyperparameterTuner, small_dataset):
        """Number of completed trials is recorded."""
        X_train, y_train, X_val, y_val = small_dataset
        result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=3)

        assert "n_trials" in result
        assert result["n_trials"] >= 1


# =============================================================================
# Tests: Edge cases and integration
# =============================================================================


class TestTunerEdgeCases:
    """Edge cases for the hyperparameter tuner."""

    def test_optuna_check(self, tuner: HyperparameterTuner):
        """Optuna check returns a boolean."""
        result = tuner._check_optuna()
        assert isinstance(result, bool)

    def test_cv_splits_stored(self):
        """Custom cv_splits value is stored."""
        tuner = HyperparameterTuner(cv_splits=5)
        assert tuner.cv_splits == 5

    def test_default_cv_splits(self):
        """Default cv_splits is 3."""
        tuner = HyperparameterTuner()
        assert tuner.cv_splits == 3
