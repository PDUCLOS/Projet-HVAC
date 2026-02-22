# -*- coding: utf-8 -*-
"""
Comprehensive tests for ModelTrainer (train.py) and ModelEvaluator (evaluate.py).
==================================================================================

Tests cover:
- ModelTrainer: initialization, temporal split, feature preparation,
  auto-regressive feature exclusion, feature name retrieval.
- ModelEvaluator: metric computation (RMSE, MAE, MAPE, R2),
  NaN handling, zero-value handling in MAPE, model comparison.

No actual model training (Ridge, LightGBM, Prophet) is performed here.
Focus is on data preparation logic and evaluation utilities.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    GeoConfig,
    ModelConfig,
    ProjectConfig,
    TimeConfig,
)
from src.models.evaluate import ModelEvaluator
from src.models.train import ModelTrainer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tmp_config(tmp_path: Path) -> ProjectConfig:
    """Create a ProjectConfig with paths pointing to tmp_path.

    The TimeConfig uses the default split dates:
        train_end  = 2024-06-30
        val_end    = 2024-12-31
    which translate to:
        train_end  = 202406  (date_id <= 202406 -> train)
        val_end    = 202412  (202406 < date_id <= 202412 -> val)
        test       = date_id > 202412
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    return ProjectConfig(
        time=TimeConfig(
            train_end="2024-06-30",
            val_end="2024-12-31",
        ),
        features_data_dir=features_dir,
        models_dir=models_dir,
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a synthetic DataFrame that mimics the features dataset.

    Contains:
    - Identifiers: date_id, dept, dept_name, city_ref, latitude, longitude
    - Metadata: n_valid_features, pct_valid_features
    - Targets: nb_installations_pac, nb_installations_clim, nb_dpe_total, nb_dpe_classe_ab
    - Numeric features: temp_mean, precip_sum, hdd_18
    - Auto-regressive features of the target: nb_installations_pac_lag_1,
      nb_installations_pac_rmean_3, nb_installations_pac_diff_1
    - Outlier columns: temp_mean_outlier_iqr, temp_mean_outlier_zscore
    - A non-numeric column: category_label

    date_id spans from 202107 to 202506 across 2 departments,
    giving enough data for train/val/test splits.
    """
    np.random.seed(42)

    # Generate monthly date_ids from 2021-07 to 2025-06
    months = pd.date_range("2021-07-01", "2025-06-01", freq="MS")
    date_ids = [int(d.strftime("%Y%m")) for d in months]
    depts = ["69", "38"]

    rows = []
    for date_id in date_ids:
        for dept in depts:
            rows.append({
                # Identifiers
                "date_id": date_id,
                "dept": dept,
                "dept_name": f"Dept-{dept}",
                "city_ref": f"City-{dept}",
                "latitude": 45.0 + np.random.randn() * 0.1,
                "longitude": 5.0 + np.random.randn() * 0.1,
                # Metadata
                "n_valid_features": 50,
                "pct_valid_features": 0.95,
                # Target variables
                "nb_installations_pac": max(10, int(100 + np.random.randn() * 20)),
                "nb_installations_clim": max(5, int(50 + np.random.randn() * 10)),
                "nb_dpe_total": max(20, int(200 + np.random.randn() * 30)),
                "nb_dpe_classe_ab": max(3, int(30 + np.random.randn() * 5)),
                # Numeric features (exogenous)
                "temp_mean": 15.0 + np.random.randn() * 5,
                "precip_sum": max(0, 50 + np.random.randn() * 20),
                "hdd_18": max(0, 200 + np.random.randn() * 100),
                "pop_total": 500000 + np.random.randint(-10000, 10000),
                "month_sin": np.sin(2 * np.pi * (date_id % 100) / 12),
                "month_cos": np.cos(2 * np.pi * (date_id % 100) / 12),
                # Auto-regressive features of the target
                "nb_installations_pac_lag_1": max(0, int(95 + np.random.randn() * 15)),
                "nb_installations_pac_lag_3": max(0, int(90 + np.random.randn() * 15)),
                "nb_installations_pac_rmean_3": 98.0 + np.random.randn() * 10,
                "nb_installations_pac_rstd_3": 5.0 + abs(np.random.randn()),
                "nb_installations_pac_diff_1": np.random.randn() * 10,
                "nb_installations_pac_pct_1": np.random.randn() * 0.05,
                # Auto-regressive features of another target (should NOT be excluded)
                "nb_installations_clim_lag_1": max(0, int(45 + np.random.randn() * 8)),
                "nb_installations_clim_rmean_3": 48.0 + np.random.randn() * 5,
                # Outlier detection columns (should be excluded)
                "temp_mean_outlier_iqr": 0,
                "temp_mean_outlier_zscore": 0,
                "precip_sum_outlier_iforest": 0,
                "hdd_18_outlier_consensus": 0,
                "pop_total_outlier_score": 0.1,
                # Non-numeric column (should be dropped by select_dtypes)
                "category_label": "residential",
            })

    return pd.DataFrame(rows)


@pytest.fixture
def trainer(tmp_config: ProjectConfig) -> ModelTrainer:
    """Create a ModelTrainer instance with default target."""
    return ModelTrainer(tmp_config, target="nb_installations_pac")


@pytest.fixture
def trainer_exclude_lags(tmp_config: ProjectConfig) -> ModelTrainer:
    """Create a ModelTrainer instance that excludes target auto-regressive features."""
    return ModelTrainer(
        tmp_config,
        target="nb_installations_pac",
        exclude_target_lags=True,
    )


@pytest.fixture
def evaluator(tmp_config: ProjectConfig) -> ModelEvaluator:
    """Create a ModelEvaluator instance."""
    return ModelEvaluator(tmp_config)


@pytest.fixture
def model_results() -> Dict:
    """Create a mock results dictionary as returned by ModelTrainer.train_all().

    Contains two fake models (model_a and model_b) with val and test metrics.
    """
    return {
        "model_a": {
            "metrics_val": {"rmse": 15.0, "mae": 10.0, "mape": 12.5, "r2": 0.85},
            "metrics_test": {"rmse": 18.0, "mae": 12.0, "mape": 14.0, "r2": 0.80},
        },
        "model_b": {
            "metrics_val": {"rmse": 20.0, "mae": 15.0, "mape": 18.0, "r2": 0.75},
            "metrics_test": {"rmse": 22.0, "mae": 17.0, "mape": 20.0, "r2": 0.70},
        },
    }


# =============================================================================
# Tests: ModelTrainer.__init__
# =============================================================================


class TestModelTrainerInit:
    """Tests for ModelTrainer initialization."""

    def test_default_target(self, tmp_config: ProjectConfig):
        """Default target is 'nb_installations_pac'."""
        trainer = ModelTrainer(tmp_config)
        assert trainer.target == "nb_installations_pac"

    def test_custom_target(self, tmp_config: ProjectConfig):
        """A custom target variable is correctly stored."""
        trainer = ModelTrainer(tmp_config, target="nb_installations_clim")
        assert trainer.target == "nb_installations_clim"

    def test_train_end_parsed(self, trainer: ModelTrainer):
        """train_end is parsed from 'YYYY-MM-DD' to YYYYMM integer."""
        # TimeConfig.train_end = "2024-06-30" -> 202406
        assert trainer.train_end == 202406

    def test_val_end_parsed(self, trainer: ModelTrainer):
        """val_end is parsed from 'YYYY-MM-DD' to YYYYMM integer."""
        # TimeConfig.val_end = "2024-12-31" -> 202412
        assert trainer.val_end == 202412

    def test_exclude_target_lags_default_false(self, trainer: ModelTrainer):
        """exclude_target_lags defaults to False."""
        assert trainer.exclude_target_lags is False

    def test_exclude_target_lags_true(self, trainer_exclude_lags: ModelTrainer):
        """exclude_target_lags can be set to True."""
        assert trainer_exclude_lags.exclude_target_lags is True

    def test_config_stored(self, trainer: ModelTrainer, tmp_config: ProjectConfig):
        """The config object is stored on the trainer."""
        assert trainer.config is tmp_config

    def test_models_dir_created(self, trainer: ModelTrainer):
        """The models directory is created during init."""
        assert trainer.models_dir.exists()


# =============================================================================
# Tests: ModelTrainer.temporal_split
# =============================================================================


class TestTemporalSplit:
    """Tests for the temporal train/val/test split."""

    def test_split_sizes(self, trainer: ModelTrainer, sample_df: pd.DataFrame):
        """The split produces non-empty train, val, and test sets."""
        df_train, df_val, df_test = trainer.temporal_split(sample_df)
        assert len(df_train) > 0, "Train set should not be empty"
        assert len(df_val) > 0, "Val set should not be empty"
        assert len(df_test) > 0, "Test set should not be empty"

    def test_split_covers_all_rows(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """All rows are assigned to exactly one split."""
        df_train, df_val, df_test = trainer.temporal_split(sample_df)
        total = len(df_train) + len(df_val) + len(df_test)
        assert total == len(sample_df), (
            f"Split total ({total}) != original ({len(sample_df)})"
        )

    def test_train_dates_before_cutoff(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """All training dates are <= train_end (202406)."""
        df_train, _, _ = trainer.temporal_split(sample_df)
        assert df_train["date_id"].max() <= 202406

    def test_val_dates_in_range(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """All validation dates are between train_end and val_end."""
        _, df_val, _ = trainer.temporal_split(sample_df)
        assert df_val["date_id"].min() > 202406
        assert df_val["date_id"].max() <= 202412

    def test_test_dates_after_val_end(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """All test dates are after val_end (202412)."""
        _, _, df_test = trainer.temporal_split(sample_df)
        assert df_test["date_id"].min() > 202412

    def test_no_temporal_leakage(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """No date overlap between train, val, and test sets."""
        df_train, df_val, df_test = trainer.temporal_split(sample_df)

        train_dates = set(df_train["date_id"].unique())
        val_dates = set(df_val["date_id"].unique())
        test_dates = set(df_test["date_id"].unique())

        assert train_dates.isdisjoint(val_dates), "Train and val dates overlap"
        assert train_dates.isdisjoint(test_dates), "Train and test dates overlap"
        assert val_dates.isdisjoint(test_dates), "Val and test dates overlap"

    def test_chronological_order(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Train max date < val min date < test min date."""
        df_train, df_val, df_test = trainer.temporal_split(sample_df)

        assert df_train["date_id"].max() < df_val["date_id"].min()
        assert df_val["date_id"].max() < df_test["date_id"].min()

    def test_split_returns_copies(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Modifying a split does not affect the original DataFrame."""
        df_train, _, _ = trainer.temporal_split(sample_df)
        original_len = len(sample_df)
        df_train.drop(df_train.index, inplace=True)
        assert len(sample_df) == original_len


# =============================================================================
# Tests: ModelTrainer.prepare_features
# =============================================================================


class TestPrepareFeatures:
    """Tests for feature preparation (column filtering)."""

    def test_excludes_identifier_columns(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """EXCLUDE_COLS (date_id, dept, dept_name, etc.) are removed from X."""
        X, _ = trainer.prepare_features(sample_df)
        for col in ModelTrainer.EXCLUDE_COLS:
            assert col not in X.columns, f"'{col}' should be excluded"

    def test_excludes_other_target_columns(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Target columns other than the active target are excluded."""
        # Active target is nb_installations_pac; the others should be dropped
        X, _ = trainer.prepare_features(sample_df)
        other_targets = ModelTrainer.TARGET_COLS - {"nb_installations_pac"}
        for col in other_targets:
            assert col not in X.columns, f"'{col}' (other target) should be excluded"

    def test_excludes_active_target_from_X(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """The active target column itself is not in X."""
        X, _ = trainer.prepare_features(sample_df)
        assert "nb_installations_pac" not in X.columns

    def test_target_in_y(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """y contains the active target variable."""
        _, y = trainer.prepare_features(sample_df)
        assert y.name == "nb_installations_pac"
        assert len(y) == len(sample_df)

    def test_excludes_outlier_columns(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Columns matching OUTLIER_PATTERNS are excluded."""
        X, _ = trainer.prepare_features(sample_df)
        outlier_cols = [
            c for c in X.columns
            if any(p in c for p in ModelTrainer.OUTLIER_PATTERNS)
        ]
        assert len(outlier_cols) == 0, (
            f"Outlier columns should be excluded, found: {outlier_cols}"
        )

    def test_keeps_exogenous_features(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Exogenous numeric features like temp_mean, precip_sum are kept."""
        X, _ = trainer.prepare_features(sample_df)
        expected = ["temp_mean", "precip_sum", "hdd_18", "pop_total",
                     "month_sin", "month_cos"]
        for col in expected:
            assert col in X.columns, f"'{col}' should be kept as a feature"

    def test_keeps_target_lag_features_by_default(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """By default, auto-regressive target features (lags, rolling) are kept."""
        X, _ = trainer.prepare_features(sample_df)
        lag_cols = [
            c for c in X.columns
            if c.startswith("nb_installations_pac")
            and any(p in c for p in ModelTrainer.TARGET_LAG_PATTERNS)
        ]
        assert len(lag_cols) > 0, "Target lag features should be kept by default"

    def test_excludes_non_numeric_columns(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Non-numeric columns are excluded by select_dtypes."""
        X, _ = trainer.prepare_features(sample_df)
        assert "category_label" not in X.columns

    def test_x_y_same_length(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """X and y have the same number of rows."""
        X, y = trainer.prepare_features(sample_df)
        assert len(X) == len(y)

    def test_all_x_columns_are_numeric(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """All columns in X are numeric dtype."""
        X, _ = trainer.prepare_features(sample_df)
        for col in X.columns:
            assert np.issubdtype(X[col].dtype, np.number), (
                f"Column '{col}' is not numeric: {X[col].dtype}"
            )


# =============================================================================
# Tests: ModelTrainer.prepare_features with exclude_target_lags=True
# =============================================================================


class TestPrepareExcludeTargetLags:
    """Tests for prepare_features when exclude_target_lags=True."""

    def test_excludes_target_lag_features(
        self, trainer_exclude_lags: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Auto-regressive features of the active target are excluded."""
        X, _ = trainer_exclude_lags.prepare_features(sample_df)

        target_lag_cols = [
            c for c in X.columns
            if c.startswith("nb_installations_pac")
            and any(p in c for p in ModelTrainer.TARGET_LAG_PATTERNS)
        ]
        assert len(target_lag_cols) == 0, (
            f"Target lag features should be excluded, found: {target_lag_cols}"
        )

    def test_keeps_other_target_lag_features(
        self, trainer_exclude_lags: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Auto-regressive features of OTHER targets are kept.

        nb_installations_clim_lag_1 should remain because the active target
        is nb_installations_pac, not nb_installations_clim.
        """
        X, _ = trainer_exclude_lags.prepare_features(sample_df)

        # nb_installations_clim is in TARGET_COLS so it is dropped as a target col,
        # but its lag features (nb_installations_clim_lag_1) should remain
        other_lag_cols = [
            c for c in X.columns
            if c.startswith("nb_installations_clim")
            and any(p in c for p in ModelTrainer.TARGET_LAG_PATTERNS)
        ]
        assert len(other_lag_cols) > 0, (
            "Lag features of non-active targets should be kept"
        )

    def test_keeps_exogenous_features(
        self, trainer_exclude_lags: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Exogenous features are still present after excluding target lags."""
        X, _ = trainer_exclude_lags.prepare_features(sample_df)
        expected = ["temp_mean", "precip_sum", "hdd_18", "pop_total"]
        for col in expected:
            assert col in X.columns, f"'{col}' should remain"

    def test_fewer_features_than_default(
        self,
        tmp_config: ProjectConfig,
        sample_df: pd.DataFrame,
    ):
        """Excluding target lags results in fewer features than the default mode."""
        trainer_default = ModelTrainer(tmp_config, target="nb_installations_pac")
        trainer_no_lags = ModelTrainer(
            tmp_config,
            target="nb_installations_pac",
            exclude_target_lags=True,
        )

        X_default, _ = trainer_default.prepare_features(sample_df)
        X_no_lags, _ = trainer_no_lags.prepare_features(sample_df)

        assert len(X_no_lags.columns) < len(X_default.columns), (
            "Excluding target lags should reduce feature count"
        )

    def test_specific_patterns_excluded(
        self, trainer_exclude_lags: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Each specific auto-regressive pattern is excluded for the active target."""
        X, _ = trainer_exclude_lags.prepare_features(sample_df)
        expected_excluded = [
            "nb_installations_pac_lag_1",
            "nb_installations_pac_lag_3",
            "nb_installations_pac_rmean_3",
            "nb_installations_pac_rstd_3",
            "nb_installations_pac_diff_1",
            "nb_installations_pac_pct_1",
        ]
        for col in expected_excluded:
            assert col not in X.columns, f"'{col}' should be excluded"


# =============================================================================
# Tests: ModelTrainer.get_feature_names
# =============================================================================


class TestGetFeatureNames:
    """Tests for the get_feature_names method."""

    def test_returns_list(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """get_feature_names returns a list."""
        names = trainer.get_feature_names(sample_df)
        assert isinstance(names, list)

    def test_returns_strings(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """All entries in the feature names list are strings."""
        names = trainer.get_feature_names(sample_df)
        assert all(isinstance(n, str) for n in names)

    def test_matches_prepare_features_columns(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """Feature names match the columns from prepare_features."""
        names = trainer.get_feature_names(sample_df)
        X, _ = trainer.prepare_features(sample_df)
        assert names == list(X.columns)

    def test_no_target_in_feature_names(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """The active target is not in the feature names."""
        names = trainer.get_feature_names(sample_df)
        assert "nb_installations_pac" not in names

    def test_no_exclude_cols_in_feature_names(
        self, trainer: ModelTrainer, sample_df: pd.DataFrame
    ):
        """No EXCLUDE_COLS are in the feature names."""
        names = trainer.get_feature_names(sample_df)
        for col in ModelTrainer.EXCLUDE_COLS:
            assert col not in names


# =============================================================================
# Tests: ModelEvaluator.compute_metrics
# =============================================================================


class TestComputeMetrics:
    """Tests for compute_metrics (RMSE, MAE, MAPE, R2)."""

    def test_perfect_prediction(self, evaluator: ModelEvaluator):
        """Perfect predictions yield RMSE=0, MAE=0, MAPE=0, R2=1."""
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        metrics = evaluator.compute_metrics(y, y)

        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mape"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_known_values(self, evaluator: ModelEvaluator):
        """Compute metrics on known input and verify approximate results."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])

        metrics = evaluator.compute_metrics(y_true, y_pred)

        # RMSE = sqrt(mean([100, 100, 100])) = sqrt(100) = 10.0
        assert metrics["rmse"] == pytest.approx(10.0, rel=1e-6)
        # MAE = mean([10, 10, 10]) = 10.0
        assert metrics["mae"] == pytest.approx(10.0, rel=1e-6)
        # R2: variance(y_true)=6666.67, SS_res=300 => 1 - 300/20000 = 0.985
        assert 0.0 < metrics["r2"] < 1.0
        # MAPE: mean([10/100, 10/200, 10/300]) * 100
        expected_mape = np.mean([10 / 100, 10 / 200, 10 / 300]) * 100
        assert metrics["mape"] == pytest.approx(expected_mape, rel=1e-4)

    def test_returns_all_keys(self, evaluator: ModelEvaluator):
        """The result dictionary contains exactly rmse, mae, mape, r2."""
        y = np.array([1.0, 2.0, 3.0])
        metrics = evaluator.compute_metrics(y, y + 0.1)
        assert set(metrics.keys()) == {"rmse", "mae", "mape", "r2"}

    def test_all_values_are_float(self, evaluator: ModelEvaluator):
        """All metric values are Python floats."""
        y = np.array([1.0, 2.0, 3.0])
        metrics = evaluator.compute_metrics(y, y + 0.5)
        for key, value in metrics.items():
            assert isinstance(value, float), f"'{key}' should be float, got {type(value)}"

    def test_rmse_greater_equal_mae(self, evaluator: ModelEvaluator):
        """RMSE is always >= MAE (by Cauchy-Schwarz / Jensen inequality)."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 35.0, 38.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)
        assert metrics["rmse"] >= metrics["mae"] - 1e-10


class TestComputeMetricsNaN:
    """Tests for compute_metrics handling of NaN values."""

    def test_nan_in_y_true(self, evaluator: ModelEvaluator):
        """NaN values in y_true are filtered out before metric computation."""
        y_true = np.array([10.0, np.nan, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 30.0, 40.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        # After filtering, perfect prediction on [10, 30, 40]
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_nan_in_y_pred(self, evaluator: ModelEvaluator):
        """NaN values in y_pred are filtered out before metric computation."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, np.nan, 30.0, 40.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_nan_in_both(self, evaluator: ModelEvaluator):
        """NaN in both arrays are handled — rows with any NaN are dropped."""
        y_true = np.array([np.nan, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, np.nan, 30.0, 40.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        # Only indices 2, 3 survive. Perfect prediction on [30, 40]
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_all_nan_returns_nan_metrics(self, evaluator: ModelEvaluator):
        """If all values are NaN after filtering, return NaN for all metrics."""
        y_true = np.array([np.nan, np.nan])
        y_pred = np.array([np.nan, np.nan])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        for key in ["rmse", "mae", "mape", "r2"]:
            assert math.isnan(metrics[key]), f"'{key}' should be NaN"

    def test_empty_arrays_return_nan(self, evaluator: ModelEvaluator):
        """Empty arrays produce NaN metrics (no valid data)."""
        y_true = np.array([])
        y_pred = np.array([])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        for key in ["rmse", "mae", "mape", "r2"]:
            assert math.isnan(metrics[key]), f"'{key}' should be NaN"


class TestComputeMetricsZeroMAPE:
    """Tests for MAPE handling when y_true contains zeros."""

    def test_mape_skips_zero_values(self, evaluator: ModelEvaluator):
        """Zero values in y_true are excluded from MAPE computation."""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([5.0, 110.0, 210.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        # MAPE computed only on [100, 200] vs [110, 210]
        # = mean([10/100, 10/200]) * 100 = mean([0.1, 0.05]) * 100 = 7.5
        expected_mape = np.mean([10 / 100, 10 / 200]) * 100
        assert metrics["mape"] == pytest.approx(expected_mape, rel=1e-4)

    def test_all_zeros_returns_nan_mape(self, evaluator: ModelEvaluator):
        """If all y_true values are zero, MAPE is NaN."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        assert math.isnan(metrics["mape"]), "MAPE should be NaN when all y_true=0"
        # Other metrics should still be computed
        assert not math.isnan(metrics["rmse"])
        assert not math.isnan(metrics["mae"])

    def test_single_nonzero_for_mape(self, evaluator: ModelEvaluator):
        """MAPE works correctly when only one y_true value is non-zero."""
        y_true = np.array([0.0, 0.0, 50.0])
        y_pred = np.array([1.0, 2.0, 55.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        # MAPE only on index 2: |50-55|/50 * 100 = 10.0
        assert metrics["mape"] == pytest.approx(10.0, rel=1e-4)


# =============================================================================
# Tests: ModelEvaluator.compare_models
# =============================================================================


class TestCompareModels:
    """Tests for the model comparison table."""

    def test_returns_dataframe(
        self, evaluator: ModelEvaluator, model_results: Dict
    ):
        """compare_models returns a pandas DataFrame."""
        df = evaluator.compare_models(model_results)
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(
        self, evaluator: ModelEvaluator, model_results: Dict
    ):
        """The comparison table has the expected column names."""
        df = evaluator.compare_models(model_results)
        expected_cols = {
            "Model",
            "Val RMSE", "Val MAE", "Val MAPE (%)", "Val R2",
            "Test RMSE", "Test MAE", "Test MAPE (%)", "Test R2",
        }
        assert set(df.columns) == expected_cols

    def test_one_row_per_model(
        self, evaluator: ModelEvaluator, model_results: Dict
    ):
        """There is one row for each model in the results."""
        df = evaluator.compare_models(model_results)
        assert len(df) == len(model_results)

    def test_model_names_in_table(
        self, evaluator: ModelEvaluator, model_results: Dict
    ):
        """All model names appear in the table."""
        df = evaluator.compare_models(model_results)
        assert set(df["Model"].values) == set(model_results.keys())

    def test_sorted_by_val_rmse(
        self, evaluator: ModelEvaluator, model_results: Dict
    ):
        """The table is sorted by Val RMSE ascending."""
        df = evaluator.compare_models(model_results)
        val_rmse_values = df["Val RMSE"].tolist()
        assert val_rmse_values == sorted(val_rmse_values), (
            "Table should be sorted by Val RMSE ascending"
        )

    def test_metric_values_match(
        self, evaluator: ModelEvaluator, model_results: Dict
    ):
        """Metric values in the table match the input results."""
        df = evaluator.compare_models(model_results)
        # model_a has Val RMSE=15.0 (lowest), so it should be first row
        row_a = df[df["Model"] == "model_a"].iloc[0]
        assert row_a["Val RMSE"] == pytest.approx(15.0)
        assert row_a["Test MAE"] == pytest.approx(12.0)
        assert row_a["Val R2"] == pytest.approx(0.85)

    def test_handles_missing_metrics(self, evaluator: ModelEvaluator):
        """Models with missing metric keys get NaN in the table."""
        results = {
            "full_model": {
                "metrics_val": {"rmse": 10.0, "mae": 8.0, "mape": 5.0, "r2": 0.9},
                "metrics_test": {"rmse": 12.0, "mae": 9.0, "mape": 6.0, "r2": 0.88},
            },
            "partial_model": {
                "metrics_val": {"rmse": 15.0},
                # metrics_test entirely missing
            },
        }
        df = evaluator.compare_models(results)
        partial_row = df[df["Model"] == "partial_model"].iloc[0]
        assert math.isnan(partial_row["Test RMSE"])
        assert math.isnan(partial_row["Val MAE"])

    def test_empty_results_raises_key_error(self, evaluator: ModelEvaluator):
        """An empty results dict raises KeyError (sort on missing 'Val RMSE' column).

        This is the current behavior: an empty results dict produces an
        empty DataFrame with no columns, which then fails when sorting
        by 'Val RMSE'. This test documents that edge case.
        """
        with pytest.raises(KeyError):
            evaluator.compare_models({})

    def test_single_model(self, evaluator: ModelEvaluator):
        """A single model produces a one-row table."""
        results = {
            "only_model": {
                "metrics_val": {"rmse": 5.0, "mae": 3.0, "mape": 2.0, "r2": 0.95},
                "metrics_test": {"rmse": 6.0, "mae": 4.0, "mape": 3.0, "r2": 0.93},
            },
        }
        df = evaluator.compare_models(results)
        assert len(df) == 1
        assert df.iloc[0]["Model"] == "only_model"


# =============================================================================
# Tests: ModelTrainer.load_dataset (file not found)
# =============================================================================


class TestLoadDataset:
    """Tests for the load_dataset method."""

    def test_raises_file_not_found(self, trainer: ModelTrainer):
        """FileNotFoundError is raised if the features CSV does not exist."""
        with pytest.raises(FileNotFoundError, match="Features dataset not found"):
            trainer.load_dataset()

    def test_loads_valid_csv(
        self,
        trainer: ModelTrainer,
        tmp_config: ProjectConfig,
        sample_df: pd.DataFrame,
    ):
        """A valid CSV file is loaded correctly."""
        csv_path = tmp_config.features_data_dir / "hvac_features_dataset.csv"
        sample_df.to_csv(csv_path, index=False)

        df = trainer.load_dataset()
        assert len(df) == len(sample_df)
        assert list(df.columns) == list(sample_df.columns)


# =============================================================================
# Tests: Regression / Security
# =============================================================================


class TestRegressionSecurity:
    """Regression tests to prevent silent changes to critical logic."""

    def test_exclude_cols_not_empty(self):
        """EXCLUDE_COLS must contain known identifier columns."""
        assert "date_id" in ModelTrainer.EXCLUDE_COLS
        assert "dept" in ModelTrainer.EXCLUDE_COLS

    def test_target_cols_not_empty(self):
        """TARGET_COLS must contain expected target variables."""
        assert "nb_installations_pac" in ModelTrainer.TARGET_COLS
        assert "nb_installations_clim" in ModelTrainer.TARGET_COLS

    def test_outlier_patterns_not_empty(self):
        """OUTLIER_PATTERNS must contain patterns for outlier columns."""
        assert len(ModelTrainer.OUTLIER_PATTERNS) >= 5
        assert "_outlier_iqr" in ModelTrainer.OUTLIER_PATTERNS

    def test_target_lag_patterns_not_empty(self):
        """TARGET_LAG_PATTERNS must contain patterns for lag features."""
        assert "_lag_" in ModelTrainer.TARGET_LAG_PATTERNS
        assert "_rmean_" in ModelTrainer.TARGET_LAG_PATTERNS

    def test_evaluator_figures_dir_created(self, evaluator: ModelEvaluator):
        """ModelEvaluator creates the figures directory on init."""
        assert evaluator.figures_dir.exists()

    def test_compute_metrics_multiplies_mape_by_100(
        self, evaluator: ModelEvaluator
    ):
        """MAPE is expressed as a percentage (multiplied by 100)."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])
        metrics = evaluator.compute_metrics(y_true, y_pred)

        # Raw MAPE = mean(|10/100|, |20/200|) = 0.10; x100 = 10.0
        assert metrics["mape"] == pytest.approx(10.0, rel=1e-4)

    def test_prepare_features_different_target(
        self, tmp_config: ProjectConfig, sample_df: pd.DataFrame
    ):
        """Changing the target changes which columns are excluded."""
        trainer_pac = ModelTrainer(tmp_config, target="nb_installations_pac")
        trainer_clim = ModelTrainer(tmp_config, target="nb_installations_clim")

        X_pac, y_pac = trainer_pac.prepare_features(sample_df)
        X_clim, y_clim = trainer_clim.prepare_features(sample_df)

        assert y_pac.name == "nb_installations_pac"
        assert y_clim.name == "nb_installations_clim"

        # nb_installations_clim should be in X_pac (as a feature, not a target)
        # Actually it is excluded because it is in TARGET_COLS - {self.target}
        assert "nb_installations_clim" not in X_pac.columns
        assert "nb_installations_pac" not in X_clim.columns
