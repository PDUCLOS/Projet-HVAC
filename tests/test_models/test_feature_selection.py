# -*- coding: utf-8 -*-
"""
Tests for FeatureSelector — Multi-method feature selection.
=============================================================

Tests cover:
- Variance-based selection (low-variance removal)
- Correlation-based selection (redundancy removal)
- RFE selection (recursive feature elimination)
- SHAP-based selection (with mock model)
- Consensus selection (full pipeline)
- Edge cases (empty features, constant columns, single feature)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.models.feature_selection import FeatureSelector


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def selector(tmp_path: Path) -> FeatureSelector:
    """Create a FeatureSelector with a temporary output directory."""
    return FeatureSelector(analysis_dir=tmp_path / "analysis")


@pytest.fixture
def sample_X() -> pd.DataFrame:
    """Create a sample feature matrix with varied characteristics.

    Features:
    - feat_important: strongly correlated with target
    - feat_moderate: moderately useful
    - feat_noise: random noise
    - feat_constant: constant value (zero variance)
    - feat_corr_a, feat_corr_b: highly correlated pair (r > 0.95)
    """
    np.random.seed(42)
    n = 100

    base = np.linspace(0, 10, n)
    return pd.DataFrame({
        "feat_important": base + np.random.randn(n) * 0.5,
        "feat_moderate": base * 0.5 + np.random.randn(n) * 2,
        "feat_noise": np.random.randn(n) * 10,
        "feat_constant": np.ones(n) * 5.0,
        "feat_corr_a": base * 2 + np.random.randn(n) * 0.1,
        "feat_corr_b": base * 2 + np.random.randn(n) * 0.15,
        "feat_low_var": np.random.randn(n) * 0.001 + 3.0,
    })


@pytest.fixture
def sample_y() -> pd.Series:
    """Create a target variable correlated with feat_important."""
    np.random.seed(42)
    n = 100
    base = np.linspace(0, 10, n)
    return pd.Series(base * 3 + np.random.randn(n) * 1, name="target")


@pytest.fixture
def trained_ridge(sample_X: pd.DataFrame, sample_y: pd.Series) -> Ridge:
    """Train a Ridge model on the sample data for SHAP tests."""
    X_clean = sample_X.fillna(0)
    model = Ridge(alpha=1.0)
    model.fit(X_clean, sample_y)
    return model


# =============================================================================
# Tests: select_by_variance
# =============================================================================


class TestSelectByVariance:
    """Tests for variance-based feature selection."""

    def test_removes_constant_features(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Constant features (zero variance) are removed."""
        result = selector.select_by_variance(sample_X, threshold=0.01)
        assert "feat_constant" not in result["selected_features"]
        assert "feat_constant" in result["removed_features"]

    def test_keeps_variable_features(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Features with sufficient variance are kept."""
        result = selector.select_by_variance(sample_X, threshold=0.01)
        assert "feat_important" in result["selected_features"]
        assert "feat_noise" in result["selected_features"]

    def test_removes_low_variance_features(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Features with variance below threshold are removed."""
        result = selector.select_by_variance(sample_X, threshold=0.01)
        assert "feat_low_var" not in result["selected_features"]

    def test_returns_correct_structure(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Result contains expected keys."""
        result = selector.select_by_variance(sample_X)
        assert "selected_features" in result
        assert "removed_features" in result
        assert "variances" in result
        assert result["method"] == "variance"

    def test_high_threshold_removes_more(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Higher threshold removes more features."""
        low = selector.select_by_variance(sample_X, threshold=0.001)
        high = selector.select_by_variance(sample_X, threshold=10.0)
        assert len(high["selected_features"]) <= len(low["selected_features"])

    def test_zero_threshold_keeps_all_variable(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Threshold of 0 keeps all features (including zero-variance)."""
        result = selector.select_by_variance(sample_X, threshold=0.0)
        # With threshold=0, all features with var >= 0 are kept (all of them)
        assert len(result["selected_features"]) == len(sample_X.columns)

    def test_empty_dataframe(self, selector: FeatureSelector):
        """Empty DataFrame returns empty results."""
        X_empty = pd.DataFrame()
        result = selector.select_by_variance(X_empty)
        assert result["selected_features"] == []
        assert result["removed_features"] == []


# =============================================================================
# Tests: select_by_correlation
# =============================================================================


class TestSelectByCorrelation:
    """Tests for correlation-based feature selection."""

    def test_removes_highly_correlated_pair(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """One of the highly correlated pair (feat_corr_a/b) is removed."""
        result = selector.select_by_correlation(sample_X, threshold=0.95)
        # At least one of the pair should be removed
        corr_a_kept = "feat_corr_a" in result["selected_features"]
        corr_b_kept = "feat_corr_b" in result["selected_features"]
        # Both cannot be kept (they are correlated > 0.95)
        assert not (corr_a_kept and corr_b_kept), (
            "Both highly correlated features should not be kept"
        )

    def test_keeps_uncorrelated_features(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Uncorrelated features are preserved."""
        result = selector.select_by_correlation(sample_X, threshold=0.95)
        assert "feat_noise" in result["selected_features"]

    def test_returns_correlation_pairs(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """High correlation pairs are reported."""
        result = selector.select_by_correlation(sample_X, threshold=0.95)
        assert len(result["correlation_pairs"]) > 0
        # Each pair is a tuple of (feat_a, feat_b, correlation)
        for pair in result["correlation_pairs"]:
            assert len(pair) == 3
            assert pair[2] > 0.95

    def test_low_threshold_removes_more(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Lower threshold (stricter) removes more features."""
        strict = selector.select_by_correlation(sample_X, threshold=0.5)
        loose = selector.select_by_correlation(sample_X, threshold=0.99)
        assert len(strict["selected_features"]) <= len(loose["selected_features"])

    def test_threshold_1_keeps_all(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Threshold of 1.0 keeps all features (no pair has |r| > 1)."""
        result = selector.select_by_correlation(sample_X, threshold=1.0)
        assert len(result["removed_features"]) == 0

    def test_returns_correct_structure(
        self, selector: FeatureSelector, sample_X: pd.DataFrame
    ):
        """Result contains expected keys."""
        result = selector.select_by_correlation(sample_X)
        assert "selected_features" in result
        assert "removed_features" in result
        assert "correlation_pairs" in result
        assert result["method"] == "correlation"

    def test_single_feature(self, selector: FeatureSelector):
        """Single-feature DataFrame is not affected."""
        X = pd.DataFrame({"only_feature": [1.0, 2.0, 3.0]})
        result = selector.select_by_correlation(X)
        assert result["selected_features"] == ["only_feature"]

    def test_empty_dataframe(self, selector: FeatureSelector):
        """Empty DataFrame returns empty results."""
        X_empty = pd.DataFrame()
        result = selector.select_by_correlation(X_empty)
        assert result["selected_features"] == []


# =============================================================================
# Tests: select_by_rfe
# =============================================================================


class TestSelectByRFE:
    """Tests for Recursive Feature Elimination."""

    def test_selects_correct_number(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
    ):
        """RFE selects the requested number of features."""
        n_select = 3
        X_clean = sample_X.fillna(0)
        result = selector.select_by_rfe(
            X_clean, sample_y, Ridge(alpha=1.0), n_features=n_select,
        )
        assert len(result["selected_features"]) == n_select

    def test_returns_ranking(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
    ):
        """RFE returns a ranking for all features."""
        X_clean = sample_X.fillna(0)
        result = selector.select_by_rfe(
            X_clean, sample_y, Ridge(alpha=1.0), n_features=3,
        )
        assert "ranking" in result
        assert len(result["ranking"]) == len(X_clean.columns)
        # Selected features have rank 1
        for feat in result["selected_features"]:
            assert result["ranking"][feat] == 1

    def test_n_features_larger_than_available(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
    ):
        """Requesting more features than available returns all."""
        X_clean = sample_X.fillna(0)
        result = selector.select_by_rfe(
            X_clean, sample_y, Ridge(alpha=1.0), n_features=100,
        )
        assert len(result["selected_features"]) == len(X_clean.columns)

    def test_returns_correct_structure(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
    ):
        """Result contains expected keys."""
        X_clean = sample_X.fillna(0)
        result = selector.select_by_rfe(
            X_clean, sample_y, Ridge(alpha=1.0), n_features=3,
        )
        assert "selected_features" in result
        assert "ranking" in result
        assert result["method"] == "rfe"


# =============================================================================
# Tests: select_by_shap
# =============================================================================


class TestSelectBySHAP:
    """Tests for SHAP-based feature selection."""

    def test_returns_correct_structure(
        self,
        selector: FeatureSelector,
        trained_ridge: Ridge,
        sample_X: pd.DataFrame,
    ):
        """Result contains expected keys regardless of SHAP availability."""
        X_clean = sample_X.fillna(0)
        result = selector.select_by_shap(trained_ridge, X_clean, top_k=3)
        assert "selected_features" in result
        assert "importance_scores" in result
        assert result["method"] == "shap"

    def test_returns_features(
        self,
        selector: FeatureSelector,
        trained_ridge: Ridge,
        sample_X: pd.DataFrame,
    ):
        """Selected features are a subset of the input columns."""
        X_clean = sample_X.fillna(0)
        result = selector.select_by_shap(trained_ridge, X_clean, top_k=5)
        for feat in result["selected_features"]:
            assert feat in X_clean.columns


# =============================================================================
# Tests: run_full_selection (consensus)
# =============================================================================


class TestRunFullSelection:
    """Tests for the full consensus feature selection pipeline."""

    def test_returns_consensus_features(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Full selection returns a consensus feature list."""
        X_clean = sample_X.fillna(0)
        result = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        assert "selected_features" in result
        assert len(result["selected_features"]) > 0

    def test_consensus_is_subset(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Consensus features are a subset of the input features."""
        X_clean = sample_X.fillna(0)
        result = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        for feat in result["selected_features"]:
            assert feat in X_clean.columns

    def test_higher_min_votes_fewer_features(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Higher min_votes produces fewer or equal consensus features."""
        X_clean = sample_X.fillna(0)
        result_1 = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        result_3 = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=3,
        )
        assert len(result_3["selected_features"]) <= len(result_1["selected_features"])

    def test_returns_votes(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Result includes vote counts per feature."""
        X_clean = sample_X.fillna(0)
        result = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        assert "votes" in result
        assert isinstance(result["votes"], dict)

    def test_returns_method_results(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Result includes individual method results."""
        X_clean = sample_X.fillna(0)
        result = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        assert "method_results" in result
        assert "variance" in result["method_results"]
        assert "correlation" in result["method_results"]
        assert "rfe" in result["method_results"]
        assert "shap" in result["method_results"]

    def test_saves_results_json(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Results are saved to a JSON file."""
        X_clean = sample_X.fillna(0)
        selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        output_path = selector.analysis_dir / "selected_features.json"
        assert output_path.exists()

    def test_report_string_included(
        self,
        selector: FeatureSelector,
        sample_X: pd.DataFrame,
        sample_y: pd.Series,
        trained_ridge: Ridge,
    ):
        """Result includes a human-readable report string."""
        X_clean = sample_X.fillna(0)
        result = selector.run_full_selection(
            X_clean, sample_y, trained_ridge,
            top_k=5, min_votes=1,
        )
        assert "report" in result
        assert "Feature Selection Report" in result["report"]


# =============================================================================
# Tests: Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_constant_columns(self, selector: FeatureSelector):
        """All-constant columns are removed by variance filter."""
        X = pd.DataFrame({
            "const_a": np.ones(50),
            "const_b": np.zeros(50),
        })
        result = selector.select_by_variance(X, threshold=0.01)
        assert len(result["selected_features"]) == 0
        assert len(result["removed_features"]) == 2

    def test_single_row(self, selector: FeatureSelector):
        """Single-row DataFrame is handled gracefully."""
        X = pd.DataFrame({"a": [1.0], "b": [2.0]})
        # Variance of a single value is NaN, treated as 0
        result = selector.select_by_variance(X, threshold=0.01)
        # Should not crash
        assert isinstance(result["selected_features"], list)

    def test_with_nan_values(
        self,
        selector: FeatureSelector,
    ):
        """NaN values do not crash variance or correlation selection."""
        np.random.seed(42)
        X = pd.DataFrame({
            "a": [1, 2, np.nan, 4, 5],
            "b": [np.nan, 2, 3, 4, 5],
            "c": [1, 2, 3, 4, 5],
        })
        # Should not raise
        var_result = selector.select_by_variance(X, threshold=0.01)
        corr_result = selector.select_by_correlation(X, threshold=0.95)
        assert isinstance(var_result["selected_features"], list)
        assert isinstance(corr_result["selected_features"], list)
