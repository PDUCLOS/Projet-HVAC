# -*- coding: utf-8 -*-
"""
Tests for the correlation analysis module (CorrelationAnalyzer).

Tests correlation computation, chart generation, text report,
and edge cases (empty data, single column, missing targets).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis.correlation import CorrelationAnalyzer


# ---------------------------------------------------------------------------
# Helper to build a sample features DataFrame
# ---------------------------------------------------------------------------

def _build_corr_df(n_months: int = 24, depts: list[str] = None) -> pd.DataFrame:
    """Build a features DataFrame suitable for correlation testing."""
    if depts is None:
        depts = ["69", "38"]
    np.random.seed(42)
    rows = []
    for dept in depts:
        for m in range(n_months):
            year = 2022 + m // 12
            month = m % 12 + 1
            date_id = year * 100 + month
            # Create correlated features
            base = np.random.uniform(10, 80)
            rows.append({
                "date_id": date_id,
                "dept": dept,
                "nb_dpe_total": base * 5 + np.random.normal(0, 10),
                "nb_installations_pac": base + np.random.normal(0, 5),
                "nb_installations_clim": base * 0.3 + np.random.normal(0, 3),
                "nb_dpe_classe_ab": base * 0.5 + np.random.normal(0, 5),
                "pct_pac": np.random.uniform(5, 20),
                "pct_clim": np.random.uniform(2, 10),
                "temp_mean": np.random.uniform(-2, 25),
                "temp_max": np.random.uniform(5, 35),
                "temp_min": np.random.uniform(-10, 15),
                "hdd_sum": np.random.uniform(0, 500),
                "cdd_sum": np.random.uniform(0, 200),
                "precipitation_sum": np.random.uniform(20, 150),
                "nb_jours_canicule": np.random.randint(0, 10),
                "nb_jours_gel": np.random.randint(0, 15),
                "confiance_menages": np.random.uniform(80, 110),
                "climat_affaires_indus": np.random.uniform(90, 110),
                "climat_affaires_bat": np.random.uniform(85, 105),
                "ipi_manufacturing": np.random.uniform(95, 115),
                "ipi_hvac_c28": np.random.uniform(90, 115),
                "ipi_hvac_c2825": np.random.uniform(85, 120),
                "month": month,
                "quarter": (month - 1) // 3 + 1,
                "is_heating": int(month in [1, 2, 3, 10, 11, 12]),
                "is_cooling": int(month in [6, 7, 8, 9]),
            })
    df = pd.DataFrame(rows)
    df["dept"] = df["dept"].astype(str).str.zfill(2)
    df["date"] = pd.to_datetime(df["date_id"].astype(str), format="%Y%m")
    return df


@pytest.fixture
def corr_analyzer(tmp_path):
    """Create a CorrelationAnalyzer with mock config and tmp output dir."""
    config = MagicMock()
    config.features_data_dir = tmp_path
    analyzer = CorrelationAnalyzer(config)
    analyzer.output_dir = tmp_path / "figures"
    analyzer.report_dir = tmp_path / "reports"
    return analyzer


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    """Tests for plot_correlation_matrix()."""

    def test_correlation_matrix_creates_file(self, corr_analyzer):
        """plot_correlation_matrix() creates a PNG file."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_correlation_matrix()
        assert path.exists()
        assert path.suffix == ".png"
        assert "correlation_matrix" in path.name

    def test_correlation_matrix_uses_base_features(self, corr_analyzer):
        """plot_correlation_matrix() uses only BASE_FEATURES columns."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        # Should not fail even with extra columns
        path = corr_analyzer.plot_correlation_matrix()
        assert path.exists()


# ---------------------------------------------------------------------------
# Top correlations
# ---------------------------------------------------------------------------

class TestTopCorrelations:
    """Tests for plot_top_correlations()."""

    def test_top_correlations_creates_file(self, corr_analyzer):
        """plot_top_correlations() creates a PNG file."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_top_correlations()
        assert path.exists()
        assert "top_correlations" in path.name

    def test_top_correlations_custom_target(self, corr_analyzer):
        """plot_top_correlations() works with custom target column."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_top_correlations(target="nb_dpe_total")
        assert path.exists()
        assert "nb_dpe_total" in path.name

    def test_top_correlations_missing_target(self, corr_analyzer):
        """plot_top_correlations() returns empty Path for missing target."""
        corr_analyzer.df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_top_correlations()
        assert path == Path()

    def test_top_correlations_respects_n_top(self, corr_analyzer):
        """plot_top_correlations() limits output to n_top features."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_top_correlations(n_top=5)
        assert path.exists()


# ---------------------------------------------------------------------------
# Correlations by department
# ---------------------------------------------------------------------------

class TestCorrelationByDept:
    """Tests for plot_correlation_by_dept()."""

    def test_correlation_by_dept_creates_file(self, corr_analyzer):
        """plot_correlation_by_dept() creates a PNG file."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_correlation_by_dept()
        assert path.exists()
        assert "correlation_by_dept" in path.name

    def test_correlation_by_dept_missing_dept_col(self, corr_analyzer):
        """plot_correlation_by_dept() returns empty Path without dept col."""
        df = _build_corr_df()
        df = df.drop(columns=["dept"])
        corr_analyzer.df = df
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_correlation_by_dept()
        assert path == Path()


# ---------------------------------------------------------------------------
# Correlations by season
# ---------------------------------------------------------------------------

class TestCorrelationBySeason:
    """Tests for plot_correlation_by_season()."""

    def test_correlation_by_season_creates_file(self, corr_analyzer):
        """plot_correlation_by_season() creates a PNG file."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_correlation_by_season()
        assert path.exists()
        assert "correlation_by_season" in path.name

    def test_correlation_by_season_missing_month(self, corr_analyzer):
        """plot_correlation_by_season() skips when month column missing."""
        df = _build_corr_df()
        df = df.drop(columns=["month"])
        corr_analyzer.df = df
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_correlation_by_season()
        assert path == Path()


# ---------------------------------------------------------------------------
# Multicollinearity
# ---------------------------------------------------------------------------

class TestMulticollinearity:
    """Tests for plot_multicollinearity()."""

    def test_multicollinearity_analysis(self, corr_analyzer):
        """plot_multicollinearity() runs without error."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        # May return Path() if no pairs > 0.8
        path = corr_analyzer.plot_multicollinearity()
        assert isinstance(path, Path)

    def test_multicollinearity_with_correlated_features(self, corr_analyzer):
        """plot_multicollinearity() detects highly correlated features."""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        df = pd.DataFrame({
            "nb_dpe_total": x * 10 + np.random.randn(n) * 0.1,
            "nb_installations_pac": x * 10 + np.random.randn(n) * 0.1,
            "temp_mean": np.random.randn(n),
        })
        corr_analyzer.df = df
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_multicollinearity()
        # Should create a chart showing the correlated pair
        assert path.exists()


# ---------------------------------------------------------------------------
# Lag correlations
# ---------------------------------------------------------------------------

class TestLagCorrelations:
    """Tests for plot_lag_correlations()."""

    def test_lag_correlations_no_lag_columns(self, corr_analyzer):
        """plot_lag_correlations() returns empty Path when no lag columns."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_lag_correlations()
        assert path == Path()

    def test_lag_correlations_with_lag_data(self, corr_analyzer):
        """plot_lag_correlations() creates chart when lag columns exist."""
        df = _build_corr_df()
        # Add lag columns
        df["temp_mean_lag_1m"] = df["temp_mean"].shift(1)
        df["temp_mean_lag_2m"] = df["temp_mean"].shift(2)
        df["temp_mean_lag_3m"] = df["temp_mean"].shift(3)
        df = df.dropna()
        corr_analyzer.df = df
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_lag_correlations()
        assert path.exists()


# ---------------------------------------------------------------------------
# Correlation report
# ---------------------------------------------------------------------------

class TestCorrelationReport:
    """Tests for generate_correlation_report()."""

    def test_report_creates_file(self, corr_analyzer):
        """generate_correlation_report() creates a text file."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.generate_correlation_report()
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "CORRELATION REPORT" in content

    def test_report_includes_recommendations(self, corr_analyzer):
        """Report includes modeling recommendations."""
        corr_analyzer.df = _build_corr_df()
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.generate_correlation_report()
        content = path.read_text(encoding="utf-8")
        assert "RECOMMENDATIONS" in content
        assert "Ridge" in content or "LightGBM" in content


# ---------------------------------------------------------------------------
# Safe plot wrapper
# ---------------------------------------------------------------------------

class TestSafePlot:
    """Tests for _safe_plot error handling."""

    def test_safe_plot_catches_errors(self, corr_analyzer):
        """_safe_plot returns None on error."""
        def failing_func():
            raise ValueError("Intentional failure")

        result = corr_analyzer._safe_plot("test", failing_func)
        assert result is None

    def test_safe_plot_returns_result(self, corr_analyzer):
        """_safe_plot returns function result on success."""
        def good_func():
            return Path("/test.png")

        result = corr_analyzer._safe_plot("test", good_func)
        assert result == Path("/test.png")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for CorrelationAnalyzer."""

    def test_empty_dataframe(self, corr_analyzer):
        """Correlation methods handle empty DataFrames."""
        corr_analyzer.df = pd.DataFrame(columns=["nb_installations_pac", "temp_mean"])
        corr_analyzer._ensure_dirs()

        # Should not crash
        path = corr_analyzer.plot_top_correlations()
        # Target column exists but is empty
        assert isinstance(path, Path)

    def test_single_column_dataframe(self, corr_analyzer):
        """Correlation methods handle single-column DataFrames."""
        corr_analyzer.df = pd.DataFrame({"nb_installations_pac": [1, 2, 3, 4, 5]})
        corr_analyzer._ensure_dirs()

        path = corr_analyzer.plot_correlation_matrix()
        assert isinstance(path, Path)

    def test_all_constant_values(self, corr_analyzer):
        """Correlation handles constant columns (std=0, correlation undefined)."""
        corr_analyzer.df = pd.DataFrame({
            "nb_installations_pac": [10, 10, 10, 10],
            "temp_mean": [5, 5, 5, 5],
            "dept": ["69", "69", "38", "38"],
            "date_id": [202301, 202302, 202301, 202302],
        })
        corr_analyzer.df["date"] = pd.to_datetime(
            corr_analyzer.df["date_id"].astype(str), format="%Y%m"
        )
        corr_analyzer._ensure_dirs()

        # Should not crash despite NaN correlations
        path = corr_analyzer.plot_top_correlations()
        assert isinstance(path, Path)

    def test_ensure_dirs_creates_directories(self, corr_analyzer):
        """_ensure_dirs() creates necessary directories."""
        corr_analyzer._ensure_dirs()
        assert corr_analyzer.output_dir.exists()
        assert corr_analyzer.report_dir.exists()
