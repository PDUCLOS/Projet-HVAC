# -*- coding: utf-8 -*-
"""
Tests for the EDA analysis module (EDAAnalyzer).

Tests chart generation methods, overview statistics, text report generation,
safe plot error handling, and edge cases with sample DataFrames.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import EDAAnalyzer


# ---------------------------------------------------------------------------
# Helper to build a sample features DataFrame
# ---------------------------------------------------------------------------

def _build_features_df(n_months: int = 12, depts: list[str] = None) -> pd.DataFrame:
    """Build a minimal features DataFrame for testing EDA."""
    if depts is None:
        depts = ["69", "38"]
    np.random.seed(42)
    rows = []
    for dept in depts:
        for m in range(1, n_months + 1):
            date_id = 202300 + m
            rows.append({
                "date_id": date_id,
                "dept": dept,
                "nb_dpe_total": np.random.randint(100, 500),
                "nb_installations_pac": np.random.randint(10, 80),
                "nb_installations_clim": np.random.randint(5, 30),
                "nb_dpe_classe_ab": np.random.randint(10, 100),
                "pct_pac": np.random.uniform(5, 20),
                "temp_mean": np.random.uniform(-2, 25),
                "hdd_sum": np.random.uniform(0, 500),
                "cdd_sum": np.random.uniform(0, 200),
                "precipitation_sum": np.random.uniform(20, 150),
                "confiance_menages": np.random.uniform(80, 110),
                "ipi_hvac_c28": np.random.uniform(90, 115),
                "nb_jours_canicule": np.random.randint(0, 10),
                "nb_jours_gel": np.random.randint(0, 15),
                "month": m,
                "quarter": (m - 1) // 3 + 1,
            })
    df = pd.DataFrame(rows)
    df["dept"] = df["dept"].astype(str).str.zfill(2)
    df["date"] = pd.to_datetime(df["date_id"].astype(str), format="%Y%m")
    return df


@pytest.fixture
def eda_analyzer(tmp_path):
    """Create an EDAAnalyzer with a mock config and tmp output dir."""
    config = MagicMock()
    config.features_data_dir = tmp_path
    analyzer = EDAAnalyzer(config)
    analyzer.output_dir = tmp_path / "figures"
    analyzer.report_dir = tmp_path / "reports"
    return analyzer


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

class TestOverview:
    """Tests for the overview() method."""

    def test_overview_returns_stats(self, eda_analyzer):
        """overview() returns a dictionary with key statistics."""
        eda_analyzer.df = _build_features_df()
        stats = eda_analyzer.overview()

        assert "shape" in stats
        assert "nan_pct" in stats
        assert stats["shape"][0] > 0

    def test_overview_with_all_columns(self, eda_analyzer):
        """overview() includes department and date info when available."""
        eda_analyzer.df = _build_features_df()
        stats = eda_analyzer.overview()

        assert "n_departments" in stats
        assert stats["n_departments"] == 2
        assert "date_range" in stats

    def test_overview_handles_missing_columns(self, eda_analyzer):
        """overview() gracefully handles missing optional columns."""
        eda_analyzer.df = pd.DataFrame({"x": [1, 2, 3]})
        stats = eda_analyzer.overview()

        assert "shape" in stats
        assert "n_departments" not in stats


# ---------------------------------------------------------------------------
# NaN heatmap
# ---------------------------------------------------------------------------

class TestNanHeatmap:
    """Tests for plot_nan_heatmap()."""

    def test_nan_heatmap_creates_file(self, eda_analyzer):
        """plot_nan_heatmap() creates a PNG file when NaN exist."""
        df = _build_features_df()
        # Inject some NaN
        df.loc[0, "temp_mean"] = np.nan
        df.loc[1, "hdd_sum"] = np.nan
        eda_analyzer.df = df
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_nan_heatmap()
        assert path.exists()
        assert path.suffix == ".png"

    def test_nan_heatmap_skips_when_no_nan(self, eda_analyzer):
        """plot_nan_heatmap() returns empty Path when no NaN."""
        eda_analyzer.df = _build_features_df()  # No NaN
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_nan_heatmap()
        assert path == Path()


# ---------------------------------------------------------------------------
# Target distributions
# ---------------------------------------------------------------------------

class TestTargetDistributions:
    """Tests for plot_target_distributions()."""

    def test_target_distributions_creates_file(self, eda_analyzer):
        """plot_target_distributions() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_target_distributions()
        assert path.exists()
        assert path.suffix == ".png"
        assert "target_distributions" in path.name


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

class TestTimeseries:
    """Tests for time series plot methods."""

    def test_timeseries_pac_creates_file(self, eda_analyzer):
        """plot_timeseries_pac() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_timeseries_pac()
        assert path.exists()
        assert "timeseries_pac" in path.name

    def test_timeseries_pac_skips_missing_column(self, eda_analyzer):
        """plot_timeseries_pac() returns empty Path when column missing."""
        eda_analyzer.df = pd.DataFrame({"dept": ["69"], "date": [pd.Timestamp("2023-01-01")]})
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_timeseries_pac()
        assert path == Path()

    def test_timeseries_dpe_total_creates_file(self, eda_analyzer):
        """plot_timeseries_dpe_total() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_timeseries_dpe_total()
        assert path.exists()

    def test_timeseries_aggregated_creates_file(self, eda_analyzer):
        """plot_timeseries_aura_aggregated() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_timeseries_aura_aggregated()
        assert path.exists()

    def test_timeseries_aggregated_skips_missing_cols(self, eda_analyzer):
        """plot_timeseries_aura_aggregated() skips when columns missing."""
        eda_analyzer.df = pd.DataFrame({"date": [pd.Timestamp("2023-01-01")]})
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_timeseries_aura_aggregated()
        assert path == Path()


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------

class TestSeasonality:
    """Tests for seasonality plot methods."""

    def test_seasonality_boxplots_creates_file(self, eda_analyzer):
        """plot_seasonality_boxplots() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_seasonality_boxplots()
        assert path.exists()

    def test_seasonality_skips_missing_month(self, eda_analyzer):
        """plot_seasonality_boxplots() skips when 'month' missing."""
        eda_analyzer.df = pd.DataFrame({"x": [1, 2]})
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_seasonality_boxplots()
        assert path == Path()

    def test_heatmap_dept_month_creates_file(self, eda_analyzer):
        """plot_heatmap_dept_month() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_heatmap_dept_month()
        assert path.exists()


# ---------------------------------------------------------------------------
# Department comparison
# ---------------------------------------------------------------------------

class TestDeptComparison:
    """Tests for department comparison plots."""

    def test_dept_comparison_creates_file(self, eda_analyzer):
        """plot_dept_comparison() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_dept_comparison()
        assert path.exists()

    def test_pct_pac_by_dept_creates_file(self, eda_analyzer):
        """plot_pct_pac_by_dept() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_pct_pac_by_dept()
        assert path.exists()

    def test_pct_pac_skips_missing_col(self, eda_analyzer):
        """plot_pct_pac_by_dept() skips when pct_pac is missing."""
        df = _build_features_df()
        df = df.drop(columns=["pct_pac"])
        eda_analyzer.df = df
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_pct_pac_by_dept()
        assert path == Path()


# ---------------------------------------------------------------------------
# Scatter features
# ---------------------------------------------------------------------------

class TestScatterFeatures:
    """Tests for scatter plot methods."""

    def test_scatter_features_creates_file(self, eda_analyzer):
        """plot_scatter_features_vs_target() creates a PNG file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_scatter_features_vs_target()
        assert path.exists()

    def test_scatter_features_skips_no_target(self, eda_analyzer):
        """plot_scatter_features_vs_target() skips when target missing."""
        eda_analyzer.df = pd.DataFrame({"temp_mean": [1.0, 2.0]})
        eda_analyzer._ensure_dirs()

        path = eda_analyzer.plot_scatter_features_vs_target()
        assert path == Path()


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

class TestTextReport:
    """Tests for text report generation."""

    def test_generate_text_report_creates_file(self, eda_analyzer):
        """generate_text_report() creates a text file."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        stats = eda_analyzer.overview()
        path = eda_analyzer.generate_text_report(stats)

        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "EDA REPORT" in content

    def test_report_contains_overview_stats(self, eda_analyzer):
        """Report file contains overview statistics."""
        eda_analyzer.df = _build_features_df()
        eda_analyzer._ensure_dirs()

        stats = {"shape": (24, 10), "nan_pct": 0.0}
        path = eda_analyzer.generate_text_report(stats)

        content = path.read_text(encoding="utf-8")
        assert "OVERVIEW" in content


# ---------------------------------------------------------------------------
# Safe plot wrapper
# ---------------------------------------------------------------------------

class TestSafePlot:
    """Tests for _safe_plot error handling wrapper."""

    def test_safe_plot_returns_result_on_success(self, eda_analyzer):
        """_safe_plot returns the function result on success."""
        def good_func():
            return Path("/test/path.png")

        result = eda_analyzer._safe_plot("test", good_func)
        assert result == Path("/test/path.png")

    def test_safe_plot_returns_none_on_failure(self, eda_analyzer):
        """_safe_plot returns None when function raises."""
        def bad_func():
            raise ValueError("Plot failed")

        result = eda_analyzer._safe_plot("test", bad_func)
        assert result is None

    def test_safe_plot_closes_figures_on_error(self, eda_analyzer):
        """_safe_plot closes all figures after an error."""
        plt.figure()  # Create a figure that should be cleaned up

        def bad_func():
            raise RuntimeError("Crash")

        eda_analyzer._safe_plot("test", bad_func)
        # All figures should be closed
        assert len(plt.get_fignums()) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for EDA."""

    def test_single_department(self, eda_analyzer):
        """EDA works with a single department."""
        eda_analyzer.df = _build_features_df(depts=["69"])
        eda_analyzer._ensure_dirs()

        stats = eda_analyzer.overview()
        assert stats["n_departments"] == 1

    def test_single_month(self, eda_analyzer):
        """EDA works with a single month of data."""
        eda_analyzer.df = _build_features_df(n_months=1)
        eda_analyzer._ensure_dirs()

        stats = eda_analyzer.overview()
        assert stats["shape"][0] == 2  # 2 depts x 1 month

    def test_ensure_dirs_creates_directories(self, eda_analyzer):
        """_ensure_dirs() creates output directories."""
        eda_analyzer._ensure_dirs()
        assert eda_analyzer.output_dir.exists()
        assert eda_analyzer.report_dir.exists()

    def test_save_fig_creates_png(self, eda_analyzer):
        """_save_fig() saves and closes the current figure."""
        eda_analyzer._ensure_dirs()
        plt.figure()
        plt.plot([1, 2, 3])

        path = eda_analyzer._save_fig("test_chart")
        assert path.exists()
        assert path.suffix == ".png"
        # Figure should be closed
        assert len(plt.get_fignums()) == 0
