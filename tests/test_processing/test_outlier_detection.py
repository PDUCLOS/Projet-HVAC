# -*- coding: utf-8 -*-
"""Tests for the outlier detection module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.settings import ProjectConfig
from src.processing.outlier_detection import OutlierDetector


@pytest.fixture
def detector(test_config: ProjectConfig) -> OutlierDetector:
    """OutlierDetector instance for tests."""
    return OutlierDetector(test_config)


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame with known outliers for testing detection."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "date_id": range(202201, 202201 + n),
        "dept": ["69"] * n,
        "nb_installations_pac": np.concatenate([
            np.random.normal(50, 10, n - 3),
            [200, 250, -10],  # 3 obvious outliers
        ]),
        "temp_mean": np.concatenate([
            np.random.normal(15, 5, n - 2),
            [60, -40],  # 2 temperature outliers
        ]),
        "hdd_sum": np.random.uniform(0, 300, n),
        "confiance_menages": np.concatenate([
            np.random.normal(95, 5, n - 1),
            [200],  # 1 outlier
        ]),
    })
    return df


@pytest.fixture
def df_clean() -> pd.DataFrame:
    """DataFrame without outliers."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "date_id": range(202201, 202201 + n),
        "dept": ["69"] * n,
        "value_a": np.random.normal(100, 5, n),
        "value_b": np.random.normal(50, 3, n),
    })


# ==================================================================
# Tests IQR
# ==================================================================

class TestDetectIQR:
    """Tests for the IQR method."""

    def test_detects_known_outliers(self, detector, df_with_outliers):
        """IQR should detect obvious outliers."""
        results = detector.detect_iqr(df_with_outliers)
        # The nb_installations_pac column has outliers (200, 250, -10)
        assert "nb_installations_pac" in results
        assert results["nb_installations_pac"]["n_outliers"] >= 2

    def test_no_outliers_on_clean_data(self, detector, df_clean):
        """No outlier on clean data (centered normal distribution)."""
        results = detector.detect_iqr(df_clean, columns=["value_a", "value_b"])
        total = sum(info["n_outliers"] for info in results.values())
        # Normal distribution has very few IQR outliers (< 5%)
        assert total < 5

    def test_returns_bounds(self, detector, df_with_outliers):
        """IQR bounds should be returned."""
        results = detector.detect_iqr(df_with_outliers, columns=["nb_installations_pac"])
        if "nb_installations_pac" in results:
            info = results["nb_installations_pac"]
            assert "lower_bound" in info
            assert "upper_bound" in info
            assert info["lower_bound"] < info["upper_bound"]

    def test_returns_indices(self, detector, df_with_outliers):
        """Outlier indices should be returned."""
        results = detector.detect_iqr(df_with_outliers)
        for col, info in results.items():
            assert "indices" in info
            assert isinstance(info["indices"], list)

    def test_empty_dataframe(self, detector):
        """Does not crash on an empty DataFrame."""
        df = pd.DataFrame({"val": []})
        results = detector.detect_iqr(df)
        assert results == {}

    def test_ignores_nan(self, detector):
        """NaN values should not be counted as outliers."""
        df = pd.DataFrame({
            "val": [1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 100]
        })
        results = detector.detect_iqr(df, columns=["val"])
        if "val" in results:
            for idx in results["val"]["indices"]:
                assert not pd.isna(df.loc[idx, "val"])

    def test_custom_factor(self, test_config):
        """A stricter IQR factor (1.0) detects more outliers."""
        strict = OutlierDetector(test_config, iqr_factor=1.0)
        normal = OutlierDetector(test_config, iqr_factor=1.5)
        df = pd.DataFrame({"val": np.random.normal(0, 1, 200)})
        strict_res = strict.detect_iqr(df, columns=["val"])
        normal_res = normal.detect_iqr(df, columns=["val"])
        strict_n = strict_res.get("val", {}).get("n_outliers", 0)
        normal_n = normal_res.get("val", {}).get("n_outliers", 0)
        assert strict_n >= normal_n


# ==================================================================
# Modified Z-score tests
# ==================================================================

class TestDetectZscoreModified:
    """Tests for the modified Z-score (MAD)."""

    def test_detects_extreme_values(self, detector, df_with_outliers):
        """The modified Z-score should detect extreme values."""
        results = detector.detect_zscore_modified(df_with_outliers)
        assert len(results) > 0

    def test_returns_z_scores(self, detector, df_with_outliers):
        """Z-scores should be returned."""
        results = detector.detect_zscore_modified(df_with_outliers)
        for col, info in results.items():
            assert "z_scores" in info
            # All Z-scores should be above the threshold
            for z in info["z_scores"]:
                assert abs(z) > detector.zscore_threshold

    def test_mad_zero_skipped(self, detector):
        """If MAD=0 (>50% identical), the column is skipped."""
        df = pd.DataFrame({"val": [1, 1, 1, 1, 1, 1, 1, 1, 100]})
        results = detector.detect_zscore_modified(df, columns=["val"])
        assert "val" not in results

    def test_no_false_positives_normal(self, detector):
        """Standard normal distribution: < 1% Z-score outliers."""
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
        results = detector.detect_zscore_modified(df, columns=["val"])
        if "val" in results:
            assert results["val"]["pct"] < 1.0


# ==================================================================
# Tests Isolation Forest
# ==================================================================

class TestDetectIsolationForest:
    """Tests for Isolation Forest."""

    def test_detects_multivariate_outliers(self, detector, df_with_outliers):
        """IF should detect multivariate outliers."""
        results = detector.detect_isolation_forest(df_with_outliers)
        assert results["n_outliers"] > 0
        assert len(results["indices"]) == results["n_outliers"]

    def test_returns_scores(self, detector, df_with_outliers):
        """Anomaly scores should be returned."""
        results = detector.detect_isolation_forest(df_with_outliers)
        assert "scores" in results

    def test_not_enough_data(self, detector):
        """With too few data points, returns 0 outliers."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        results = detector.detect_isolation_forest(df)
        assert results["n_outliers"] == 0

    def test_single_column(self, detector):
        """With a single numeric column, returns 0 outliers."""
        df = pd.DataFrame({"a": range(50)})
        results = detector.detect_isolation_forest(df, columns=["a"])
        assert results["n_outliers"] == 0


# ==================================================================
# Combined detection + flagging tests
# ==================================================================

class TestDetectAndFlag:
    """Tests for combined detection."""

    def test_adds_flag_columns(self, detector, df_with_outliers):
        """Flag columns should be added."""
        df_flagged, report = detector.detect_and_flag(df_with_outliers)
        assert "_outlier_iqr" in df_flagged.columns
        assert "_outlier_zscore" in df_flagged.columns
        assert "_outlier_iforest" in df_flagged.columns
        assert "_outlier_consensus" in df_flagged.columns
        assert "_outlier_score" in df_flagged.columns

    def test_consensus_requires_two_methods(self, detector, df_with_outliers):
        """Consensus requires at least 2 methods in agreement."""
        df_flagged, _ = detector.detect_and_flag(df_with_outliers)
        # Verify that consensus => at least 2 methods
        consensus_rows = df_flagged[df_flagged["_outlier_consensus"]]
        for _, row in consensus_rows.iterrows():
            n_methods = (
                int(row["_outlier_iqr"])
                + int(row["_outlier_zscore"])
                + int(row["_outlier_iforest"])
            )
            assert n_methods >= 2

    def test_report_structure(self, detector, df_with_outliers):
        """The report should have the correct structure."""
        _, report = detector.detect_and_flag(df_with_outliers)
        assert "n_rows" in report
        assert "iqr" in report
        assert "zscore" in report
        assert "isolation_forest" in report
        assert "consensus" in report

    def test_does_not_modify_data(self, detector, df_with_outliers):
        """Detection should not modify the original data."""
        original = df_with_outliers["nb_installations_pac"].copy()
        df_flagged, _ = detector.detect_and_flag(df_with_outliers)
        pd.testing.assert_series_equal(
            df_flagged["nb_installations_pac"], original,
        )


# ==================================================================
# Treatment tests
# ==================================================================

class TestDetectAndTreat:
    """Tests for outlier treatment."""

    def test_strategy_flag(self, detector, df_with_outliers):
        """The 'flag' strategy does not modify values."""
        original_sum = df_with_outliers["nb_installations_pac"].sum()
        df_treated, _ = detector.detect_and_treat(
            df_with_outliers, strategy="flag",
        )
        assert df_treated["nb_installations_pac"].sum() == original_sum

    def test_strategy_clip(self, detector, df_with_outliers):
        """The 'clip' strategy brings outliers back to IQR bounds."""
        df_treated, report = detector.detect_and_treat(
            df_with_outliers, strategy="clip",
        )
        assert report["strategy"] == "clip"
        # Extreme values should have been attenuated
        assert df_treated["nb_installations_pac"].max() < 250

    def test_strategy_remove(self, detector, df_with_outliers):
        """The 'remove' strategy deletes rows."""
        original_len = len(df_with_outliers)
        df_treated, report = detector.detect_and_treat(
            df_with_outliers, strategy="remove",
        )
        assert len(df_treated) <= original_len

    def test_preserves_row_count_flag(self, detector, df_with_outliers):
        """Flag and clip preserve all rows."""
        for strategy in ["flag", "clip"]:
            df_treated, _ = detector.detect_and_treat(
                df_with_outliers, strategy=strategy,
            )
            assert len(df_treated) == len(df_with_outliers)


# ==================================================================
# Temporal anomaly tests
# ==================================================================

class TestTemporalAnomalies:
    """Tests for temporal anomaly detection."""

    def test_detects_spike(self, detector):
        """Detects a sudden spike (doubling from the previous month)."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303, 202304],
            "dept": ["69"] * 4,
            "nb_installations_pac": [100, 100, 100, 300],  # +200% in month 4
        })
        result = detector.detect_temporal_anomalies(df, threshold_pct=50.0)
        anomalies = result["anomalies"]
        assert len(anomalies) >= 1
        assert any(a["type"] == "spike" for a in anomalies)

    def test_detects_drop(self, detector):
        """Detects a sudden drop."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303],
            "dept": ["69"] * 3,
            "nb_installations_pac": [100, 100, 20],  # -80% in month 3
        })
        result = detector.detect_temporal_anomalies(df, threshold_pct=50.0)
        anomalies = result["anomalies"]
        assert len(anomalies) >= 1
        assert any(a["type"] == "drop" for a in anomalies)

    def test_no_anomaly_stable(self, detector):
        """No anomaly on stable data."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303, 202304],
            "dept": ["69"] * 4,
            "nb_installations_pac": [100, 105, 98, 103],  # < 10% variation
        })
        result = detector.detect_temporal_anomalies(df, threshold_pct=50.0)
        assert len(result["anomalies"]) == 0


# ==================================================================
# Report tests
# ==================================================================

class TestReport:
    """Tests for report generation."""

    def test_generates_report_file(self, detector, df_with_outliers, tmp_path):
        """The report should be generated as a text file."""
        detector.report_dir = tmp_path
        df_flagged, report = detector.detect_and_flag(df_with_outliers)
        path = detector.generate_report(df_flagged, report)
        assert path.exists()
        content = path.read_text()
        assert "OUTLIER DETECTION REPORT" in content
        assert "IQR" in content
        assert "Z-score" in content
        assert "Isolation Forest" in content

    def test_report_includes_temporal(self, detector, df_with_outliers, tmp_path):
        """The report includes temporal anomalies if provided."""
        detector.report_dir = tmp_path
        df_flagged, report = detector.detect_and_flag(df_with_outliers)
        temporal = {"anomalies": [{"dept": "69", "date_id": 202301,
                                   "value": 200, "pct_change": 150.0,
                                   "type": "spike"}],
                    "threshold_pct": 50.0}
        path = detector.generate_report(df_flagged, report, temporal)
        content = path.read_text()
        assert "TEMPORAL ANOMALIES" in content


# ==================================================================
# run_full_analysis tests
# ==================================================================

class TestRunFullAnalysis:
    """Tests for full analysis."""

    def test_returns_treated_df_and_report(self, detector, df_with_outliers, tmp_path):
        """Full analysis returns the treated DataFrame and the report."""
        detector.report_dir = tmp_path
        df_treated, report_path = detector.run_full_analysis(
            df_with_outliers, strategy="clip",
        )
        assert isinstance(df_treated, pd.DataFrame)
        assert report_path.exists()
        assert len(df_treated) == len(df_with_outliers)


# ==================================================================
# Utility tests
# ==================================================================

class TestUtilities:
    """Tests for utility methods."""

    def test_get_numeric_columns(self, detector):
        """Returns only the relevant numeric columns."""
        df = pd.DataFrame({
            "date_id": [1, 2, 3],
            "dept": ["69", "38", "69"],
            "value": [10.0, 20.0, 30.0],
            "_flag": [True, False, True],
        })
        cols = detector._get_numeric_columns(df)
        assert "value" in cols
        assert "date_id" not in cols  # excluded
        assert "dept" not in cols      # non-numeric
        assert "_flag" not in cols     # _ prefix
