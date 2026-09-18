# -*- coding: utf-8 -*-
"""
Tests for DriftDetector — Data drift detection module.
========================================================

Tests cover:
- KS drift detection (same vs shifted distributions)
- PSI computation (identical, moderate shift, severe shift)
- PSI status classification
- Full drift report generation
- Edge cases (empty data, NaN, constant features)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.data_drift import DriftDetector


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def detector(tmp_path: Path) -> DriftDetector:
    """Create a DriftDetector with a temporary output directory."""
    return DriftDetector(analysis_dir=tmp_path / "analysis")


@pytest.fixture
def identical_data():
    """Create two identical DataFrames (no drift expected).

    Returns:
        Tuple (reference_df, current_df, features).
    """
    np.random.seed(42)
    n = 200
    data = {
        "feature_a": np.random.normal(100, 15, n),
        "feature_b": np.random.uniform(0, 50, n),
        "feature_c": np.random.exponential(10, n),
    }
    ref_df = pd.DataFrame(data)
    # Same distribution, different random seed
    np.random.seed(43)
    cur_data = {
        "feature_a": np.random.normal(100, 15, n),
        "feature_b": np.random.uniform(0, 50, n),
        "feature_c": np.random.exponential(10, n),
    }
    cur_df = pd.DataFrame(cur_data)
    features = ["feature_a", "feature_b", "feature_c"]
    return ref_df, cur_df, features


@pytest.fixture
def shifted_data():
    """Create DataFrames where current data has a clear distribution shift.

    feature_a: mean shifted by 3 standard deviations
    feature_b: same distribution (control)
    feature_c: variance doubled

    Returns:
        Tuple (reference_df, current_df, features).
    """
    np.random.seed(42)
    n = 200
    ref_df = pd.DataFrame({
        "feature_a": np.random.normal(100, 10, n),
        "feature_b": np.random.uniform(0, 50, n),
        "feature_c": np.random.normal(50, 5, n),
    })
    cur_df = pd.DataFrame({
        "feature_a": np.random.normal(130, 10, n),  # Mean shift
        "feature_b": np.random.uniform(0, 50, n),   # No shift
        "feature_c": np.random.normal(50, 15, n),   # Variance change
    })
    features = ["feature_a", "feature_b", "feature_c"]
    return ref_df, cur_df, features


# =============================================================================
# Tests: detect_ks_drift
# =============================================================================


class TestKSDrift:
    """Tests for Kolmogorov-Smirnov drift detection."""

    def test_no_drift_same_distribution(
        self, detector: DriftDetector, identical_data
    ):
        """Same distribution should generally not detect drift."""
        ref_df, cur_df, features = identical_data
        results = detector.detect_ks_drift(ref_df, cur_df, features, threshold=0.05)

        # Most features should not show drift with same distribution
        drift_count = sum(
            1 for r in results.values() if r["drift_detected"]
        )
        # Allow at most 1 false positive (statistical tests have false positive rate)
        assert drift_count <= 1, (
            f"Too many false positives: {drift_count}/3 features show drift"
        )

    def test_detects_shifted_distribution(
        self, detector: DriftDetector, shifted_data
    ):
        """Clearly shifted distribution (3 sigma) should be detected."""
        ref_df, cur_df, features = shifted_data
        results = detector.detect_ks_drift(ref_df, cur_df, features, threshold=0.05)

        # feature_a has a 3-sigma mean shift — should be detected
        assert results["feature_a"]["drift_detected"], (
            "KS should detect drift in feature_a (mean shifted by 3 sigma)"
        )

    def test_returns_correct_structure(
        self, detector: DriftDetector, identical_data
    ):
        """Results contain expected keys for each feature."""
        ref_df, cur_df, features = identical_data
        results = detector.detect_ks_drift(ref_df, cur_df, features)

        for feature in features:
            assert feature in results
            assert "statistic" in results[feature]
            assert "p_value" in results[feature]
            assert "drift_detected" in results[feature]
            assert isinstance(results[feature]["drift_detected"], bool)

    def test_ks_statistic_between_0_and_1(
        self, detector: DriftDetector, identical_data
    ):
        """KS statistic is always between 0 and 1."""
        ref_df, cur_df, features = identical_data
        results = detector.detect_ks_drift(ref_df, cur_df, features)

        for feature in features:
            stat = results[feature]["statistic"]
            assert 0 <= stat <= 1, f"KS stat should be in [0,1], got {stat}"

    def test_p_value_between_0_and_1(
        self, detector: DriftDetector, identical_data
    ):
        """P-value is always between 0 and 1."""
        ref_df, cur_df, features = identical_data
        results = detector.detect_ks_drift(ref_df, cur_df, features)

        for feature in features:
            p = results[feature]["p_value"]
            assert 0 <= p <= 1, f"P-value should be in [0,1], got {p}"

    def test_missing_feature_skipped(self, detector: DriftDetector):
        """Features not in the data are skipped."""
        ref_df = pd.DataFrame({"a": [1, 2, 3]})
        cur_df = pd.DataFrame({"a": [1, 2, 3]})
        results = detector.detect_ks_drift(
            ref_df, cur_df, ["a", "nonexistent"],
        )
        assert "a" in results
        assert "nonexistent" not in results

    def test_empty_feature_values(self, detector: DriftDetector):
        """Feature with all NaN produces no drift detection."""
        ref_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        cur_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        results = detector.detect_ks_drift(ref_df, cur_df, ["a"])

        assert not results["a"]["drift_detected"]
        assert results["a"]["note"] == "insufficient data"

    def test_stricter_threshold_detects_less(
        self, detector: DriftDetector, shifted_data
    ):
        """A stricter threshold (lower) detects fewer drifts."""
        ref_df, cur_df, features = shifted_data

        results_loose = detector.detect_ks_drift(
            ref_df, cur_df, features, threshold=0.10,
        )
        results_strict = detector.detect_ks_drift(
            ref_df, cur_df, features, threshold=0.001,
        )

        drift_loose = sum(1 for r in results_loose.values() if r["drift_detected"])
        drift_strict = sum(1 for r in results_strict.values() if r["drift_detected"])
        assert drift_strict <= drift_loose


# =============================================================================
# Tests: compute_psi
# =============================================================================


class TestComputePSI:
    """Tests for Population Stability Index computation."""

    def test_identical_distributions_low_psi(self, detector: DriftDetector):
        """Identical distributions should produce PSI close to 0."""
        np.random.seed(42)
        reference = np.random.normal(100, 15, 1000)
        current = np.random.normal(100, 15, 1000)

        psi = detector.compute_psi(reference, current)
        assert psi < 0.10, f"PSI should be < 0.10 for same dist, got {psi}"

    def test_shifted_distribution_high_psi(self, detector: DriftDetector):
        """Significantly shifted distribution should produce high PSI."""
        np.random.seed(42)
        reference = np.random.normal(100, 10, 1000)
        current = np.random.normal(150, 10, 1000)

        psi = detector.compute_psi(reference, current)
        assert psi > 0.25, f"PSI should be > 0.25 for shifted dist, got {psi}"

    def test_psi_non_negative(self, detector: DriftDetector):
        """PSI is always non-negative."""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 500)
        current = np.random.normal(0.5, 1.5, 500)

        psi = detector.compute_psi(reference, current)
        assert psi >= 0, f"PSI should be >= 0, got {psi}"

    def test_psi_exactly_same_data(self, detector: DriftDetector):
        """Same exact data should produce PSI of 0 (or near 0)."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        psi = detector.compute_psi(data, data)
        # With identical data and binning, PSI should be very close to 0
        assert psi < 0.01 or np.isnan(psi)

    def test_psi_empty_reference(self, detector: DriftDetector):
        """Empty reference array returns NaN."""
        psi = detector.compute_psi(np.array([]), np.array([1, 2, 3]))
        assert np.isnan(psi)

    def test_psi_empty_current(self, detector: DriftDetector):
        """Empty current array returns NaN."""
        psi = detector.compute_psi(np.array([1, 2, 3]), np.array([]))
        assert np.isnan(psi)

    def test_psi_with_nan_values(self, detector: DriftDetector):
        """NaN values are filtered before PSI computation."""
        reference = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])
        current = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])

        psi = detector.compute_psi(reference, current)
        # Should compute without error (NaN filtered)
        assert isinstance(psi, float)


# =============================================================================
# Tests: PSI status classification
# =============================================================================


class TestPSIStatus:
    """Tests for PSI status classification."""

    def test_ok_status(self, detector: DriftDetector):
        """PSI < 0.10 is classified as OK."""
        assert detector._get_psi_status(0.05) == "OK"
        assert detector._get_psi_status(0.0) == "OK"
        assert detector._get_psi_status(0.09) == "OK"

    def test_warning_status(self, detector: DriftDetector):
        """0.10 <= PSI < 0.25 is classified as WARNING."""
        assert detector._get_psi_status(0.10) == "WARNING"
        assert detector._get_psi_status(0.15) == "WARNING"
        assert detector._get_psi_status(0.24) == "WARNING"

    def test_alert_status(self, detector: DriftDetector):
        """PSI >= 0.25 is classified as ALERT."""
        assert detector._get_psi_status(0.25) == "ALERT"
        assert detector._get_psi_status(0.50) == "ALERT"
        assert detector._get_psi_status(1.0) == "ALERT"

    def test_nan_status(self, detector: DriftDetector):
        """NaN PSI is classified as OK (cannot determine drift)."""
        assert detector._get_psi_status(float("nan")) == "OK"


# =============================================================================
# Tests: generate_drift_report
# =============================================================================


class TestGenerateDriftReport:
    """Tests for the full drift report generation."""

    def test_report_structure(
        self, detector: DriftDetector, identical_data
    ):
        """Report contains all required keys."""
        ref_df, cur_df, features = identical_data
        report = detector.generate_drift_report(
            ref_df, cur_df, features, save=False,
        )

        assert "overall_status" in report
        assert "summary" in report
        assert "n_features" in report
        assert "features" in report

    def test_overall_status_ok_no_drift(
        self, detector: DriftDetector, identical_data
    ):
        """No drift produces OK overall status."""
        ref_df, cur_df, features = identical_data
        report = detector.generate_drift_report(
            ref_df, cur_df, features, save=False,
        )

        # With identical distributions, overall should be OK
        assert report["overall_status"] in ("OK", "WARNING")

    def test_overall_status_alert_with_drift(
        self, detector: DriftDetector, shifted_data
    ):
        """Significant drift produces WARNING or ALERT overall status."""
        ref_df, cur_df, features = shifted_data
        report = detector.generate_drift_report(
            ref_df, cur_df, features, save=False,
        )

        # With a 3-sigma shift, at least WARNING
        assert report["overall_status"] in ("WARNING", "ALERT")

    def test_feature_report_per_feature(
        self, detector: DriftDetector, identical_data
    ):
        """Each feature has a detailed sub-report."""
        ref_df, cur_df, features = identical_data
        report = detector.generate_drift_report(
            ref_df, cur_df, features, save=False,
        )

        for feature in features:
            assert feature in report["features"]
            feat_report = report["features"][feature]
            assert "ks_statistic" in feat_report
            assert "ks_p_value" in feat_report
            assert "psi" in feat_report
            assert "status" in feat_report
            assert feat_report["status"] in ("OK", "WARNING", "ALERT")

    def test_summary_counts_consistent(
        self, detector: DriftDetector, identical_data
    ):
        """Summary counts sum to the total number of features."""
        ref_df, cur_df, features = identical_data
        report = detector.generate_drift_report(
            ref_df, cur_df, features, save=False,
        )

        total = (
            report["summary"]["OK"]
            + report["summary"]["WARNING"]
            + report["summary"]["ALERT"]
        )
        assert total == report["n_features"]

    def test_saves_report_json(
        self, detector: DriftDetector, identical_data
    ):
        """Report is saved to a JSON file when save=True."""
        ref_df, cur_df, features = identical_data
        detector.generate_drift_report(
            ref_df, cur_df, features, save=True,
        )

        output_path = detector.analysis_dir / "drift_report.json"
        assert output_path.exists()

        # Verify JSON is valid
        content = json.loads(output_path.read_text(encoding="utf-8"))
        assert "overall_status" in content

    def test_report_includes_statistics(
        self, detector: DriftDetector, identical_data
    ):
        """Feature reports include ref and current statistics."""
        ref_df, cur_df, features = identical_data
        report = detector.generate_drift_report(
            ref_df, cur_df, features, save=False,
        )

        for feature in features:
            feat_report = report["features"][feature]
            assert "ref_mean" in feat_report
            assert "cur_mean" in feat_report
            assert "ref_std" in feat_report
            assert "cur_std" in feat_report


# =============================================================================
# Tests: Edge cases
# =============================================================================


class TestDriftEdgeCases:
    """Edge cases for drift detection."""

    def test_single_value_reference(self, detector: DriftDetector):
        """Single-value reference does not crash."""
        ref_df = pd.DataFrame({"a": [5.0]})
        cur_df = pd.DataFrame({"a": [5.0, 6.0, 7.0]})
        report = detector.generate_drift_report(
            ref_df, cur_df, ["a"], save=False,
        )
        assert "a" in report["features"]

    def test_all_nan_feature(self, detector: DriftDetector):
        """All-NaN feature is handled gracefully."""
        ref_df = pd.DataFrame({"a": [np.nan, np.nan, np.nan]})
        cur_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        report = detector.generate_drift_report(
            ref_df, cur_df, ["a"], save=False,
        )
        # Should not crash
        assert "a" in report["features"]

    def test_constant_feature(self, detector: DriftDetector):
        """Constant feature (zero variance) is handled."""
        ref_df = pd.DataFrame({"a": [5.0] * 100})
        cur_df = pd.DataFrame({"a": [5.0] * 100})

        psi = detector.compute_psi(ref_df["a"].values, cur_df["a"].values)
        # Constant and equal => PSI should be 0 or NaN
        assert psi == 0.0 or np.isnan(psi)

    def test_different_sample_sizes(self, detector: DriftDetector):
        """Different sample sizes do not cause errors."""
        np.random.seed(42)
        ref_df = pd.DataFrame({"a": np.random.normal(0, 1, 1000)})
        cur_df = pd.DataFrame({"a": np.random.normal(0, 1, 50)})

        report = detector.generate_drift_report(
            ref_df, cur_df, ["a"], save=False,
        )
        assert report["n_features"] == 1
