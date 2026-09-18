# -*- coding: utf-8 -*-
"""
Data Drift Detection Module — Monitor feature distribution shifts.
====================================================================

Detects distribution changes between a reference (training) dataset
and a current (production/new) dataset. Essential for model monitoring
and retraining decisions.

Methods:
    1. Kolmogorov-Smirnov (KS) test — Non-parametric distribution comparison
    2. Population Stability Index (PSI) — Binned distribution divergence

PSI interpretation thresholds (industry standard):
    - PSI < 0.10   — OK:      No significant drift
    - 0.10 <= PSI < 0.25 — WARNING:  Moderate drift, investigate
    - PSI >= 0.25  — ALERT:   Significant drift, consider retraining

Usage:
    >>> from src.models.data_drift import DriftDetector
    >>> detector = DriftDetector()
    >>> report = detector.generate_drift_report(ref_df, curr_df, features)
    >>> drifted = [f for f, r in report["features"].items() if r["status"] == "ALERT"]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats


class DriftDetector:
    """Detect data drift between reference and current datasets.

    Uses statistical tests (KS test, PSI) to identify features
    whose distributions have shifted significantly, which may
    degrade model performance.

    Attributes:
        logger: Module logger instance.
        analysis_dir: Output directory for drift reports.
    """

    # PSI interpretation thresholds
    PSI_OK_THRESHOLD = 0.10
    PSI_WARNING_THRESHOLD = 0.25

    def __init__(self, analysis_dir: Union[str, Path] = "data/analysis") -> None:
        """Initialize the drift detector.

        Args:
            analysis_dir: Directory where drift reports are saved.
        """
        self.logger = logging.getLogger("models.data_drift")
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def detect_ks_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str],
        threshold: float = 0.05,
    ) -> Dict[str, Dict[str, Any]]:
        """Kolmogorov-Smirnov test for distribution shift per feature.

        The KS test compares two empirical distributions and returns
        a test statistic (max distance between CDFs) and a p-value.
        Drift is detected when p-value < threshold.

        Args:
            reference_df: Reference (training) dataset.
            current_df: Current (new) dataset.
            features: List of feature names to test.
            threshold: P-value threshold for drift detection.

        Returns:
            Dictionary {feature_name: {statistic, p_value, drift_detected}}.
        """
        self.logger.info(
            "KS drift detection on %d features (threshold=%.3f)...",
            len(features), threshold,
        )

        results = {}
        n_drift = 0

        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                self.logger.warning("  Feature '%s' missing from data, skipped", feature)
                continue

            ref_values = reference_df[feature].dropna().values
            cur_values = current_df[feature].dropna().values

            if len(ref_values) == 0 or len(cur_values) == 0:
                results[feature] = {
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "drift_detected": False,
                    "note": "insufficient data",
                }
                continue

            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
            drift_detected = bool(p_value < threshold)

            if drift_detected:
                n_drift += 1

            results[feature] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": drift_detected,
            }

        self.logger.info(
            "  KS test: %d/%d features show drift (p < %.3f)",
            n_drift, len(results), threshold,
        )

        return results

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """Compute Population Stability Index (PSI) for a single feature.

        PSI measures how much the distribution of a variable has shifted
        between two samples. It uses binning based on the reference
        distribution's quantiles.

        Formula:
            PSI = sum((current_pct - reference_pct) * ln(current_pct / reference_pct))

        Args:
            reference: Reference sample values.
            current: Current sample values.
            bins: Number of bins for discretization.

        Returns:
            PSI value (float). Lower is better (0 = identical distributions).
        """
        # Remove NaN
        reference = np.array(reference, dtype=float)
        current = np.array(current, dtype=float)
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        if len(reference) == 0 or len(current) == 0:
            return float("nan")

        # Create bins from reference quantiles
        quantiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(reference, quantiles)

        # Ensure unique bin edges (handle constant features)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Constant feature: PSI is 0 if both constant and equal
            if np.std(current) == 0 and np.mean(reference) == np.mean(current):
                return 0.0
            return float("nan")

        # Compute proportions per bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / len(reference)
        cur_pct = cur_counts / len(current)

        # Replace zeros to avoid log(0) and division by zero
        epsilon = 1e-6
        ref_pct = np.clip(ref_pct, epsilon, None)
        cur_pct = np.clip(cur_pct, epsilon, None)

        # PSI formula
        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

        return psi

    def _get_psi_status(self, psi_value: float) -> str:
        """Classify a PSI value into OK / WARNING / ALERT.

        Args:
            psi_value: PSI value for a feature.

        Returns:
            Status string: "OK", "WARNING", or "ALERT".
        """
        if np.isnan(psi_value):
            return "OK"
        if psi_value < self.PSI_OK_THRESHOLD:
            return "OK"
        if psi_value < self.PSI_WARNING_THRESHOLD:
            return "WARNING"
        return "ALERT"

    def generate_drift_report(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str],
        ks_threshold: float = 0.05,
        psi_bins: int = 10,
        save: bool = True,
    ) -> Dict[str, Any]:
        """Generate a comprehensive drift report for all features.

        Combines KS test results and PSI values into a unified report
        with per-feature status (OK / WARNING / ALERT).

        Overall status:
            - OK: no feature has an ALERT
            - WARNING: at least one WARNING but no ALERT
            - ALERT: at least one feature has ALERT status

        Args:
            reference_df: Reference (training) dataset.
            current_df: Current (new) dataset.
            features: List of feature names to analyze.
            ks_threshold: P-value threshold for KS test.
            psi_bins: Number of bins for PSI computation.
            save: Whether to save the report to JSON.

        Returns:
            Dictionary with:
                - overall_status: "OK", "WARNING", or "ALERT"
                - summary: counts per status
                - features: {feature: {ks_*, psi, status}}
                - n_features: total features analyzed
        """
        self.logger.info("=" * 60)
        self.logger.info("  DRIFT DETECTION REPORT")
        self.logger.info(
            "  Reference: %d rows, Current: %d rows",
            len(reference_df), len(current_df),
        )
        self.logger.info("  Features: %d", len(features))
        self.logger.info("=" * 60)

        # Run KS tests
        ks_results = self.detect_ks_drift(
            reference_df, current_df, features, ks_threshold,
        )

        # Compute PSI per feature and determine status
        feature_reports = {}
        status_counts = {"OK": 0, "WARNING": 0, "ALERT": 0}

        for feature in features:
            if feature not in reference_df.columns or feature not in current_df.columns:
                continue

            ref_values = reference_df[feature].dropna().values
            cur_values = current_df[feature].dropna().values

            psi_value = self.compute_psi(ref_values, cur_values, bins=psi_bins)
            psi_status = self._get_psi_status(psi_value)

            ks_data = ks_results.get(feature, {})
            ks_drift = ks_data.get("drift_detected", False)

            # Combined status: worst of PSI and KS
            if psi_status == "ALERT" or (ks_drift and psi_status != "OK"):
                combined_status = "ALERT"
            elif psi_status == "WARNING" or ks_drift:
                combined_status = "WARNING"
            else:
                combined_status = "OK"

            status_counts[combined_status] += 1

            feature_reports[feature] = {
                "ks_statistic": ks_data.get("statistic", float("nan")),
                "ks_p_value": ks_data.get("p_value", float("nan")),
                "ks_drift_detected": ks_drift,
                "psi": round(psi_value, 6) if not np.isnan(psi_value) else None,
                "psi_status": psi_status,
                "status": combined_status,
                "ref_mean": float(np.mean(ref_values)) if len(ref_values) > 0 else None,
                "cur_mean": float(np.mean(cur_values)) if len(cur_values) > 0 else None,
                "ref_std": float(np.std(ref_values)) if len(ref_values) > 0 else None,
                "cur_std": float(np.std(cur_values)) if len(cur_values) > 0 else None,
            }

        # Overall status
        if status_counts["ALERT"] > 0:
            overall_status = "ALERT"
        elif status_counts["WARNING"] > 0:
            overall_status = "WARNING"
        else:
            overall_status = "OK"

        report = {
            "overall_status": overall_status,
            "summary": status_counts,
            "n_features": len(feature_reports),
            "ks_threshold": ks_threshold,
            "psi_bins": psi_bins,
            "features": feature_reports,
        }

        # Log summary
        self.logger.info(
            "  Overall status: %s", overall_status,
        )
        self.logger.info(
            "  OK=%d, WARNING=%d, ALERT=%d",
            status_counts["OK"],
            status_counts["WARNING"],
            status_counts["ALERT"],
        )

        # Log features with issues
        for feat, data in feature_reports.items():
            if data["status"] != "OK":
                self.logger.warning(
                    "  %s: %s (PSI=%.4f, KS p=%.4f)",
                    feat, data["status"],
                    data.get("psi") or 0.0,
                    data.get("ks_p_value") or 0.0,
                )

        if save:
            self._save_report(report)

        return report

    def _save_report(self, report: Dict[str, Any]) -> Path:
        """Save drift report to JSON file.

        Args:
            report: Drift report dictionary.

        Returns:
            Path of the saved file.
        """
        output_path = self.analysis_dir / "drift_report.json"

        # Make serializable (handle NaN)
        def _clean(obj: Any) -> Any:
            if isinstance(obj, float) and np.isnan(obj):
                return None
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            return obj

        clean_report = _clean(report)
        output_path.write_text(
            json.dumps(clean_report, indent=2, default=str),
            encoding="utf-8",
        )
        self.logger.info("Drift report saved -> %s", output_path)
        return output_path
