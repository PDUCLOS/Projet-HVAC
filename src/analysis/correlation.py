# -*- coding: utf-8 -*-
"""
Correlation studies — Phase 3.2.
====================================

This module analyzes correlations between features and target variables
to guide feature selection in Phase 4 (ML modeling).

Analyses produced:
    1. Full correlation matrix (heatmap)
    2. Top correlations with each target variable
    3. Correlations by department (geographic stability)
    4. Correlations by season (seasonal variations)
    5. Multicollinearity analysis (VIF)
    6. Correlation matrix for lag and rolling features

Usage:
    >>> from src.analysis.correlation import CorrelationAnalyzer
    >>> corr = CorrelationAnalyzer(config)
    >>> corr.run_full_correlation()

    # Or via CLI
    python -m src.pipeline eda
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Correlation analyzer for the HVAC dataset.

    Attributes:
        config: Project configuration.
        df: Features dataset DataFrame.
        output_dir: Output directory for charts.
    """

    DEPT_NAMES = {
        "01": "Ain", "07": "Ardèche", "26": "Drôme", "38": "Isère",
        "42": "Loire", "69": "Rhône", "73": "Savoie", "74": "Haute-Savoie",
    }

    # Base features (excluding lags/rolling) for the main matrix
    BASE_FEATURES = [
        "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
        "nb_dpe_classe_ab", "pct_pac", "pct_clim",
        "temp_mean", "temp_max", "temp_min", "hdd_sum", "cdd_sum",
        "precipitation_sum", "nb_jours_canicule", "nb_jours_gel",
        "confiance_menages", "climat_affaires_indus", "climat_affaires_bat",
        "ipi_manufacturing", "ipi_hvac_c28", "ipi_hvac_c2825",
        "month", "quarter", "is_heating", "is_cooling",
    ]

    TARGET_COL = "nb_installations_pac"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.output_dir = Path("data/analysis/figures")
        self.report_dir = Path("data/analysis")
        self.df: Optional[pd.DataFrame] = None

        sns.set_theme(style="whitegrid", font_scale=1.0)
        plt.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        })

    def _load_data(self) -> pd.DataFrame:
        """Load the features dataset."""
        filepath = self.config.features_data_dir / "hvac_features_dataset.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Features dataset not found: {filepath}")

        df = pd.read_csv(filepath)
        df["dept"] = df["dept"].astype(str).str.zfill(2)
        df["date"] = pd.to_datetime(df["date_id"].astype(str), format="%Y%m")
        return df

    def _ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, name: str) -> Path:
        path = self.output_dir / f"{name}.png"
        plt.savefig(path)
        plt.close()
        logger.info("  → %s", path)
        return path

    # ------------------------------------------------------------------
    # 1. Full correlation matrix
    # ------------------------------------------------------------------

    def plot_correlation_matrix(self) -> Path:
        """Correlation matrix for base features (excluding lags).

        Returns:
            Path to the saved chart.
        """
        df = self.df
        cols = [c for c in self.BASE_FEATURES if c in df.columns]
        corr_matrix = df[cols].corr()

        # Mask the upper half (redundant)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size": 8},
        )
        ax.set_title("Correlation matrix — Base features", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        return self._save_fig("20_correlation_matrix_base")

    # ------------------------------------------------------------------
    # 2. Top correlations with target
    # ------------------------------------------------------------------

    def plot_top_correlations(self, target: str = None, n_top: int = 20) -> Path:
        """Bar chart of the strongest correlations with the target.

        Args:
            target: Name of the target column (default: nb_installations_pac).
            n_top: Number of features to display.

        Returns:
            Path to the saved chart.
        """
        target = target or self.TARGET_COL
        df = self.df
        numeric = df.select_dtypes(include=[np.number])

        if target not in numeric.columns:
            logger.warning("Target '%s' not found in dataset.", target)
            return Path()

        corr = numeric.corrwith(numeric[target])
        corr = corr.drop(target, errors="ignore")

        # Exclude auxiliary columns
        exclude = ["date_id", "year_trend", "n_valid_features", "pct_valid_features"]
        corr = corr.drop(exclude, errors="ignore")

        # Top N by absolute value
        top = corr.abs().sort_values(ascending=False).head(n_top)
        top_signed = corr.loc[top.index]

        fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.35)))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_signed.values]
        ax.barh(range(len(top_signed)), top_signed.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_signed)))
        ax.set_yticklabels(top_signed.index, fontsize=9)
        ax.set_xlabel("Correlation coefficient (Pearson)")
        ax.set_title(f"Top {n_top} correlations with {target}", fontsize=14)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        # Annotations
        for i, (val, feat) in enumerate(zip(top_signed.values, top_signed.index)):
            ax.text(val + 0.01 * np.sign(val), i, f"{val:.3f}",
                    va="center", fontsize=8)

        return self._save_fig(f"21_top_correlations_{target}")

    # ------------------------------------------------------------------
    # 3. Correlations by department
    # ------------------------------------------------------------------

    def plot_correlation_by_dept(self) -> Path:
        """Heatmap of feature-to-PAC correlations by department.

        Allows checking whether relationships are geographically stable.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        target = self.TARGET_COL
        features = [
            "temp_mean", "hdd_sum", "cdd_sum", "precipitation_sum",
            "confiance_menages", "ipi_hvac_c28",
            "nb_jours_canicule", "nb_jours_gel", "month",
        ]
        features = [f for f in features if f in df.columns]

        # Correlation by department
        corr_by_dept = {}
        for dept in sorted(df["dept"].unique()):
            subset = df[df["dept"] == dept]
            if len(subset) > 5 and target in subset.columns:
                corr = subset[features].corrwith(subset[target])
                corr_by_dept[f"{dept} — {self.DEPT_NAMES.get(dept, dept)}"] = corr

        if not corr_by_dept:
            logger.warning("Not enough data for correlations by department.")
            return Path()

        corr_df = pd.DataFrame(corr_by_dept).T

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title(
            f"Feature correlations with {target} by department",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)

        return self._save_fig("22_correlation_by_dept")

    # ------------------------------------------------------------------
    # 4. Correlations by season
    # ------------------------------------------------------------------

    def plot_correlation_by_season(self) -> Path:
        """Heatmap of correlations by season (winter/summer/etc.).

        Returns:
            Path to the saved chart.
        """
        df = self.df.copy()
        target = self.TARGET_COL

        df["saison"] = df["month"].map(lambda m: (
            "1_Winter" if m in [12, 1, 2] else
            "2_Spring" if m in [3, 4, 5] else
            "3_Summer" if m in [6, 7, 8] else "4_Autumn"
        ))

        features = [
            "temp_mean", "hdd_sum", "cdd_sum", "precipitation_sum",
            "confiance_menages", "ipi_hvac_c28",
            "nb_jours_canicule", "nb_jours_gel",
        ]
        features = [f for f in features if f in df.columns]

        corr_by_season = {}
        for saison in sorted(df["saison"].unique()):
            subset = df[df["saison"] == saison]
            if len(subset) > 5 and target in subset.columns:
                corr = subset[features].corrwith(subset[target])
                corr_by_season[saison.split("_")[1]] = corr

        if not corr_by_season:
            return Path()

        corr_df = pd.DataFrame(corr_by_season).T

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 10},
        )
        ax.set_title(
            f"Feature correlations with {target} by season",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)

        return self._save_fig("23_correlation_by_season")

    # ------------------------------------------------------------------
    # 5. Multicollinearity analysis
    # ------------------------------------------------------------------

    def plot_multicollinearity(self) -> Path:
        """Identify pairs of highly correlated features (|r| > 0.8).

        Returns:
            Path to the saved chart.
        """
        df = self.df
        features = [c for c in self.BASE_FEATURES if c in df.columns]
        corr = df[features].corr()

        # Extract pairs with |r| > 0.8 (excluding diagonal)
        pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                r = corr.iloc[i, j]
                if abs(r) > 0.8:
                    pairs.append((features[i], features[j], r))

        if not pairs:
            logger.info("  No pair with |r| > 0.8 found.")
            return Path()

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        fig, ax = plt.subplots(figsize=(12, max(5, len(pairs) * 0.5)))
        labels = [f"{a} ↔ {b}" for a, b, _ in pairs]
        values = [r for _, _, r in pairs]
        colors = ["#e74c3c" if r > 0 else "#3498db" for r in values]

        ax.barh(range(len(pairs)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Correlation coefficient")
        ax.set_title("Multicollinear pairs (|r| > 0.8)", fontsize=14)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        for i, val in enumerate(values):
            ax.text(val + 0.01 * np.sign(val), i, f"{val:.3f}",
                    va="center", fontsize=9)

        return self._save_fig("24_multicollinearity")

    # ------------------------------------------------------------------
    # 6. Lag correlations
    # ------------------------------------------------------------------

    def plot_lag_correlations(self) -> Path:
        """Heatmap of correlations between different lags and the target.

        Helps determine which lag horizons are the most predictive.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        target = self.TARGET_COL

        # Find all lag columns
        lag_cols = [c for c in df.columns if "_lag_" in c]
        if not lag_cols:
            logger.info("  No lag column found.")
            return Path()

        # Organize by variable and horizon
        lag_data = {}
        for col in lag_cols:
            # Format: variable_lag_Xm
            parts = col.rsplit("_lag_", 1)
            if len(parts) == 2:
                var = parts[0]
                horizon = parts[1]
                if var not in lag_data:
                    lag_data[var] = {}
                if target in df.columns:
                    lag_data[var][horizon] = df[col].corr(df[target])

        if not lag_data:
            return Path()

        # Create a DataFrame for the heatmap
        lag_df = pd.DataFrame(lag_data).T
        # Sort columns by horizon
        col_order = sorted(lag_df.columns, key=lambda x: int(x.replace("m", "")))
        lag_df = lag_df[col_order]

        fig, ax = plt.subplots(figsize=(10, max(5, len(lag_df) * 0.6)))
        sns.heatmap(
            lag_df, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Lag correlations with {target}", fontsize=14)
        ax.set_xlabel("Horizon du lag")
        ax.set_ylabel("Variable")

        return self._save_fig("25_lag_correlations")

    # ------------------------------------------------------------------
    # Correlation report
    # ------------------------------------------------------------------

    def generate_correlation_report(self) -> Path:
        """Generate a text-based correlation report.

        Returns:
            Path to the saved report.
        """
        df = self.df
        target = self.TARGET_COL
        report_path = self.report_dir / "correlation_report.txt"

        lines = []
        lines.append("=" * 70)
        lines.append("  CORRELATION REPORT — HVAC Market Analysis")
        lines.append("=" * 70)

        # Top 20 correlations with the target
        numeric = df.select_dtypes(include=[np.number])
        if target in numeric.columns:
            corr = numeric.corrwith(numeric[target])
            corr = corr.drop(target, errors="ignore")
            top = corr.abs().sort_values(ascending=False).head(20)

            lines.append(f"\n\n1. TOP 20 CORRELATIONS WITH {target}")
            lines.append("-" * 50)
            for feat in top.index:
                val = corr[feat]
                lines.append(f"  {'+' if val > 0 else '-'}{abs(val):.3f}  {feat}")

        # Multicollinear pairs
        features = [c for c in self.BASE_FEATURES if c in df.columns]
        corr_matrix = df[features].corr()
        pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.8:
                    pairs.append((features[i], features[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        lines.append(f"\n\n2. MULTICOLLINEAR PAIRS (|r| > 0.8) — {len(pairs)} found")
        lines.append("-" * 50)
        for a, b, r in pairs:
            lines.append(f"  {r:+.3f}  {a} ↔ {b}")

        # Recommendations
        lines.append("\n\n3. RECOMMENDATIONS FOR MODELING")
        lines.append("-" * 50)
        if pairs:
            lines.append("  ⚠ Multicollinearity detected:")
            seen = set()
            for a, b, r in pairs:
                if a not in seen and b not in seen:
                    lines.append(f"    → Keep {a}, consider removing {b} (r={r:.3f})")
                    seen.add(b)

        lines.append("  ✓ Use Ridge or Lasso to handle collinearity")
        lines.append("  ✓ LightGBM is naturally robust to collinearity")
        lines.append("  ✓ Verify correlation stability across departments")

        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("Correlation report saved: %s", report_path)

        return report_path

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_full_correlation(self) -> Dict[str, Any]:
        """Execute all correlation analyses.

        Returns:
            Dictionary with the paths of generated files.
        """
        logger.info("=" * 60)
        logger.info("  Phase 3.2 — Correlation analysis")
        logger.info("=" * 60)

        self._ensure_dirs()
        self.df = self._load_data()

        results = {"figures": []}

        logger.info("Correlation matrix...")
        results["figures"].append(str(self.plot_correlation_matrix()))

        logger.info("Top correlations vs target...")
        results["figures"].append(str(self.plot_top_correlations()))

        logger.info("Correlations by department...")
        fig = self.plot_correlation_by_dept()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Correlations by season...")
        fig = self.plot_correlation_by_season()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Multicollinearity analysis...")
        fig = self.plot_multicollinearity()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Lag correlations...")
        fig = self.plot_lag_correlations()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Correlation report...")
        report = self.generate_correlation_report()
        results["report"] = str(report)

        logger.info(
            "Correlation analysis complete: %d charts.",
            len(results["figures"]),
        )
        return results
