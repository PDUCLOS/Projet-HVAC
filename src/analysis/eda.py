# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) — Phase 3.1.
==============================================

This module automatically generates all visualizations and
descriptive statistics for the HVAC dataset. Charts are
saved in data/analysis/figures/.

Analysis categories:
    1. Overview          — shape, dtypes, NaN, descriptive statistics
    2. Distributions     — histograms of target variables and key features
    3. Time series       — monthly evolution of heat pump/AC installations
    4. Seasonality       — boxplots by month, month x department heatmap
    5. Geography         — department comparisons
    6. Relationships     — scatter plots features vs target

Usage:
    >>> from src.analysis.eda import EDAAnalyzer
    >>> eda = EDAAnalyzer(config)
    >>> eda.run_full_eda()

    # Or via CLI
    python -m src.pipeline eda
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file generation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """Exploratory analysis generator for the HVAC dataset.

    Attributes:
        config: Project configuration (ProjectConfig).
        df: DataFrame loaded from the features dataset.
        output_dir: Output directory for charts.
        dept_names: Department code → name mapping.
    """

    # Consistent color palette for departments
    DEPT_COLORS = {
        "01": "#1f77b4", "07": "#ff7f0e", "26": "#2ca02c", "38": "#d62728",
        "42": "#9467bd", "69": "#8c564b", "73": "#e377c2", "74": "#7f7f7f",
    }

    DEPT_NAMES = {
        "01": "Ain", "07": "Ardèche", "26": "Drôme", "38": "Isère",
        "42": "Loire", "69": "Rhône", "73": "Savoie", "74": "Haute-Savoie",
    }

    # Target variables
    TARGET_COLS = [
        "nb_dpe_total", "nb_installations_pac",
        "nb_installations_clim", "nb_dpe_classe_ab",
    ]

    # Key features for analysis
    KEY_FEATURES = [
        "temp_mean", "hdd_sum", "cdd_sum", "precipitation_sum",
        "confiance_menages", "ipi_hvac_c28", "nb_jours_canicule", "nb_jours_gel",
    ]

    def __init__(self, config: Any) -> None:
        """Initialize the EDA analyzer.

        Args:
            config: ProjectConfig instance.
        """
        self.config = config
        self.output_dir = Path("data/analysis/figures")
        self.report_dir = Path("data/analysis")
        self.df: Optional[pd.DataFrame] = None

        # Matplotlib style
        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "figure.figsize": (14, 8),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
        })

    # Minimum columns required for any EDA analysis
    REQUIRED_COLS = {"dept", "date_id"}

    def _load_data(self) -> pd.DataFrame:
        """Load the features dataset and validate required columns.

        Returns:
            DataFrame of the features dataset.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns (dept, date_id) are missing.
        """
        filepath = self.config.features_data_dir / "hvac_features_dataset.csv"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Features dataset not found: {filepath}. "
                f"Run 'python -m src.pipeline process' first."
            )

        df = pd.read_csv(filepath)

        # Validate required columns
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"Features dataset is missing required columns: {missing}. "
                f"Available: {list(df.columns)[:20]}..."
            )

        df["dept"] = df["dept"].astype(str).str.zfill(2)
        df["date"] = pd.to_datetime(df["date_id"].astype(str), format="%Y%m")

        logger.info("Dataset loaded: %d rows × %d columns", len(df), len(df.columns))
        return df

    def _ensure_dirs(self) -> None:
        """Create output directories if necessary."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, name: str) -> Path:
        """Save the current figure and close it.

        Args:
            name: File name (without extension).

        Returns:
            Path to the saved file.
        """
        path = self.output_dir / f"{name}.png"
        plt.savefig(path)
        plt.close()
        logger.info("  → %s", path)
        return path

    # ------------------------------------------------------------------
    # 1. Overview
    # ------------------------------------------------------------------

    def overview(self) -> Dict[str, Any]:
        """Generate global descriptive statistics.

        Gracefully handles missing columns by reporting 'N/A' for
        unavailable metrics instead of crashing.

        Returns:
            Dictionary with key dataset metrics.
        """
        df = self.df
        logger.info("=== OVERVIEW ===")

        stats = {
            "shape": df.shape,
            "nan_pct": round(df.isna().mean().mean() * 100, 2),
        }

        if "dept" in df.columns:
            stats["n_departments"] = df["dept"].nunique()
            stats["departments"] = sorted(df["dept"].unique().tolist())

        if "date" in df.columns:
            stats["date_range"] = f"{df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}"

        if "date_id" in df.columns:
            stats["n_months"] = df["date_id"].nunique()

        if "nb_dpe_total" in df.columns:
            stats["total_dpe"] = int(df["nb_dpe_total"].sum())

        if "nb_installations_pac" in df.columns:
            stats["total_pac"] = int(df["nb_installations_pac"].sum())

        if "nb_installations_pac" in df.columns and "nb_dpe_total" in df.columns:
            total_dpe = df["nb_dpe_total"].sum()
            if total_dpe > 0:
                stats["pct_pac_global"] = round(
                    df["nb_installations_pac"].sum() / total_dpe * 100, 2
                )

        for key, val in stats.items():
            logger.info("  %s: %s", key, val)

        return stats

    def plot_nan_heatmap(self) -> Path:
        """Heatmap of missing values by column.

        Returns:
            Path to the saved chart.
        """
        df = self.df

        # Keep only columns with at least 1 NaN
        nan_cols = df.columns[df.isna().any()].tolist()
        if not nan_cols:
            logger.info("  No missing values — NaN heatmap skipped.")
            return Path()

        # Calculate % of NaN per column
        nan_pct = df[nan_cols].isna().mean().sort_values(ascending=False) * 100

        fig, ax = plt.subplots(figsize=(12, max(4, len(nan_cols) * 0.4)))
        bars = ax.barh(range(len(nan_pct)), nan_pct.values, color="#e74c3c", alpha=0.8)
        ax.set_yticks(range(len(nan_pct)))
        ax.set_yticklabels(nan_pct.index, fontsize=9)
        ax.set_xlabel("% Missing Values")
        ax.set_title("Missing Values by Feature")
        ax.invert_yaxis()

        # Add values on the bars
        for bar, val in zip(bars, nan_pct.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)

        return self._save_fig("01_nan_heatmap")

    # ------------------------------------------------------------------
    # 2. Distributions
    # ------------------------------------------------------------------

    def plot_target_distributions(self) -> Path:
        """Histograms of the 4 target variables.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        targets = [c for c in self.TARGET_COLS if c in df.columns]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Target Variable Distributions", fontsize=16, y=1.02)

        for ax, col in zip(axes.flat, targets):
            data = df[col].dropna()
            ax.hist(data, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
            ax.axvline(data.mean(), color="#e74c3c", linestyle="--", label=f"Mean: {data.mean():.0f}")
            ax.axvline(data.median(), color="#2ecc71", linestyle="--", label=f"Median: {data.median():.0f}")
            ax.set_title(col)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=9)

        plt.tight_layout()
        return self._save_fig("02_target_distributions")

    def plot_feature_distributions(self) -> Path:
        """Histograms of key features (weather + economy).

        Returns:
            Path to the saved chart.
        """
        df = self.df
        features = [c for c in self.KEY_FEATURES if c in df.columns]
        n = len(features)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        fig.suptitle("Key Feature Distributions", fontsize=16, y=1.02)

        for i, (ax, col) in enumerate(zip(axes.flat, features)):
            data = df[col].dropna()
            ax.hist(data, bins=25, color="#9b59b6", alpha=0.7, edgecolor="white")
            ax.set_title(col, fontsize=11)
            ax.tick_params(labelsize=9)

        # Hide empty axes
        for j in range(i + 1, len(axes.flat)):
            axes.flat[j].set_visible(False)

        plt.tight_layout()
        return self._save_fig("03_feature_distributions")

    # ------------------------------------------------------------------
    # 3. Time series
    # ------------------------------------------------------------------

    def plot_timeseries_pac(self) -> Path:
        """Monthly evolution of heat pump installations by department.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        for col in ("dept", "date", "nb_installations_pac"):
            if col not in df.columns:
                logger.warning("Skipping timeseries PAC: '%s' column missing", col)
                return Path()

        fig, ax = plt.subplots(figsize=(16, 8))

        for dept in sorted(df["dept"].unique()):
            mask = df["dept"] == dept
            subset = df[mask].sort_values("date")
            ax.plot(
                subset["date"], subset["nb_installations_pac"],
                color=self.DEPT_COLORS.get(dept, "#333"),
                label=f"{dept} — {self.DEPT_NAMES.get(dept, dept)}",
                linewidth=1.5, alpha=0.85,
            )

        ax.set_title("Heat Pump Installations by Month and Department", fontsize=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("Heat Pump Installation Count")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        return self._save_fig("04_timeseries_pac")

    def plot_timeseries_dpe_total(self) -> Path:
        """Monthly evolution of total DPE count by department.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        for col in ("dept", "date", "nb_dpe_total"):
            if col not in df.columns:
                logger.warning("Skipping timeseries DPE total: '%s' column missing", col)
                return Path()

        fig, ax = plt.subplots(figsize=(16, 8))

        for dept in sorted(df["dept"].unique()):
            mask = df["dept"] == dept
            subset = df[mask].sort_values("date")
            ax.plot(
                subset["date"], subset["nb_dpe_total"],
                color=self.DEPT_COLORS.get(dept, "#333"),
                label=f"{dept} — {self.DEPT_NAMES.get(dept, dept)}",
                linewidth=1.5, alpha=0.85,
            )

        ax.set_title("Total DPE by Month and Department", fontsize=15)
        ax.set_xlabel("Date")
        ax.set_ylabel("DPE Count")
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        return self._save_fig("05_timeseries_dpe_total")

    def plot_timeseries_aura_aggregated(self) -> Path:
        """Aggregated time series: heat pumps + AC + total DPE.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        required = {"date", "nb_dpe_total", "nb_installations_pac", "nb_installations_clim"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("Skipping aggregated timeseries: missing columns %s", missing)
            return Path()

        agg = df.groupby("date").agg({
            "nb_dpe_total": "sum",
            "nb_installations_pac": "sum",
            "nb_installations_clim": "sum",
        }).reset_index().sort_values("date")

        fig, ax1 = plt.subplots(figsize=(16, 8))

        ax1.bar(agg["date"], agg["nb_dpe_total"], width=25, color="#bdc3c7",
                alpha=0.5, label="DPE total")
        ax1.set_ylabel("DPE total (barres)", color="#7f8c8d")
        ax1.tick_params(axis="y", labelcolor="#7f8c8d")

        ax2 = ax1.twinx()
        ax2.plot(agg["date"], agg["nb_installations_pac"], color="#e74c3c",
                 linewidth=2, marker="o", markersize=3, label="PAC")
        ax2.plot(agg["date"], agg["nb_installations_clim"], color="#3498db",
                 linewidth=2, marker="s", markersize=3, label="Climatisation")
        ax2.set_ylabel("Heat Pump / AC Installations (lines)")

        ax1.set_title("France — DPE Volume and Heat Pump/AC Installations", fontsize=15)
        ax1.set_xlabel("Date")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

        plt.xticks(rotation=45)
        return self._save_fig("06_timeseries_aura_aggregated")

    # ------------------------------------------------------------------
    # 4. Seasonality
    # ------------------------------------------------------------------

    def plot_seasonality_boxplots(self) -> Path:
        """Boxplots of heat pump and AC installations by month (seasonality).

        Returns:
            Path to the saved chart.
        """
        df = self.df
        required = {"month", "nb_installations_pac", "nb_installations_clim"}
        available = required & set(df.columns)
        if "month" not in df.columns:
            logger.warning("Skipping seasonality boxplots: 'month' column missing")
            return Path()

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Heat pumps by month
        if "nb_installations_pac" in df.columns:
            sns.boxplot(data=df, x="month", y="nb_installations_pac",
                        palette="coolwarm", ax=axes[0])
            axes[0].set_title("Seasonality of Heat Pump Installations")
        axes[0].set_xlabel("Month")
        axes[0].set_ylabel("Heat Pump Installations")

        # Air conditioning by month
        if "nb_installations_clim" in df.columns:
            sns.boxplot(data=df, x="month", y="nb_installations_clim",
                        palette="YlOrRd", ax=axes[1])
            axes[1].set_title("Seasonality of AC Installations")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("AC Installations")

        plt.tight_layout()
        return self._save_fig("07_seasonality_boxplots")

    def plot_heatmap_dept_month(self) -> Path:
        """Heatmap of heat pump installations: department x month.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        for col in ("nb_installations_pac", "dept", "month"):
            if col not in df.columns:
                logger.warning("Skipping heatmap dept×month: '%s' column missing", col)
                return Path()

        pivot = df.pivot_table(
            values="nb_installations_pac",
            index="dept",
            columns="month",
            aggfunc="mean",
        )
        # Rename index with department names
        pivot.index = [f"{d} — {self.DEPT_NAMES.get(d, d)}" for d in pivot.index]

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(
            pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
        )
        ax.set_title("Average Heat Pump Installations: Department × Month", fontsize=14)
        ax.set_xlabel("Month")
        ax.set_ylabel("Department")

        return self._save_fig("08_heatmap_dept_month")

    # ------------------------------------------------------------------
    # 5. Geography — Department comparison
    # ------------------------------------------------------------------

    def plot_dept_comparison(self) -> Path:
        """Volume comparison by department (stacked bars).

        Returns:
            Path to the saved chart.
        """
        df = self.df
        required = {"dept", "nb_dpe_total", "nb_installations_pac", "nb_installations_clim"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("Skipping dept comparison: missing columns %s", missing)
            return Path()

        agg = df.groupby("dept").agg({
            "nb_dpe_total": "sum",
            "nb_installations_pac": "sum",
            "nb_installations_clim": "sum",
        }).reset_index()

        agg["dept_label"] = agg["dept"].map(
            lambda d: f"{d}\n{self.DEPT_NAMES.get(d, d)}"
        )
        agg = agg.sort_values("nb_dpe_total", ascending=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle("Department Comparison", fontsize=16, y=1.02)

        for ax, col, color, title in zip(
            axes,
            ["nb_dpe_total", "nb_installations_pac", "nb_installations_clim"],
            ["#3498db", "#e74c3c", "#2ecc71"],
            ["DPE total", "Installations PAC", "Installations Clim"],
        ):
            ax.barh(agg["dept_label"], agg[col], color=color, alpha=0.8)
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Total Volume")
            # Add values
            for i, v in enumerate(agg[col]):
                ax.text(v + agg[col].max() * 0.01, i, f"{v:,.0f}",
                        va="center", fontsize=9)

        plt.tight_layout()
        return self._save_fig("09_dept_comparison")

    def plot_pct_pac_by_dept(self) -> Path:
        """Average heat pump rate by department.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        if "pct_pac" not in df.columns or "dept" not in df.columns:
            logger.warning("Skipping pct_pac_by_dept: 'pct_pac' or 'dept' missing")
            return Path()

        pct = df.groupby("dept")["pct_pac"].mean().sort_values(ascending=True)
        labels = [f"{d} — {self.DEPT_NAMES.get(d, d)}" for d in pct.index]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(labels, pct.values, color=[self.DEPT_COLORS.get(d, "#333") for d in pct.index])
        ax.set_xlabel("Average Heat Pump Rate (%)")
        ax.set_title("Average Heat Pump Rate by Department", fontsize=14)

        for bar, val in zip(bars, pct.values):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}%", va="center", fontsize=10)

        return self._save_fig("10_pct_pac_by_dept")

    # ------------------------------------------------------------------
    # 6. Feature / target relationships
    # ------------------------------------------------------------------

    def plot_scatter_features_vs_target(self) -> Path:
        """Scatter plots of key features vs nb_installations_pac.

        Returns:
            Path to the saved chart.
        """
        df = self.df
        if "nb_installations_pac" not in df.columns:
            logger.warning("Skipping scatter features: 'nb_installations_pac' missing")
            return Path()

        features = [c for c in self.KEY_FEATURES if c in df.columns]
        if not features:
            logger.warning("Skipping scatter features: no KEY_FEATURES found")
            return Path()

        n = len(features)
        ncols = 4
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        fig.suptitle("Key Features vs Heat Pump Installations", fontsize=16, y=1.02)

        for i, (ax, feat) in enumerate(zip(axes.flat, features)):
            valid = df[[feat, "nb_installations_pac"]].dropna()
            ax.scatter(
                valid[feat], valid["nb_installations_pac"],
                alpha=0.4, s=15, c="#3498db",
            )
            # Simple linear regression
            if len(valid) > 10:
                z = np.polyfit(valid[feat], valid["nb_installations_pac"], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid[feat].min(), valid[feat].max(), 100)
                ax.plot(x_line, p(x_line), color="#e74c3c", linewidth=2, alpha=0.8)

                # Correlation coefficient
                corr = valid[feat].corr(valid["nb_installations_pac"])
                ax.set_title(f"{feat}\nr = {corr:.3f}", fontsize=11)
            else:
                ax.set_title(feat, fontsize=11)

            ax.set_xlabel(feat, fontsize=9)
            ax.set_ylabel("PAC", fontsize=9)

        # Hide empty axes
        for j in range(i + 1, len(axes.flat)):
            axes.flat[j].set_visible(False)

        plt.tight_layout()
        return self._save_fig("11_scatter_features_vs_pac")

    def plot_temp_vs_pac_by_season(self) -> Path:
        """Scatter plot of temperature vs heat pumps colored by season (winter/summer).

        Returns:
            Path to the saved chart.
        """
        df = self.df.copy()

        # Determine which temperature column is available
        temp_col = None
        for candidate in ["temp_mean", "temperature_2m_mean"]:
            if candidate in df.columns:
                temp_col = candidate
                break

        if temp_col is None or "nb_installations_pac" not in df.columns:
            logger.warning(
                "Skipping temp_vs_pac_by_season: missing columns "
                "(need temp_mean or temperature_2m_mean + nb_installations_pac)"
            )
            return self._save_fig("12_temp_vs_pac_by_season")

        df["season"] = df["month"].map(lambda m: (
            "Winter" if m in [12, 1, 2] else
            "Spring" if m in [3, 4, 5] else
            "Summer" if m in [6, 7, 8] else "Fall"
        ))
        season_colors = {
            "Winter": "#3498db", "Spring": "#2ecc71",
            "Summer": "#e74c3c", "Fall": "#f39c12",
        }

        fig, ax = plt.subplots(figsize=(12, 8))
        for season, color in season_colors.items():
            mask = df["season"] == season
            ax.scatter(
                df.loc[mask, temp_col],
                df.loc[mask, "nb_installations_pac"],
                c=color, label=season, alpha=0.6, s=30,
            )

        ax.set_title("Average Temperature vs Heat Pump Installations (by Season)", fontsize=14)
        ax.set_xlabel("Average Temperature (°C)")
        ax.set_ylabel("Heat Pump Installations")
        ax.legend(fontsize=11)

        return self._save_fig("12_temp_vs_pac_by_season")

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def generate_text_report(self, stats: Dict[str, Any]) -> Path:
        """Generate a summary text report for the EDA.

        Args:
            stats: Dictionary of overview() statistics.

        Returns:
            Path to the saved report.
        """
        df = self.df
        report_path = self.report_dir / "eda_report.txt"

        buf = StringIO()
        buf.write("=" * 70 + "\n")
        buf.write("  EDA REPORT — HVAC Market Analysis (France)\n")
        buf.write("=" * 70 + "\n\n")

        # 1. Overview
        buf.write("1. OVERVIEW\n")
        buf.write("-" * 40 + "\n")
        for key, val in stats.items():
            buf.write(f"  {key}: {val}\n")

        # 2. Descriptive statistics of targets
        buf.write("\n\n2. DESCRIPTIVE STATISTICS — TARGET VARIABLES\n")
        buf.write("-" * 40 + "\n")
        targets = [c for c in self.TARGET_COLS if c in df.columns]
        buf.write(df[targets].describe().to_string())

        # 3. Statistics by department
        agg_cols = {}
        for col in ["nb_dpe_total", "nb_installations_pac"]:
            if col in df.columns:
                agg_cols[col] = ["sum", "mean"]
        if "pct_pac" in df.columns:
            agg_cols["pct_pac"] = "mean"

        if agg_cols and "dept" in df.columns:
            buf.write("\n\n\n3. VOLUMES BY DEPARTMENT\n")
            buf.write("-" * 40 + "\n")
            dept_stats = df.groupby("dept").agg(agg_cols).round(2)
            buf.write(dept_stats.to_string())

        # 4. Seasonality
        if "month" in df.columns and "nb_installations_pac" in df.columns:
            buf.write("\n\n\n4. SEASONALITY — HEAT PUMPS BY MONTH (ALL DEPT AVERAGE)\n")
            buf.write("-" * 40 + "\n")
            month_avg = df.groupby("month")["nb_installations_pac"].mean().round(1)
            for m, v in month_avg.items():
                months_en = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                buf.write(f"  {months_en[m-1]:>3}: {v:>8.1f}\n")

        # 5. Top correlations
        buf.write("\n\n5. TOP CORRELATIONS WITH nb_installations_pac\n")
        buf.write("-" * 40 + "\n")
        numeric = df.select_dtypes(include=[np.number])
        if "nb_installations_pac" in numeric.columns:
            corr = numeric.corrwith(numeric["nb_installations_pac"]).abs()
            corr = corr.drop("nb_installations_pac", errors="ignore")
            top = corr.sort_values(ascending=False).head(15)
            for feat, val in top.items():
                sign = "+" if numeric[feat].corr(numeric["nb_installations_pac"]) > 0 else "-"
                buf.write(f"  {sign}{val:.3f}  {feat}\n")

        report_text = buf.getvalue()
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("EDA report saved: %s", report_path)

        return report_path

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def _safe_plot(self, name: str, func, *args, **kwargs) -> Optional[Path]:
        """Execute a plot function with error handling.

        If the plot fails, logs the error and returns None instead of
        crashing the entire EDA pipeline.

        Args:
            name: Human-readable name for the plot (used in logs).
            func: Plot method to call.
            *args, **kwargs: Arguments forwarded to the plot method.

        Returns:
            Path to the saved figure, or None if the plot failed.
        """
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            logger.error("  ✗ Plot '%s' failed: %s", name, exc)
            plt.close("all")  # Prevent leaked figures
            return None

    def run_full_eda(self) -> Dict[str, Any]:
        """Execute the full suite of EDA analyses.

        Each plot is wrapped in error handling so that one failing chart
        does not crash the entire pipeline. Failed plots are logged as
        errors and skipped.

        Returns:
            Dictionary with paths to generated files and stats.
        """
        logger.info("=" * 60)
        logger.info("  Phase 3 — Exploratory Data Analysis (EDA)")
        logger.info("=" * 60)

        self._ensure_dirs()
        self.df = self._load_data()

        results = {"figures": [], "errors": []}

        # 1. Overview
        stats = self._safe_plot("overview", self.overview) or {}
        results["stats"] = stats

        # All plot steps: (label, method)
        plot_steps = [
            ("NaN heatmap", self.plot_nan_heatmap),
            ("Target distributions", self.plot_target_distributions),
            ("Feature distributions", self.plot_feature_distributions),
            ("Timeseries PAC", self.plot_timeseries_pac),
            ("Timeseries DPE total", self.plot_timeseries_dpe_total),
            ("Timeseries aggregated", self.plot_timeseries_aura_aggregated),
            ("Seasonality boxplots", self.plot_seasonality_boxplots),
            ("Heatmap dept × month", self.plot_heatmap_dept_month),
            ("Department comparison", self.plot_dept_comparison),
            ("Heat pump rate by dept", self.plot_pct_pac_by_dept),
            ("Scatter features vs target", self.plot_scatter_features_vs_target),
            ("Temp vs PAC by season", self.plot_temp_vs_pac_by_season),
        ]

        for label, method in plot_steps:
            logger.info("%s...", label)
            fig_path = self._safe_plot(label, method)
            if fig_path is not None and fig_path.name:
                results["figures"].append(str(fig_path))
            elif fig_path is None:
                results["errors"].append(label)

        # Text report
        logger.info("Generating text report...")
        report_path = self._safe_plot("Text report", self.generate_text_report, stats)
        if report_path is not None:
            results["report"] = str(report_path)

        n_ok = len(results["figures"])
        n_err = len(results["errors"])
        logger.info(
            "EDA complete: %d charts generated, %d skipped/failed.", n_ok, n_err,
        )
        if n_err > 0:
            logger.warning("  Failed plots: %s", results["errors"])

        return results
