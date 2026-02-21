# -*- coding: utf-8 -*-
"""
Feature Engineering — Building advanced features for ML (Phase 3.3).
=====================================================================

This module transforms the ML-ready dataset (Phase 2.4 merge) by adding
advanced features designed to improve predictive models.

Feature categories:

    1. **Temporal lags**: time-shifted values
       - lag_1m, lag_3m, lag_6m for the target variable and key features
       - Capture auto-correlation and trend

    2. **Rolling windows**: rolling means and standard deviations
       - rolling_mean_3m, rolling_mean_6m, rolling_std_3m
       - Smooth noise and capture medium-term dynamics

    3. **Variations**: differences and growth rates
       - diff_1m (absolute month-to-month difference)
       - pct_change_1m (relative variation in %)
       - Capture market acceleration/deceleration

    4. **Interaction features**: products and ratios between variables
       - hdd x confiance_menages (cold weather + favorable context)
       - cdd x ipi_hvac (heatwave + industrial activity)

    5. **Cyclic encoding**: sin/cos transformations
       - Already added in merge_datasets (month_sin, month_cos)

IMPORTANT — NaN handling:
    Lags and rolling create NaN at the beginning of each series. We do NOT
    remove them here (the model will handle them via imputation or temporal
    split). The `n_valid_features` column allows filtering afterwards if needed.

Usage:
    >>> from src.processing.feature_engineering import FeatureEngineer
    >>> fe = FeatureEngineer(config)
    >>> df_features = fe.engineer(df_ml)
    >>> # Or directly from file:
    >>> df_features = fe.engineer_from_file()

Extensibility:
    To add new features:
    1. Create a _add_xxx_features(df) method in FeatureEngineer
    2. Call it in engineer()
    3. Document the feature in the docstring and the data dictionary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class FeatureEngineer:
    """Generate advanced features for ML modeling.

    Works on the ML-ready dataset (granularity = month x department)
    and adds temporal, statistical, and interaction features.

    Attributes:
        config: Project configuration (lags, rolling windows, etc.).
        logger: Structured logger.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize the feature engineer.

        Args:
            config: Centralized project configuration.
        """
        self.config = config
        self.logger = logging.getLogger("processing.features")

        # Parameters from the ML config
        self.max_lag = config.model.max_lag_months        # 6 months max
        self.rolling_windows = config.model.rolling_windows  # [3, 6]

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations.

        The DataFrame MUST contain at minimum:
        - date_id (int YYYYMM)
        - dept (str, department code)
        - nb_dpe_total, nb_installations_pac (target variables)
        - temp_mean, hdd_sum, cdd_sum (weather features)

        Args:
            df: ML-ready dataset (output of DatasetMerger.build_ml_dataset).

        Returns:
            Enriched DataFrame with all engineered features.
        """
        self.logger.info("=" * 60)
        self.logger.info("  FEATURE ENGINEERING")
        self.logger.info("=" * 60)
        self.logger.info("  Input dataset: %d rows × %d columns", len(df), len(df.columns))

        n_cols_start = len(df.columns)

        # Ensure sorting (critical for temporal operations)
        df = df.sort_values(["dept", "date_id"]).reset_index(drop=True)

        # 1. Temporal lags
        df = self._add_lag_features(df)

        # 2. Rolling windows
        df = self._add_rolling_features(df)

        # 3. Variations (diff + pct_change)
        df = self._add_variation_features(df)

        # 4. Interaction features
        df = self._add_interaction_features(df)

        # 5. Trend features
        df = self._add_trend_features(df)

        # 6. Completeness feature
        df = self._add_completeness_flag(df)

        n_cols_end = len(df.columns)
        self.logger.info(
            "  Features added: %d new columns (%d → %d)",
            n_cols_end - n_cols_start, n_cols_start, n_cols_end,
        )

        # Save the enriched dataset
        output_path = self._save_features(df)
        self.logger.info("  ✓ Features saved → %s", output_path)
        self.logger.info("=" * 60)

        return df

    def engineer_from_file(self) -> pd.DataFrame:
        """Load the ML dataset and apply feature engineering.

        Convenience method that reads the hvac_ml_dataset.csv file
        then calls engineer().

        Returns:
            DataFrame with engineered features.

        Raises:
            FileNotFoundError: If the ML dataset does not exist.
        """
        ml_path = self.config.features_data_dir / "hvac_ml_dataset.csv"
        if not ml_path.exists():
            raise FileNotFoundError(
                f"ML dataset not found: {ml_path}. "
                f"Run 'python -m src.pipeline merge' first."
            )

        df = pd.read_csv(ml_path)
        self.logger.info("  Loaded %s: %d rows", ml_path.name, len(df))
        return self.engineer(df)

    # ==================================================================
    # 1. Temporal lags
    # ==================================================================

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features (time-shifted values).

        For each target variable and key feature, creates lag_Xm
        columns containing the value from X months before.

        Lags generated: 1, 3, 6 months (configurable via max_lag_months).

        IMPORTANT: lags create NaN at the beginning of the series for
        each department. These NaN will be handled by the model or
        removed at train/test split time.

        Args:
            df: DataFrame sorted by dept + date_id.

        Returns:
            DataFrame with lag_Xm columns added.
        """
        # Columns on which to create lags
        target_cols = [
            "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
        ]
        feature_cols = [
            "temp_mean", "hdd_sum", "cdd_sum",
            "confiance_menages",
        ]
        lag_cols = [c for c in target_cols + feature_cols if c in df.columns]

        # Lags to generate
        lags = [1, 3, 6]
        lags = [l for l in lags if l <= self.max_lag]

        n_features = 0
        for col in lag_cols:
            for lag in lags:
                lag_name = f"{col}_lag_{lag}m"
                df[lag_name] = df.groupby("dept")[col].shift(lag)
                n_features += 1

        self.logger.info(
            "  Lags: %d features (columns=%d, lags=%s)",
            n_features, len(lag_cols), lags,
        )
        return df

    # ==================================================================
    # 2. Rolling windows (rolling means and standard deviations)
    # ==================================================================

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features.

        For each target variable and key feature, computes the rolling
        mean and rolling standard deviation over 3 and 6-month windows.

        Rolling is computed PER DEPARTMENT (groupby dept).

        NOTE: min_periods=1 to avoid creating unnecessary NaN
        at the beginning of the series (the window is simply reduced).

        Args:
            df: DataFrame sorted by dept + date_id.

        Returns:
            DataFrame with rolling columns added.
        """
        roll_cols = [
            "nb_dpe_total", "nb_installations_pac",
            "temp_mean", "hdd_sum", "cdd_sum",
        ]
        roll_cols = [c for c in roll_cols if c in df.columns]

        n_features = 0
        for col in roll_cols:
            for window in self.rolling_windows:
                # Rolling mean
                mean_name = f"{col}_rmean_{window}m"
                df[mean_name] = df.groupby("dept")[col].transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                ).round(2)
                n_features += 1

                # Rolling standard deviation (volatility measure)
                std_name = f"{col}_rstd_{window}m"
                df[std_name] = df.groupby("dept")[col].transform(
                    lambda s: s.rolling(window, min_periods=2).std()
                ).round(2)
                n_features += 1

        self.logger.info(
            "  Rolling: %d features (columns=%d, windows=%s)",
            n_features, len(roll_cols), self.rolling_windows,
        )
        return df

    # ==================================================================
    # 3. Variations (diff + pct_change)
    # ==================================================================

    def _add_variation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal variation features.

        For target variables:
        - diff_1m: absolute difference from the previous month
        - pct_change_1m: relative variation in % from the previous month

        Capture market dynamics (acceleration/deceleration).

        Args:
            df: DataFrame sorted by dept + date_id.

        Returns:
            DataFrame with variation columns added.
        """
        var_cols = [
            "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
        ]
        var_cols = [c for c in var_cols if c in df.columns]

        n_features = 0
        for col in var_cols:
            # Absolute difference
            diff_name = f"{col}_diff_1m"
            df[diff_name] = df.groupby("dept")[col].diff(1)
            n_features += 1

            # Relative variation (%)
            pct_name = f"{col}_pct_1m"
            df[pct_name] = (
                df.groupby("dept")[col].pct_change(1) * 100
            ).round(2)
            # Clip extreme variations (divisions by ~0)
            if pct_name in df.columns:
                df[pct_name] = df[pct_name].clip(-200, 500)
            n_features += 1

        self.logger.info(
            "  Variations: %d features (columns=%d)", n_features, len(var_cols),
        )
        return df

    # ==================================================================
    # 4. Interaction features
    # ==================================================================

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables.

        Business hypotheses:
        - A cold winter (high HDD) + high household confidence
          -> more heating replacements (heat pumps)
        - A hot summer (high CDD) + strong HVAC industrial activity
          -> more air conditioning installations
        - Household confidence x building business climate
          -> proxy of household investment intent

        Args:
            df: DataFrame with weather and economic features.

        Returns:
            DataFrame with interaction features added.
        """
        n_features = 0

        # HDD x household confidence (cold winter + confidence -> heating investment)
        if "hdd_sum" in df.columns and "confiance_menages" in df.columns:
            # Normalize before multiplying to avoid disparate scales
            hdd_max = max(df["hdd_sum"].max(), 1)
            hdd_norm = df["hdd_sum"] / hdd_max
            conf_norm = df["confiance_menages"] / 100  # Base 100
            df["interact_hdd_confiance"] = (hdd_norm * conf_norm).round(4)
            n_features += 1

        # CDD x IPI HVAC (heat + industrial production -> AC installations)
        if "cdd_sum" in df.columns and "ipi_hvac_c28" in df.columns:
            cdd_max = max(df["cdd_sum"].max(), 1)
            cdd_norm = df["cdd_sum"] / cdd_max
            ipi_norm = df["ipi_hvac_c28"] / 100  # Index base ~100
            df["interact_cdd_ipi"] = (cdd_norm * ipi_norm).round(4)
            n_features += 1

        # Confidence x building climate (dual investment proxy)
        if "confiance_menages" in df.columns and "climat_affaires_bat" in df.columns:
            df["interact_confiance_bat"] = (
                (df["confiance_menages"] / 100) * (df["climat_affaires_bat"] / 100)
            ).round(4)
            n_features += 1

        # Composite extreme temperature flag
        if "nb_jours_canicule" in df.columns and "nb_jours_gel" in df.columns:
            df["jours_extremes"] = df["nb_jours_canicule"] + df["nb_jours_gel"]
            n_features += 1

        self.logger.info("  Interactions: %d features", n_features)
        return df

    # ==================================================================
    # 5. Trend features
    # ==================================================================

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add longer-term trend features.

        Features added:
        - year_trend: normalized year (2021=0, 2022=1, ...)
          Captures the secular trend (heat pump market growth)
        - delta_temp_vs_mean: temperature deviation from the
          department's historical average
          Captures weather anomalies that trigger purchases

        Args:
            df: DataFrame with year and temp_mean columns.

        Returns:
            Enriched DataFrame.
        """
        n_features = 0

        # Annual trend (linear)
        if "year" in df.columns:
            year_min = df["year"].min()
            df["year_trend"] = df["year"] - year_min
            n_features += 1

        # Temperature deviation vs department average
        if "temp_mean" in df.columns and "dept" in df.columns and "month" in df.columns:
            # Compute the historical average per dept x month
            temp_avg = df.groupby(["dept", "month"])["temp_mean"].transform("mean")
            df["delta_temp_vs_mean"] = (df["temp_mean"] - temp_avg).round(2)
            n_features += 1

        self.logger.info("  Trend: %d features", n_features)
        return df

    # ==================================================================
    # 6. Completeness flag
    # ==================================================================

    def _add_completeness_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a feature completeness indicator.

        The `n_valid_features` column counts the number of non-NaN
        features for each row. Useful for filtering rows with too
        few data points (beginning of series where lags create NaN).

        Args:
            df: DataFrame with all features.

        Returns:
            DataFrame with n_valid_features column.
        """
        # Exclude non-feature columns (identifiers, metadata)
        exclude_cols = {
            "date_id", "dept", "dept_name", "city_ref",
            "latitude", "longitude", "year", "month", "quarter",
        }
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        df["n_valid_features"] = df[feature_cols].notna().sum(axis=1)
        df["pct_valid_features"] = (
            100 * df["n_valid_features"] / len(feature_cols)
        ).round(1)

        self.logger.info(
            "  Completeness: min=%.0f%%, median=%.0f%%, max=%.0f%%",
            df["pct_valid_features"].min(),
            df["pct_valid_features"].median(),
            df["pct_valid_features"].max(),
        )
        return df

    # ==================================================================
    # Save
    # ==================================================================

    def _save_features(self, df: pd.DataFrame) -> Path:
        """Save the dataset with engineered features.

        Args:
            df: Enriched dataset.

        Returns:
            Path of the saved file.
        """
        output_dir = self.config.features_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hvac_features_dataset.csv"

        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "Features dataset: %d rows × %d cols → %s (%.1f MB)",
            len(df), len(df.columns), output_path, size_mb,
        )
        return output_path

    # ==================================================================
    # Utilities
    # ==================================================================

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a statistical summary of all features.

        Useful for diagnostics and quick exploration.

        Args:
            df: Dataset with features.

        Returns:
            DataFrame with count, mean, std, min, max, %NaN per feature.
        """
        summary = df.describe().T
        summary["pct_nan"] = (df.isna().mean() * 100).round(1)
        summary["dtype"] = df.dtypes
        return summary.sort_values("pct_nan", ascending=False)
