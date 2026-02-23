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

    5. **PAC efficiency features**: heat pump viability indicators
       - cop_proxy: estimated COP based on temperature and altitude
       - is_mountain: binary flag for high-altitude departments (>800m)
       - pac_viability_score: composite score combining temp, altitude, housing type
       - interact_altitude_frost: altitude x frost days interaction

    6. **Cyclic encoding**: sin/cos transformations
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

        # 5. PAC efficiency features
        df = self._add_pac_efficiency_features(df)

        # 6. Trend features
        df = self._add_trend_features(df)

        # 7. Completeness feature
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
    # 5. PAC (heat pump) efficiency features
    # ==================================================================

    def _add_pac_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features modeling heat pump viability by climate and geography.

        Domain knowledge:
        - Air-source heat pump (PAC air-air/air-eau) COP degrades in cold weather.
        - Below -7°C, COP drops below ~2.0, making PAC economically uncompetitive.
        - At high altitude (>800m), base temperatures are lower year-round.
        - Mountain departments have structurally different HVAC adoption patterns.
        - Population density inversely correlates with altitude/mountain zones.

        Features added (from prefecture altitude):
        - cop_proxy: estimated COP based on frost days and department mean altitude
          Formula: COP = 4.5 - 0.08 * frost_days - 0.0005 * altitude_mean, clipped [1.0, 5.0]
        - is_mountain: binary flag (altitude_mean > 800m OR pct_zone_montagne > 50%)
        - pac_viability_score: composite [0, 1] score combining COP, housing type, frost

        Features added (from altitude distribution):
        - pct_zone_montagne: % territory classified as mountain zone (loi montagne)
        - altitude_mean: mean altitude of the department (IGN BD ALTI)
        - densite_pop: population density (hab/km2)
        - interact_altitude_frost: altitude_mean × frost interaction
        - interact_maisons_altitude: pct_maisons × altitude_mean
        - interact_montagne_densite: mountain zone × inverse density
          (sparse mountain areas = strongest PAC constraint)

        Args:
            df: DataFrame with weather, geographic, and reference features.

        Returns:
            DataFrame with PAC efficiency features added.
        """
        n_features = 0

        # --- Determine best altitude source ---
        # Prefer altitude_mean (department average) over altitude (prefecture only)
        has_alt_mean = "altitude_mean" in df.columns
        has_altitude = "altitude" in df.columns
        alt_col = "altitude_mean" if has_alt_mean else ("altitude" if has_altitude else None)

        has_frost = "nb_jours_gel" in df.columns
        has_pac_days = "nb_jours_pac_inefficient" in df.columns
        has_montagne = "pct_zone_montagne" in df.columns
        has_densite = "densite_pop" in df.columns

        # --- COP proxy estimation ---
        # Based on simplified ASHRAE/SEPEMO model for air-source heat pumps
        if has_frost or alt_col:
            cop = np.full(len(df), 4.5)  # Ideal COP at mild temperature

            if has_frost:
                # Each frost day degrades COP
                cop -= 0.08 * df["nb_jours_gel"].fillna(0)

            if alt_col:
                # Higher altitude = colder base temperature = lower COP
                cop -= 0.0005 * df[alt_col].fillna(0)

            df["cop_proxy"] = np.clip(cop, 1.0, 5.0).round(2)
            n_features += 1

        # --- Mountain flag ---
        # Uses altitude_mean (more representative than prefecture altitude)
        # Also considers pct_zone_montagne: >50% mountain territory = mountain dept
        if alt_col or has_montagne:
            is_mt = np.zeros(len(df), dtype=int)
            if alt_col:
                is_mt = is_mt | (df[alt_col].fillna(0) > 800).astype(int)
            if has_montagne:
                is_mt = is_mt | (df["pct_zone_montagne"].fillna(0) > 50).astype(int)
            df["is_mountain"] = is_mt
            n_features += 1

        # --- PAC viability score (composite [0, 1]) ---
        # High score = favorable for PAC adoption
        # Combines: COP quality, housing structure (houses > apartments for PAC),
        #           and absence of extreme cold days
        if "cop_proxy" in df.columns:
            # Normalize COP to [0, 1] (1.0 -> 0, 5.0 -> 1)
            cop_norm = ((df["cop_proxy"] - 1.0) / 4.0).clip(0, 1)

            # Housing factor: more houses = more PAC potential
            if "pct_maisons" in df.columns:
                housing_norm = (df["pct_maisons"].fillna(50) / 100).clip(0, 1)
            else:
                housing_norm = 0.5

            # Frost penalty: many frost days = less favorable
            if has_frost:
                frost_max = max(df["nb_jours_gel"].max(), 1)
                frost_penalty = 1 - (df["nb_jours_gel"].fillna(0) / frost_max).clip(0, 1)
            else:
                frost_penalty = 0.5

            df["pac_viability_score"] = (
                0.5 * cop_norm + 0.3 * housing_norm + 0.2 * frost_penalty
            ).round(4)
            n_features += 1

        # --- Interaction: altitude × frost days ---
        # Mountains with cold winters = very hostile for PAC
        if alt_col and has_frost:
            alt_max = max(df[alt_col].max(), 1)
            frost_max = max(df["nb_jours_gel"].max(), 1)
            df["interact_altitude_frost"] = (
                (df[alt_col].fillna(0) / alt_max)
                * (df["nb_jours_gel"].fillna(0) / frost_max)
            ).round(4)
            n_features += 1

        # --- Interaction: pct_maisons × altitude ---
        # Houses in mountains: potential for PAC but efficiency concern
        if alt_col and "pct_maisons" in df.columns:
            alt_max = max(df[alt_col].max(), 1)
            df["interact_maisons_altitude"] = (
                (df["pct_maisons"].fillna(50) / 100)
                * (df[alt_col].fillna(0) / alt_max)
            ).round(4)
            n_features += 1

        # --- Interaction: mountain zone × inverse density ---
        # Sparse mountain areas are the strongest constraint for PAC adoption:
        # few inhabitants, high altitude, cold climate, hard to service
        if has_montagne and has_densite:
            dens_max = max(df["densite_pop"].max(), 1)
            inv_density = 1 - (df["densite_pop"].fillna(0) / dens_max).clip(0, 1)
            df["interact_montagne_densite"] = (
                (df["pct_zone_montagne"].fillna(0) / 100)
                * inv_density
            ).round(4)
            n_features += 1

        # --- PAC inefficient days ratio ---
        if has_pac_days:
            # Ratio of PAC-critical days over total days in month (~30)
            df["pct_jours_pac_inefficient"] = (
                df["nb_jours_pac_inefficient"].fillna(0) / 30 * 100
            ).round(1)
            n_features += 1

        # --- COP x HDD interaction ---
        # Cold demand (HDD) vs heat pump efficiency (COP):
        # High HDD + low COP = worst case (high heating need but PAC inefficient)
        if "cop_proxy" in df.columns and "hdd_sum" in df.columns:
            hdd_max = max(df["hdd_sum"].max(), 1)
            df["interact_cop_hdd"] = (
                (df["cop_proxy"] / 5.0) * (df["hdd_sum"] / hdd_max)
            ).round(4)
            n_features += 1

        self.logger.info("  PAC efficiency: %d features", n_features)
        return df

    # ==================================================================
    # 6. Trend features
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
