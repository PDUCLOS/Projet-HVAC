# -*- coding: utf-8 -*-
"""
Multi-source merging — Building the ML-ready dataset (Phase 2.4).
==================================================================

This module merges cleaned data from all sources into a single
dataset ready for feature engineering and ML modeling.

The final dataset granularity is: **month x department** (96 metropolitan France departments).
For each (month, department) pair, we have:

    - Target variables (Y): nb_dpe_total, nb_installations_pac, etc.
      -> Source: ADEME DPE aggregated
    - Weather features: temp_mean, HDD, CDD, precipitation, extreme days
      -> Source: Open-Meteo aggregated by month x department
    - Economic features: confiance_menages, climat_affaires, IPI
      -> Source: INSEE + Eurostat (national -> duplicated per department)

Join architecture:
    DPE (aggregated month x dept)
    LEFT JOIN Weather (aggregated month x dept)    ON date_id + dept
    LEFT JOIN INSEE (month)                        ON date_id
    LEFT JOIN Eurostat (month x nace)              ON date_id
    JOIN dim_time                                  ON date_id
    JOIN dim_geo                                   ON dept

Usage:
    >>> from src.processing.merge_datasets import DatasetMerger
    >>> merger = DatasetMerger(config)
    >>> df_ml = merger.build_ml_dataset()
    >>> df_ml.to_csv("data/features/hvac_ml_dataset.csv", index=False)

Extensibility:
    To add a new source to the ML dataset:
    1. Add a _prepare_xxx() method that returns a DataFrame
       with date_id (+ dept if local data)
    2. Integrate it into build_ml_dataset() via merge/join
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class DatasetMerger:
    """Merges cleaned data into an ML-ready dataset.

    The final granularity is month x department. Only months where
    the target variable (DPE) is available are kept (>= July 2021).

    Attributes:
        config: Project configuration.
        logger: Structured logger.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize the merger with the project configuration.

        Args:
            config: Centralized project configuration.
        """
        self.config = config
        self.logger = logging.getLogger("processing.merge")

    def build_ml_dataset(self) -> pd.DataFrame:
        """Build the complete ML-ready dataset.

        Steps:
        1. Prepare aggregated DPE (target variable + volume)
        2. Prepare weather aggregated by month x department
        3. Prepare economic indicators (INSEE + Eurostat)
        4. Merge all sources on date_id + dept
        5. Add temporal and geographic metadata
        6. Filter the period of interest (DPE v2: >= 2021-07)
        7. Save the final dataset

        Returns:
            ML-ready DataFrame with all features and target variables.
        """
        self.logger.info("=" * 60)
        self.logger.info("  PHASE 2.4 — Multi-source Merge → ML Dataset")
        self.logger.info("=" * 60)

        # 1. Target variable: DPE aggregated by month x department
        df_dpe = self._prepare_dpe_target()
        if df_dpe is None or df_dpe.empty:
            self.logger.error("Cannot build dataset: DPE data missing")
            return pd.DataFrame()
        self.logger.info(
            "  DPE target: %d rows (month × dept)", len(df_dpe),
        )

        # 2. Weather features (month x department)
        df_meteo = self._prepare_weather_features()
        if df_meteo is not None:
            self.logger.info(
                "  Weather features: %d rows", len(df_meteo),
            )

        # 3. Economic features (month, national)
        df_eco = self._prepare_economic_features()
        if df_eco is not None:
            self.logger.info(
                "  Economic features: %d rows", len(df_eco),
            )

        # 4. Progressive merging
        # Base = DPE (contains date_id + dept)
        df = df_dpe.copy()

        # Join weather (same granularity: date_id + dept)
        if df_meteo is not None:
            df = df.merge(
                df_meteo,
                on=["date_id", "dept"],
                how="left",
                suffixes=("", "_meteo"),
            )
            self.logger.info(
                "  After weather merge: %d rows × %d columns",
                len(df), len(df.columns),
            )

        # Join economics (granularity = date_id only -> broadcast across departments)
        if df_eco is not None:
            df = df.merge(
                df_eco,
                on="date_id",
                how="left",
                suffixes=("", "_eco"),
            )
            self.logger.info(
                "  After economic merge: %d rows × %d columns",
                len(df), len(df.columns),
            )

        # 5. Add temporal metadata
        df = self._add_time_features(df)

        # 6. Add geographic metadata
        df = self._add_geo_features(df)

        # 7. Filter DPE v2 period (>= 2021-07)
        dpe_start = int(
            self.config.time.dpe_start_date[:4]
            + self.config.time.dpe_start_date[5:7]
        )
        df = df[df["date_id"] >= dpe_start].copy()
        self.logger.info(
            "  After DPE v2 filter (>= %d): %d rows", dpe_start, len(df),
        )

        # 8. Final sort
        df = df.sort_values(["date_id", "dept"]).reset_index(drop=True)

        # 9. Log the final dataset
        self.logger.info("=" * 60)
        self.logger.info("  DATASET ML-READY")
        self.logger.info("  Dimensions: %d rows × %d columns", len(df), len(df.columns))
        self.logger.info("  Columns: %s", list(df.columns))
        self.logger.info("  Period: %d → %d", df["date_id"].min(), df["date_id"].max())
        self.logger.info(
            "  Departments: %s",
            sorted(df["dept"].unique().tolist()),
        )

        # NaN statistics
        null_pct = df.isna().mean() * 100
        cols_with_nulls = null_pct[null_pct > 0].sort_values(ascending=False)
        if len(cols_with_nulls) > 0:
            self.logger.info("  Columns with NaN:")
            for col, pct in cols_with_nulls.items():
                self.logger.info("    %-30s : %.1f%% NaN", col, pct)
        else:
            self.logger.info("  No NaN in dataset ✓")
        self.logger.info("=" * 60)

        # 10. Save
        output_path = self._save_ml_dataset(df)
        self.logger.info("  ✓ ML dataset saved → %s", output_path)

        return df

    # ==================================================================
    # Preparation of each source
    # ==================================================================

    def _prepare_dpe_target(self) -> Optional[pd.DataFrame]:
        """Prepare the target variable: DPE aggregated by month x department.

        Aggregates cleaned DPE to compute per month x department:
        - nb_dpe_total: total number of DPE
        - nb_installations_pac: DPE with detected heat pump
        - nb_installations_clim: DPE with air conditioning
        - nb_dpe_classe_ab: DPE class A or B
        - pct_pac: percentage of heat pumps among DPE
        - pct_clim: percentage of air conditioning

        Returns:
            Aggregated DataFrame, or None if the file is missing.
        """
        # Look for the cleaned file first, otherwise the raw one
        clean_path = self.config.processed_data_dir / "dpe" / "dpe_france_clean.csv"
        raw_path = self.config.raw_data_dir / "dpe" / "dpe_france_all.csv"

        filepath = clean_path if clean_path.exists() else raw_path
        if not filepath.exists():
            self.logger.error("DPE file not found: neither %s nor %s", clean_path, raw_path)
            return None

        self.logger.info("  Reading DPE from %s...", filepath.name)

        # Detect if the file contains pre-computed columns
        # (cleaned file has them, raw file does not)
        sample = pd.read_csv(filepath, nrows=2)
        has_precalc = "is_pac" in sample.columns
        self.logger.info(
            "  Pre-computed columns: %s", "yes" if has_precalc else "no",
        )

        # Read in chunks (low_memory=False to avoid DtypeWarning)
        chunks_agg = []
        for chunk in pd.read_csv(filepath, chunksize=200_000, low_memory=False):
            # If derived columns do not exist, compute them
            if "date_id" not in chunk.columns:
                chunk["date_etablissement_dpe"] = pd.to_datetime(
                    chunk["date_etablissement_dpe"], errors="coerce"
                )
                chunk = chunk.dropna(subset=["date_etablissement_dpe"])
                chunk["date_id"] = (
                    chunk["date_etablissement_dpe"].dt.year * 100
                    + chunk["date_etablissement_dpe"].dt.month
                )

            if "is_pac" not in chunk.columns:
                pac_pattern = r"(?i)PAC |PAC$|pompe.*chaleur|thermodynamique"
                chauffage_str = chunk["type_generateur_chauffage_principal"].fillna("")
                froid_str = chunk["type_generateur_froid"].fillna("")
                chunk["is_pac"] = (
                    chauffage_str.str.contains(pac_pattern, regex=True)
                    | froid_str.str.contains(pac_pattern, regex=True)
                ).astype(int)
                chunk["is_clim"] = (froid_str.str.len() > 0).astype(int)
                chunk["is_classe_ab"] = chunk["etiquette_dpe"].isin(["A", "B"]).astype(int)

            # Prepare the department column
            chunk["dept"] = (
                chunk["code_departement_ban"].astype(str).str.zfill(2)
            )

            # Aggregate the chunk
            agg = chunk.groupby(["date_id", "dept"]).agg(
                nb_dpe_total=("is_pac", "count"),
                nb_installations_pac=("is_pac", "sum"),
                nb_installations_clim=("is_clim", "sum"),
                nb_dpe_classe_ab=("is_classe_ab", "sum"),
            ).reset_index()

            chunks_agg.append(agg)

        # Merge all aggregated chunks
        df_agg = pd.concat(chunks_agg, ignore_index=True)

        # Re-aggregate (sum) because the same month x dept may appear in multiple chunks
        df_agg = df_agg.groupby(["date_id", "dept"]).agg({
            "nb_dpe_total": "sum",
            "nb_installations_pac": "sum",
            "nb_installations_clim": "sum",
            "nb_dpe_classe_ab": "sum",
        }).reset_index()

        # Convert to int
        for col in ["nb_dpe_total", "nb_installations_pac",
                     "nb_installations_clim", "nb_dpe_classe_ab"]:
            df_agg[col] = df_agg[col].astype(int)

        # Derived percentages (useful features for ML)
        df_agg["pct_pac"] = (
            100 * df_agg["nb_installations_pac"] / df_agg["nb_dpe_total"].clip(lower=1)
        ).round(2)
        df_agg["pct_clim"] = (
            100 * df_agg["nb_installations_clim"] / df_agg["nb_dpe_total"].clip(lower=1)
        ).round(2)
        df_agg["pct_classe_ab"] = (
            100 * df_agg["nb_dpe_classe_ab"] / df_agg["nb_dpe_total"].clip(lower=1)
        ).round(2)

        return df_agg

    def _prepare_weather_features(self) -> Optional[pd.DataFrame]:
        """Prepare weather features aggregated by month x department.

        Aggregates daily weather data into monthly metrics:
        - temp_mean, temp_max, temp_min: temperature statistics
        - hdd_sum, cdd_sum: cumulative degree-days
        - precipitation_sum: monthly precipitation
        - nb_jours_canicule: days with T > 35°C
        - nb_jours_gel: days with T < 0°C
        - wind_max: maximum wind speed

        Returns:
            Aggregated DataFrame, or None if the file is missing.
        """
        clean_path = self.config.processed_data_dir / "weather" / "weather_france.csv"
        raw_path = self.config.raw_data_dir / "weather" / "weather_france.csv"

        filepath = clean_path if clean_path.exists() else raw_path
        if not filepath.exists():
            self.logger.warning("Weather file not found")
            return None

        df = pd.read_csv(filepath)

        # Identify columns
        date_col = "time" if "time" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])

        if "date_id" not in df.columns:
            df["date_id"] = df[date_col].dt.year * 100 + df[date_col].dt.month

        if "dept" in df.columns:
            df["dept"] = df["dept"].astype(str).str.zfill(2)
        else:
            self.logger.error("  Column 'dept' missing in weather data")
            return None

        # Aggregation columns
        agg_dict = {}

        col_map = {
            "temperature_2m_mean": ("temp_mean", "mean"),
            "temperature_2m_max": ("temp_max", "max"),
            "temperature_2m_min": ("temp_min", "min"),
            "precipitation_sum": ("precipitation_sum", "sum"),
            "wind_speed_10m_max": ("wind_max", "max"),
            "hdd": ("hdd_sum", "sum"),
            "cdd": ("cdd_sum", "sum"),
        }

        for src_col, (dst_col, agg_func) in col_map.items():
            if src_col in df.columns:
                agg_dict[src_col] = agg_func

        # Binary indicators to count
        if "temperature_2m_max" in df.columns:
            df["_canicule"] = (df["temperature_2m_max"] > 35).astype(int)
            agg_dict["_canicule"] = "sum"
        if "temperature_2m_min" in df.columns:
            df["_gel"] = (df["temperature_2m_min"] < 0).astype(int)
            agg_dict["_gel"] = "sum"

        # Aggregation by month x department
        monthly = df.groupby(["date_id", "dept"]).agg(agg_dict).reset_index()

        # Rename columns
        rename = {}
        for src_col, (dst_col, _) in col_map.items():
            if src_col in monthly.columns:
                rename[src_col] = dst_col
        if "_canicule" in monthly.columns:
            rename["_canicule"] = "nb_jours_canicule"
        if "_gel" in monthly.columns:
            rename["_gel"] = "nb_jours_gel"

        monthly = monthly.rename(columns=rename)

        # Round
        float_cols = monthly.select_dtypes(include=["float64"]).columns
        monthly[float_cols] = monthly[float_cols].round(2)

        return monthly

    def _prepare_economic_features(self) -> Optional[pd.DataFrame]:
        """Prepare economic features (INSEE + Eurostat).

        Economic indicators are national (not departmental).
        They are merged into a single DataFrame at monthly granularity (date_id).
        When joining with the main dataset, they will be
        broadcast across all departments.

        Returns:
            Economic DataFrame, or None if files are missing.
        """
        df_eco = None

        # --- INSEE ---
        clean_insee = self.config.processed_data_dir / "insee" / "indicateurs_economiques.csv"
        raw_insee = self.config.raw_data_dir / "insee" / "indicateurs_economiques.csv"
        insee_path = clean_insee if clean_insee.exists() else raw_insee

        if insee_path.exists():
            df_insee = pd.read_csv(insee_path)

            # Filter monthly periods
            if "period" in df_insee.columns:
                mask = df_insee["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
                df_insee = df_insee[mask].copy()

            if "date_id" not in df_insee.columns:
                df_insee["date_id"] = df_insee["period"].str.replace("-", "").astype(int)

            # Rename columns for ML
            col_map = {
                "confiance_menages": "confiance_menages",
                "climat_affaires_industrie": "climat_affaires_indus",
                "climat_affaires_batiment": "climat_affaires_bat",
                "opinion_achats_importants": "opinion_achats",
                "situation_financiere_future": "situation_fin_future",
                "ipi_industrie_manuf": "ipi_manufacturing",
            }
            for old, new in col_map.items():
                if old in df_insee.columns and old != new:
                    df_insee = df_insee.rename(columns={old: new})

            # Keep only useful columns
            keep_cols = ["date_id"] + [v for v in col_map.values() if v in df_insee.columns]
            df_eco = df_insee[keep_cols].copy()

        # --- Eurostat ---
        clean_euro = self.config.processed_data_dir / "eurostat" / "ipi_hvac_france.csv"
        raw_euro = self.config.raw_data_dir / "eurostat" / "ipi_hvac_france.csv"
        euro_path = clean_euro if clean_euro.exists() else raw_euro

        if euro_path.exists():
            df_euro = pd.read_csv(euro_path)

            if "period" in df_euro.columns:
                mask = df_euro["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
                df_euro = df_euro[mask].copy()

            if "date_id" not in df_euro.columns:
                df_euro["date_id"] = df_euro["period"].str.replace("-", "").astype(int)

            # Pivot: one column per NACE code
            if "nace_r2" in df_euro.columns and "ipi_value" in df_euro.columns:
                pivot = df_euro.pivot_table(
                    index="date_id", columns="nace_r2",
                    values="ipi_value", aggfunc="first",
                ).reset_index()

                rename_map = {}
                if "C28" in pivot.columns:
                    rename_map["C28"] = "ipi_hvac_c28"
                if "C2825" in pivot.columns:
                    rename_map["C2825"] = "ipi_hvac_c2825"
                pivot = pivot.rename(columns=rename_map)

                # Merge with INSEE
                if df_eco is not None:
                    euro_cols = ["date_id"]
                    if "ipi_hvac_c28" in pivot.columns:
                        euro_cols.append("ipi_hvac_c28")
                    if "ipi_hvac_c2825" in pivot.columns:
                        euro_cols.append("ipi_hvac_c2825")

                    df_eco = df_eco.merge(
                        pivot[euro_cols],
                        on="date_id",
                        how="outer",
                    )
                else:
                    df_eco = pivot

        if df_eco is not None:
            df_eco = df_eco.sort_values("date_id").reset_index(drop=True)

            # Round
            float_cols = df_eco.select_dtypes(include=["float64"]).columns
            df_eco[float_cols] = df_eco[float_cols].round(2)

        return df_eco

    # ==================================================================
    # Enrichment
    # ==================================================================

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the dataset.

        Features added:
        - year, month, quarter
        - is_heating: heating season (October-March)
        - is_cooling: cooling season (June-September)
        - month_sin, month_cos: cyclic encoding of the month

        Args:
            df: DataFrame with date_id column.

        Returns:
            Enriched DataFrame.
        """
        df["year"] = df["date_id"] // 100
        df["month"] = df["date_id"] % 100
        df["quarter"] = ((df["month"] - 1) // 3) + 1

        # HVAC seasons
        df["is_heating"] = df["month"].isin([1, 2, 3, 10, 11, 12]).astype(int)
        df["is_cooling"] = df["month"].isin([6, 7, 8, 9]).astype(int)

        # Cyclic encoding (captures the December <-> January continuity)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)

        return df

    def _add_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geographic metadata.

        Features added:
        - dept_name: department name
        - city_ref: weather reference city
        - latitude, longitude: reference city coordinates

        Args:
            df: DataFrame with dept column.

        Returns:
            Enriched DataFrame.
        """
        geo_data = []
        for city, info in self.config.geo.cities.items():
            geo_data.append({
                "dept": info["dept"],
                "dept_name": self._dept_name(info["dept"]),
                "city_ref": city,
                "latitude": info["lat"],
                "longitude": info["lon"],
            })
        df_geo = pd.DataFrame(geo_data)

        df = df.merge(df_geo, on="dept", how="left")
        return df

    @staticmethod
    def _dept_name(code: str) -> str:
        """Return the name of a department from its code.

        Args:
            code: Department code (01, 07, etc.).

        Returns:
            Department name.
        """
        names = {
            "01": "Ain",
            "07": "Ardèche",
            "26": "Drôme",
            "38": "Isère",
            "42": "Loire",
            "69": "Rhône",
            "73": "Savoie",
            "74": "Haute-Savoie",
        }
        return names.get(code, f"Dept-{code}")

    # ==================================================================
    # Save
    # ==================================================================

    def _save_ml_dataset(self, df: pd.DataFrame) -> Path:
        """Save the final ML dataset.

        The file is saved in data/features/ (ML directory).

        Args:
            df: Complete ML-ready dataset.

        Returns:
            Path of the saved file.
        """
        output_dir = self.config.features_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hvac_ml_dataset.csv"

        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "ML dataset: %d rows × %d cols → %s (%.1f MB)",
            len(df), len(df.columns), output_path, size_mb,
        )
        return output_path
