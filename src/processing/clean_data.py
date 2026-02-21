# -*- coding: utf-8 -*-
"""
Raw data cleaning — Phase 2.1.
===============================

This module centralizes the cleaning of each raw data source.
Each cleaning function:
1. Reads the raw CSV (data/raw/)
2. Removes duplicates and outliers
3. Converts types (dates, numerics)
4. Handles missing values (NaN)
5. Saves the cleaned result (data/processed/)

The functions are designed to be idempotent: re-running them
produces the same result.

Usage:
    >>> from src.processing.clean_data import DataCleaner
    >>> cleaner = DataCleaner(config)
    >>> cleaner.clean_all()
    >>> # Or source by source:
    >>> cleaner.clean_weather()

Extensibility:
    To add cleaning for a new source:
    1. Add a clean_xxx() method in DataCleaner
    2. Call it from clean_all()
    3. Document the cleaning rules in the docstring
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class DataCleaner:
    """Central cleaner for the HVAC project raw data.

    Each clean_xxx() method processes a specific source.
    Cleaned data is saved in data/processed/
    with the same filename as the raw file.

    Attributes:
        config: Project configuration (paths, parameters).
        logger: Structured logger for operation tracking.
        stats: Dictionary of cleaning statistics per source.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialize the cleaner with the project configuration.

        Args:
            config: Centralized project configuration.
        """
        self.config = config
        self.logger = logging.getLogger("processing.clean")
        self.stats: Dict[str, Dict] = {}

        # Ensure the output directory exists
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def clean_all(self) -> Dict[str, Dict]:
        """Clean all data sources in the recommended order.

        The order does not matter here (no inter-source dependency),
        but we follow the pipeline order for log consistency.

        Returns:
            Dictionary {source: {rows_in, rows_out, dropped, ...}}.
        """
        self.logger.info("=" * 60)
        self.logger.info("  PHASE 2.1 — Raw Data Cleaning")
        self.logger.info("=" * 60)

        self.clean_weather()
        self.clean_insee()
        self.clean_eurostat()
        self.clean_dpe()

        # Summary
        self.logger.info("=" * 60)
        self.logger.info("  CLEANING SUMMARY")
        self.logger.info("=" * 60)
        for source, stat in self.stats.items():
            self.logger.info(
                "  %-15s : %6d → %6d rows (-%d duplicates, -%d outliers)",
                source,
                stat.get("rows_in", 0),
                stat.get("rows_out", 0),
                stat.get("duplicates_removed", 0),
                stat.get("outliers_removed", 0),
            )
        self.logger.info("=" * 60)

        return self.stats

    # ==================================================================
    # WEATHER cleaning (Open-Meteo)
    # ==================================================================

    def clean_weather(self) -> Optional[pd.DataFrame]:
        """Clean raw weather data.

        Cleaning rules:
        1. Convert the date column to datetime
        2. Remove duplicates (date x city)
        3. Remove rows with NaN temperature (critical)
        4. Clip outlier values:
           - Temperature: [-30, +50] °C (physical bounds for AURA region)
           - Precipitation: [0, 300] mm/day
           - Wind speed: [0, 200] km/h
        5. Recompute HDD/CDD if necessary (base 18°C)
        6. Add derived columns (year, month, date_id)

        Returns:
            Cleaned DataFrame, or None if the source file is missing.
        """
        self.logger.info("Cleaning WEATHER...")
        filepath = self.config.raw_data_dir / "weather" / "weather_france.csv"

        if not filepath.exists():
            self.logger.warning("  Missing file: %s", filepath)
            return None

        df = pd.read_csv(filepath)
        rows_in = len(df)
        self.logger.info("  Raw rows: %d", rows_in)

        # 1. Date conversion
        date_col = "time" if "time" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        null_dates = df[date_col].isna().sum()
        if null_dates > 0:
            self.logger.warning("  %d invalid dates removed", null_dates)
            df = df.dropna(subset=[date_col])

        # 2. Duplicates (date x city)
        key_cols = [date_col, "city"] if "city" in df.columns else [date_col]
        dups = df.duplicated(subset=key_cols).sum()
        df = df.drop_duplicates(subset=key_cols, keep="last")
        self.logger.info("  Duplicates removed: %d", dups)

        # 3. Critical missing values (temperature)
        temp_cols = [c for c in df.columns if "temperature" in c.lower()]
        null_temps = df[temp_cols].isna().all(axis=1).sum() if temp_cols else 0
        if null_temps > 0:
            df = df.dropna(subset=temp_cols, how="all")
            self.logger.info("  Rows with no temperature: %d removed", null_temps)

        # 4. Clip outlier values
        outliers_removed = 0
        clip_rules = {
            "temperature_2m_max": (-30, 50),
            "temperature_2m_min": (-30, 50),
            "temperature_2m_mean": (-30, 50),
            "precipitation_sum": (0, 300),
            "wind_speed_10m_max": (0, 200),
        }
        for col, (vmin, vmax) in clip_rules.items():
            if col in df.columns:
                mask_out = (df[col] < vmin) | (df[col] > vmax)
                n_out = mask_out.sum()
                if n_out > 0:
                    self.logger.info(
                        "  %s: %d values outside [%s, %s] clipped",
                        col, n_out, vmin, vmax,
                    )
                    outliers_removed += n_out
                df[col] = df[col].clip(vmin, vmax)

        # 5. Recompute HDD/CDD (base 18°C) for consistency
        if "temperature_2m_mean" in df.columns:
            df["hdd"] = np.maximum(0, 18.0 - df["temperature_2m_mean"]).round(2)
            df["cdd"] = np.maximum(0, df["temperature_2m_mean"] - 18.0).round(2)

        # 6. Derived columns to facilitate joins
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["date_id"] = df["year"] * 100 + df["month"]

        # Ensure dept is a zero-padded string
        if "dept" in df.columns:
            df["dept"] = df["dept"].astype(str).str.zfill(2)

        # Chronological sort
        df = df.sort_values([date_col, "city"] if "city" in df.columns else [date_col])
        df = df.reset_index(drop=True)

        # Round floating-point values
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].round(2)

        rows_out = len(df)
        self._save_cleaned(df, "weather", "weather_france.csv")

        self.stats["weather"] = {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "duplicates_removed": dups,
            "outliers_removed": outliers_removed,
            "null_dates_removed": null_dates,
        }
        self.logger.info(
            "  ✓ Weather cleaned: %d → %d rows", rows_in, rows_out,
        )
        return df

    # ==================================================================
    # INSEE cleaning (Economic indicators)
    # ==================================================================

    def clean_insee(self) -> Optional[pd.DataFrame]:
        """Clean INSEE economic indicators.

        Cleaning rules:
        1. Filter monthly periods only (YYYY-MM)
           — excludes quarterly formats (2019Q1)
        2. Remove duplicates on the period
        3. Linear interpolation of missing values
           (max 3 consecutive months, beyond that = NaN preserved)
        4. Convert to date_id (YYYYMM) for joining
        5. Chronological sort

        Returns:
            Cleaned DataFrame, or None if the source file is missing.
        """
        self.logger.info("Cleaning INSEE...")
        filepath = self.config.raw_data_dir / "insee" / "indicateurs_economiques.csv"

        if not filepath.exists():
            self.logger.warning("  Missing file: %s", filepath)
            return None

        df = pd.read_csv(filepath)
        rows_in = len(df)
        self.logger.info("  Raw rows: %d", rows_in)
        self.logger.info("  Columns: %s", list(df.columns))

        # 1. Filter monthly periods (YYYY-MM)
        if "period" in df.columns:
            mask_monthly = df["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
            non_monthly = (~mask_monthly).sum()
            if non_monthly > 0:
                self.logger.info(
                    "  %d non-monthly rows filtered (quarterly, etc.)",
                    non_monthly,
                )
            df = df[mask_monthly].copy()

        # 2. Duplicates
        dups = df.duplicated(subset=["period"]).sum()
        df = df.drop_duplicates(subset=["period"], keep="last")

        # 3. Convert to date_id
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # 4. Chronological sort (required for interpolation)
        df = df.sort_values("date_id").reset_index(drop=True)

        # 5. Linear interpolation for short gaps (max 3 months)
        # This fills occasional gaps in the INSEE series
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        indicator_cols = [c for c in numeric_cols if c not in ["date_id"]]
        nulls_before = df[indicator_cols].isna().sum().sum()

        for col in indicator_cols:
            df[col] = df[col].interpolate(method="linear", limit=3)

        nulls_after = df[indicator_cols].isna().sum().sum()
        interpolated = nulls_before - nulls_after

        if interpolated > 0:
            self.logger.info(
                "  %d values interpolated (gaps ≤ 3 months)", interpolated,
            )

        # 6. Log remaining NaN per column
        remaining_nulls = df[indicator_cols].isna().sum()
        for col, count in remaining_nulls.items():
            if count > 0:
                self.logger.warning(
                    "  ⚠ %s: %d remaining NaN (%.1f%%)",
                    col, count, 100 * count / len(df),
                )

        # Round
        df[indicator_cols] = df[indicator_cols].round(2)

        rows_out = len(df)
        self._save_cleaned(df, "insee", "indicateurs_economiques.csv")

        self.stats["insee"] = {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "duplicates_removed": dups,
            "outliers_removed": 0,
            "interpolated": interpolated,
            "non_monthly_filtered": non_monthly if "period" in df.columns else 0,
        }
        self.logger.info(
            "  ✓ INSEE cleaned: %d → %d rows", rows_in, rows_out,
        )
        return df

    # ==================================================================
    # EUROSTAT cleaning (IPI)
    # ==================================================================

    def clean_eurostat(self) -> Optional[pd.DataFrame]:
        """Clean Eurostat IPI data.

        Cleaning rules:
        1. Filter monthly periods (YYYY-MM)
        2. Remove duplicates (period x nace_r2)
        3. Detect and flag series breaks
           (variation > 30% from one month to the next = suspect)
        4. Linear interpolation of short gaps
        5. Convert to date_id

        Returns:
            Cleaned DataFrame, or None if the source file is missing.
        """
        self.logger.info("Cleaning EUROSTAT...")
        filepath = self.config.raw_data_dir / "eurostat" / "ipi_hvac_france.csv"

        if not filepath.exists():
            self.logger.warning("  Missing file: %s", filepath)
            return None

        df = pd.read_csv(filepath)
        rows_in = len(df)
        self.logger.info("  Raw rows: %d", rows_in)
        self.logger.info("  Columns: %s", list(df.columns))

        # 1. Filter monthly periods
        if "period" in df.columns:
            mask_monthly = df["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
            non_monthly = (~mask_monthly).sum()
            if non_monthly > 0:
                self.logger.info(
                    "  %d non-monthly rows filtered", non_monthly,
                )
            df = df[mask_monthly].copy()

        # 2. Duplicates
        key_cols = ["period", "nace_r2"] if "nace_r2" in df.columns else ["period"]
        dups = df.duplicated(subset=key_cols).sum()
        df = df.drop_duplicates(subset=key_cols, keep="last")

        # 3. Convert to date_id
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # 4. Sort
        df = df.sort_values(["nace_r2", "date_id"] if "nace_r2" in df.columns else ["date_id"])
        df = df.reset_index(drop=True)

        # 5. Interpolation per NACE series
        interpolated = 0
        if "nace_r2" in df.columns and "ipi_value" in df.columns:
            nulls_before = df["ipi_value"].isna().sum()

            # Interpolate per NACE group
            df["ipi_value"] = df.groupby("nace_r2")["ipi_value"].transform(
                lambda s: s.interpolate(method="linear", limit=3)
            )

            nulls_after = df["ipi_value"].isna().sum()
            interpolated = nulls_before - nulls_after
            if interpolated > 0:
                self.logger.info(
                    "  %d IPI values interpolated", interpolated,
                )

            # 6. Detect suspicious variations (> 30%)
            df["ipi_pct_change"] = df.groupby("nace_r2")["ipi_value"].transform(
                lambda s: s.pct_change().abs()
            )
            suspects = (df["ipi_pct_change"] > 0.30).sum()
            if suspects > 0:
                self.logger.warning(
                    "  ⚠ %d IPI variations > 30%% detected (potential breaks)",
                    suspects,
                )
            # Keep the column for diagnosis but do not remove the rows
            df = df.drop(columns=["ipi_pct_change"])

        # Round
        if "ipi_value" in df.columns:
            df["ipi_value"] = df["ipi_value"].round(2)

        rows_out = len(df)
        self._save_cleaned(df, "eurostat", "ipi_hvac_france.csv")

        self.stats["eurostat"] = {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "duplicates_removed": dups,
            "outliers_removed": 0,
            "interpolated": interpolated,
        }
        self.logger.info(
            "  ✓ Eurostat cleaned: %d → %d rows", rows_in, rows_out,
        )
        return df

    # ==================================================================
    # DPE cleaning (ADEME)
    # ==================================================================

    def clean_dpe(self) -> Optional[pd.DataFrame]:
        """Clean raw ADEME DPE data.

        This is the most voluminous source (~1.4M rows).
        Cleaning is done in chunks to manage memory.

        Cleaning rules:
        1. Remove duplicates on numero_dpe
        2. Convert and validate dates
           - date_etablissement_dpe must be >= 2021-07-01 (DPE v2)
           - date_etablissement_dpe must be <= today
        3. Validate DPE/GES labels (A-G only)
        4. Clip numeric values:
           - surface_habitable_logement: [5, 1000] m²
           - conso_5_usages_par_m2_ep: [0, 1000] kWh/m².year
           - cout_total_5_usages: [0, 50000] EUR/year
        5. Clean character strings (strip, partial lowercase)
        6. Validate the department code
        7. Derived columns: year, month, date_id, is_pac, is_clim

        Returns:
            Cleaned DataFrame, or None if the source file is missing.
        """
        self.logger.info("Cleaning DPE (~1.4M rows)...")
        filepath = self.config.raw_data_dir / "dpe" / "dpe_france_all.csv"

        if not filepath.exists():
            self.logger.warning("  Missing file: %s", filepath)
            return None

        # Read in chunks to manage memory
        chunks = []
        rows_in = 0
        dups = 0
        date_invalid = 0
        etiquette_invalid = 0
        outliers_removed = 0

        chunk_size = 200_000
        self.logger.info("  Reading in chunks of %d rows...", chunk_size)

        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
            rows_in += len(chunk)

            # 1. Duplicates on numero_dpe
            n_before = len(chunk)
            chunk = chunk.drop_duplicates(subset=["numero_dpe"], keep="last")
            dups += n_before - len(chunk)

            # 2. Dates
            chunk["date_etablissement_dpe"] = pd.to_datetime(
                chunk["date_etablissement_dpe"], errors="coerce"
            )
            # Filter: DPE v2 only (>= 2021-07-01) and not in the future
            date_min = pd.Timestamp("2021-07-01")
            date_max = pd.Timestamp.now()
            mask_date = (
                chunk["date_etablissement_dpe"].notna()
                & (chunk["date_etablissement_dpe"] >= date_min)
                & (chunk["date_etablissement_dpe"] <= date_max)
            )
            n_invalid_date = (~mask_date).sum()
            date_invalid += n_invalid_date
            chunk = chunk[mask_date].copy()

            # 3. Valid DPE/GES labels (A-G)
            valid_labels = {"A", "B", "C", "D", "E", "F", "G"}
            if "etiquette_dpe" in chunk.columns:
                mask_etiq = chunk["etiquette_dpe"].isin(valid_labels)
                n_invalid_etiq = (~mask_etiq).sum()
                etiquette_invalid += n_invalid_etiq
                chunk = chunk[mask_etiq].copy()

            # 4. Clip numeric values
            clip_rules = {
                "surface_habitable_logement": (5, 1000),
                "conso_5_usages_par_m2_ep": (0, 1000),
                "conso_5_usages_par_m2_ef": (0, 1000),
                "emission_ges_5_usages_par_m2": (0, 500),
                "cout_total_5_usages": (0, 50000),
                "cout_chauffage": (0, 30000),
                "cout_ecs": (0, 10000),
                "hauteur_sous_plafond": (1.5, 8.0),
            }
            for col, (vmin, vmax) in clip_rules.items():
                if col in chunk.columns:
                    mask_out = chunk[col].notna() & (
                        (chunk[col] < vmin) | (chunk[col] > vmax)
                    )
                    outliers_removed += mask_out.sum()
                    # Clip rather than delete (preserve the DPE record)
                    chunk[col] = chunk[col].clip(vmin, vmax)

            # 5. Clean strings (strip whitespace)
            str_cols = chunk.select_dtypes(include=["object"]).columns
            for col in str_cols:
                chunk[col] = chunk[col].str.strip()

            # 6. Validate department code
            if "code_departement_ban" in chunk.columns:
                valid_depts = set(self.config.geo.departments)
                chunk["code_departement_ban"] = (
                    chunk["code_departement_ban"].astype(str).str.zfill(2)
                )
                chunk = chunk[
                    chunk["code_departement_ban"].isin(valid_depts)
                ].copy()

            # 7. Derived columns
            chunk["year"] = chunk["date_etablissement_dpe"].dt.year
            chunk["month"] = chunk["date_etablissement_dpe"].dt.month
            chunk["date_id"] = chunk["year"] * 100 + chunk["month"]

            # Heat pump detection (same logic as in db_manager)
            pac_pattern = r"(?i)PAC |PAC$|pompe.*chaleur|thermodynamique"
            chauffage_str = chunk["type_generateur_chauffage_principal"].fillna("")
            froid_str = chunk["type_generateur_froid"].fillna("")
            chunk["is_pac"] = (
                chauffage_str.str.contains(pac_pattern, regex=True)
                | froid_str.str.contains(pac_pattern, regex=True)
            ).astype(int)

            # Air conditioning = cold generator filled in
            chunk["is_clim"] = (froid_str.str.len() > 0).astype(int)

            # Class A-B (high-performance building)
            chunk["is_classe_ab"] = chunk["etiquette_dpe"].isin(["A", "B"]).astype(int)

            # Remove the _score column if it exists (API artifact)
            if "_score" in chunk.columns:
                chunk = chunk.drop(columns=["_score"])

            chunks.append(chunk)

            if (i + 1) % 5 == 0:
                self.logger.info(
                    "  Chunk %d: %d rows processed total", i + 1, rows_in,
                )

        if not chunks:
            self.logger.error("  No DPE data after cleaning")
            return None

        df = pd.concat(chunks, ignore_index=True)

        # Global deduplication (chunks may have inter-chunk duplicates)
        n_before_global = len(df)
        df = df.drop_duplicates(subset=["numero_dpe"], keep="last")
        dups += n_before_global - len(df)

        # Round floating-point values
        float_cols = df.select_dtypes(include=["float64"]).columns
        df[float_cols] = df[float_cols].round(2)

        rows_out = len(df)
        self._save_cleaned(df, "dpe", "dpe_france_clean.csv")

        self.stats["dpe"] = {
            "rows_in": rows_in,
            "rows_out": rows_out,
            "duplicates_removed": dups,
            "date_invalid_removed": date_invalid,
            "etiquette_invalid_removed": etiquette_invalid,
            "outliers_removed": outliers_removed,
        }

        # Detailed DPE log
        self.logger.info(
            "  ✓ DPE cleaned: %d → %d rows", rows_in, rows_out,
        )
        self.logger.info("    - Duplicates removed: %d", dups)
        self.logger.info("    - Invalid dates: %d", date_invalid)
        self.logger.info("    - Invalid labels: %d", etiquette_invalid)
        self.logger.info("    - Values clipped: %d", outliers_removed)

        # DPE label distribution after cleaning
        if "etiquette_dpe" in df.columns:
            distrib = df["etiquette_dpe"].value_counts().sort_index()
            self.logger.info("    Cleaned DPE distribution:\n%s", distrib.to_string())

        # Heat pump rate after cleaning
        n_pac = df["is_pac"].sum()
        self.logger.info(
            "    Heat pumps detected: %d (%.1f%%)", n_pac, 100 * n_pac / max(len(df), 1),
        )

        return df

    # ==================================================================
    # Utilities
    # ==================================================================

    def _save_cleaned(
        self,
        df: pd.DataFrame,
        subdir: str,
        filename: str,
    ) -> Path:
        """Save a cleaned DataFrame to data/processed/.

        Args:
            df: Cleaned DataFrame.
            subdir: Subdirectory (e.g., 'weather', 'insee').
            filename: CSV filename.

        Returns:
            Full path of the saved file.
        """
        output_dir = self.config.processed_data_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "  Saved → %s (%.1f MB)", output_path, size_mb,
        )
        return output_path
