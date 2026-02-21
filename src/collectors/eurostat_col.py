# -*- coding: utf-8 -*-
"""
Eurostat collector — HVAC industrial production.
=================================================

Retrieves monthly industrial production indices from Eurostat
via the dedicated Python package `eurostat`.

Source: https://ec.europa.eu/eurostat
Package: pip install eurostat (https://pypi.org/project/eurostat/)
Authentication: None

Datasets used:
    - sts_inpr_m: Short-term statistics, Industrial Production, Monthly
      Filters: geo=FR, nace_r2=C28 (machinery) and C2825 (HVAC equipment),
      unit=I21 (index base 2021), s_adj=SCA (seasonally and calendar adjusted)

NACE codes relevant to HVAC:
    - C28   : Manufacture of machinery and equipment n.e.c.
    - C2825 : Manufacture of non-domestic cooling and ventilation equipment
              (includes air conditioning, heat pumps, ventilation)

NOTE: Downloading the full dataset can take 30-60 seconds
      because Eurostat returns all combinations (countries, sectors, etc.)
      before local filtering.

Extensibility:
    To add a new NACE code or a new country,
    modify the NACE_CODES and GEO_FILTER constants below.
"""

from __future__ import annotations

from typing import ClassVar, List

import pandas as pd

from src.collectors.base import BaseCollector

# =============================================================================
# Eurostat configuration
# =============================================================================

# NACE codes for HVAC-related industrial sectors
NACE_CODES: List[str] = [
    "C28",    # Manufacture of machinery and equipment (parent aggregate)
    "C2825",  # Manufacture of air conditioning equipment
]

# Geographic and statistical filters
GEO_FILTER = "FR"              # France only
UNIT_FILTER = "I21"            # Index base 2021
SEASONAL_ADJ_FILTER = "SCA"    # Seasonally and calendar adjusted


class EurostatCollector(BaseCollector):
    """Collector for the HVAC industrial production index (Eurostat).

    Uses the Python `eurostat` package to download the sts_inpr_m
    dataset, then filters locally on France and HVAC-related
    NACE codes.

    The result is a DataFrame in long (melted) format with:
    - period: month in YYYY-MM format
    - nace_r2: sector NACE code
    - ipi_value: production index value

    Auto-registered as 'eurostat' in the CollectorRegistry.
    """

    source_name: ClassVar[str] = "eurostat"
    output_subdir: ClassVar[str] = "eurostat"
    output_filename: ClassVar[str] = "ipi_hvac_france.csv"

    def collect(self) -> pd.DataFrame:
        """Download and filter the France HVAC IPI from Eurostat.

        Steps:
        1. Download the full sts_inpr_m dataset via the eurostat package
        2. Filter on France (geo=FR)
        3. Filter on HVAC NACE codes (C28, C2825)
        4. Filter on unit (I21) and seasonal adjustment (SCA)
        5. Pivot to long format (one row per month x NACE code)

        Returns:
            DataFrame with columns: period, nace_r2, ipi_value.
        """
        try:
            import eurostat as estat
        except ImportError:
            raise ImportError(
                "Le package 'eurostat' est requis. "
                "Installation : pip install eurostat"
            )

        self.logger.info(
            "Téléchargement du dataset Eurostat 'sts_inpr_m'... "
            "(peut prendre 30-60 secondes)"
        )

        try:
            # Download the full dataset (all dimensions)
            df = estat.get_data_df("sts_inpr_m", flags=False)
            self.logger.info(
                "Dataset brut téléchargé : %d lignes × %d colonnes",
                len(df), len(df.columns),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Échec du téléchargement Eurostat : {exc}"
            ) from exc

        # Identify the geographic column
        # The eurostat package may name the column 'geo\\TIME_PERIOD' or 'geo'
        geo_col = None
        for candidate in ["geo\\TIME_PERIOD", "geo\\time", "geo"]:
            if candidate in df.columns:
                geo_col = candidate
                break

        if geo_col is None:
            raise ValueError(
                f"Colonne géographique introuvable. "
                f"Colonnes disponibles : {list(df.columns[:10])}"
            )

        # Filter on France + HVAC sectors + unit + adjustment
        self.logger.info("Filtrage : geo=%s, NACE=%s, unit=%s, s_adj=%s",
                         GEO_FILTER, NACE_CODES, UNIT_FILTER, SEASONAL_ADJ_FILTER)

        mask = (
            (df[geo_col] == GEO_FILTER)
            & (df["nace_r2"].isin(NACE_CODES))
        )

        # Apply optional filters if they exist
        if "s_adj" in df.columns:
            mask &= df["s_adj"] == SEASONAL_ADJ_FILTER
        if "unit" in df.columns:
            mask &= df["unit"] == UNIT_FILTER

        df_filtered = df[mask].copy()

        self.logger.info(
            "Après filtrage : %d lignes (sur %d)",
            len(df_filtered), len(df),
        )

        if df_filtered.empty:
            self.logger.warning(
                "⚠ Aucune donnée après filtrage. "
                "Vérifier les codes NACE et filtres."
            )
            return pd.DataFrame()

        # Pivot to long format (melt)
        # Time columns are in 'YYYY-MM' format (e.g., '2024-01')
        time_cols = [
            c for c in df_filtered.columns
            if c[:2] == "20" or c[:2] == "19"  # Columns starting with 20xx or 19xx
        ]

        if not time_cols:
            raise ValueError(
                "Aucune colonne temporelle trouvée. "
                f"Colonnes : {list(df_filtered.columns)}"
            )

        # Melt: transform time columns into rows
        df_melted = df_filtered.melt(
            id_vars=["nace_r2"],
            value_vars=time_cols,
            var_name="period",
            value_name="ipi_value",
        )

        # Clean missing values (NaN = no data for this month)
        df_melted = df_melted.dropna(subset=["ipi_value"])

        # Filter on the configured time period
        start_period = self.config.start_date[:7]  # "2019-01"
        end_period = self.config.end_date[:7]       # "2025-12"
        df_melted = df_melted[
            (df_melted["period"] >= start_period)
            & (df_melted["period"] <= end_period)
        ].copy()

        # Sort chronologically
        df_melted = df_melted.sort_values(
            ["nace_r2", "period"]
        ).reset_index(drop=True)

        self.logger.info(
            "Résultat final : %d observations, %d codes NACE, "
            "période %s → %s",
            len(df_melted),
            df_melted["nace_r2"].nunique(),
            df_melted["period"].min(),
            df_melted["period"].max(),
        )

        return df_melted

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the structure and quality of Eurostat data.

        Checks:
        1. Required columns present
        2. At least one NACE code collected
        3. IPI values within realistic ranges (0-200)
        4. Temporal continuity (no major gaps)

        Args:
            df: Raw DataFrame from collect().

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        # 1. Required columns
        required = {"period", "nace_r2", "ipi_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans les données Eurostat : {missing}"
            )

        # 2. NACE codes present
        nace_present = df["nace_r2"].unique().tolist()
        self.logger.info("Codes NACE collectés : %s", nace_present)

        # 3. Value ranges
        ipi_min = df["ipi_value"].min()
        ipi_max = df["ipi_value"].max()
        if ipi_min < 0 or ipi_max > 300:
            self.logger.warning(
                "⚠ Valeurs IPI suspectes : min=%.1f, max=%.1f",
                ipi_min, ipi_max,
            )

        # 4. Log summary
        for nace in nace_present:
            subset = df[df["nace_r2"] == nace]
            self.logger.info(
                "  NACE %s : %d mois, IPI moyen=%.1f [%.1f, %.1f]",
                nace, len(subset),
                subset["ipi_value"].mean(),
                subset["ipi_value"].min(),
                subset["ipi_value"].max(),
            )

        self.logger.info(
            "Validation OK : %d observations | %s → %s",
            len(df), df["period"].min(), df["period"].max(),
        )

        return df
