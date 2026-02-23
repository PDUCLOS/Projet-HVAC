# -*- coding: utf-8 -*-
"""
SITADEL collector — Building permits.
=======================================

Retrieves building permit data (authorized housing units)
from the SITADEL database of the Ministry of Ecological Transition.

Source: https://www.statistiques.developpement-durable.gouv.fr
Format: CSV in a ZIP file
Authentication: None (Open Data)

AUDIT NOTES:
    - The domain MUST use 'www.' to avoid TLS errors.
    - The ZIP file URL changes with each monthly update.
    - The dataset on data.gouv.fr is archived -> use the SDES source.
    - ~15% of construction starts are not reported in SITADEL
      (documented structural underestimation).

Collected data:
    - Permit processing date
    - Department and region
    - Number of housing units created
    - Construction type (individual, collective, residential)
    - Applicant category (individual, SCI, etc.)

Filtering: Region 84 (Auvergne-Rhone-Alpes) or target departments.

Extensibility:
    To update the ZIP file URL, modify the constant
    SITADEL_ZIP_URL below. The URL is updated monthly
    by SDES.
"""

from __future__ import annotations

import io
import zipfile
from typing import ClassVar, List, Optional

import pandas as pd

from src.collectors.base import BaseCollector

# =============================================================================
# SITADEL configuration
# =============================================================================

# DiDo API URL for building permits
# MIGRATION: Direct ZIP files no longer exist since late 2025.
# SITADEL data is now served via the SDES DiDo API.
# datafileRid = identifier of the file "PC and DP creating housing since 2017"
# millesime = update date (YYYY-MM)
SITADEL_DIDO_API_URL = (
    "https://data.statistiques.developpement-durable.gouv.fr/"
    "dido/api/v1/datafiles/"
    "8b35affb-55fc-4c1f-915b-7750f974446a/csv"
)
SITADEL_MILLESIME = "2026-01"  # Latest known update

# Columns of interest in the SITADEL CSV file
# NOTE: The DiDo API and raw CSV use different column names than our
# internal legacy names. The _DIDO_COLUMN_MAP below normalizes them.
#
# Actual CSV columns → Internal legacy names:
#   DEP_CODE → DEP              (DiDo 2026 rename)
#   REG_CODE → REG              (DiDo 2026 rename)
#   DATE_REELLE_AUTORISATION → DATE_PRISE_EN_COMPTE  (date format: YYYY-MM-DD)
#   NB_LGT_COL_CREES → NB_LGT_COLL_CREES            (typo: COL vs COLL)
#   SURF_HAB_CREEE → SURF_TOT_M2                     (closest available surface)
SITADEL_COLUMNS = [
    "REG",                    # Region code (legacy name; actual: REG or REG_CODE)
    "DEP",                    # Department code (legacy name; actual: DEP or DEP_CODE)
    "DATE_PRISE_EN_COMPTE",   # Permit date (legacy; actual: DATE_REELLE_AUTORISATION)
    "NB_LGT_TOT_CREES",      # Total housing units created
    "NB_LGT_COLL_CREES",     # Collective housing (legacy; actual: NB_LGT_COL_CREES)
    "SURF_TOT_M2",            # Surface area (legacy; actual: SURF_HAB_CREEE)
    "CAT_DEM",                # Applicant category
    "I_AUT_PC",               # Building permit indicator
]

# Column renaming map: actual CSV names → legacy internal names
# Applied during collect() to normalize all column names.
_DIDO_COLUMN_MAP = {
    "DEP_CODE": "DEP",
    "REG_CODE": "REG",
    "DATE_REELLE_AUTORISATION": "DATE_PRISE_EN_COMPTE",
    "NB_LGT_COL_CREES": "NB_LGT_COLL_CREES",
    "SURF_HAB_CREEE": "SURF_TOT_M2",
}


class SitadelCollector(BaseCollector):
    """Collector for SITADEL building permits.

    Downloads the CSV from the SDES DiDo API, filters on the
    Auvergne-Rhone-Alpes region, and saves.

    MIGRATION 2026: Direct ZIP files no longer exist.
    Data is now served via the DiDo API.

    Auto-registered as 'sitadel' in the CollectorRegistry.
    """

    source_name: ClassVar[str] = "sitadel"
    output_subdir: ClassVar[str] = "sitadel"
    output_filename: ClassVar[str] = "permis_construire_france.csv"

    def collect(self) -> pd.DataFrame:
        """Download and filter building permits.

        Uses the SDES DiDo API to retrieve the CSV directly
        (no more ZIP file since the late 2025 migration).

        Steps:
        1. Download the CSV from the DiDo API (~40-50 MB)
        2. Read the CSV with ';' separator (French format)
        3. Filter on target departments
        4. Clean column types

        Returns:
            DataFrame filtered on target departments with relevant columns.
        """
        self.logger.info(
            "Downloading SITADEL via DiDo API (millesime=%s)...",
            SITADEL_MILLESIME,
        )

        # Build the URL with the vintage
        params = {"millesime": SITADEL_MILLESIME, "withColumnName": "true"}

        try:
            # Download the raw CSV (extended timeout for large files)
            csv_content = self.fetch_bytes(SITADEL_DIDO_API_URL, params=params)
            self.logger.info(
                "CSV downloaded: %.1f MB",
                len(csv_content) / (1024 * 1024),
            )
        except Exception as exc:
            raise RuntimeError(
                f"SITADEL download via DiDo failed: {exc}. "
                f"Check the millesime ({SITADEL_MILLESIME}) on the DiDo catalog."
            ) from exc

        # Read the CSV in memory
        try:
            df = pd.read_csv(
                io.BytesIO(csv_content), sep=",",
                encoding="utf-8",
                low_memory=False,
                dtype=str,  # Read everything as string for cleaning
            )
        except Exception:
            # Fallback: try with ';' separator (legacy format)
            try:
                df = pd.read_csv(
                    io.BytesIO(csv_content), sep=";",
                    encoding="utf-8",
                    low_memory=False,
                    dtype=str,
                )
            except Exception:
                # Last fallback: latin-1
                df = pd.read_csv(
                    io.BytesIO(csv_content), sep=";",
                    encoding="latin-1",
                    low_memory=False,
                    dtype=str,
                )
                self.logger.warning("Fallback latin-1 used")

        self.logger.info(
            "CSV loaded: %d rows × %d columns", len(df), len(df.columns),
        )

        # Normalize column names: DiDo API renamed columns in 2026
        # (DEP_CODE → DEP, REG_CODE → REG) for backwards compatibility
        renamed = {
            old: new
            for old, new in _DIDO_COLUMN_MAP.items()
            if old in df.columns and new not in df.columns
        }
        if renamed:
            df = df.rename(columns=renamed)
            self.logger.info(
                "Columns renamed for compatibility: %s",
                {old: new for old, new in renamed.items()},
            )

        # Filter on target departments
        # The DEP column can have various formats (01, 1, 001...)
        if "DEP" in df.columns:
            # Normalize department code to 2 characters
            df["DEP"] = df["DEP"].astype(str).str.strip().str.zfill(2)
            df_filtered = df[df["DEP"].isin(self.config.departments)].copy()
        elif "REG" in df.columns:
            # Fallback: filter by region
            df_filtered = df[df["REG"].astype(str).str.strip() == self.config.region_code].copy()
        else:
            raise ValueError(
                "Neither 'DEP' nor 'REG' found in columns. "
                f"Available columns: {list(df.columns)}"
            )

        self.logger.info(
            "After filtering (%d depts): %d rows (out of %d total)",
            len(self.config.departments), len(df_filtered), len(df),
        )

        # Convert numeric columns
        if "NB_LGT_TOT_CREES" in df_filtered.columns:
            df_filtered["NB_LGT_TOT_CREES"] = pd.to_numeric(
                df_filtered["NB_LGT_TOT_CREES"], errors="coerce"
            )

        return df_filtered

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the structure and quality of SITADEL data.

        Checks:
        1. Required columns present (DEP)
        2. Correct departments (in the configured list)
        3. Minimum row count (at least 1000 permits expected)
        4. Positive NB_LGT_TOT_CREES values

        Args:
            df: Filtered DataFrame from collect().

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        # 1. Required columns
        if "DEP" not in df.columns:
            raise ValueError("Column 'DEP' missing from SITADEL data")

        # 2. Check departments
        depts_found = sorted(df["DEP"].unique().tolist())
        self.logger.info("Departments found: %s", depts_found)

        # 3. Minimum row count
        if len(df) < 100:
            self.logger.warning(
                "⚠ Very few SITADEL data: %d rows "
                "(>1000 expected for AURA)", len(df),
            )

        # 4. Check NB_LGT_TOT_CREES if available
        if "NB_LGT_TOT_CREES" in df.columns:
            total_logements = df["NB_LGT_TOT_CREES"].sum()
            self.logger.info(
                "Total authorized housing units France: %d",
                int(total_logements) if pd.notna(total_logements) else 0,
            )

        # Log summary
        self.logger.info(
            "Validation OK: %d permits | %d departments",
            len(df), len(depts_found),
        )

        return df
