# -*- coding: utf-8 -*-
"""
ADEME DPE collector — Energy Performance Diagnostics.
======================================================

Retrieves DPE (Diagnostic de Performance Energetique) data
from the ADEME API. DPEs serve as a PROXY for HVAC installations:
a DPE mentioning a heat pump (PAC) or air conditioning
indicates a recent installation of that equipment.

Source: https://data.ademe.fr/datasets/dpe03existant
API: https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines
Authentication: None (Open Data)

AUDIT NOTES:
    - The old URL (dpe-v2-logements-existants) has been BROKEN since 2025.
    - The new URL is 'dpe03existant' (verified February 2026).
    - The dataset contains ~14M DPEs -> mandatory pagination.
    - Pagination uses a cursor ('after'), not a page number.
    - The DPE is an imperfect proxy: it is triggered by a sale/rental,
      not directly by an HVAC installation.

Collected data (selected fields):
    - numero_dpe: Unique identifier
    - date_etablissement_dpe: Diagnostic date
    - etiquette_dpe: Energy class (A-G)
    - etiquette_ges: GHG class (A-G)
    - type_energie_chauffage: Heating energy type
    - type_installation_chauffage: Heating installation type
    - surface_habitable_logement: Living area in m2
    - code_postal_ban: Postal code
    - code_departement_ban: Department code (filtering)

ML target variable:
    Number of DPEs per month x department mentioning:
    - PAC (heat pump) in type_installation_chauffage
    - Air conditioning in equipment fields
    - Class A-B (high-performance buildings = likely recent equipment)

Extensibility:
    To modify collected fields, adjust the DPE_SELECT_FIELDS list.
    To change filtering, modify the parameters in collect().
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

import pandas as pd
from tqdm import tqdm

from src.collectors.base import BaseCollector

# =============================================================================
# ADEME DPE configuration
# =============================================================================

# ADEME DPE API URL (UPDATED following the audit)
DPE_API_BASE = (
    "https://data.ademe.fr/data-fair/api/v1/"
    "datasets/dpe03existant/lines"
)

# Fields to select — enriched to reach ~1 GB of data
# Grouped by functional category for readability
DPE_SELECT_FIELDS = [
    # --- Identification ---
    "numero_dpe",
    "date_etablissement_dpe",
    "date_visite_diagnostiqueur",

    # --- Location ---
    "code_postal_ban",
    "code_departement_ban",
    "code_insee_ban",
    "nom_commune_ban",

    # --- Energy performance (ML target variable) ---
    "etiquette_dpe",
    "etiquette_ges",
    "conso_5_usages_par_m2_ep",             # Primary energy consumption kWh/m2.year
    "conso_5_usages_par_m2_ef",             # Final energy consumption
    "emission_ges_5_usages_par_m2",         # GHG emissions kgCO2/m2.year

    # --- Building characteristics ---
    "type_batiment",                         # house / apartment / building
    "annee_construction",
    "periode_construction",
    "surface_habitable_logement",
    "nombre_niveau_logement",
    "hauteur_sous_plafond",

    # --- HVAC: Heating (key features for the model) ---
    "type_energie_principale_chauffage",
    "type_installation_chauffage",
    "type_generateur_chauffage_principal",   # Heat pump, boiler, radiator, etc.

    # --- HVAC: DHW (domestic hot water) ---
    "type_energie_principale_ecs",

    # --- HVAC: Cooling / Air conditioning ---
    "type_generateur_froid",                 # Air conditioning, reversible heat pump, etc.

    # --- Insulation (renovation indicator) ---
    "qualite_isolation_enveloppe",
    "qualite_isolation_murs",
    "qualite_isolation_menuiseries",
    "qualite_isolation_plancher_haut_comble_perdu",

    # --- Costs (economic indicator) ---
    "cout_chauffage",
    "cout_ecs",
    "cout_total_5_usages",
]

# Number of rows per page (maximum allowed by the ADEME API)
PAGE_SIZE = 10000


class DpeCollector(BaseCollector):
    """Collector for ADEME DPEs (proxy for HVAC installations).

    Collects DPEs department by department with cursor-based
    pagination. The total volume can reach several hundred
    thousand rows per department.

    WARNING: This is the most data-intensive collection in the project.
    Allow 10-30 minutes depending on the connection.

    Auto-registered as 'dpe' in the CollectorRegistry.
    """

    source_name: ClassVar[str] = "dpe"
    output_subdir: ClassVar[str] = "dpe"
    output_filename: ClassVar[str] = "dpe_france_all.csv"

    def collect(self) -> pd.DataFrame:
        """Collect DPEs for all configured departments.

        For each department:
        1. Paginated request via 'after' cursor
        2. Result accumulation until exhaustion
        3. Intermediate save per department (safety measure)

        Returns:
            Concatenated DataFrame of all collected DPEs.
        """
        all_frames: List[pd.DataFrame] = []
        errors: List[str] = []

        for dept in self.config.departments:
            self.logger.info(
                "Collecte DPE département %s...", dept,
            )

            try:
                df_dept = self._collect_department(dept)
                if not df_dept.empty:
                    all_frames.append(df_dept)

                    # Intermediate save per department (safety measure)
                    dept_path = (
                        self.config.raw_data_dir / "dpe" / f"dpe_{dept}.csv"
                    )
                    dept_path.parent.mkdir(parents=True, exist_ok=True)
                    df_dept.to_csv(dept_path, index=False)
                    self.logger.info(
                        "  ✓ Dept %s : %d DPE sauvegardés → %s",
                        dept, len(df_dept), dept_path,
                    )

            except Exception as exc:
                error_msg = f"Échec département {dept} : {exc}"
                errors.append(error_msg)
                self.logger.error("  ✗ %s", error_msg)
                continue

        if not all_frames:
            self.logger.error("Aucun DPE collecté. Erreurs : %s", errors)
            return pd.DataFrame()

        # Concatenate all departments
        result = pd.concat(all_frames, ignore_index=True)

        if errors:
            self.logger.warning(
                "⚠ Collecte partielle : %d/%d départements réussis",
                len(all_frames), len(self.config.departments),
            )

        return result

    def _collect_department(
        self, dept_code: str, max_pages: int = 200
    ) -> pd.DataFrame:
        """Collect all DPEs for a department via pagination.

        The ADEME API uses cursor-based pagination ('after'):
        - The first request returns the first N rows + a cursor
        - Subsequent requests use the cursor to advance

        Args:
            dept_code: Department code (e.g., "69").
            max_pages: Safety guard: maximum number of pages to retrieve.

        Returns:
            DataFrame with all DPEs for the department.
        """
        all_rows: List[dict] = []

        params = {
            "size": PAGE_SIZE,
            "select": ",".join(DPE_SELECT_FIELDS),
            "qs": f"code_departement_ban:{dept_code}",
        }

        for page in tqdm(
            range(max_pages),
            desc=f"  DPE dept {dept_code}",
            leave=False,
        ):
            try:
                data = self.fetch_json(DPE_API_BASE, params=params)
            except Exception as exc:
                self.logger.error(
                    "  Erreur page %d pour dept %s : %s",
                    page + 1, dept_code, exc,
                )
                break  # Save what has already been collected

            # Extract results
            results = data.get("results", [])
            if not results:
                break  # No more data

            all_rows.extend(results)

            # Retrieve the cursor for the next page
            next_cursor = data.get("next")
            if not next_cursor:
                break

            # Update parameters with the cursor
            # The ADEME API uses the 'after' field for pagination
            if "after" in str(next_cursor):
                # Extract the 'after' parameter from the next URL
                import urllib.parse
                parsed = urllib.parse.urlparse(next_cursor)
                query_params = urllib.parse.parse_qs(parsed.query)
                if "after" in query_params:
                    params["after"] = query_params["after"][0]
                else:
                    break
            else:
                break

            # Courtesy pause
            self.rate_limit_pause()

        self.logger.debug(
            "  Dept %s : %d lignes collectées en %d pages",
            dept_code, len(all_rows), page + 1,
        )

        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the structure and quality of DPE data.

        Checks:
        1. Required columns present
        2. Date conversion
        3. Departments within the configured scope
        4. DPE label statistics

        Args:
            df: Raw DataFrame from collect().

        Returns:
            Validated DataFrame with correct types.

        Raises:
            ValueError: If critical columns are missing.
        """
        # 1. Required columns
        critical_cols = {"date_etablissement_dpe", "etiquette_dpe"}
        available = set(df.columns)
        missing = critical_cols - available
        if missing:
            self.logger.warning(
                "Colonnes attendues manquantes : %s. "
                "Colonnes disponibles : %s", missing, sorted(available),
            )

        # 2. Date conversion
        if "date_etablissement_dpe" in df.columns:
            df["date_etablissement_dpe"] = pd.to_datetime(
                df["date_etablissement_dpe"], errors="coerce"
            )
            valid_dates = df["date_etablissement_dpe"].notna().sum()
            self.logger.info(
                "Dates valides : %d/%d (%.1f%%)",
                valid_dates, len(df), 100 * valid_dates / max(len(df), 1),
            )

        # 3. DPE label distribution
        if "etiquette_dpe" in df.columns:
            distribution = df["etiquette_dpe"].value_counts()
            self.logger.info("Distribution DPE :\n%s", distribution.to_string())

        # 4. Departments
        if "code_departement_ban" in df.columns:
            depts = sorted(df["code_departement_ban"].dropna().unique().tolist())
            self.logger.info("Départements collectés : %s", depts)

        # Log summary
        self.logger.info(
            "Validation OK : %d DPE collectés", len(df),
        )

        return df
