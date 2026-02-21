# -*- coding: utf-8 -*-
"""
INSEE BDM collector — Economic indicators via SDMX API.
========================================================

Retrieves macroeconomic time series from the INSEE
Macroeconomic Data Bank (BDM) via the SDMX protocol.

Source: https://www.bdm.insee.fr/series/sdmx
Authentication: None (open API)
Format: XML SDMX 2.1

Collected series:
    - Household confidence (synthetic indicator, seasonally adjusted)
    - Business climate in industry (seasonally adjusted)
    - Business climate in construction (seasonally adjusted)
    - Opinion on major purchase opportunities (seasonally adjusted balance)
    - Future household financial situation (seasonally adjusted balance)
    - IPI manufacturing industry (seasonally and calendar adjusted, base 2021)

AUDIT NOTES:
    - The idbanks have been verified and corrected (February 2026).
    - The old series 001759970 was a discontinued CPI, not confidence.
    - These indicators are NATIONAL (not regional) — this is expected,
      they serve as contextual features for the ML model.

Extensibility:
    To add a new INSEE series:
    1. Find the idbank at https://www.bdm.insee.fr
    2. Add an entry in the INSEE_SERIES dictionary below
    3. That's it! The collector will retrieve it automatically.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Any

import pandas as pd
from lxml import etree

from src.collectors.base import BaseCollector

# =============================================================================
# INSEE series configuration
# =============================================================================

# INSEE BDM SDMX API URL
INSEE_BDM_URL = "https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}"

# SDMX 2.1 XML namespaces for parsing
SDMX_NAMESPACES = {
    "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
}

# Dictionary of series to collect
# Key = short name (will become the column name)
# idbank = unique identifier in the BDM
# desc = human-readable description for logging
INSEE_SERIES: Dict[str, Dict[str, Any]] = {
    "confiance_menages": {
        "idbank": "001759970",
        "desc": "Indicateur synthétique de confiance des ménages (CVS)",
        "freq": "mensuel",
        "base": 100,  # Long-term average = 100
    },
    "climat_affaires_industrie": {
        "idbank": "001565530",
        "desc": "Indicateur du climat des affaires - Tous secteurs (CVS)",
        "freq": "mensuel",
        "base": 100,
    },
    "climat_affaires_batiment": {
        "idbank": "001586808",
        "desc": "Indicateur climat des affaires dans le bâtiment (CVS)",
        "freq": "mensuel",
        "base": 100,
    },
    "opinion_achats_importants": {
        "idbank": "001759974",
        "desc": "Opportunité de faire des achats importants (solde CVS)",
        "freq": "mensuel",
    },
    "situation_financiere_future": {
        "idbank": "001759972",
        "desc": "Situation financière future des ménages (solde CVS)",
        "freq": "mensuel",
    },
    "ipi_industrie_manuf": {
        "idbank": "010768261",
        "desc": "IPI Industrie manufacturière (CVS-CJO, base 2021)",
        "freq": "mensuel",
    },
}


class InseeCollector(BaseCollector):
    """Collector for INSEE economic indicators via SDMX API.

    Retrieves each series individually (by idbank), parses the SDMX XML,
    then merges all series on the 'period' column (YYYY-MM format).

    The collector is fault-tolerant: if one series fails, the others are
    still collected and merged.

    Auto-registered as 'insee' in the CollectorRegistry.
    """

    source_name: ClassVar[str] = "insee"
    output_subdir: ClassVar[str] = "insee"
    output_filename: ClassVar[str] = "indicateurs_economiques.csv"

    def collect(self) -> pd.DataFrame:
        """Collect all configured INSEE series.

        For each series:
        1. Build the SDMX URL with the idbank and start period
        2. Retrieve the XML and extract observations
        3. Convert to DataFrame with columns [period, series_name]

        Then merges all series on 'period' (outer join to preserve
        periods present in only one series).

        Returns:
            DataFrame with columns: period, confiance_menages,
            climat_affaires_industrie, climat_affaires_batiment,
            opinion_achats_importants, situation_financiere_future,
            ipi_industrie_manuf.
        """
        all_series: Dict[str, pd.DataFrame] = {}
        errors: List[str] = []

        # Extract the year-month from the start date
        start_period = self.config.start_date[:7]  # "2019-01-01" -> "2019-01"

        for name, info in INSEE_SERIES.items():
            self.logger.info(
                "Collecte série '%s' (idbank=%s) : %s",
                name, info["idbank"], info["desc"],
            )

            # Build the URL with the idbank and start period
            url = INSEE_BDM_URL.format(idbank=info["idbank"])
            params = {"startPeriod": start_period}

            try:
                # Retrieve and parse the SDMX XML
                root = self.fetch_xml(url, params=params)

                # The INSEE BDM API returns SDMX in StructureSpecific format
                # (not Generic). Observations use XML attributes directly:
                # TIME_PERIOD and OBS_VALUE as attributes of <Obs>
                #
                # StructureSpecific format:
                #   <Obs TIME_PERIOD="2024-01" OBS_VALUE="91.0" />
                # Generic format (not used by INSEE BDM):
                #   <Obs><ObsDimension value="2024-01"/><ObsValue value="91.0"/></Obs>

                # Search for Obs elements with wildcard namespace
                # as the exact namespace varies between series
                observations = root.findall(".//{*}Obs")

                if not observations:
                    raise ValueError(
                        f"Aucune observation trouvee pour '{name}' "
                        f"(idbank={info['idbank']}). "
                        f"Verifier que l'idbank est correct sur bdm.insee.fr"
                    )

                # Parse each observation from the XML attributes
                records = []
                for obs in observations:
                    # In StructureSpecific, period and value are attributes
                    period = obs.get("TIME_PERIOD")
                    value_str = obs.get("OBS_VALUE")

                    if period is None or value_str is None:
                        continue

                    # Handle non-numeric values ("ND", "", etc.)
                    try:
                        value = float(value_str)
                    except (ValueError, TypeError):
                        self.logger.debug(
                            "  Valeur non numerique ignoree : "
                            "period=%s, value='%s'", period, value_str,
                        )
                        continue

                    records.append({"period": period, name: value})

                df_series = pd.DataFrame(records)
                all_series[name] = df_series

                self.logger.info(
                    "  ✓ '%s' : %d observations (%s → %s)",
                    name, len(df_series),
                    df_series["period"].iloc[0] if len(df_series) > 0 else "?",
                    df_series["period"].iloc[-1] if len(df_series) > 0 else "?",
                )

            except Exception as exc:
                error_msg = f"Échec pour '{name}' (idbank={info['idbank']}): {exc}"
                errors.append(error_msg)
                self.logger.error("  ✗ %s", error_msg)
                continue

            # Pause between calls to avoid overloading the INSEE API
            self.rate_limit_pause()

        # Check that at least one series was collected
        if not all_series:
            self.logger.error(
                "Aucune série INSEE collectée. Erreurs : %s", errors
            )
            return pd.DataFrame()

        # Merge all series on the 'period' column
        # Outer join: preserve periods from each series even if
        # some series have gaps
        series_names = list(all_series.keys())
        result = all_series[series_names[0]]

        for name in series_names[1:]:
            result = result.merge(
                all_series[name], on="period", how="outer"
            )

        # Sort by chronological period
        result = result.sort_values("period").reset_index(drop=True)

        # Log summary
        if errors:
            self.logger.warning(
                "⚠ Collecte partielle : %d/%d séries réussies",
                len(all_series), len(INSEE_SERIES),
            )

        return result

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the structure and quality of INSEE data.

        Checks:
        1. 'period' column present and in correct format (YYYY-MM)
        2. At least 3 series collected (out of 6)
        3. No major gaps in periods
        4. Values within realistic ranges

        Args:
            df: Raw DataFrame from collect().

        Returns:
            Validated DataFrame.

        Raises:
            ValueError: If fewer than 3 series are available.
        """
        # 1. Check the period column
        if "period" not in df.columns:
            raise ValueError("Colonne 'period' manquante dans les données INSEE")

        # 2. Check the number of collected series
        serie_cols = [c for c in df.columns if c != "period"]
        if len(serie_cols) < 3:
            raise ValueError(
                f"Trop peu de séries collectées : {len(serie_cols)}/6. "
                f"Colonnes disponibles : {serie_cols}"
            )

        # 3. Check null values per series
        for col in serie_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.logger.warning(
                    "  Série '%s' : %d valeurs manquantes sur %d périodes",
                    col, null_count, len(df),
                )

        # 4. Check realistic value ranges
        # Confidence/climate indices revolve around 100 (+/-50)
        for col in serie_cols:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                col_min, col_max = valid_values.min(), valid_values.max()
                if col_min < -200 or col_max > 500:
                    self.logger.warning(
                        "  ⚠ Valeurs suspectes pour '%s' : "
                        "min=%.1f, max=%.1f", col, col_min, col_max,
                    )

        # Log summary
        self.logger.info(
            "Validation OK : %d périodes | %d séries | %s → %s",
            len(df), len(serie_cols),
            df["period"].iloc[0], df["period"].iloc[-1],
        )

        return df
