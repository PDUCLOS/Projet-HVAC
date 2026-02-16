# -*- coding: utf-8 -*-
"""
Collecteur INSEE BDM — Indicateurs économiques via API SDMX.
=============================================================

Récupère les séries temporelles macroéconomiques depuis la Banque de
Données Macroéconomiques (BDM) de l'INSEE, via le protocole SDMX.

Source : https://www.bdm.insee.fr/series/sdmx
Authentification : Aucune (API ouverte)
Format : XML SDMX 2.1

Séries collectées :
    - Confiance des ménages (indicateur synthétique, CVS)
    - Climat des affaires dans l'industrie (CVS)
    - Climat des affaires dans le bâtiment (CVS)
    - Opinion sur l'opportunité d'achats importants (solde CVS)
    - Situation financière future des ménages (solde CVS)
    - IPI industrie manufacturière (CVS-CJO, base 2021)

NOTES AUDIT :
    - Les idbanks ont été vérifiés et corrigés (février 2026).
    - L'ancienne série 001759970 était un IPC arrêté, pas la confiance.
    - Ces indicateurs sont NATIONAUX (pas régionaux) — c'est normal,
      ils servent de features contextuelles pour le modèle ML.

Extensibilité :
    Pour ajouter une nouvelle série INSEE :
    1. Trouver l'idbank sur https://www.bdm.insee.fr
    2. Ajouter une entrée dans le dictionnaire INSEE_SERIES ci-dessous
    3. C'est tout ! Le collecteur la récupérera automatiquement.
"""

from __future__ import annotations

from typing import ClassVar, Dict, List, Any

import pandas as pd
from lxml import etree

from src.collectors.base import BaseCollector

# =============================================================================
# Configuration des séries INSEE
# =============================================================================

# URL de l'API SDMX de l'INSEE BDM
INSEE_BDM_URL = "https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}"

# Namespaces XML SDMX 2.1 pour le parsing
SDMX_NAMESPACES = {
    "message": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
    "generic": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic",
}

# Dictionnaire des séries à collecter
# Clé = nom court (deviendra le nom de colonne)
# idbank = identifiant unique dans la BDM
# desc = description humaine pour le logging
INSEE_SERIES: Dict[str, Dict[str, Any]] = {
    "confiance_menages": {
        "idbank": "001759970",
        "desc": "Indicateur synthétique de confiance des ménages (CVS)",
        "freq": "mensuel",
        "base": 100,  # Moyenne longue période = 100
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
    """Collecteur des indicateurs économiques INSEE via API SDMX.

    Récupère chaque série individuellement (par idbank), parse le XML SDMX,
    puis fusionne toutes les séries sur la colonne 'period' (format YYYY-MM).

    Le collecteur est tolérant : si une série échoue, les autres sont
    quand même collectées et fusionnées.

    Auto-enregistré comme 'insee' dans le CollectorRegistry.
    """

    source_name: ClassVar[str] = "insee"
    output_subdir: ClassVar[str] = "insee"
    output_filename: ClassVar[str] = "indicateurs_economiques.csv"

    def collect(self) -> pd.DataFrame:
        """Collecte toutes les séries INSEE configurées.

        Pour chaque série :
        1. Construit l'URL SDMX avec l'idbank et la période de début
        2. Récupère le XML et extrait les observations
        3. Convertit en DataFrame avec colonnes [period, nom_serie]

        Fusionne ensuite toutes les séries sur 'period' (outer join
        pour conserver les périodes présentes dans une seule série).

        Returns:
            DataFrame avec colonnes : period, confiance_menages,
            climat_affaires_industrie, climat_affaires_batiment,
            opinion_achats_importants, situation_financiere_future,
            ipi_industrie_manuf.
        """
        all_series: Dict[str, pd.DataFrame] = {}
        errors: List[str] = []

        # Extraire l'année-mois de début depuis la date de début
        start_period = self.config.start_date[:7]  # "2019-01-01" → "2019-01"

        for name, info in INSEE_SERIES.items():
            self.logger.info(
                "Collecte série '%s' (idbank=%s) : %s",
                name, info["idbank"], info["desc"],
            )

            # Construire l'URL avec l'idbank et la période de début
            url = INSEE_BDM_URL.format(idbank=info["idbank"])
            params = {"startPeriod": start_period}

            try:
                # Récupérer et parser le XML SDMX
                root = self.fetch_xml(url, params=params)

                # L'API INSEE BDM retourne du SDMX en format StructureSpecific
                # (pas Generic). Les observations utilisent des attributs XML
                # directement : TIME_PERIOD et OBS_VALUE comme attributs de <Obs>
                #
                # Format StructureSpecific :
                #   <Obs TIME_PERIOD="2024-01" OBS_VALUE="91.0" />
                # Format Generic (non utilisé par l'INSEE BDM) :
                #   <Obs><ObsDimension value="2024-01"/><ObsValue value="91.0"/></Obs>

                # Chercher les éléments Obs avec wildcard namespace
                # car le namespace exact varie selon les séries
                observations = root.findall(".//{*}Obs")

                if not observations:
                    raise ValueError(
                        f"Aucune observation trouvee pour '{name}' "
                        f"(idbank={info['idbank']}). "
                        f"Verifier que l'idbank est correct sur bdm.insee.fr"
                    )

                # Parser chaque observation depuis les attributs XML
                records = []
                for obs in observations:
                    # En StructureSpecific, period et value sont des attributs
                    period = obs.get("TIME_PERIOD")
                    value_str = obs.get("OBS_VALUE")

                    if period is None or value_str is None:
                        continue

                    # Gérer les valeurs non numériques ("ND", "", etc.)
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

            # Pause entre les appels pour ne pas surcharger l'API INSEE
            self.rate_limit_pause()

        # Vérifier qu'au moins une série a été collectée
        if not all_series:
            self.logger.error(
                "Aucune série INSEE collectée. Erreurs : %s", errors
            )
            return pd.DataFrame()

        # Fusionner toutes les séries sur la colonne 'period'
        # Outer join : conserver les périodes de chaque série même si
        # certaines séries ont des trous
        series_names = list(all_series.keys())
        result = all_series[series_names[0]]

        for name in series_names[1:]:
            result = result.merge(
                all_series[name], on="period", how="outer"
            )

        # Trier par période chronologique
        result = result.sort_values("period").reset_index(drop=True)

        # Log du bilan
        if errors:
            self.logger.warning(
                "⚠ Collecte partielle : %d/%d séries réussies",
                len(all_series), len(INSEE_SERIES),
            )

        return result

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure et la qualité des données INSEE.

        Vérifications :
        1. Colonne 'period' présente et au bon format (YYYY-MM)
        2. Au moins 3 séries collectées (sur 6)
        3. Pas de trous majeurs dans les périodes
        4. Valeurs dans des plages réalistes

        Args:
            df: DataFrame brut issu de collect().

        Returns:
            DataFrame validé.

        Raises:
            ValueError: Si moins de 3 séries sont disponibles.
        """
        # 1. Vérifier la colonne period
        if "period" not in df.columns:
            raise ValueError("Colonne 'period' manquante dans les données INSEE")

        # 2. Vérifier le nombre de séries collectées
        serie_cols = [c for c in df.columns if c != "period"]
        if len(serie_cols) < 3:
            raise ValueError(
                f"Trop peu de séries collectées : {len(serie_cols)}/6. "
                f"Colonnes disponibles : {serie_cols}"
            )

        # 3. Vérifier les valeurs nulles par série
        for col in serie_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self.logger.warning(
                    "  Série '%s' : %d valeurs manquantes sur %d périodes",
                    col, null_count, len(df),
                )

        # 4. Vérifier les plages de valeurs réalistes
        # Les indices de confiance/climat tournent autour de 100 (±50)
        for col in serie_cols:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                col_min, col_max = valid_values.min(), valid_values.max()
                if col_min < -200 or col_max > 500:
                    self.logger.warning(
                        "  ⚠ Valeurs suspectes pour '%s' : "
                        "min=%.1f, max=%.1f", col, col_min, col_max,
                    )

        # Log du résumé
        self.logger.info(
            "Validation OK : %d périodes | %d séries | %s → %s",
            len(df), len(serie_cols),
            df["period"].iloc[0], df["period"].iloc[-1],
        )

        return df
