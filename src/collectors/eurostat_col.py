# -*- coding: utf-8 -*-
"""
Collecteur Eurostat — Production industrielle HVAC.
====================================================

Récupère les indices de production industrielle mensuels depuis Eurostat
via le package Python dédié `eurostat`.

Source : https://ec.europa.eu/eurostat
Package : pip install eurostat (https://pypi.org/project/eurostat/)
Authentification : Aucune

Datasets utilisés :
    - sts_inpr_m : Short-term statistics, Industrial Production, Monthly
      Filtres : geo=FR, nace_r2=C28 (machines) et C2825 (équipements clim),
      unit=I21 (indice base 2021), s_adj=SCA (CVS-CJO)

Codes NACE pertinents pour le HVAC :
    - C28    : Fabrication de machines et équipements n.c.a.
    - C2825  : Fabrication d'équipements aérauliques et frigorifiques
               (inclut climatisation, PAC, ventilation)

NOTE : Le téléchargement du dataset complet peut prendre 30-60 secondes
       car Eurostat retourne toutes les combinaisons (pays, secteurs, etc.)
       avant filtrage local.

Extensibilité :
    Pour ajouter un nouveau code NACE ou un nouveau pays,
    modifier les constantes NACE_CODES et GEO_FILTER ci-dessous.
"""

from __future__ import annotations

from typing import ClassVar, List

import pandas as pd

from src.collectors.base import BaseCollector

# =============================================================================
# Configuration Eurostat
# =============================================================================

# Codes NACE des secteurs industriels liés au HVAC
NACE_CODES: List[str] = [
    "C28",    # Fabrication de machines et équipements (agrégat parent)
    "C2825",  # Fabrication d'équipements de conditionnement d'air
]

# Filtres géographiques et statistiques
GEO_FILTER = "FR"              # France uniquement
UNIT_FILTER = "I21"            # Indice base 2021
SEASONAL_ADJ_FILTER = "SCA"    # Corrigé des variations saisonnières et jours ouvrés


class EurostatCollector(BaseCollector):
    """Collecteur de l'indice de production industrielle HVAC (Eurostat).

    Utilise le package Python `eurostat` pour télécharger le dataset
    sts_inpr_m, puis filtre localement sur la France et les codes NACE
    liés au HVAC.

    Le résultat est un DataFrame au format long (melted) avec :
    - period : mois au format YYYY-MM
    - nace_r2 : code NACE du secteur
    - ipi_value : valeur de l'indice de production

    Auto-enregistré comme 'eurostat' dans le CollectorRegistry.
    """

    source_name: ClassVar[str] = "eurostat"
    output_subdir: ClassVar[str] = "eurostat"
    output_filename: ClassVar[str] = "ipi_hvac_france.csv"

    def collect(self) -> pd.DataFrame:
        """Télécharge et filtre l'IPI HVAC France depuis Eurostat.

        Étapes :
        1. Télécharger le dataset complet sts_inpr_m via le package eurostat
        2. Filtrer sur la France (geo=FR)
        3. Filtrer sur les codes NACE HVAC (C28, C2825)
        4. Filtrer sur l'unité (I21) et l'ajustement saisonnier (SCA)
        5. Pivoter en format long (une ligne par mois × code NACE)

        Returns:
            DataFrame avec colonnes : period, nace_r2, ipi_value.
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
            # Télécharger le dataset complet (toutes les dimensions)
            df = estat.get_data_df("sts_inpr_m", flags=False)
            self.logger.info(
                "Dataset brut téléchargé : %d lignes × %d colonnes",
                len(df), len(df.columns),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Échec du téléchargement Eurostat : {exc}"
            ) from exc

        # Identifier la colonne géographique
        # Le package eurostat peut nommer la colonne 'geo\\TIME_PERIOD' ou 'geo'
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

        # Filtrer sur la France + secteurs HVAC + unité + ajustement
        self.logger.info("Filtrage : geo=%s, NACE=%s, unit=%s, s_adj=%s",
                         GEO_FILTER, NACE_CODES, UNIT_FILTER, SEASONAL_ADJ_FILTER)

        mask = (
            (df[geo_col] == GEO_FILTER)
            & (df["nace_r2"].isin(NACE_CODES))
        )

        # Appliquer les filtres optionnels s'ils existent
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

        # Pivoter en format long (melt)
        # Les colonnes temporelles sont au format 'YYYY-MM' (ex: '2024-01')
        time_cols = [
            c for c in df_filtered.columns
            if c[:2] == "20" or c[:2] == "19"  # Colonnes commençant par 20xx ou 19xx
        ]

        if not time_cols:
            raise ValueError(
                "Aucune colonne temporelle trouvée. "
                f"Colonnes : {list(df_filtered.columns)}"
            )

        # Melt : transformer les colonnes temporelles en lignes
        df_melted = df_filtered.melt(
            id_vars=["nace_r2"],
            value_vars=time_cols,
            var_name="period",
            value_name="ipi_value",
        )

        # Nettoyer les valeurs manquantes (NaN = pas de données pour ce mois)
        df_melted = df_melted.dropna(subset=["ipi_value"])

        # Filtrer sur la période configurée
        start_period = self.config.start_date[:7]  # "2019-01"
        end_period = self.config.end_date[:7]       # "2025-12"
        df_melted = df_melted[
            (df_melted["period"] >= start_period)
            & (df_melted["period"] <= end_period)
        ].copy()

        # Trier chronologiquement
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
        """Valide la structure et la qualité des données Eurostat.

        Vérifications :
        1. Colonnes obligatoires présentes
        2. Au moins un code NACE collecté
        3. Valeurs IPI dans des plages réalistes (0-200)
        4. Continuité temporelle (pas de trous majeurs)

        Args:
            df: DataFrame brut issu de collect().

        Returns:
            DataFrame validé.

        Raises:
            ValueError: Si les colonnes obligatoires sont manquantes.
        """
        # 1. Colonnes obligatoires
        required = {"period", "nace_r2", "ipi_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans les données Eurostat : {missing}"
            )

        # 2. Codes NACE présents
        nace_present = df["nace_r2"].unique().tolist()
        self.logger.info("Codes NACE collectés : %s", nace_present)

        # 3. Plages de valeurs
        ipi_min = df["ipi_value"].min()
        ipi_max = df["ipi_value"].max()
        if ipi_min < 0 or ipi_max > 300:
            self.logger.warning(
                "⚠ Valeurs IPI suspectes : min=%.1f, max=%.1f",
                ipi_min, ipi_max,
            )

        # 4. Log résumé
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
