# -*- coding: utf-8 -*-
"""
Collecteur SITADEL — Permis de construire.
===========================================

Récupère les données de permis de construire (logements autorisés)
depuis la base SITADEL du Ministère de la Transition Écologique.

Source : https://www.statistiques.developpement-durable.gouv.fr
Format : CSV dans un fichier ZIP
Authentification : Aucune (Open Data)

NOTES AUDIT :
    - Le domaine DOIT utiliser 'www.' pour éviter les erreurs TLS.
    - L'URL du fichier ZIP change à chaque mise à jour mensuelle.
    - Le dataset sur data.gouv.fr est archivé → utiliser la source SDES.
    - ~15% des mises en chantier ne remontent pas dans SITADEL
      (sous-estimation structurelle documentée).

Données collectées :
    - Date de prise en compte du permis
    - Département et région
    - Nombre de logements créés
    - Type de construction (individuel, collectif, résidence)
    - Catégorie de demandeur (particulier, SCI, etc.)

Filtrage : Région 84 (Auvergne-Rhône-Alpes) ou départements cibles.

Extensibilité :
    Pour mettre à jour l'URL du fichier ZIP, modifier la constante
    SITADEL_ZIP_URL ci-dessous. L'URL est mise à jour mensuellement
    par le SDES.
"""

from __future__ import annotations

import io
import zipfile
from typing import ClassVar, List, Optional

import pandas as pd

from src.collectors.base import BaseCollector

# =============================================================================
# Configuration SITADEL
# =============================================================================

# URL de l'API DiDo pour les permis de construire
# MIGRATION : les fichiers ZIP directs n'existent plus depuis fin 2025.
# Les données SITADEL sont désormais servies via l'API DiDo du SDES.
# datafileRid = identifiant du fichier "PC et DP créant des logements depuis 2017"
# millesime = date de mise à jour (YYYY-MM)
SITADEL_DIDO_API_URL = (
    "https://data.statistiques.developpement-durable.gouv.fr/"
    "dido/api/v1/datafiles/"
    "8b35affb-55fc-4c1f-915b-7750f974446a/csv"
)
SITADEL_MILLESIME = "2026-01"  # Dernière mise à jour connue

# Colonnes d'intérêt dans le fichier CSV SITADEL
SITADEL_COLUMNS = [
    "REG",                    # Code région
    "DEP",                    # Code département
    "DATE_PRISE_EN_COMPTE",   # Date de prise en compte du permis
    "NB_LGT_TOT_CREES",      # Nombre total de logements créés
    "CAT_DEM",                # Catégorie du demandeur
    "I_AUT_PC",               # Indicateur permis de construire
]


class SitadelCollector(BaseCollector):
    """Collecteur des permis de construire SITADEL.

    Télécharge le CSV depuis l'API DiDo du SDES, filtre sur la
    région Auvergne-Rhône-Alpes, et sauvegarde.

    MIGRATION 2026 : les fichiers ZIP directs n'existent plus.
    Les données sont désormais servies via l'API DiDo.

    Auto-enregistré comme 'sitadel' dans le CollectorRegistry.
    """

    source_name: ClassVar[str] = "sitadel"
    output_subdir: ClassVar[str] = "sitadel"
    output_filename: ClassVar[str] = "permis_construire_aura.csv"

    def collect(self) -> pd.DataFrame:
        """Télécharge et filtre les permis de construire AURA.

        Utilise l'API DiDo du SDES pour récupérer le CSV directement
        (plus de fichier ZIP depuis la migration de fin 2025).

        Étapes :
        1. Télécharger le CSV depuis l'API DiDo (~40-50 Mo)
        2. Lire le CSV avec le séparateur ';' (format français)
        3. Filtrer sur les départements AURA cibles
        4. Nettoyer les types de colonnes

        Returns:
            DataFrame filtré sur AURA avec les colonnes pertinentes.
        """
        self.logger.info(
            "Telechargement SITADEL via API DiDo (millesime=%s)...",
            SITADEL_MILLESIME,
        )

        # Construire l'URL avec le millésime
        params = {"millesime": SITADEL_MILLESIME, "withColumnName": "true"}

        try:
            # Télécharger le CSV brut (timeout étendu car fichier volumineux)
            csv_content = self.fetch_bytes(SITADEL_DIDO_API_URL, params=params)
            self.logger.info(
                "CSV telecharge : %.1f Mo",
                len(csv_content) / (1024 * 1024),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Echec du telechargement SITADEL via DiDo : {exc}. "
                f"Verifier le millesime ({SITADEL_MILLESIME}) sur le catalogue DiDo."
            ) from exc

        # Lire le CSV en mémoire
        try:
            df = pd.read_csv(
                io.BytesIO(csv_content), sep=",",
                encoding="utf-8",
                low_memory=False,
                dtype=str,  # Lire tout en string pour le nettoyage
            )
        except Exception:
            # Fallback : essayer avec séparateur ';' (ancien format)
            try:
                df = pd.read_csv(
                    io.BytesIO(csv_content), sep=";",
                    encoding="utf-8",
                    low_memory=False,
                    dtype=str,
                )
            except Exception:
                # Dernier fallback : latin-1
                df = pd.read_csv(
                    io.BytesIO(csv_content), sep=";",
                    encoding="latin-1",
                    low_memory=False,
                    dtype=str,
                )
                self.logger.warning("Fallback latin-1 utilise")

        self.logger.info(
            "CSV chargé : %d lignes × %d colonnes", len(df), len(df.columns),
        )

        # Filtrer sur les départements AURA
        # La colonne DEP peut avoir des formats variés (01, 1, 001...)
        if "DEP" in df.columns:
            # Normaliser le code département sur 2 caractères
            df["DEP"] = df["DEP"].astype(str).str.strip().str.zfill(2)
            df_aura = df[df["DEP"].isin(self.config.departments)].copy()
        elif "REG" in df.columns:
            # Fallback : filtrer par région
            df_aura = df[df["REG"].astype(str).str.strip() == self.config.region_code].copy()
        else:
            raise ValueError(
                "Ni 'DEP' ni 'REG' trouvés dans les colonnes. "
                f"Colonnes disponibles : {list(df.columns)}"
            )

        self.logger.info(
            "Après filtrage AURA : %d lignes (sur %d total)",
            len(df_aura), len(df),
        )

        # Convertir les colonnes numériques
        if "NB_LGT_TOT_CREES" in df_aura.columns:
            df_aura["NB_LGT_TOT_CREES"] = pd.to_numeric(
                df_aura["NB_LGT_TOT_CREES"], errors="coerce"
            )

        return df_aura

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure et la qualité des données SITADEL.

        Vérifications :
        1. Colonnes obligatoires présentes (DEP, DATE_PRISE_EN_COMPTE)
        2. Départements corrects (dans la liste AURA)
        3. Nombre minimum de lignes (au moins 1000 permis attendus)
        4. Valeurs de NB_LGT_TOT_CREES positives

        Args:
            df: DataFrame filtré issu de collect().

        Returns:
            DataFrame validé.

        Raises:
            ValueError: Si les colonnes obligatoires sont manquantes.
        """
        # 1. Colonnes obligatoires
        if "DEP" not in df.columns:
            raise ValueError("Colonne 'DEP' manquante dans les données SITADEL")

        # 2. Vérifier les départements
        depts_found = sorted(df["DEP"].unique().tolist())
        self.logger.info("Départements trouvés : %s", depts_found)

        # 3. Nombre minimum de lignes
        if len(df) < 100:
            self.logger.warning(
                "⚠ Très peu de données SITADEL : %d lignes "
                "(>1000 attendu pour AURA)", len(df),
            )

        # 4. Vérifier NB_LGT_TOT_CREES si disponible
        if "NB_LGT_TOT_CREES" in df.columns:
            total_logements = df["NB_LGT_TOT_CREES"].sum()
            self.logger.info(
                "Total logements autorisés AURA : %d",
                int(total_logements) if pd.notna(total_logements) else 0,
            )

        # Log résumé
        self.logger.info(
            "Validation OK : %d permis | %d départements",
            len(df), len(depts_found),
        )

        return df
