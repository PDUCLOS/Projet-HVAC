# -*- coding: utf-8 -*-
"""
Nettoyage des données brutes — Phase 2.1.
==========================================

Ce module centralise le nettoyage de chaque source de données brute.
Chaque fonction de nettoyage :
1. Lit le CSV brut (data/raw/)
2. Supprime les doublons et valeurs aberrantes
3. Convertit les types (dates, numériques)
4. Gère les valeurs manquantes (NaN)
5. Sauvegarde le résultat nettoyé (data/processed/)

Les fonctions sont conçues pour être idempotentes : les relancer
produit le même résultat.

Usage :
    >>> from src.processing.clean_data import DataCleaner
    >>> cleaner = DataCleaner(config)
    >>> cleaner.clean_all()
    >>> # Ou source par source :
    >>> cleaner.clean_weather()

Extensibilité :
    Pour ajouter le nettoyage d'une nouvelle source :
    1. Ajouter une méthode clean_xxx() dans DataCleaner
    2. L'appeler depuis clean_all()
    3. Documenter les règles de nettoyage dans le docstring
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class DataCleaner:
    """Nettoyeur central des données brutes du projet HVAC.

    Chaque méthode clean_xxx() traite une source spécifique.
    Les données nettoyées sont sauvegardées dans data/processed/
    avec le même nom de fichier que le brut.

    Attributes:
        config: Configuration du projet (chemins, paramètres).
        logger: Logger structuré pour le suivi des opérations.
        stats: Dictionnaire des statistiques de nettoyage par source.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialise le nettoyeur avec la configuration projet.

        Args:
            config: Configuration centralisée du projet.
        """
        self.config = config
        self.logger = logging.getLogger("processing.clean")
        self.stats: Dict[str, Dict] = {}

        # S'assurer que le répertoire de sortie existe
        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def clean_all(self) -> Dict[str, Dict]:
        """Nettoie toutes les sources de données dans l'ordre recommandé.

        L'ordre n'a pas d'importance ici (pas de dépendance inter-sources),
        mais on suit l'ordre du pipeline pour la cohérence des logs.

        Returns:
            Dictionnaire {source: {rows_in, rows_out, dropped, ...}}.
        """
        self.logger.info("=" * 60)
        self.logger.info("  PHASE 2.1 — Nettoyage des données brutes")
        self.logger.info("=" * 60)

        self.clean_weather()
        self.clean_insee()
        self.clean_eurostat()
        self.clean_dpe()

        # Bilan
        self.logger.info("=" * 60)
        self.logger.info("  BILAN NETTOYAGE")
        self.logger.info("=" * 60)
        for source, stat in self.stats.items():
            self.logger.info(
                "  %-15s : %6d → %6d lignes (-%d doublons, -%d aberrants)",
                source,
                stat.get("rows_in", 0),
                stat.get("rows_out", 0),
                stat.get("duplicates_removed", 0),
                stat.get("outliers_removed", 0),
            )
        self.logger.info("=" * 60)

        return self.stats

    # ==================================================================
    # Nettoyage MÉTÉO (Open-Meteo)
    # ==================================================================

    def clean_weather(self) -> Optional[pd.DataFrame]:
        """Nettoie les données météo brutes.

        Règles de nettoyage :
        1. Conversion de la colonne date en datetime
        2. Suppression des doublons (date × ville)
        3. Suppression des lignes avec température NaN (critique)
        4. Clipping des valeurs aberrantes :
           - Température : [-30, +50] °C (bornes physiques AURA)
           - Précipitations : [0, 300] mm/jour
           - Vitesse vent : [0, 200] km/h
        5. Recalcul HDD/CDD si nécessaire (base 18°C)
        6. Ajout de colonnes dérivées (year, month, date_id)

        Returns:
            DataFrame nettoyé, ou None si le fichier source manque.
        """
        self.logger.info("Nettoyage MÉTÉO...")
        filepath = self.config.raw_data_dir / "weather" / "weather_france.csv"

        if not filepath.exists():
            self.logger.warning("  Fichier manquant : %s", filepath)
            return None

        df = pd.read_csv(filepath)
        rows_in = len(df)
        self.logger.info("  Lignes brutes : %d", rows_in)

        # 1. Conversion date
        date_col = "time" if "time" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        null_dates = df[date_col].isna().sum()
        if null_dates > 0:
            self.logger.warning("  %d dates invalides supprimées", null_dates)
            df = df.dropna(subset=[date_col])

        # 2. Doublons (date × ville)
        key_cols = [date_col, "city"] if "city" in df.columns else [date_col]
        dups = df.duplicated(subset=key_cols).sum()
        df = df.drop_duplicates(subset=key_cols, keep="last")
        self.logger.info("  Doublons supprimés : %d", dups)

        # 3. Valeurs manquantes critiques (température)
        temp_cols = [c for c in df.columns if "temperature" in c.lower()]
        null_temps = df[temp_cols].isna().all(axis=1).sum() if temp_cols else 0
        if null_temps > 0:
            df = df.dropna(subset=temp_cols, how="all")
            self.logger.info("  Lignes sans aucune température : %d supprimées", null_temps)

        # 4. Clipping des valeurs aberrantes
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
                        "  %s : %d valeurs hors [%s, %s] clippées",
                        col, n_out, vmin, vmax,
                    )
                    outliers_removed += n_out
                df[col] = df[col].clip(vmin, vmax)

        # 5. Recalcul HDD/CDD (base 18°C) pour cohérence
        if "temperature_2m_mean" in df.columns:
            df["hdd"] = np.maximum(0, 18.0 - df["temperature_2m_mean"]).round(2)
            df["cdd"] = np.maximum(0, df["temperature_2m_mean"] - 18.0).round(2)

        # 6. Colonnes dérivées pour faciliter les jointures
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["date_id"] = df["year"] * 100 + df["month"]

        # S'assurer que dept est un string avec padding
        if "dept" in df.columns:
            df["dept"] = df["dept"].astype(str).str.zfill(2)

        # Tri chronologique
        df = df.sort_values([date_col, "city"] if "city" in df.columns else [date_col])
        df = df.reset_index(drop=True)

        # Arrondir les flottants
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
            "  ✓ Météo nettoyée : %d → %d lignes", rows_in, rows_out,
        )
        return df

    # ==================================================================
    # Nettoyage INSEE (Indicateurs économiques)
    # ==================================================================

    def clean_insee(self) -> Optional[pd.DataFrame]:
        """Nettoie les indicateurs économiques INSEE.

        Règles de nettoyage :
        1. Filtrage des périodes mensuelles uniquement (YYYY-MM)
           — exclut les formats trimestriels (2019Q1)
        2. Suppression des doublons sur la période
        3. Interpolation linéaire des valeurs manquantes
           (max 3 mois consécutifs, au-delà = NaN conservé)
        4. Conversion date_id (YYYYMM) pour jointure
        5. Tri chronologique

        Returns:
            DataFrame nettoyé, ou None si le fichier source manque.
        """
        self.logger.info("Nettoyage INSEE...")
        filepath = self.config.raw_data_dir / "insee" / "indicateurs_economiques.csv"

        if not filepath.exists():
            self.logger.warning("  Fichier manquant : %s", filepath)
            return None

        df = pd.read_csv(filepath)
        rows_in = len(df)
        self.logger.info("  Lignes brutes : %d", rows_in)
        self.logger.info("  Colonnes : %s", list(df.columns))

        # 1. Filtrer les périodes mensuelles (YYYY-MM)
        if "period" in df.columns:
            mask_monthly = df["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
            non_monthly = (~mask_monthly).sum()
            if non_monthly > 0:
                self.logger.info(
                    "  %d lignes non mensuelles filtrées (trimestrielles, etc.)",
                    non_monthly,
                )
            df = df[mask_monthly].copy()

        # 2. Doublons
        dups = df.duplicated(subset=["period"]).sum()
        df = df.drop_duplicates(subset=["period"], keep="last")

        # 3. Conversion date_id
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # 4. Tri chronologique (nécessaire pour l'interpolation)
        df = df.sort_values("date_id").reset_index(drop=True)

        # 5. Interpolation linéaire des gaps courts (max 3 mois)
        # Cela comble les trous ponctuels dans les séries INSEE
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        indicator_cols = [c for c in numeric_cols if c not in ["date_id"]]
        nulls_before = df[indicator_cols].isna().sum().sum()

        for col in indicator_cols:
            df[col] = df[col].interpolate(method="linear", limit=3)

        nulls_after = df[indicator_cols].isna().sum().sum()
        interpolated = nulls_before - nulls_after

        if interpolated > 0:
            self.logger.info(
                "  %d valeurs interpolées (gaps ≤ 3 mois)", interpolated,
            )

        # 6. Log des NaN restants par colonne
        remaining_nulls = df[indicator_cols].isna().sum()
        for col, count in remaining_nulls.items():
            if count > 0:
                self.logger.warning(
                    "  ⚠ %s : %d NaN restants (%.1f%%)",
                    col, count, 100 * count / len(df),
                )

        # Arrondir
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
            "  ✓ INSEE nettoyé : %d → %d lignes", rows_in, rows_out,
        )
        return df

    # ==================================================================
    # Nettoyage EUROSTAT (IPI)
    # ==================================================================

    def clean_eurostat(self) -> Optional[pd.DataFrame]:
        """Nettoie les données Eurostat IPI.

        Règles de nettoyage :
        1. Filtrage des périodes mensuelles (YYYY-MM)
        2. Suppression des doublons (period × nace_r2)
        3. Détection et flagging des ruptures de série
           (variation > 30% d'un mois à l'autre = suspect)
        4. Interpolation linéaire des gaps courts
        5. Conversion date_id

        Returns:
            DataFrame nettoyé, ou None si le fichier source manque.
        """
        self.logger.info("Nettoyage EUROSTAT...")
        filepath = self.config.raw_data_dir / "eurostat" / "ipi_hvac_france.csv"

        if not filepath.exists():
            self.logger.warning("  Fichier manquant : %s", filepath)
            return None

        df = pd.read_csv(filepath)
        rows_in = len(df)
        self.logger.info("  Lignes brutes : %d", rows_in)
        self.logger.info("  Colonnes : %s", list(df.columns))

        # 1. Filtrer les périodes mensuelles
        if "period" in df.columns:
            mask_monthly = df["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
            non_monthly = (~mask_monthly).sum()
            if non_monthly > 0:
                self.logger.info(
                    "  %d lignes non mensuelles filtrées", non_monthly,
                )
            df = df[mask_monthly].copy()

        # 2. Doublons
        key_cols = ["period", "nace_r2"] if "nace_r2" in df.columns else ["period"]
        dups = df.duplicated(subset=key_cols).sum()
        df = df.drop_duplicates(subset=key_cols, keep="last")

        # 3. Conversion date_id
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # 4. Tri
        df = df.sort_values(["nace_r2", "date_id"] if "nace_r2" in df.columns else ["date_id"])
        df = df.reset_index(drop=True)

        # 5. Interpolation par série NACE
        interpolated = 0
        if "nace_r2" in df.columns and "ipi_value" in df.columns:
            nulls_before = df["ipi_value"].isna().sum()

            # Interpoler par groupe NACE
            df["ipi_value"] = df.groupby("nace_r2")["ipi_value"].transform(
                lambda s: s.interpolate(method="linear", limit=3)
            )

            nulls_after = df["ipi_value"].isna().sum()
            interpolated = nulls_before - nulls_after
            if interpolated > 0:
                self.logger.info(
                    "  %d valeurs IPI interpolées", interpolated,
                )

            # 6. Détection des variations suspectes (> 30%)
            df["ipi_pct_change"] = df.groupby("nace_r2")["ipi_value"].transform(
                lambda s: s.pct_change().abs()
            )
            suspects = (df["ipi_pct_change"] > 0.30).sum()
            if suspects > 0:
                self.logger.warning(
                    "  ⚠ %d variations IPI > 30%% détectées (ruptures potentielles)",
                    suspects,
                )
            # Garder la colonne pour diagnostic mais ne pas supprimer les lignes
            df = df.drop(columns=["ipi_pct_change"])

        # Arrondir
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
            "  ✓ Eurostat nettoyé : %d → %d lignes", rows_in, rows_out,
        )
        return df

    # ==================================================================
    # Nettoyage DPE (ADEME)
    # ==================================================================

    def clean_dpe(self) -> Optional[pd.DataFrame]:
        """Nettoie les DPE bruts ADEME.

        C'est la source la plus volumineuse (~1.4M lignes).
        Le nettoyage est fait par chunks pour gérer la mémoire.

        Règles de nettoyage :
        1. Suppression des doublons sur numero_dpe
        2. Conversion et validation des dates
           - date_etablissement_dpe doit être ≥ 2021-07-01 (DPE v2)
           - date_etablissement_dpe doit être ≤ aujourd'hui
        3. Validation des étiquettes DPE/GES (A-G uniquement)
        4. Clipping des valeurs numériques :
           - surface_habitable_logement : [5, 1000] m²
           - conso_5_usages_par_m2_ep : [0, 1000] kWh/m².an
           - cout_total_5_usages : [0, 50000] EUR/an
        5. Nettoyage des chaînes de caractères (strip, lowercase partiel)
        6. Validation du département (01-74, dans AURA)
        7. Colonnes dérivées : year, month, date_id, is_pac, is_clim

        Returns:
            DataFrame nettoyé, ou None si le fichier source manque.
        """
        self.logger.info("Nettoyage DPE (volumétrie ~1.4M lignes)...")
        filepath = self.config.raw_data_dir / "dpe" / "dpe_france_all.csv"

        if not filepath.exists():
            self.logger.warning("  Fichier manquant : %s", filepath)
            return None

        # Lecture par chunks pour gérer la mémoire
        chunks = []
        rows_in = 0
        dups = 0
        date_invalid = 0
        etiquette_invalid = 0
        outliers_removed = 0

        chunk_size = 200_000
        self.logger.info("  Lecture par chunks de %d lignes...", chunk_size)

        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
            rows_in += len(chunk)

            # 1. Doublons sur numero_dpe
            n_before = len(chunk)
            chunk = chunk.drop_duplicates(subset=["numero_dpe"], keep="last")
            dups += n_before - len(chunk)

            # 2. Dates
            chunk["date_etablissement_dpe"] = pd.to_datetime(
                chunk["date_etablissement_dpe"], errors="coerce"
            )
            # Filtrer : DPE v2 seulement (≥ 2021-07-01) et pas dans le futur
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

            # 3. Étiquettes DPE/GES valides (A-G)
            valid_labels = {"A", "B", "C", "D", "E", "F", "G"}
            if "etiquette_dpe" in chunk.columns:
                mask_etiq = chunk["etiquette_dpe"].isin(valid_labels)
                n_invalid_etiq = (~mask_etiq).sum()
                etiquette_invalid += n_invalid_etiq
                chunk = chunk[mask_etiq].copy()

            # 4. Clipping des valeurs numériques
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
                    # On clipe plutôt que de supprimer (conserver le DPE)
                    chunk[col] = chunk[col].clip(vmin, vmax)

            # 5. Nettoyage des chaînes (strip whitespace)
            str_cols = chunk.select_dtypes(include=["object"]).columns
            for col in str_cols:
                chunk[col] = chunk[col].str.strip()

            # 6. Validation département AURA
            if "code_departement_ban" in chunk.columns:
                valid_depts = set(self.config.geo.departments)
                chunk["code_departement_ban"] = (
                    chunk["code_departement_ban"].astype(str).str.zfill(2)
                )
                chunk = chunk[
                    chunk["code_departement_ban"].isin(valid_depts)
                ].copy()

            # 7. Colonnes dérivées
            chunk["year"] = chunk["date_etablissement_dpe"].dt.year
            chunk["month"] = chunk["date_etablissement_dpe"].dt.month
            chunk["date_id"] = chunk["year"] * 100 + chunk["month"]

            # Détection PAC (même logique que dans db_manager)
            pac_pattern = r"(?i)PAC |PAC$|pompe.*chaleur|thermodynamique"
            chauffage_str = chunk["type_generateur_chauffage_principal"].fillna("")
            froid_str = chunk["type_generateur_froid"].fillna("")
            chunk["is_pac"] = (
                chauffage_str.str.contains(pac_pattern, regex=True)
                | froid_str.str.contains(pac_pattern, regex=True)
            ).astype(int)

            # Climatisation = generateur froid renseigné
            chunk["is_clim"] = (froid_str.str.len() > 0).astype(int)

            # Classe A-B (bâtiment performant)
            chunk["is_classe_ab"] = chunk["etiquette_dpe"].isin(["A", "B"]).astype(int)

            # Supprimer la colonne _score si elle existe (artefact API)
            if "_score" in chunk.columns:
                chunk = chunk.drop(columns=["_score"])

            chunks.append(chunk)

            if (i + 1) % 5 == 0:
                self.logger.info(
                    "  Chunk %d : %d lignes traitées au total", i + 1, rows_in,
                )

        if not chunks:
            self.logger.error("  Aucune donnée DPE après nettoyage")
            return None

        df = pd.concat(chunks, ignore_index=True)

        # Dédoublonnage global (les chunks peuvent avoir des doublons inter-chunks)
        n_before_global = len(df)
        df = df.drop_duplicates(subset=["numero_dpe"], keep="last")
        dups += n_before_global - len(df)

        # Arrondir les flottants
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

        # Log détaillé DPE
        self.logger.info(
            "  ✓ DPE nettoyé : %d → %d lignes", rows_in, rows_out,
        )
        self.logger.info("    - Doublons supprimés : %d", dups)
        self.logger.info("    - Dates invalides : %d", date_invalid)
        self.logger.info("    - Étiquettes invalides : %d", etiquette_invalid)
        self.logger.info("    - Valeurs clippées : %d", outliers_removed)

        # Distribution des étiquettes DPE après nettoyage
        if "etiquette_dpe" in df.columns:
            distrib = df["etiquette_dpe"].value_counts().sort_index()
            self.logger.info("    Distribution DPE nettoyée :\n%s", distrib.to_string())

        # Taux de PAC après nettoyage
        n_pac = df["is_pac"].sum()
        self.logger.info(
            "    PAC détectées : %d (%.1f%%)", n_pac, 100 * n_pac / max(len(df), 1),
        )

        return df

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def _save_cleaned(
        self,
        df: pd.DataFrame,
        subdir: str,
        filename: str,
    ) -> Path:
        """Sauvegarde un DataFrame nettoyé dans data/processed/.

        Args:
            df: DataFrame nettoyé.
            subdir: Sous-répertoire (ex: 'weather', 'insee').
            filename: Nom du fichier CSV.

        Returns:
            Chemin complet du fichier sauvegardé.
        """
        output_dir = self.config.processed_data_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "  Sauvegardé → %s (%.1f Mo)", output_path, size_mb,
        )
        return output_path
