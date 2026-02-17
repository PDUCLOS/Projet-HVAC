# -*- coding: utf-8 -*-
"""
Fusion multi-sources — Construction du dataset ML-ready (Phase 2.4).
====================================================================

Ce module fusionne les données nettoyées de toutes les sources en un
dataset unique prêt pour le feature engineering et la modélisation ML.

Le dataset final a le grain : **mois × département** (8 départements AURA).
Pour chaque couple (mois, département), on a :

    - Variables cibles (Y) : nb_dpe_total, nb_installations_pac, etc.
      → Source : DPE ADEME agrégés
    - Features météo : temp_mean, HDD, CDD, précipitations, jours extrêmes
      → Source : Open-Meteo agrégé par mois × département
    - Features économiques : confiance_menages, climat_affaires, IPI
      → Source : INSEE + Eurostat (nationales → dupliquées par département)

Architecture de jointure :
    DPE (agrégé mois × dept)
    LEFT JOIN Météo (agrégé mois × dept)    ON date_id + dept
    LEFT JOIN INSEE (mois)                   ON date_id
    LEFT JOIN Eurostat (mois × nace)         ON date_id
    JOIN dim_time                             ON date_id
    JOIN dim_geo                              ON dept

Usage :
    >>> from src.processing.merge_datasets import DatasetMerger
    >>> merger = DatasetMerger(config)
    >>> df_ml = merger.build_ml_dataset()
    >>> df_ml.to_csv("data/features/hvac_ml_dataset.csv", index=False)

Extensibilité :
    Pour ajouter une nouvelle source au dataset ML :
    1. Ajouter une méthode _prepare_xxx() qui retourne un DataFrame
       avec date_id (+ dept si données locales)
    2. L'intégrer dans build_ml_dataset() via merge/join
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class DatasetMerger:
    """Fusionne les données nettoyées en dataset ML-ready.

    Le grain final est mois × département. On ne garde que les mois
    où la variable cible (DPE) est disponible (≥ juillet 2021).

    Attributes:
        config: Configuration du projet.
        logger: Logger structuré.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialise le merger avec la configuration projet.

        Args:
            config: Configuration centralisée du projet.
        """
        self.config = config
        self.logger = logging.getLogger("processing.merge")

    def build_ml_dataset(self) -> pd.DataFrame:
        """Construit le dataset ML-ready complet.

        Étapes :
        1. Préparer les DPE agrégés (variable cible + volume)
        2. Préparer la météo agrégée par mois × département
        3. Préparer les indicateurs économiques (INSEE + Eurostat)
        4. Fusionner toutes les sources sur date_id + dept
        5. Ajouter les métadonnées temporelles et géographiques
        6. Filtrer la période d'intérêt (DPE v2 : ≥ 2021-07)
        7. Sauvegarder le dataset final

        Returns:
            DataFrame ML-ready avec toutes les features et variables cibles.
        """
        self.logger.info("=" * 60)
        self.logger.info("  PHASE 2.4 — Fusion multi-sources → Dataset ML")
        self.logger.info("=" * 60)

        # 1. Variable cible : DPE agrégés par mois × département
        df_dpe = self._prepare_dpe_target()
        if df_dpe is None or df_dpe.empty:
            self.logger.error("Impossible de construire le dataset : DPE manquants")
            return pd.DataFrame()
        self.logger.info(
            "  DPE cible : %d lignes (mois × dept)", len(df_dpe),
        )

        # 2. Features météo (mois × département)
        df_meteo = self._prepare_weather_features()
        if df_meteo is not None:
            self.logger.info(
                "  Météo features : %d lignes", len(df_meteo),
            )

        # 3. Features économiques (mois, nationales)
        df_eco = self._prepare_economic_features()
        if df_eco is not None:
            self.logger.info(
                "  Économie features : %d lignes", len(df_eco),
            )

        # 4. Fusion progressive
        # Base = DPE (contient date_id + dept)
        df = df_dpe.copy()

        # Joindre météo (même grain : date_id + dept)
        if df_meteo is not None:
            df = df.merge(
                df_meteo,
                on=["date_id", "dept"],
                how="left",
                suffixes=("", "_meteo"),
            )
            self.logger.info(
                "  Après fusion météo : %d lignes × %d colonnes",
                len(df), len(df.columns),
            )

        # Joindre économie (grain = date_id seul → broadcast sur les départements)
        if df_eco is not None:
            df = df.merge(
                df_eco,
                on="date_id",
                how="left",
                suffixes=("", "_eco"),
            )
            self.logger.info(
                "  Après fusion économie : %d lignes × %d colonnes",
                len(df), len(df.columns),
            )

        # 5. Ajouter les métadonnées temporelles
        df = self._add_time_features(df)

        # 6. Ajouter les métadonnées géographiques
        df = self._add_geo_features(df)

        # 7. Filtrer la période DPE v2 (≥ 2021-07)
        dpe_start = int(
            self.config.time.dpe_start_date[:4]
            + self.config.time.dpe_start_date[5:7]
        )
        df = df[df["date_id"] >= dpe_start].copy()
        self.logger.info(
            "  Après filtre DPE v2 (>= %d) : %d lignes", dpe_start, len(df),
        )

        # 8. Tri final
        df = df.sort_values(["date_id", "dept"]).reset_index(drop=True)

        # 9. Log du dataset final
        self.logger.info("=" * 60)
        self.logger.info("  DATASET ML-READY")
        self.logger.info("  Dimensions : %d lignes × %d colonnes", len(df), len(df.columns))
        self.logger.info("  Colonnes : %s", list(df.columns))
        self.logger.info("  Période : %d → %d", df["date_id"].min(), df["date_id"].max())
        self.logger.info(
            "  Départements : %s",
            sorted(df["dept"].unique().tolist()),
        )

        # Stats NaN
        null_pct = df.isna().mean() * 100
        cols_with_nulls = null_pct[null_pct > 0].sort_values(ascending=False)
        if len(cols_with_nulls) > 0:
            self.logger.info("  Colonnes avec NaN :")
            for col, pct in cols_with_nulls.items():
                self.logger.info("    %-30s : %.1f%% NaN", col, pct)
        else:
            self.logger.info("  Aucun NaN dans le dataset ✓")
        self.logger.info("=" * 60)

        # 10. Sauvegarder
        output_path = self._save_ml_dataset(df)
        self.logger.info("  ✓ Dataset ML sauvegardé → %s", output_path)

        return df

    # ==================================================================
    # Préparation de chaque source
    # ==================================================================

    def _prepare_dpe_target(self) -> Optional[pd.DataFrame]:
        """Prépare la variable cible : DPE agrégés par mois × département.

        Agrège les DPE nettoyés pour calculer par mois × département :
        - nb_dpe_total : nombre total de DPE
        - nb_installations_pac : DPE avec PAC détectée
        - nb_installations_clim : DPE avec climatisation
        - nb_dpe_classe_ab : DPE classe A ou B
        - pct_pac : pourcentage de PAC parmi les DPE
        - pct_clim : pourcentage de climatisation

        Returns:
            DataFrame agrégé, ou None si le fichier est manquant.
        """
        # Chercher d'abord le fichier nettoyé, sinon le brut
        clean_path = self.config.processed_data_dir / "dpe" / "dpe_france_clean.csv"
        raw_path = self.config.raw_data_dir / "dpe" / "dpe_france_all.csv"

        filepath = clean_path if clean_path.exists() else raw_path
        if not filepath.exists():
            self.logger.error("Fichier DPE introuvable : ni %s ni %s", clean_path, raw_path)
            return None

        self.logger.info("  Lecture DPE depuis %s...", filepath.name)

        # Détecter si le fichier contient les colonnes pré-calculées
        # (fichier nettoyé les a, le brut non)
        sample = pd.read_csv(filepath, nrows=2)
        has_precalc = "is_pac" in sample.columns
        self.logger.info(
            "  Colonnes pré-calculées : %s", "oui" if has_precalc else "non",
        )

        # Lecture par chunks (low_memory=False pour éviter les DtypeWarning)
        chunks_agg = []
        for chunk in pd.read_csv(filepath, chunksize=200_000, low_memory=False):
            # Si les colonnes dérivées n'existent pas, les calculer
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

            # Préparer la colonne département
            chunk["dept"] = (
                chunk["code_departement_ban"].astype(str).str.zfill(2)
            )

            # Agréger le chunk
            agg = chunk.groupby(["date_id", "dept"]).agg(
                nb_dpe_total=("is_pac", "count"),
                nb_installations_pac=("is_pac", "sum"),
                nb_installations_clim=("is_clim", "sum"),
                nb_dpe_classe_ab=("is_classe_ab", "sum"),
            ).reset_index()

            chunks_agg.append(agg)

        # Fusionner tous les chunks agrégés
        df_agg = pd.concat(chunks_agg, ignore_index=True)

        # Re-agréger (somme) car un même mois×dept peut apparaître dans plusieurs chunks
        df_agg = df_agg.groupby(["date_id", "dept"]).agg({
            "nb_dpe_total": "sum",
            "nb_installations_pac": "sum",
            "nb_installations_clim": "sum",
            "nb_dpe_classe_ab": "sum",
        }).reset_index()

        # Convertir en int
        for col in ["nb_dpe_total", "nb_installations_pac",
                     "nb_installations_clim", "nb_dpe_classe_ab"]:
            df_agg[col] = df_agg[col].astype(int)

        # Pourcentages dérivés (features utiles pour le ML)
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
        """Prépare les features météo agrégées par mois × département.

        Agrège les données météo quotidiennes en métriques mensuelles :
        - temp_mean, temp_max, temp_min : statistiques de température
        - hdd_sum, cdd_sum : degrés-jours cumulés
        - precipitation_sum : précipitations mensuelles
        - nb_jours_canicule : jours avec T > 35°C
        - nb_jours_gel : jours avec T < 0°C
        - wind_max : vitesse max de vent

        Returns:
            DataFrame agrégé, ou None si le fichier est manquant.
        """
        clean_path = self.config.processed_data_dir / "weather" / "weather_france.csv"
        raw_path = self.config.raw_data_dir / "weather" / "weather_france.csv"

        filepath = clean_path if clean_path.exists() else raw_path
        if not filepath.exists():
            self.logger.warning("Fichier météo introuvable")
            return None

        df = pd.read_csv(filepath)

        # Identifier les colonnes
        date_col = "time" if "time" in df.columns else "date"
        df[date_col] = pd.to_datetime(df[date_col])

        if "date_id" not in df.columns:
            df["date_id"] = df[date_col].dt.year * 100 + df[date_col].dt.month

        if "dept" in df.columns:
            df["dept"] = df["dept"].astype(str).str.zfill(2)
        else:
            self.logger.error("  Colonne 'dept' manquante dans météo")
            return None

        # Colonnes d'agrégation
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

        # Indicateurs binaires à compter
        if "temperature_2m_max" in df.columns:
            df["_canicule"] = (df["temperature_2m_max"] > 35).astype(int)
            agg_dict["_canicule"] = "sum"
        if "temperature_2m_min" in df.columns:
            df["_gel"] = (df["temperature_2m_min"] < 0).astype(int)
            agg_dict["_gel"] = "sum"

        # Agrégation mois × département
        monthly = df.groupby(["date_id", "dept"]).agg(agg_dict).reset_index()

        # Renommer les colonnes
        rename = {}
        for src_col, (dst_col, _) in col_map.items():
            if src_col in monthly.columns:
                rename[src_col] = dst_col
        if "_canicule" in monthly.columns:
            rename["_canicule"] = "nb_jours_canicule"
        if "_gel" in monthly.columns:
            rename["_gel"] = "nb_jours_gel"

        monthly = monthly.rename(columns=rename)

        # Arrondir
        float_cols = monthly.select_dtypes(include=["float64"]).columns
        monthly[float_cols] = monthly[float_cols].round(2)

        return monthly

    def _prepare_economic_features(self) -> Optional[pd.DataFrame]:
        """Prépare les features économiques (INSEE + Eurostat).

        Les indicateurs économiques sont nationaux (pas départementaux).
        On les fusionne en un seul DataFrame au grain mensuel (date_id).
        Lors de la jointure avec le dataset principal, ils seront
        broadcast sur tous les départements.

        Returns:
            DataFrame économique, ou None si les fichiers sont manquants.
        """
        df_eco = None

        # --- INSEE ---
        clean_insee = self.config.processed_data_dir / "insee" / "indicateurs_economiques.csv"
        raw_insee = self.config.raw_data_dir / "insee" / "indicateurs_economiques.csv"
        insee_path = clean_insee if clean_insee.exists() else raw_insee

        if insee_path.exists():
            df_insee = pd.read_csv(insee_path)

            # Filtrer les périodes mensuelles
            if "period" in df_insee.columns:
                mask = df_insee["period"].astype(str).str.match(r"^\d{4}-\d{2}$")
                df_insee = df_insee[mask].copy()

            if "date_id" not in df_insee.columns:
                df_insee["date_id"] = df_insee["period"].str.replace("-", "").astype(int)

            # Renommer les colonnes pour le ML
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

            # Garder uniquement les colonnes utiles
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

            # Pivoter : une colonne par code NACE
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

                # Fusionner avec INSEE
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

            # Arrondir
            float_cols = df_eco.select_dtypes(include=["float64"]).columns
            df_eco[float_cols] = df_eco[float_cols].round(2)

        return df_eco

    # ==================================================================
    # Enrichissement
    # ==================================================================

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features temporelles au dataset.

        Features ajoutées :
        - year, month, quarter
        - is_heating : saison de chauffage (octobre-mars)
        - is_cooling : saison de climatisation (juin-septembre)
        - month_sin, month_cos : encoding cyclique du mois

        Args:
            df: DataFrame avec colonne date_id.

        Returns:
            DataFrame enrichi.
        """
        df["year"] = df["date_id"] // 100
        df["month"] = df["date_id"] % 100
        df["quarter"] = ((df["month"] - 1) // 3) + 1

        # Saisons HVAC
        df["is_heating"] = df["month"].isin([1, 2, 3, 10, 11, 12]).astype(int)
        df["is_cooling"] = df["month"].isin([6, 7, 8, 9]).astype(int)

        # Encoding cyclique (capture la continuité décembre ↔ janvier)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)

        return df

    def _add_geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les métadonnées géographiques.

        Features ajoutées :
        - dept_name : nom du département
        - city_ref : ville de référence météo
        - latitude, longitude : coordonnées de la ville de référence

        Args:
            df: DataFrame avec colonne dept.

        Returns:
            DataFrame enrichi.
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
        """Retourne le nom d'un département AURA à partir de son code.

        Args:
            code: Code département (01, 07, etc.).

        Returns:
            Nom du département.
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
    # Sauvegarde
    # ==================================================================

    def _save_ml_dataset(self, df: pd.DataFrame) -> Path:
        """Sauvegarde le dataset ML final.

        Le fichier est sauvegardé dans data/features/ (répertoire ML).

        Args:
            df: Dataset ML-ready complet.

        Returns:
            Chemin du fichier sauvegardé.
        """
        output_dir = self.config.features_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hvac_ml_dataset.csv"

        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "Dataset ML : %d lignes × %d cols → %s (%.1f Mo)",
            len(df), len(df.columns), output_path, size_mb,
        )
        return output_path
