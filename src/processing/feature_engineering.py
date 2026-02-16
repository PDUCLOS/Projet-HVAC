# -*- coding: utf-8 -*-
"""
Feature Engineering — Construction des features avancées pour le ML (Phase 3.3).
================================================================================

Ce module transforme le dataset ML-ready (fusion Phase 2.4) en ajoutant
des features avancées conçues pour améliorer les modèles prédictifs.

Catégories de features :

    1. **Lags temporels** : valeurs décalées dans le temps
       - lag_1m, lag_3m, lag_6m pour la variable cible et les features clés
       - Permettent de capturer l'auto-corrélation et la tendance

    2. **Rolling windows** : moyennes et écarts-types glissants
       - rolling_mean_3m, rolling_mean_6m, rolling_std_3m
       - Lissent le bruit et capturent la dynamique de moyen terme

    3. **Variations** : différences et taux de croissance
       - diff_1m (différence absolue mois à mois)
       - pct_change_1m (variation relative en %)
       - Capturent l'accélération/décélération du marché

    4. **Features d'interaction** : produits et ratios entre variables
       - hdd × confiance_menages (météo froide + contexte favorable)
       - cdd × ipi_hvac (canicule + activité industrielle)

    5. **Encoding cyclique** : transformations sin/cos
       - Déjà ajouté dans merge_datasets (month_sin, month_cos)

IMPORTANT — Gestion des NaN :
    Les lags et rolling créent des NaN en début de série. On ne les supprime
    PAS ici (le modèle gèrera via imputation ou split temporel). La colonne
    `n_valid_features` permet de filtrer a posteriori si besoin.

Usage :
    >>> from src.processing.feature_engineering import FeatureEngineer
    >>> fe = FeatureEngineer(config)
    >>> df_features = fe.engineer(df_ml)
    >>> # Ou directement depuis le fichier :
    >>> df_features = fe.engineer_from_file()

Extensibilité :
    Pour ajouter de nouvelles features :
    1. Créer une méthode _add_xxx_features(df) dans FeatureEngineer
    2. L'appeler dans engineer()
    3. Documenter la feature dans le docstring et le data dictionary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class FeatureEngineer:
    """Génère les features avancées pour la modélisation ML.

    Travaille sur le dataset ML-ready (grain = mois × département)
    et ajoute des features temporelles, statistiques et d'interaction.

    Attributes:
        config: Configuration du projet (lags, rolling windows, etc.).
        logger: Logger structuré.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialise le feature engineer.

        Args:
            config: Configuration centralisée du projet.
        """
        self.config = config
        self.logger = logging.getLogger("processing.features")

        # Paramètres depuis la config ML
        self.max_lag = config.model.max_lag_months        # 6 mois max
        self.rolling_windows = config.model.rolling_windows  # [3, 6]

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique toutes les transformations de features.

        Le DataFrame DOIT contenir au minimum :
        - date_id (int YYYYMM)
        - dept (str, code département)
        - nb_dpe_total, nb_installations_pac (variables cibles)
        - temp_mean, hdd_sum, cdd_sum (features météo)

        Args:
            df: Dataset ML-ready (sortie de DatasetMerger.build_ml_dataset).

        Returns:
            DataFrame enrichi avec toutes les features engineerées.
        """
        self.logger.info("=" * 60)
        self.logger.info("  FEATURE ENGINEERING")
        self.logger.info("=" * 60)
        self.logger.info("  Dataset en entrée : %d lignes × %d colonnes", len(df), len(df.columns))

        n_cols_start = len(df.columns)

        # S'assurer du tri (critique pour les opérations temporelles)
        df = df.sort_values(["dept", "date_id"]).reset_index(drop=True)

        # 1. Lags temporels
        df = self._add_lag_features(df)

        # 2. Rolling windows
        df = self._add_rolling_features(df)

        # 3. Variations (diff + pct_change)
        df = self._add_variation_features(df)

        # 4. Features d'interaction
        df = self._add_interaction_features(df)

        # 5. Features de tendance
        df = self._add_trend_features(df)

        # 6. Feature de complétude
        df = self._add_completeness_flag(df)

        n_cols_end = len(df.columns)
        self.logger.info(
            "  Features ajoutées : %d nouvelles colonnes (%d → %d)",
            n_cols_end - n_cols_start, n_cols_start, n_cols_end,
        )

        # Sauvegarder le dataset enrichi
        output_path = self._save_features(df)
        self.logger.info("  ✓ Features sauvegardées → %s", output_path)
        self.logger.info("=" * 60)

        return df

    def engineer_from_file(self) -> pd.DataFrame:
        """Charge le dataset ML et applique le feature engineering.

        Méthode de convenance qui lit le fichier hvac_ml_dataset.csv
        puis appelle engineer().

        Returns:
            DataFrame avec features engineerées.

        Raises:
            FileNotFoundError: Si le dataset ML n'existe pas.
        """
        ml_path = self.config.features_data_dir / "hvac_ml_dataset.csv"
        if not ml_path.exists():
            raise FileNotFoundError(
                f"Dataset ML introuvable : {ml_path}. "
                f"Lancer d'abord 'python -m src.pipeline merge'."
            )

        df = pd.read_csv(ml_path)
        self.logger.info("  Chargé %s : %d lignes", ml_path.name, len(df))
        return self.engineer(df)

    # ==================================================================
    # 1. Lags temporels
    # ==================================================================

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features de lag (valeurs décalées dans le temps).

        Pour chaque variable cible et feature clé, crée des colonnes
        lag_Xm contenant la valeur de X mois avant.

        Lags générés : 1, 3, 6 mois (configurable via max_lag_months).

        IMPORTANT : les lags créent des NaN en début de série pour
        chaque département. Ces NaN seront gérés par le modèle ou
        supprimés au moment du split train/test.

        Args:
            df: DataFrame trié par dept + date_id.

        Returns:
            DataFrame avec colonnes lag_Xm ajoutées.
        """
        # Colonnes sur lesquelles créer des lags
        target_cols = [
            "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
        ]
        feature_cols = [
            "temp_mean", "hdd_sum", "cdd_sum",
            "confiance_menages",
        ]
        lag_cols = [c for c in target_cols + feature_cols if c in df.columns]

        # Lags à générer
        lags = [1, 3, 6]
        lags = [l for l in lags if l <= self.max_lag]

        n_features = 0
        for col in lag_cols:
            for lag in lags:
                lag_name = f"{col}_lag_{lag}m"
                df[lag_name] = df.groupby("dept")[col].shift(lag)
                n_features += 1

        self.logger.info(
            "  Lags : %d features (colonnes=%d, lags=%s)",
            n_features, len(lag_cols), lags,
        )
        return df

    # ==================================================================
    # 2. Rolling windows (moyennes et écarts-types glissants)
    # ==================================================================

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features de fenêtre glissante.

        Pour chaque variable cible et feature clé, calcule la moyenne
        glissante et l'écart-type glissant sur des fenêtres de 3 et 6 mois.

        Le rolling est calculé PAR DÉPARTEMENT (groupby dept).

        NOTE : min_periods=1 pour ne pas créer de NaN inutiles
        en début de série (la fenêtre est simplement réduite).

        Args:
            df: DataFrame trié par dept + date_id.

        Returns:
            DataFrame avec colonnes rolling ajoutées.
        """
        roll_cols = [
            "nb_dpe_total", "nb_installations_pac",
            "temp_mean", "hdd_sum", "cdd_sum",
        ]
        roll_cols = [c for c in roll_cols if c in df.columns]

        n_features = 0
        for col in roll_cols:
            for window in self.rolling_windows:
                # Moyenne glissante
                mean_name = f"{col}_rmean_{window}m"
                df[mean_name] = df.groupby("dept")[col].transform(
                    lambda s: s.rolling(window, min_periods=1).mean()
                ).round(2)
                n_features += 1

                # Écart-type glissant (mesure de volatilité)
                std_name = f"{col}_rstd_{window}m"
                df[std_name] = df.groupby("dept")[col].transform(
                    lambda s: s.rolling(window, min_periods=2).std()
                ).round(2)
                n_features += 1

        self.logger.info(
            "  Rolling : %d features (colonnes=%d, windows=%s)",
            n_features, len(roll_cols), self.rolling_windows,
        )
        return df

    # ==================================================================
    # 3. Variations (diff + pct_change)
    # ==================================================================

    def _add_variation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features de variation temporelle.

        Pour les variables cibles :
        - diff_1m : différence absolue avec le mois précédent
        - pct_change_1m : variation relative en % avec le mois précédent

        Capturent la dynamique du marché (accélération/décélération).

        Args:
            df: DataFrame trié par dept + date_id.

        Returns:
            DataFrame avec colonnes de variation ajoutées.
        """
        var_cols = [
            "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
        ]
        var_cols = [c for c in var_cols if c in df.columns]

        n_features = 0
        for col in var_cols:
            # Différence absolue
            diff_name = f"{col}_diff_1m"
            df[diff_name] = df.groupby("dept")[col].diff(1)
            n_features += 1

            # Variation relative (%)
            pct_name = f"{col}_pct_1m"
            df[pct_name] = (
                df.groupby("dept")[col].pct_change(1) * 100
            ).round(2)
            # Clipper les variations extrêmes (divisions par ~0)
            if pct_name in df.columns:
                df[pct_name] = df[pct_name].clip(-200, 500)
            n_features += 1

        self.logger.info(
            "  Variations : %d features (colonnes=%d)", n_features, len(var_cols),
        )
        return df

    # ==================================================================
    # 4. Features d'interaction
    # ==================================================================

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les features d'interaction entre variables.

        Hypothèses métier :
        - Un hiver froid (HDD élevé) + confiance des ménages élevée
          → plus de remplacement de chauffage (PAC)
        - Un été chaud (CDD élevé) + activité industrielle HVAC forte
          → plus d'installations de climatisation
        - Confiance des ménages × climat du bâtiment
          → proxy de l'intention d'investissement des ménages

        Args:
            df: DataFrame avec features météo et économiques.

        Returns:
            DataFrame avec features d'interaction ajoutées.
        """
        n_features = 0

        # HDD × confiance ménages (hiver froid + confiance → investissement chauffage)
        if "hdd_sum" in df.columns and "confiance_menages" in df.columns:
            # Normaliser avant de multiplier pour éviter les échelles disparates
            hdd_max = max(df["hdd_sum"].max(), 1)
            hdd_norm = df["hdd_sum"] / hdd_max
            conf_norm = df["confiance_menages"] / 100  # Base 100
            df["interact_hdd_confiance"] = (hdd_norm * conf_norm).round(4)
            n_features += 1

        # CDD × IPI HVAC (chaleur + production industrielle → installations clim)
        if "cdd_sum" in df.columns and "ipi_hvac_c28" in df.columns:
            cdd_max = max(df["cdd_sum"].max(), 1)
            cdd_norm = df["cdd_sum"] / cdd_max
            ipi_norm = df["ipi_hvac_c28"] / 100  # Indice base ~100
            df["interact_cdd_ipi"] = (cdd_norm * ipi_norm).round(4)
            n_features += 1

        # Confiance × climat bâtiment (double proxy d'investissement)
        if "confiance_menages" in df.columns and "climat_affaires_bat" in df.columns:
            df["interact_confiance_bat"] = (
                (df["confiance_menages"] / 100) * (df["climat_affaires_bat"] / 100)
            ).round(4)
            n_features += 1

        # Température extrême flag composite
        if "nb_jours_canicule" in df.columns and "nb_jours_gel" in df.columns:
            df["jours_extremes"] = df["nb_jours_canicule"] + df["nb_jours_gel"]
            n_features += 1

        # Ratio prix gaz / électricité (driver principal adoption PAC)
        if "ipc_gaz" in df.columns and "ipc_electricite" in df.columns:
            elec = df["ipc_electricite"].clip(lower=1)
            df["ratio_prix_gaz_elec"] = (df["ipc_gaz"] / elec).round(4)
            n_features += 1

        # Interaction prix gaz × HDD (gaz cher + hiver froid → switch PAC)
        if "ipc_gaz" in df.columns and "hdd_sum" in df.columns:
            gaz_norm = df["ipc_gaz"] / 100
            hdd_max = max(df["hdd_sum"].max(), 1)
            df["interact_gaz_hdd"] = (gaz_norm * (df["hdd_sum"] / hdd_max)).round(4)
            n_features += 1

        self.logger.info("  Interactions : %d features", n_features)
        return df

    # ==================================================================
    # 5. Features de tendance
    # ==================================================================

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des features de tendance à plus long terme.

        Features ajoutées :
        - year_trend : année normalisée (2021=0, 2022=1, ...)
          Capture la tendance séculaire (croissance du marché PAC)
        - delta_temp_vs_mean : écart de température par rapport
          à la moyenne historique du département
          Capture les anomalies météo qui déclenchent des achats

        Args:
            df: DataFrame avec colonnes year et temp_mean.

        Returns:
            DataFrame enrichi.
        """
        n_features = 0

        # Tendance annuelle (linéaire)
        if "year" in df.columns:
            year_min = df["year"].min()
            df["year_trend"] = df["year"] - year_min
            n_features += 1

        # Écart de température vs moyenne du département
        if "temp_mean" in df.columns and "dept" in df.columns and "month" in df.columns:
            # Calculer la moyenne historique par dept × mois
            temp_avg = df.groupby(["dept", "month"])["temp_mean"].transform("mean")
            df["delta_temp_vs_mean"] = (df["temp_mean"] - temp_avg).round(2)
            n_features += 1

        self.logger.info("  Tendance : %d features", n_features)
        return df

    # ==================================================================
    # 6. Flag de complétude
    # ==================================================================

    def _add_completeness_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute un indicateur de complétude des features.

        La colonne `n_valid_features` compte le nombre de features
        non-NaN pour chaque ligne. Utile pour filtrer les lignes
        avec trop peu de données (début de série où les lags créent
        des NaN).

        Args:
            df: DataFrame avec toutes les features.

        Returns:
            DataFrame avec colonne n_valid_features.
        """
        # Exclure les colonnes non-feature (identifiants, métadonnées)
        exclude_cols = {
            "date_id", "dept", "dept_name", "city_ref",
            "latitude", "longitude", "year", "month", "quarter",
        }
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        df["n_valid_features"] = df[feature_cols].notna().sum(axis=1)
        df["pct_valid_features"] = (
            100 * df["n_valid_features"] / len(feature_cols)
        ).round(1)

        self.logger.info(
            "  Complétude : min=%.0f%%, median=%.0f%%, max=%.0f%%",
            df["pct_valid_features"].min(),
            df["pct_valid_features"].median(),
            df["pct_valid_features"].max(),
        )
        return df

    # ==================================================================
    # Sauvegarde
    # ==================================================================

    def _save_features(self, df: pd.DataFrame) -> Path:
        """Sauvegarde le dataset avec features engineerées.

        Args:
            df: Dataset enrichi.

        Returns:
            Chemin du fichier sauvegardé.
        """
        output_dir = self.config.features_data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "hvac_features_dataset.csv"

        df.to_csv(output_path, index=False)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        self.logger.info(
            "Features dataset : %d lignes × %d cols → %s (%.1f Mo)",
            len(df), len(df.columns), output_path, size_mb,
        )
        return output_path

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retourne un résumé statistique de toutes les features.

        Utile pour le diagnostic et l'exploration rapide.

        Args:
            df: Dataset avec features.

        Returns:
            DataFrame avec count, mean, std, min, max, %NaN par feature.
        """
        summary = df.describe().T
        summary["pct_nan"] = (df.isna().mean() * 100).round(1)
        summary["dtype"] = df.dtypes
        return summary.sort_values("pct_nan", ascending=False)
