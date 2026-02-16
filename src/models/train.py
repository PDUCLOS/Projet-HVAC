# -*- coding: utf-8 -*-
"""
Pipeline d'entrainement — Phase 4 : Modelisation ML.
=====================================================

Orchestre l'entrainement de tous les modeles (baseline + DL) en respectant
le split temporel defini dans la configuration :

    Train  : 2021-07 -> 2024-06 (36 mois × 8 depts = ~288 lignes)
    Val    : 2024-07 -> 2024-12 (6 mois × 8 depts = ~48 lignes)
    Test   : 2025-01 -> 2025-12 (12 mois × 8 depts = ~96 lignes)

Le grain est mois × departement. Les NaN en debut de serie (lags, rolling)
sont supprimes a l'entrainement.

Usage :
    >>> from src.models.train import ModelTrainer
    >>> trainer = ModelTrainer(config)
    >>> results = trainer.train_all()

    # Ou via CLI
    python -m src.pipeline train
    python -m src.pipeline train --target nb_installations_pac
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import ProjectConfig


class ModelTrainer:
    """Orchestre l'entrainement et la validation de tous les modeles.

    Responsabilites :
    - Charger le dataset features
    - Appliquer le split temporel (train / val / test)
    - Selectionner les features pertinentes (supprimer identifiants, NaN)
    - Entrainer chaque modele avec cross-validation temporelle
    - Sauvegarder les modeles entraines et les metriques

    Attributes:
        config: Configuration du projet.
        target: Variable cible a predire.
        models_dir: Repertoire de sauvegarde des modeles.
    """

    # Features exclues de l'entrainement (identifiants, metadata, cibles)
    EXCLUDE_COLS = {
        "date_id", "dept", "dept_name", "city_ref",
        "latitude", "longitude",
        "n_valid_features", "pct_valid_features",
    }

    # Variables cibles disponibles
    TARGET_COLS = {
        "nb_installations_pac", "nb_installations_clim",
        "nb_dpe_total", "nb_dpe_classe_ab",
    }

    def __init__(
        self,
        config: ProjectConfig,
        target: str = "nb_installations_pac",
    ) -> None:
        self.config = config
        self.target = target
        self.logger = logging.getLogger("models.train")
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Split dates (YYYYMM format)
        self.train_end = int(
            config.time.train_end[:4] + config.time.train_end[5:7]
        )
        self.val_end = int(
            config.time.val_end[:4] + config.time.val_end[5:7]
        )

    def load_dataset(self) -> pd.DataFrame:
        """Charge le dataset features depuis le CSV.

        Returns:
            DataFrame avec toutes les features engineerees.

        Raises:
            FileNotFoundError: Si le dataset n'existe pas.
        """
        path = self.config.features_data_dir / "hvac_features_dataset.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset features introuvable : {path}. "
                f"Lancer 'python -m src.pipeline features' d'abord."
            )
        df = pd.read_csv(path)
        self.logger.info("Dataset charge : %d lignes x %d colonnes", len(df), len(df.columns))
        return df

    def temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Applique le split temporel train / val / test.

        Le split respecte la chronologie (pas de fuite temporelle).

        Args:
            df: Dataset complet avec colonne date_id.

        Returns:
            Tuple (df_train, df_val, df_test).
        """
        df_train = df[df["date_id"] <= self.train_end].copy()
        df_val = df[
            (df["date_id"] > self.train_end)
            & (df["date_id"] <= self.val_end)
        ].copy()
        df_test = df[df["date_id"] > self.val_end].copy()

        self.logger.info(
            "Split temporel : train=%d, val=%d, test=%d",
            len(df_train), len(df_val), len(df_test),
        )
        return df_train, df_val, df_test

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Separe features (X) et cible (y), supprime les colonnes non pertinentes.

        Args:
            df: DataFrame avec toutes les colonnes.

        Returns:
            Tuple (X, y) prets pour l'entrainement.
        """
        # Colonnes a exclure
        drop_cols = self.EXCLUDE_COLS | (self.TARGET_COLS - {self.target})

        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and c != self.target
        ]

        # Ne garder que les colonnes numeriques
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        y = df[self.target].copy()

        return X, y

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Retourne la liste des noms de features utilisees.

        Args:
            df: DataFrame source.

        Returns:
            Liste des noms de colonnes features.
        """
        X, _ = self.prepare_features(df)
        return list(X.columns)

    def train_all(
        self,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Entraine tous les modeles et retourne les resultats.

        Pipeline complet :
        1. Charger le dataset
        2. Split temporel
        3. Supprimer les lignes avec NaN dans la cible
        4. Entrainer Ridge, LightGBM, Prophet
        5. Entrainer LSTM (si disponible)
        6. Evaluer sur validation et test
        7. Sauvegarder modeles et metriques

        Args:
            target: Variable cible (override de self.target).

        Returns:
            Dictionnaire avec resultats par modele.
        """
        if target:
            self.target = target

        self.logger.info("=" * 60)
        self.logger.info("  PHASE 4 — Entrainement des modeles")
        self.logger.info("  Cible : %s", self.target)
        self.logger.info("=" * 60)

        # 1. Charger
        df = self.load_dataset()

        # 2. Split temporel
        df_train, df_val, df_test = self.temporal_split(df)

        # 3. Preparer features
        X_train, y_train = self.prepare_features(df_train)
        X_val, y_val = self.prepare_features(df_val)
        X_test, y_test = self.prepare_features(df_test)

        # 4. Supprimer les lignes avec NaN dans la cible
        mask_train = y_train.notna()
        X_train, y_train = X_train[mask_train], y_train[mask_train]
        mask_val = y_val.notna()
        X_val, y_val = X_val[mask_val], y_val[mask_val]
        mask_test = y_test.notna()
        X_test, y_test = X_test[mask_test], y_test[mask_test]

        self.logger.info(
            "Apres suppression NaN cible : train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        # 5. Imputer les NaN dans les features (median)
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_train_imp = pd.DataFrame(
            imputer.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index,
        )
        X_val_imp = pd.DataFrame(
            imputer.transform(X_val),
            columns=X_val.columns, index=X_val.index,
        )
        X_test_imp = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )

        # 6. Scaler (pour Ridge et LSTM)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_imp),
            columns=X_train_imp.columns, index=X_train_imp.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val_imp),
            columns=X_val_imp.columns, index=X_val_imp.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test_imp),
            columns=X_test_imp.columns, index=X_test_imp.index,
        )

        # Sauvegarder imputer et scaler
        self._save_artifact(imputer, "imputer.pkl")
        self._save_artifact(scaler, "scaler.pkl")

        # 7. Entrainer les modeles
        results = {}

        # --- Modeles baseline ---
        from src.models.baseline import BaselineModels

        baseline = BaselineModels(self.config, self.target)

        # Ridge Regression (utilise donnees scalees)
        self.logger.info("-" * 40)
        self.logger.info("Entrainement Ridge Regression...")
        ridge_result = baseline.train_ridge(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test,
        )
        results["ridge"] = ridge_result
        self._save_artifact(ridge_result["model"], "ridge_model.pkl")

        # LightGBM (utilise donnees imputees, pas scalees)
        self.logger.info("-" * 40)
        self.logger.info("Entrainement LightGBM...")
        lgbm_result = baseline.train_lightgbm(
            X_train_imp, y_train,
            X_val_imp, y_val,
            X_test_imp, y_test,
        )
        results["lightgbm"] = lgbm_result
        self._save_artifact(lgbm_result["model"], "lightgbm_model.pkl")

        # Prophet
        self.logger.info("-" * 40)
        self.logger.info("Entrainement Prophet...")
        prophet_result = baseline.train_prophet(
            df_train, df_val, df_test,
        )
        results["prophet"] = prophet_result

        # --- LSTM (exploratoire) ---
        try:
            from src.models.deep_learning import LSTMModel

            self.logger.info("-" * 40)
            self.logger.info("Entrainement LSTM (exploratoire)...")
            lstm_model = LSTMModel(self.config, self.target)
            lstm_result = lstm_model.train_and_evaluate(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
            )
            results["lstm"] = lstm_result
        except ImportError:
            self.logger.warning(
                "PyTorch non disponible — LSTM ignore. "
                "Installer via : pip install -r requirements-dl.txt"
            )
        except Exception as e:
            self.logger.warning("LSTM echoue : %s", e)

        # 8. Cross-validation temporelle (sur Ridge et LightGBM)
        self.logger.info("-" * 40)
        self.logger.info("Cross-validation temporelle...")
        cv_results = self._cross_validate(
            X_train_imp, y_train, X_train_scaled, y_train, baseline,
        )
        for model_name, cv_scores in cv_results.items():
            results[model_name]["cv_scores"] = cv_scores

        # 9. Sauvegarder le resume
        self._save_results_summary(results)

        self.logger.info("=" * 60)
        self.logger.info("  Entrainement termine — %d modeles", len(results))
        for name, res in results.items():
            val_rmse = res.get("metrics_val", {}).get("rmse", float("nan"))
            test_rmse = res.get("metrics_test", {}).get("rmse", float("nan"))
            self.logger.info(
                "  %-12s | Val RMSE=%.2f | Test RMSE=%.2f",
                name, val_rmse, test_rmse,
            )
        self.logger.info("=" * 60)

        return results

    def _cross_validate(
        self,
        X_train_imp: pd.DataFrame,
        y_train: pd.Series,
        X_train_scaled: pd.DataFrame,
        y_train_scaled: pd.Series,
        baseline: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Cross-validation temporelle avec TimeSeriesSplit.

        Args:
            X_train_imp: Features imputees (pour LightGBM).
            y_train: Cible.
            X_train_scaled: Features scalees (pour Ridge).
            y_train_scaled: Cible (identique).
            baseline: Instance de BaselineModels.

        Returns:
            Scores de CV par modele.
        """
        from src.models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(self.config)
        n_splits = min(3, len(X_train_imp) // 20)
        if n_splits < 2:
            self.logger.warning("Pas assez de donnees pour la cross-validation")
            return {}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results: Dict[str, List[Dict]] = {"ridge": [], "lightgbm": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_imp)):
            self.logger.info("  CV fold %d/%d", fold + 1, n_splits)

            # Ridge
            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(
                X_train_scaled.iloc[train_idx],
                y_train_scaled.iloc[train_idx],
            )
            y_pred_ridge = ridge.predict(X_train_scaled.iloc[val_idx])
            metrics_ridge = evaluator.compute_metrics(
                y_train.iloc[val_idx].values, y_pred_ridge,
            )
            cv_results["ridge"].append(metrics_ridge)

            # LightGBM
            import lightgbm as lgb
            lgb_model = lgb.LGBMRegressor(
                **self.config.model.lightgbm_params,
                verbose=-1,
            )
            lgb_model.fit(
                X_train_imp.iloc[train_idx],
                y_train.iloc[train_idx],
            )
            y_pred_lgb = lgb_model.predict(X_train_imp.iloc[val_idx])
            metrics_lgb = evaluator.compute_metrics(
                y_train.iloc[val_idx].values, y_pred_lgb,
            )
            cv_results["lightgbm"].append(metrics_lgb)

        # Agreger les scores
        aggregated = {}
        for model_name, folds in cv_results.items():
            if folds:
                agg = {}
                for metric in folds[0]:
                    values = [f[metric] for f in folds]
                    agg[f"{metric}_mean"] = float(np.mean(values))
                    agg[f"{metric}_std"] = float(np.std(values))
                aggregated[model_name] = agg

        return aggregated

    def _save_artifact(self, obj: Any, filename: str) -> Path:
        """Sauvegarde un artefact (modele, scaler) en pickle.

        Args:
            obj: Objet a serialiser.
            filename: Nom du fichier.

        Returns:
            Chemin du fichier sauvegarde.
        """
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        self.logger.info("  Artefact sauvegarde → %s", path)
        return path

    def _save_results_summary(self, results: Dict[str, Any]) -> Path:
        """Sauvegarde un resume des resultats en CSV.

        Args:
            results: Dictionnaire des resultats par modele.

        Returns:
            Chemin du fichier CSV.
        """
        rows = []
        for model_name, res in results.items():
            row = {"model": model_name, "target": self.target}
            for split in ["val", "test"]:
                metrics = res.get(f"metrics_{split}", {})
                for metric, value in metrics.items():
                    row[f"{split}_{metric}"] = value
            # CV scores
            cv = res.get("cv_scores", {})
            for key, value in cv.items():
                row[f"cv_{key}"] = value
            rows.append(row)

        df_results = pd.DataFrame(rows)
        path = self.models_dir / "training_results.csv"
        df_results.to_csv(path, index=False)
        self.logger.info("Resume des resultats → %s", path)
        return path
