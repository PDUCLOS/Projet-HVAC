# -*- coding: utf-8 -*-
"""
Training pipeline — Phase 4: ML Modeling.
==========================================

Orchestrates the training of all models (baseline + DL) respecting
the temporal split defined in the configuration:

    Train  : 2021-07 -> 2024-06 (36 months x 8 depts = ~288 rows)
    Val    : 2024-07 -> 2024-12 (6 months x 8 depts = ~48 rows)
    Test   : 2025-01 -> 2025-12 (12 months x 8 depts = ~96 rows)

The granularity is month x department. NaN at the beginning of the series
(lags, rolling) are removed during training.

Usage:
    >>> from src.models.train import ModelTrainer
    >>> trainer = ModelTrainer(config)
    >>> results = trainer.train_all()

    # Or via CLI
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
from sklearn.preprocessing import RobustScaler

from config.settings import ProjectConfig


class ModelTrainer:
    """Orchestrates training and validation of all models.

    Responsibilities:
    - Load the features dataset
    - Apply the temporal split (train / val / test)
    - Select relevant features (remove identifiers, NaN)
    - Train each model with temporal cross-validation
    - Save trained models and metrics

    Attributes:
        config: Project configuration.
        target: Target variable to predict.
        models_dir: Directory for saving models.
    """

    # Features excluded from training (identifiers, metadata, targets)
    EXCLUDE_COLS = {
        "date_id", "dept", "dept_name", "city_ref",
        "latitude", "longitude",
        "n_valid_features", "pct_valid_features",
    }

    # Available target variables
    TARGET_COLS = {
        "nb_installations_pac", "nb_installations_clim",
        "nb_dpe_total", "nb_dpe_classe_ab",
    }

    # Auto-regressive features of the target (lags, rolling, diff of the target)
    # Excluded in "no_target_leakage" mode to evaluate exogenous features
    TARGET_LAG_PATTERNS = [
        "_lag_", "_rmean_", "_rstd_", "_diff_", "_pct_",
    ]

    def __init__(
        self,
        config: ProjectConfig,
        target: str = "nb_installations_pac",
        exclude_target_lags: bool = False,
    ) -> None:
        self.config = config
        self.target = target
        self.exclude_target_lags = exclude_target_lags
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
        """Load the features dataset from CSV.

        Returns:
            DataFrame with all engineered features.

        Raises:
            FileNotFoundError: If the dataset does not exist.
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
        """Apply the temporal train / val / test split.

        The split respects chronological order (no temporal leakage).

        Args:
            df: Complete dataset with date_id column.

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
        """Separate features (X) and target (y), remove irrelevant columns.

        If exclude_target_lags=True, also excludes features derived from the
        target variable (lags, rolling, diff, pct_change). This allows evaluating
        the contribution of exogenous features (weather, economics) without
        the target's auto-correlation which otherwise dominates predictions.

        Args:
            df: DataFrame with all columns.

        Returns:
            Tuple (X, y) ready for training.
        """
        # Columns to exclude
        drop_cols = self.EXCLUDE_COLS | (self.TARGET_COLS - {self.target})

        feature_cols = [
            c for c in df.columns
            if c not in drop_cols and c != self.target
        ]

        # Exclude auto-regressive features of the target if requested
        if self.exclude_target_lags:
            target_prefix = self.target
            feature_cols = [
                c for c in feature_cols
                if not (
                    c.startswith(target_prefix)
                    and any(p in c for p in self.TARGET_LAG_PATTERNS)
                )
            ]

        # Keep only numeric columns
        X = df[feature_cols].select_dtypes(include=[np.number]).copy()
        y = df[self.target].copy()

        return X, y

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return the list of feature names used.

        Args:
            df: Source DataFrame.

        Returns:
            List of feature column names.
        """
        X, _ = self.prepare_features(df)
        return list(X.columns)

    def train_all(
        self,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train all models and return results.

        Complete pipeline:
        1. Load the dataset
        2. Temporal split
        3. Remove rows with NaN in the target
        4. Train Ridge, LightGBM, Prophet
        5. Train LSTM (if available)
        6. Evaluate on validation and test
        7. Save models and metrics

        Args:
            target: Target variable (overrides self.target).

        Returns:
            Dictionary with results per model.
        """
        if target:
            self.target = target

        self.logger.info("=" * 60)
        self.logger.info("  PHASE 4 — Entrainement des modeles")
        self.logger.info("  Cible : %s", self.target)
        self.logger.info("=" * 60)

        # 1. Load
        df = self.load_dataset()

        # 2. Temporal split
        df_train, df_val, df_test = self.temporal_split(df)

        # 3. Prepare features
        X_train, y_train = self.prepare_features(df_train)
        X_val, y_val = self.prepare_features(df_val)
        X_test, y_test = self.prepare_features(df_test)

        # 4. Remove rows with NaN in the target
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

        # 5. Impute NaN in features (median)
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

        # 6. Scaler (for Ridge and LSTM)
        # RobustScaler uses median and IQR instead of mean/std,
        # making it resistant to outliers (unlike StandardScaler)
        scaler = RobustScaler()
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

        # Save imputer and scaler
        self._save_artifact(imputer, "imputer.pkl")
        self._save_artifact(scaler, "scaler.pkl")

        # 7. Train models
        results = {}

        # --- Baseline models ---
        from src.models.baseline import BaselineModels

        baseline = BaselineModels(self.config, self.target)

        # Ridge Regression (uses scaled data)
        self.logger.info("-" * 40)
        self.logger.info("Entrainement Ridge Regression...")
        ridge_result = baseline.train_ridge(
            X_train_scaled, y_train,
            X_val_scaled, y_val,
            X_test_scaled, y_test,
        )
        results["ridge"] = ridge_result
        self._save_artifact(ridge_result["model"], "ridge_model.pkl")

        # Ridge "exogenous" — without target lags/rolling/diff
        # Allows evaluating the true contribution of exogenous features
        # (weather, economics) without the auto-correlation that dominates R²
        self.logger.info("-" * 40)
        self.logger.info("Entrainement Ridge (exogenes, sans lags cible)...")
        exo_trainer = ModelTrainer(
            self.config, target=self.target, exclude_target_lags=True,
        )
        X_train_exo, _ = exo_trainer.prepare_features(df_train)
        X_val_exo, _ = exo_trainer.prepare_features(df_val)
        X_test_exo, _ = exo_trainer.prepare_features(df_test)

        # Same target NaN masks
        X_train_exo = X_train_exo[mask_train]
        X_val_exo = X_val_exo[mask_val]
        X_test_exo = X_test_exo[mask_test]

        # Imputer + scaler for exogenous features
        imputer_exo = SimpleImputer(strategy="median")
        X_train_exo_imp = pd.DataFrame(
            imputer_exo.fit_transform(X_train_exo),
            columns=X_train_exo.columns, index=X_train_exo.index,
        )
        X_val_exo_imp = pd.DataFrame(
            imputer_exo.transform(X_val_exo),
            columns=X_val_exo.columns, index=X_val_exo.index,
        )
        X_test_exo_imp = pd.DataFrame(
            imputer_exo.transform(X_test_exo),
            columns=X_test_exo.columns, index=X_test_exo.index,
        )

        scaler_exo = RobustScaler()
        X_train_exo_sc = pd.DataFrame(
            scaler_exo.fit_transform(X_train_exo_imp),
            columns=X_train_exo_imp.columns, index=X_train_exo_imp.index,
        )
        X_val_exo_sc = pd.DataFrame(
            scaler_exo.transform(X_val_exo_imp),
            columns=X_val_exo_imp.columns, index=X_val_exo_imp.index,
        )
        X_test_exo_sc = pd.DataFrame(
            scaler_exo.transform(X_test_exo_imp),
            columns=X_test_exo_imp.columns, index=X_test_exo_imp.index,
        )

        ridge_exo_result = baseline.train_ridge(
            X_train_exo_sc, y_train,
            X_val_exo_sc, y_val,
            X_test_exo_sc, y_test,
        )
        results["ridge_exogenes"] = ridge_exo_result
        n_exo = len(X_train_exo.columns)
        n_all = len(X_train.columns)
        self.logger.info(
            "  Ridge exogenes : %d features (vs %d avec lags cible)",
            n_exo, n_all,
        )

        # LightGBM (uses imputed data, not scaled)
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

        # --- LSTM (exploratory) ---
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

        # 8. Temporal cross-validation (on Ridge and LightGBM)
        self.logger.info("-" * 40)
        self.logger.info("Cross-validation temporelle...")
        cv_results = self._cross_validate(
            X_train_imp, y_train, X_train_scaled, y_train, baseline,
        )
        for model_name, cv_scores in cv_results.items():
            results[model_name]["cv_scores"] = cv_scores

        # 9. Save summary
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
        """Temporal cross-validation with TimeSeriesSplit.

        Args:
            X_train_imp: Imputed features (for LightGBM).
            y_train: Target.
            X_train_scaled: Scaled features (for Ridge).
            y_train_scaled: Target (identical).
            baseline: BaselineModels instance.

        Returns:
            CV scores per model.
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

        # Aggregate scores
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
        """Save an artifact (model, scaler) as pickle.

        Args:
            obj: Object to serialize.
            filename: Filename.

        Returns:
            Path of the saved file.
        """
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        self.logger.info("  Artefact sauvegarde → %s", path)
        return path

    def _save_results_summary(self, results: Dict[str, Any]) -> Path:
        """Save a results summary as CSV.

        Args:
            results: Dictionary of results per model.

        Returns:
            Path of the CSV file.
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
