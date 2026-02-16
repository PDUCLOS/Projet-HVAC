# -*- coding: utf-8 -*-
"""
Modeles baseline — Ridge Regression, LightGBM, Prophet (Phase 4.1).
===================================================================

Trois modeles adaptes au volume de donnees (~288 lignes train) :

    1. Ridge Regression (Tier 1 — robuste)
       - Regression lineaire regularisee L2
       - Tres stable sur petit dataset
       - Sert de reference minimale

    2. LightGBM (Tier 2 — faisable)
       - Gradient boosting regularise
       - Capture les non-linearites
       - Hyperparametres contraints (max_depth=4, min_child_samples=20)

    3. Prophet (Tier 1 — series temporelles)
       - Modele additif de Facebook/Meta
       - Capture tendance + saisonnalite annuelle
       - Regresseurs externes (meteo, confiance)
       - Entraine par departement (univarie + regresseurs)

Usage :
    >>> from src.models.baseline import BaselineModels
    >>> baseline = BaselineModels(config, target="nb_installations_pac")
    >>> result = baseline.train_ridge(X_train, y_train, X_val, y_val, X_test, y_test)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from config.settings import ProjectConfig


class BaselineModels:
    """Conteneur pour les modeles baseline ML.

    Chaque methode train_xxx() entraine un modele et retourne un dictionnaire
    standardise avec :
    - model : l'objet modele entraine
    - metrics_val : metriques sur le set de validation
    - metrics_test : metriques sur le set de test
    - predictions_val : predictions sur validation
    - predictions_test : predictions sur test
    - feature_importance : importance des features (si disponible)

    Attributes:
        config: Configuration du projet.
        target: Variable cible.
    """

    def __init__(self, config: ProjectConfig, target: str) -> None:
        self.config = config
        self.target = target
        self.logger = logging.getLogger("models.baseline")

    def train_ridge(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Entraine une Ridge Regression.

        La regularisation L2 empeche l'overfitting sur les features
        correles (lags, rolling). Alpha est choisi par cross-validation
        temporelle.

        Args:
            X_train: Features d'entrainement (scalees).
            y_train: Cible d'entrainement.
            X_val: Features de validation.
            y_val: Cible de validation.
            X_test: Features de test.
            y_test: Cible de test.

        Returns:
            Dictionnaire avec modele, metriques et predictions.
        """
        from src.models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(self.config)

        # Selection de alpha par CV temporelle
        best_alpha = self._select_ridge_alpha(X_train, y_train)
        self.logger.info("  Ridge alpha selectionne : %.4f", best_alpha)

        # Entrainement final
        model = Ridge(alpha=best_alpha)
        model.fit(X_train, y_train)

        # Predictions
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # Clipper les predictions negatives (des comptages ne sont pas negatifs)
        y_pred_val = np.clip(y_pred_val, 0, None)
        y_pred_test = np.clip(y_pred_test, 0, None)

        # Metriques
        metrics_val = evaluator.compute_metrics(y_val.values, y_pred_val)
        metrics_test = evaluator.compute_metrics(y_test.values, y_pred_test)

        # Feature importance (coefficients absolus)
        importance = pd.Series(
            np.abs(model.coef_), index=X_train.columns,
        ).sort_values(ascending=False)

        self.logger.info(
            "  Ridge — Val RMSE=%.2f, Test RMSE=%.2f, R2_val=%.3f",
            metrics_val["rmse"], metrics_test["rmse"], metrics_val["r2"],
        )
        self.logger.info("  Top-5 features : %s", list(importance.head(5).index))

        return {
            "model": model,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "predictions_val": y_pred_val,
            "predictions_test": y_pred_test,
            "feature_importance": importance,
            "alpha": best_alpha,
        }

    def _select_ridge_alpha(
        self, X: pd.DataFrame, y: pd.Series
    ) -> float:
        """Selectionne le meilleur alpha par cross-validation temporelle.

        Args:
            X: Features d'entrainement.
            y: Cible.

        Returns:
            Meilleur alpha.
        """
        from sklearn.metrics import mean_squared_error

        alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        n_splits = min(3, len(X) // 20)
        if n_splits < 2:
            return 1.0  # Defaut si pas assez de donnees

        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_alpha = 1.0
        best_rmse = float("inf")

        for alpha in alphas:
            rmses = []
            for train_idx, val_idx in tscv.split(X):
                model = Ridge(alpha=alpha)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                y_pred = model.predict(X.iloc[val_idx])
                rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], y_pred))
                rmses.append(rmse)
            avg_rmse = np.mean(rmses)
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_alpha = alpha

        return best_alpha

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Entraine un modele LightGBM regularise.

        Les hyperparametres sont contraints pour un petit dataset :
        - max_depth=4, num_leaves=15 (evite les arbres trop profonds)
        - min_child_samples=20 (force des feuilles suffisamment peuplees)
        - reg_alpha=0.1, reg_lambda=0.1 (regularisation L1 + L2)
        - subsample=0.8 (bagging pour reduire la variance)

        Early stopping sur le set de validation.

        Args:
            X_train: Features d'entrainement (imputees, PAS scalees).
            y_train: Cible d'entrainement.
            X_val: Features de validation.
            y_val: Cible de validation.
            X_test: Features de test.
            y_test: Cible de test.

        Returns:
            Dictionnaire avec modele, metriques et predictions.
        """
        import lightgbm as lgb

        from src.models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(self.config)

        # Parametres depuis la config (regularises pour petit dataset)
        params = dict(self.config.model.lightgbm_params)
        params["verbose"] = -1
        params["random_state"] = 42

        model = lgb.LGBMRegressor(**params)

        # Entrainement avec early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predictions
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        y_pred_val = np.clip(y_pred_val, 0, None)
        y_pred_test = np.clip(y_pred_test, 0, None)

        # Metriques
        metrics_val = evaluator.compute_metrics(y_val.values, y_pred_val)
        metrics_test = evaluator.compute_metrics(y_test.values, y_pred_test)

        # Feature importance (gain-based)
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns,
        ).sort_values(ascending=False)

        self.logger.info(
            "  LightGBM — Val RMSE=%.2f, Test RMSE=%.2f, R2_val=%.3f",
            metrics_val["rmse"], metrics_test["rmse"], metrics_val["r2"],
        )
        self.logger.info(
            "  Best iteration : %d / %d",
            model.best_iteration_, params.get("n_estimators", 200),
        )
        self.logger.info("  Top-5 features : %s", list(importance.head(5).index))

        return {
            "model": model,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "predictions_val": y_pred_val,
            "predictions_test": y_pred_test,
            "feature_importance": importance,
            "best_iteration": model.best_iteration_,
        }

    def train_prophet(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Entraine un modele Prophet par departement.

        Prophet est un modele de series temporelles additif :
            y(t) = tendance(t) + saisonnalite(t) + regresseurs(t) + bruit

        On entraine un modele par departement, puis on agrege les metriques.
        Des regresseurs externes (meteo, confiance) sont ajoutes si disponibles.

        Args:
            df_train: DataFrame d'entrainement complet (avec date_id, dept).
            df_val: DataFrame de validation.
            df_test: DataFrame de test.

        Returns:
            Dictionnaire avec metriques agregees et predictions par dept.
        """
        from src.models.evaluate import ModelEvaluator

        evaluator = ModelEvaluator(self.config)

        # Regresseurs a utiliser (si disponibles)
        regressors = [
            "temp_mean", "hdd_sum", "cdd_sum",
            "confiance_menages", "ipi_hvac_c28",
        ]

        departments = sorted(df_train["dept"].unique())
        all_preds_val = []
        all_actual_val = []
        all_preds_test = []
        all_actual_test = []
        models_by_dept = {}

        for dept in departments:
            try:
                result = self._train_prophet_dept(
                    dept, df_train, df_val, df_test, regressors,
                )
                if result is not None:
                    models_by_dept[dept] = result["model"]
                    all_preds_val.extend(result["preds_val"])
                    all_actual_val.extend(result["actual_val"])
                    all_preds_test.extend(result["preds_test"])
                    all_actual_test.extend(result["actual_test"])
            except Exception as e:
                self.logger.warning("  Prophet dept %s echoue : %s", dept, e)

        # Metriques agregees
        if all_actual_val:
            metrics_val = evaluator.compute_metrics(
                np.array(all_actual_val), np.array(all_preds_val),
            )
            metrics_test = evaluator.compute_metrics(
                np.array(all_actual_test), np.array(all_preds_test),
            )
        else:
            self.logger.error("Prophet : aucun departement entraine avec succes")
            metrics_val = {"rmse": float("nan"), "mae": float("nan"),
                           "mape": float("nan"), "r2": float("nan")}
            metrics_test = metrics_val.copy()

        self.logger.info(
            "  Prophet — Val RMSE=%.2f, Test RMSE=%.2f, R2_val=%.3f",
            metrics_val.get("rmse", float("nan")),
            metrics_test.get("rmse", float("nan")),
            metrics_val.get("r2", float("nan")),
        )

        return {
            "model": models_by_dept,
            "metrics_val": metrics_val,
            "metrics_test": metrics_test,
            "predictions_val": np.array(all_preds_val) if all_preds_val else np.array([]),
            "predictions_test": np.array(all_preds_test) if all_preds_test else np.array([]),
        }

    def _train_prophet_dept(
        self,
        dept: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        regressors: list,
    ) -> Optional[Dict[str, Any]]:
        """Entraine Prophet pour un departement.

        Args:
            dept: Code departement.
            df_train: Donnees d'entrainement.
            df_val: Donnees de validation.
            df_test: Donnees de test.
            regressors: Liste des colonnes regresseurs.

        Returns:
            Dictionnaire avec modele et predictions, ou None si echec.
        """
        from prophet import Prophet

        # Filtrer le departement
        train = df_train[df_train["dept"] == dept].copy()
        val = df_val[df_val["dept"] == dept].copy()
        test = df_test[df_test["dept"] == dept].copy()

        if len(train) < 12:
            self.logger.warning("  Dept %s : trop peu de donnees (%d)", dept, len(train))
            return None

        # Construire le DataFrame Prophet (ds, y + regresseurs)
        def to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
            date_id = df["date_id"].astype(str)
            ds = pd.to_datetime(date_id.str[:4] + "-" + date_id.str[4:6] + "-01")
            pdf = pd.DataFrame({"ds": ds, "y": df[self.target].values})
            for reg in regressors:
                if reg in df.columns:
                    pdf[reg] = df[reg].values
            return pdf.reset_index(drop=True)

        pdf_train = to_prophet_df(train)
        pdf_val = to_prophet_df(val)
        pdf_test = to_prophet_df(test)

        # Imputer les NaN dans les regresseurs (forward fill puis median)
        available_regs = [r for r in regressors if r in pdf_train.columns]
        for reg in available_regs:
            for pdf in [pdf_train, pdf_val, pdf_test]:
                pdf[reg] = pdf[reg].ffill().bfill()
                if pdf[reg].isna().any():
                    pdf[reg] = pdf[reg].fillna(pdf_train[reg].median())

        # Configurer Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # Regularise les changements de tendance
            seasonality_prior_scale=5.0,
        )
        for reg in available_regs:
            model.add_regressor(reg)

        # Supprimer les logs Prophet
        import logging as _logging
        _logging.getLogger("prophet").setLevel(_logging.WARNING)
        _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

        # Entrainer
        model.fit(pdf_train)

        # Predire
        forecast_val = model.predict(pdf_val)
        forecast_test = model.predict(pdf_test)

        preds_val = np.clip(forecast_val["yhat"].values, 0, None)
        preds_test = np.clip(forecast_test["yhat"].values, 0, None)

        return {
            "model": model,
            "preds_val": list(preds_val),
            "actual_val": list(pdf_val["y"].values),
            "preds_test": list(preds_test),
            "actual_test": list(pdf_test["y"].values),
        }
