# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning Module — Optuna-based optimization.
===========================================================

Provides systematic hyperparameter tuning for Ridge and LightGBM
using Optuna with TimeSeriesSplit cross-validation.

Key design choices:
    - TimeSeriesSplit for temporal data (no future leakage)
    - RMSE as the optimization objective
    - Graceful degradation when Optuna is not installed
    - Configurable number of trials for budget control

Usage:
    >>> from src.models.hyperparameter_tuning import HyperparameterTuner
    >>> tuner = HyperparameterTuner()
    >>> result = tuner.tune_ridge(X_train, y_train, X_val, y_val, n_trials=50)
    >>> best_params = result["best_params"]

    # Or tune LightGBM:
    >>> result = tuner.tune_lightgbm(X_train, y_train, X_val, y_val, n_trials=50)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class HyperparameterTuner:
    """Optuna-based hyperparameter tuner for ML models.

    Optimizes model hyperparameters using Bayesian optimization
    (via Optuna) with temporal cross-validation to avoid data leakage.

    Attributes:
        logger: Module logger instance.
        cv_splits: Number of TimeSeriesSplit folds.
    """

    def __init__(self, cv_splits: int = 3) -> None:
        """Initialize the hyperparameter tuner.

        Args:
            cv_splits: Number of cross-validation splits for TimeSeriesSplit.
        """
        self.logger = logging.getLogger("models.hyperparameter_tuning")
        self.cv_splits = cv_splits

    def _check_optuna(self) -> bool:
        """Check if Optuna is available.

        Returns:
            True if optuna can be imported, False otherwise.
        """
        try:
            import optuna  # noqa: F401
            return True
        except ImportError:
            self.logger.warning(
                "Optuna is not installed. Install via: pip install optuna. "
                "Falling back to default parameters."
            )
            return False

    def tune_ridge(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Tune Ridge regression alpha using Optuna.

        The search space covers alpha from 1e-4 to 1000 (log-uniform),
        optimizing RMSE on the validation set with internal
        TimeSeriesSplit cross-validation.

        Args:
            X_train: Training features (scaled).
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            n_trials: Number of Optuna trials.

        Returns:
            Dictionary with:
                - best_params: dict with optimal alpha
                - best_score: best validation RMSE
                - study_results: trial history summary
                - n_trials: number of trials completed
        """
        self.logger.info("Tuning Ridge (n_trials=%d)...", n_trials)

        if not self._check_optuna():
            # Fallback: grid search over a few values
            return self._fallback_ridge(X_train, y_train, X_val, y_val)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective for Ridge alpha selection."""
            alpha = trial.suggest_float("alpha", 1e-4, 1000.0, log=True)

            from sklearn.linear_model import Ridge

            # Internal cross-validation on training data
            n_splits = min(self.cv_splits, len(X_train) // 10)
            if n_splits < 2:
                # Not enough data for CV, evaluate on validation set
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                return float(np.sqrt(mean_squared_error(y_val, y_pred)))

            tscv = TimeSeriesSplit(n_splits=n_splits)
            rmses = []

            for train_idx, val_idx in tscv.split(X_train):
                model = Ridge(alpha=alpha)
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                y_pred = model.predict(X_train.iloc[val_idx])
                rmse = float(np.sqrt(
                    mean_squared_error(y_train.iloc[val_idx], y_pred),
                ))
                rmses.append(rmse)

            return float(np.mean(rmses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value

        # Evaluate best params on the held-out validation set
        from sklearn.linear_model import Ridge

        best_model = Ridge(alpha=best_params["alpha"])
        best_model.fit(X_train, y_train)
        val_pred = best_model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        self.logger.info(
            "  Ridge best alpha=%.6f, CV RMSE=%.4f, Val RMSE=%.4f",
            best_params["alpha"], best_score, val_rmse,
        )

        # Build trial history summary
        study_results = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials[:10]  # Keep top 10 for summary
        ]

        return {
            "best_params": best_params,
            "best_score": best_score,
            "val_rmse": val_rmse,
            "study_results": study_results,
            "n_trials": len(study.trials),
        }

    def _fallback_ridge(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Fallback Ridge tuning without Optuna (simple grid search).

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            Dictionary with best_params, best_score, etc.
        """
        from sklearn.linear_model import Ridge

        alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]
        best_alpha = 1.0
        best_rmse = float("inf")

        for alpha in alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha

        self.logger.info(
            "  Ridge fallback: best alpha=%.4f, Val RMSE=%.4f",
            best_alpha, best_rmse,
        )

        return {
            "best_params": {"alpha": best_alpha},
            "best_score": best_rmse,
            "val_rmse": best_rmse,
            "study_results": [],
            "n_trials": len(alphas),
        }

    def tune_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Tune LightGBM hyperparameters using Optuna.

        Search space:
            - max_depth: [2, 8]
            - num_leaves: [7, 127]
            - learning_rate: [0.01, 0.3] (log-uniform)
            - n_estimators: [50, 500]
            - min_child_samples: [5, 50]
            - reg_alpha: [1e-4, 10.0] (log-uniform, L1)
            - reg_lambda: [1e-4, 10.0] (log-uniform, L2)
            - subsample: [0.5, 1.0]

        Args:
            X_train: Training features (imputed, NOT scaled).
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            n_trials: Number of Optuna trials.

        Returns:
            Dictionary with:
                - best_params: dict with optimal hyperparameters
                - best_score: best validation RMSE
                - study_results: trial history summary
                - n_trials: number of trials completed
        """
        self.logger.info("Tuning LightGBM (n_trials=%d)...", n_trials)

        if not self._check_optuna():
            return self._fallback_lightgbm(X_train, y_train, X_val, y_val)

        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            """Optuna objective for LightGBM hyperparameter search."""
            import lightgbm as lgb

            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "num_leaves": trial.suggest_int("num_leaves", 7, 127),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True,
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 50,
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-4, 10.0, log=True,
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-4, 10.0, log=True,
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "verbose": -1,
                "random_state": 42,
            }

            # Internal cross-validation on training data
            n_splits = min(self.cv_splits, len(X_train) // 10)
            if n_splits < 2:
                # Not enough data for CV, evaluate on validation set
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=10, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                y_pred = model.predict(X_val)
                return float(np.sqrt(mean_squared_error(y_val, y_pred)))

            tscv = TimeSeriesSplit(n_splits=n_splits)
            rmses = []

            for train_idx, val_idx in tscv.split(X_train):
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train.iloc[train_idx],
                    y_train.iloc[train_idx],
                    eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=10, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                y_pred = model.predict(X_train.iloc[val_idx])
                rmse = float(np.sqrt(
                    mean_squared_error(y_train.iloc[val_idx], y_pred),
                ))
                rmses.append(rmse)

            return float(np.mean(rmses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_score = study.best_value

        # Evaluate best params on the held-out validation set
        import lightgbm as lgb

        best_params_full = dict(best_params)
        best_params_full["verbose"] = -1
        best_params_full["random_state"] = 42

        best_model = lgb.LGBMRegressor(**best_params_full)
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        val_pred = best_model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

        self.logger.info(
            "  LightGBM best: CV RMSE=%.4f, Val RMSE=%.4f", best_score, val_rmse,
        )
        self.logger.info("  Best params: %s", best_params)

        # Build trial history summary
        study_results = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials[:10]
        ]

        return {
            "best_params": best_params,
            "best_score": best_score,
            "val_rmse": val_rmse,
            "study_results": study_results,
            "n_trials": len(study.trials),
        }

    def _fallback_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Fallback LightGBM tuning without Optuna.

        Uses the default config parameters as a baseline.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.

        Returns:
            Dictionary with best_params, best_score, etc.
        """
        import lightgbm as lgb

        # Use sensible defaults for small datasets
        default_params = {
            "max_depth": 4,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "subsample": 0.8,
            "verbose": -1,
            "random_state": 42,
        }

        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        y_pred = model.predict(X_val)
        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        # Remove non-serializable keys from params
        best_params = {
            k: v for k, v in default_params.items()
            if k not in ("verbose", "random_state")
        }

        self.logger.info(
            "  LightGBM fallback: Val RMSE=%.4f (default params)", val_rmse,
        )

        return {
            "best_params": best_params,
            "best_score": val_rmse,
            "val_rmse": val_rmse,
            "study_results": [],
            "n_trials": 1,
        }
