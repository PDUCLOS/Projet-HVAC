# -*- coding: utf-8 -*-
"""
Feature Selection Module — Automated multi-method feature selection.
=====================================================================

Provides a systematic approach to reducing feature dimensionality
using multiple complementary strategies:

    1. SHAP-based selection     — Model-aware importance ranking
    2. Recursive Feature Elim.  — Greedy backward elimination
    3. Variance threshold       — Remove near-constant features
    4. Correlation filtering    — Remove redundant highly correlated pairs

A consensus method combines all approaches to identify the most
robustly important features.

Usage:
    >>> from src.models.feature_selection import FeatureSelector
    >>> selector = FeatureSelector()
    >>> result = selector.run_full_selection(X, y, model)
    >>> selected = result["selected_features"]

    # Or individual methods:
    >>> shap_result = selector.select_by_shap(model, X, top_k=20)
    >>> rfe_result = selector.select_by_rfe(X, y, estimator, n_features=20)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


class FeatureSelector:
    """Multi-method feature selector for ML pipeline.

    Combines SHAP importance, recursive feature elimination,
    variance filtering, and correlation filtering to identify
    the most relevant features for prediction.

    Attributes:
        logger: Module logger instance.
        analysis_dir: Output directory for selection reports.
    """

    def __init__(self, analysis_dir: Union[str, Path] = "data/analysis") -> None:
        """Initialize the feature selector.

        Args:
            analysis_dir: Directory where reports and results are saved.
        """
        self.logger = logging.getLogger("models.feature_selection")
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    def select_by_shap(
        self,
        model: Any,
        X: pd.DataFrame,
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Select top-k features by SHAP importance.

        Uses TreeExplainer for tree-based models (LightGBM, XGBoost)
        or KernelExplainer as a fallback for other models.

        Args:
            model: Trained model (must support predict).
            X: Feature matrix (DataFrame with named columns).
            top_k: Number of top features to select.

        Returns:
            Dictionary with:
                - selected_features: list of top-k feature names
                - importance_scores: dict {feature: mean_abs_shap}
                - method: "shap"
        """
        try:
            import shap
        except ImportError:
            self.logger.warning(
                "SHAP not installed. Install via: pip install shap. "
                "Returning all features."
            )
            return {
                "selected_features": list(X.columns),
                "importance_scores": {},
                "method": "shap",
                "error": "shap not installed",
            }

        self.logger.info("SHAP feature selection (top_k=%d)...", top_k)

        try:
            # Prefer TreeExplainer for tree-based models
            model_type = type(model).__name__
            if hasattr(model, "booster_") or "LGBM" in model_type or "XGB" in model_type:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                # KernelExplainer (slower, use subsample)
                sample = X.sample(min(100, len(X)), random_state=42)
                explainer = shap.KernelExplainer(model.predict, sample)
                shap_values = explainer.shap_values(
                    X.sample(min(200, len(X)), random_state=42),
                )

            # Compute mean absolute SHAP values per feature
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance = pd.Series(
                mean_abs_shap, index=X.columns,
            ).sort_values(ascending=False)

            top_k_capped = min(top_k, len(importance))
            selected = list(importance.head(top_k_capped).index)

            self.logger.info(
                "  SHAP: selected %d features (top importance: %s)",
                len(selected), selected[:5],
            )

            return {
                "selected_features": selected,
                "importance_scores": importance.to_dict(),
                "method": "shap",
            }

        except Exception as e:
            self.logger.warning("SHAP selection failed: %s", e)
            return {
                "selected_features": list(X.columns),
                "importance_scores": {},
                "method": "shap",
                "error": str(e),
            }

    def select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        estimator: Any,
        n_features: int = 20,
    ) -> Dict[str, Any]:
        """Recursive Feature Elimination (RFE).

        Iteratively removes the least important features using
        the estimator's feature importance (coef_ or feature_importances_).

        Args:
            X: Feature matrix.
            y: Target variable.
            estimator: Scikit-learn compatible estimator with fit/predict.
            n_features: Target number of features to select.

        Returns:
            Dictionary with:
                - selected_features: list of selected feature names
                - ranking: dict {feature: rank} (1 = selected)
                - method: "rfe"
        """
        from sklearn.feature_selection import RFE

        self.logger.info("RFE feature selection (n_features=%d)...", n_features)

        n_features_capped = min(n_features, X.shape[1])

        try:
            rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_features_capped,
                step=1,
            )
            rfe.fit(X, y)

            selected = list(X.columns[rfe.support_])
            ranking = dict(zip(X.columns, rfe.ranking_.tolist()))

            self.logger.info(
                "  RFE: selected %d features", len(selected),
            )

            return {
                "selected_features": selected,
                "ranking": ranking,
                "method": "rfe",
            }

        except Exception as e:
            self.logger.warning("RFE selection failed: %s", e)
            return {
                "selected_features": list(X.columns),
                "ranking": {},
                "method": "rfe",
                "error": str(e),
            }

    def select_by_variance(
        self,
        X: pd.DataFrame,
        threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """Remove low-variance features.

        Features with variance below the threshold are considered
        near-constant and provide no discriminative power.

        Args:
            X: Feature matrix.
            threshold: Minimum variance to keep a feature.

        Returns:
            Dictionary with:
                - selected_features: list of features above threshold
                - removed_features: list of dropped features
                - variances: dict {feature: variance}
                - method: "variance"
        """
        self.logger.info(
            "Variance-based selection (threshold=%.4f)...", threshold,
        )

        variances = X.var()
        mask = variances >= threshold
        selected = list(X.columns[mask])
        removed = list(X.columns[~mask])

        self.logger.info(
            "  Variance: kept %d, removed %d low-variance features",
            len(selected), len(removed),
        )
        if removed:
            self.logger.info("  Removed: %s", removed[:10])

        return {
            "selected_features": selected,
            "removed_features": removed,
            "variances": variances.to_dict(),
            "method": "variance",
        }

    def select_by_correlation(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """Remove highly correlated features (keep one of each pair).

        For each pair with |correlation| > threshold, the feature with
        lower mean absolute correlation with all other features is kept.

        Args:
            X: Feature matrix.
            threshold: Maximum absolute correlation allowed.

        Returns:
            Dictionary with:
                - selected_features: list of features to keep
                - removed_features: list of dropped features
                - correlation_pairs: list of (feat_a, feat_b, corr)
                - method: "correlation"
        """
        self.logger.info(
            "Correlation-based selection (threshold=%.2f)...", threshold,
        )

        if X.empty or X.shape[1] < 2:
            return {
                "selected_features": list(X.columns),
                "removed_features": [],
                "correlation_pairs": [],
                "method": "correlation",
            }

        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1),
        )

        # Find pairs exceeding the threshold
        high_corr_pairs = []
        to_drop = set()

        for col in upper_tri.columns:
            correlated = upper_tri.index[upper_tri[col] > threshold].tolist()
            for corr_col in correlated:
                corr_val = float(corr_matrix.loc[corr_col, col])
                high_corr_pairs.append((col, corr_col, corr_val))

                # Drop the feature with higher mean correlation to all others
                mean_corr_col = corr_matrix[col].mean()
                mean_corr_corr = corr_matrix[corr_col].mean()
                if mean_corr_col > mean_corr_corr:
                    to_drop.add(col)
                else:
                    to_drop.add(corr_col)

        selected = [c for c in X.columns if c not in to_drop]
        removed = list(to_drop)

        self.logger.info(
            "  Correlation: kept %d, removed %d redundant features",
            len(selected), len(removed),
        )

        return {
            "selected_features": selected,
            "removed_features": removed,
            "correlation_pairs": high_corr_pairs,
            "method": "correlation",
        }

    def run_full_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: Any,
        top_k: int = 20,
        rfe_estimator: Optional[Any] = None,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        min_votes: int = 2,
    ) -> Dict[str, Any]:
        """Run all selection methods and return consensus features.

        A feature is included in the consensus if it is selected by
        at least `min_votes` methods.

        Args:
            X: Feature matrix.
            y: Target variable.
            model: Trained model (used for SHAP).
            top_k: Number of top features for SHAP and RFE.
            rfe_estimator: Estimator for RFE (defaults to Ridge).
            variance_threshold: Variance threshold for low-variance removal.
            correlation_threshold: Correlation threshold for redundancy removal.
            min_votes: Minimum number of methods that must select a feature.

        Returns:
            Dictionary with:
                - selected_features: consensus feature list
                - method_results: dict of individual method results
                - votes: dict {feature: vote_count}
                - report: summary string
        """
        self.logger.info("=" * 60)
        self.logger.info("  FULL FEATURE SELECTION")
        self.logger.info("  Features: %d, Samples: %d", X.shape[1], X.shape[0])
        self.logger.info("=" * 60)

        # Handle NaN for selection methods
        X_clean = X.fillna(X.median())

        method_results = {}

        # 1. Variance filtering (fast, always run first)
        var_result = self.select_by_variance(X_clean, variance_threshold)
        method_results["variance"] = var_result

        # Work only with variance-surviving features
        X_var = X_clean[var_result["selected_features"]]

        # 2. Correlation filtering
        corr_result = self.select_by_correlation(X_var, correlation_threshold)
        method_results["correlation"] = corr_result

        # 3. SHAP importance
        shap_result = self.select_by_shap(model, X_var, top_k)
        method_results["shap"] = shap_result

        # 4. RFE
        if rfe_estimator is None:
            from sklearn.linear_model import Ridge
            rfe_estimator = Ridge(alpha=1.0)
        rfe_result = self.select_by_rfe(X_var, y, rfe_estimator, top_k)
        method_results["rfe"] = rfe_result

        # Consensus: count votes per feature
        all_features = set(X.columns)
        votes = {f: 0 for f in all_features}

        for method_name, result in method_results.items():
            for feature in result.get("selected_features", []):
                if feature in votes:
                    votes[feature] += 1

        # Select features with enough votes
        consensus = [
            f for f, v in sorted(votes.items(), key=lambda x: -x[1])
            if v >= min_votes
        ]

        # Build summary report
        report_lines = [
            "Feature Selection Report",
            "=" * 50,
            f"Input features: {X.shape[1]}",
            f"Variance filter: {len(var_result['selected_features'])} kept"
            f" (removed {len(var_result.get('removed_features', []))})",
            f"Correlation filter: {len(corr_result['selected_features'])} kept"
            f" (removed {len(corr_result.get('removed_features', []))})",
            f"SHAP top-{top_k}: {len(shap_result['selected_features'])} selected",
            f"RFE top-{top_k}: {len(rfe_result['selected_features'])} selected",
            f"Consensus (min_votes={min_votes}): {len(consensus)} features",
            "",
            "Consensus features:",
        ]
        for f in consensus:
            report_lines.append(f"  - {f} (votes={votes[f]})")

        report = "\n".join(report_lines)
        self.logger.info("\n%s", report)

        # Save results
        output = {
            "selected_features": consensus,
            "method_results": {
                name: {
                    "selected_features": r["selected_features"],
                    "method": r["method"],
                }
                for name, r in method_results.items()
            },
            "votes": votes,
            "report": report,
        }

        self._save_results(output)
        return output

    def _save_results(self, results: Dict[str, Any]) -> Path:
        """Save feature selection results to JSON.

        Args:
            results: Selection results dictionary.

        Returns:
            Path of the saved file.
        """
        output_path = self.analysis_dir / "selected_features.json"

        # Make serializable (remove non-JSON items)
        serializable = {
            "selected_features": results["selected_features"],
            "votes": results["votes"],
            "method_results": results.get("method_results", {}),
        }

        output_path.write_text(
            json.dumps(serializable, indent=2, default=str),
            encoding="utf-8",
        )
        self.logger.info("Feature selection results saved -> %s", output_path)
        return output_path
