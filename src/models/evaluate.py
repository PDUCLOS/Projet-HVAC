# -*- coding: utf-8 -*-
"""
Model evaluation and comparison — Phase 4.3-4.4.
==================================================

This module provides:

    1. Standard metric computation (RMSE, MAE, MAPE, R2)
    2. Model comparison (summary table)
    3. SHAP analysis (interpretable feature importance)
    4. Visualizations (predictions vs actual, residuals, comparison)

Charts are saved in data/models/figures/.

Usage:
    >>> from src.models.evaluate import ModelEvaluator
    >>> evaluator = ModelEvaluator(config)
    >>> metrics = evaluator.compute_metrics(y_true, y_pred)
    >>> evaluator.plot_comparison(results)

    # Or via CLI
    python -m src.pipeline evaluate
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from config.settings import ProjectConfig


class ModelEvaluator:
    """Evaluate and compare Phase 4 models.

    Attributes:
        config: Project configuration.
        figures_dir: Directory for saving figures.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("models.evaluate")
        self.figures_dir = Path("data/models/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            Dictionary with RMSE, MAE, MAPE, R2.
        """
        # Filter NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"rmse": float("nan"), "mae": float("nan"),
                    "mape": float("nan"), "r2": float("nan")}

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE: avoid division by zero
        mask_nonzero = y_true != 0
        if mask_nonzero.any():
            mape = float(
                mean_absolute_percentage_error(
                    y_true[mask_nonzero], y_pred[mask_nonzero]
                ) * 100
            )
        else:
            mape = float("nan")

        return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    def compare_models(
        self, results: Dict[str, Any]
    ) -> pd.DataFrame:
        """Create a comparative table of models.

        Args:
            results: Dictionary {model_name: results} from ModelTrainer.

        Returns:
            DataFrame with val and test metrics for each model.
        """
        rows = []
        for model_name, res in results.items():
            row = {"Model": model_name}
            for split, prefix in [("val", "Val"), ("test", "Test")]:
                metrics = res.get(f"metrics_{split}", {})
                row[f"{prefix} RMSE"] = metrics.get("rmse", float("nan"))
                row[f"{prefix} MAE"] = metrics.get("mae", float("nan"))
                row[f"{prefix} MAPE (%)"] = metrics.get("mape", float("nan"))
                row[f"{prefix} R2"] = metrics.get("r2", float("nan"))
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("Val RMSE")

        self.logger.info("\n%s", df.to_string(index=False))
        return df

    def plot_predictions_vs_actual(
        self,
        results: Dict[str, Any],
        y_val: np.ndarray,
        y_test: np.ndarray,
        target_name: str = "nb_installations_pac",
    ) -> List[Path]:
        """Plot predictions vs actual values for each model.

        Args:
            results: Model results.
            y_val: Validation actual values.
            y_test: Test actual values.
            target_name: Target variable name.

        Returns:
            List of saved figure paths.
        """
        figures = []

        for model_name, res in results.items():
            preds_val = res.get("predictions_val", np.array([]))
            preds_test = res.get("predictions_test", np.array([]))

            if len(preds_val) == 0 and len(preds_test) == 0:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{model_name} — Predictions vs Actual ({target_name})")

            # Validation
            if len(preds_val) > 0:
                n = min(len(y_val), len(preds_val))
                axes[0].plot(range(n), y_val[:n], "b-o", label="Actual", markersize=4)
                axes[0].plot(range(n), preds_val[:n], "r--s", label="Predicted", markersize=4)
                axes[0].set_title("Validation")
                axes[0].set_xlabel("Time index")
                axes[0].set_ylabel(target_name)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Test
            if len(preds_test) > 0:
                n = min(len(y_test), len(preds_test))
                axes[1].plot(range(n), y_test[:n], "b-o", label="Actual", markersize=4)
                axes[1].plot(range(n), preds_test[:n], "r--s", label="Predicted", markersize=4)
                axes[1].set_title("Test")
                axes[1].set_xlabel("Time index")
                axes[1].set_ylabel(target_name)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            path = self.figures_dir / f"predictions_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append(path)
            self.logger.info("  Figure saved → %s", path)

        return figures

    def plot_model_comparison(
        self, results: Dict[str, Any]
    ) -> Path:
        """Comparative chart of metrics per model.

        Args:
            results: Model results.

        Returns:
            Path of the saved figure.
        """
        df = self.compare_models(results)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Model Comparison", fontsize=14)

        metrics = [
            ("Val RMSE", "Test RMSE", "RMSE"),
            ("Val MAE", "Test MAE", "MAE"),
            ("Val MAPE (%)", "Test MAPE (%)", "MAPE (%)"),
            ("Val R2", "Test R2", "R2"),
        ]

        for ax, (val_col, test_col, title) in zip(axes.flat, metrics):
            x = np.arange(len(df))
            width = 0.35
            bars_val = ax.bar(x - width / 2, df[val_col], width, label="Validation")
            bars_test = ax.bar(x + width / 2, df[test_col], width, label="Test")
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(df["Model"], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # Add values on bars
            for bar in bars_val:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                ha="center", va="bottom", fontsize=8)
            for bar in bars_test:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                                ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        path = self.figures_dir / "model_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("  Comparison figure → %s", path)
        return path

    def plot_residuals(
        self,
        results: Dict[str, Any],
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Path:
        """Plot residuals for each model (diagnostics).

        Args:
            results: Model results.
            y_val: Validation actual values.
            y_test: Test actual values.

        Returns:
            Path of the saved figure.
        """
        model_names = [
            name for name, res in results.items()
            if len(res.get("predictions_test", [])) > 0
        ]

        if not model_names:
            self.logger.warning("No predictions available for residuals")
            return self.figures_dir / "residuals.png"

        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle("Residual Analysis", fontsize=14)

        for i, name in enumerate(model_names):
            res = results[name]
            preds_test = res.get("predictions_test", np.array([]))
            n = min(len(y_test), len(preds_test))
            if n == 0:
                continue

            residuals = y_test[:n] - preds_test[:n]

            # Residual distribution
            axes[i, 0].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
            axes[i, 0].axvline(0, color="red", linestyle="--")
            axes[i, 0].set_title(f"{name} — Residual Distribution")
            axes[i, 0].set_xlabel("Residual (actual - predicted)")

            # Residuals vs predictions
            axes[i, 1].scatter(preds_test[:n], residuals, alpha=0.6, s=30)
            axes[i, 1].axhline(0, color="red", linestyle="--")
            axes[i, 1].set_title(f"{name} — Residuals vs Predictions")
            axes[i, 1].set_xlabel("Prediction")
            axes[i, 1].set_ylabel("Residual")
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.figures_dir / "residuals.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("  Residuals figure → %s", path)
        return path

    def plot_feature_importance(
        self,
        results: Dict[str, Any],
        top_n: int = 20,
    ) -> List[Path]:
        """Plot feature importance for models that support it.

        Args:
            results: Model results.
            top_n: Number of features to display.

        Returns:
            List of figure paths.
        """
        figures = []

        for model_name, res in results.items():
            importance = res.get("feature_importance")
            if importance is None or len(importance) == 0:
                continue

            fig, ax = plt.subplots(figsize=(10, 8))
            top = importance.head(top_n)
            top.iloc[::-1].plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"{model_name} — Top {top_n} Features")
            ax.set_xlabel("Importance")
            ax.grid(True, alpha=0.3, axis="x")

            plt.tight_layout()
            path = self.figures_dir / f"feature_importance_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append(path)
            self.logger.info("  Feature importance %s → %s", model_name, path)

        return figures

    def shap_analysis(
        self,
        model: Any,
        X: pd.DataFrame,
        model_name: str = "lightgbm",
        max_display: int = 20,
    ) -> Optional[Path]:
        """SHAP analysis: impact of each feature on predictions.

        Uses TreeExplainer for LightGBM (fast) or
        KernelExplainer for other models (slow).

        Args:
            model: Trained model.
            X: Features (DataFrame with column names).
            model_name: Model name (for title and filename).
            max_display: Maximum number of features to display.

        Returns:
            Path of the SHAP figure, or None if SHAP fails.
        """
        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP not available. pip install shap")
            return None

        self.logger.info("  SHAP analysis for %s...", model_name)

        try:
            if model_name == "lightgbm":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                # KernelExplainer (slow, use a subsample)
                sample = X.sample(min(50, len(X)), random_state=42)
                explainer = shap.KernelExplainer(model.predict, sample)
                shap_values = explainer.shap_values(X)

            # Summary plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X,
                max_display=max_display,
                show=False,
            )
            path = self.figures_dir / f"shap_{model_name}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            self.logger.info("  SHAP %s → %s", model_name, path)
            return path

        except Exception as e:
            self.logger.warning("  SHAP failed for %s: %s", model_name, e)
            return None

    def generate_full_report(
        self, results: Dict[str, Any], target: str = "nb_installations_pac"
    ) -> Path:
        """Generate a complete text report of the results.

        Args:
            results: Results from all models.
            target: Target variable used.

        Returns:
            Path of the report file.
        """
        report_lines = [
            "=" * 70,
            "EVALUATION REPORT — Phase 4 ML Modeling",
            f"Target variable: {target}",
            "=" * 70,
            "",
        ]

        # Comparative table
        df_comp = self.compare_models(results)
        report_lines.append("MODEL COMPARISON:")
        report_lines.append("-" * 70)
        report_lines.append(df_comp.to_string(index=False))
        report_lines.append("")

        # Details per model
        for model_name, res in results.items():
            report_lines.append(f"\n--- {model_name.upper()} ---")
            for split in ["val", "test"]:
                metrics = res.get(f"metrics_{split}", {})
                report_lines.append(f"  {split.capitalize()} :")
                for metric, value in metrics.items():
                    report_lines.append(f"    {metric:>6s} = {value:.4f}")

            # CV scores
            cv = res.get("cv_scores", {})
            if cv:
                report_lines.append("  Cross-validation :")
                for key, value in cv.items():
                    report_lines.append(f"    {key:>20s} = {value:.4f}")

            # Feature importance (top 10)
            importance = res.get("feature_importance")
            if importance is not None:
                report_lines.append("  Top-10 features :")
                for feat, val in importance.head(10).items():
                    report_lines.append(f"    {feat:>40s} : {val:.4f}")

        # Recommendation
        report_lines.append("\n" + "=" * 70)
        report_lines.append("RECOMMENDATION:")

        # Find the best model on val
        best_model = min(
            results.items(),
            key=lambda x: x[1].get("metrics_val", {}).get("rmse", float("inf")),
        )
        report_lines.append(
            f"  Best model (Val RMSE): {best_model[0]} "
            f"(RMSE={best_model[1].get('metrics_val', {}).get('rmse', 0):.2f})"
        )
        report_lines.append("=" * 70)

        # Save
        report = "\n".join(report_lines)
        path = Path("data/models") / "evaluation_report.txt"
        path.write_text(report, encoding="utf-8")
        self.logger.info("Evaluation report → %s", path)
        return path
