# -*- coding: utf-8 -*-
"""
Evaluation et comparaison des modeles — Phase 4.3-4.4.
======================================================

Ce module fournit :

    1. Calcul des metriques standard (RMSE, MAE, MAPE, R2)
    2. Comparaison entre modeles (tableau recapitulatif)
    3. Analyse SHAP (feature importance interpretable)
    4. Visualisations (predictions vs reel, residus, comparaison)

Les graphiques sont sauvegardes dans data/models/figures/.

Usage :
    >>> from src.models.evaluate import ModelEvaluator
    >>> evaluator = ModelEvaluator(config)
    >>> metrics = evaluator.compute_metrics(y_true, y_pred)
    >>> evaluator.plot_comparison(results)

    # Ou via CLI
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
    """Evalue et compare les modeles de la Phase 4.

    Attributes:
        config: Configuration du projet.
        figures_dir: Repertoire de sauvegarde des figures.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.logger = logging.getLogger("models.evaluate")
        self.figures_dir = Path("data/models/figures")
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calcule les metriques de regression.

        Args:
            y_true: Valeurs reelles.
            y_pred: Valeurs predites.

        Returns:
            Dictionnaire avec RMSE, MAE, MAPE, R2.
        """
        # Filtrer les NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"rmse": float("nan"), "mae": float("nan"),
                    "mape": float("nan"), "r2": float("nan")}

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        # MAPE : eviter la division par zero
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
        """Cree un tableau comparatif des modeles.

        Args:
            results: Dictionnaire {nom_modele: resultats} depuis ModelTrainer.

        Returns:
            DataFrame avec metriques val et test pour chaque modele.
        """
        rows = []
        for model_name, res in results.items():
            row = {"Modele": model_name}
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
        """Trace les predictions vs valeurs reelles pour chaque modele.

        Args:
            results: Resultats des modeles.
            y_val: Valeurs reelles validation.
            y_test: Valeurs reelles test.
            target_name: Nom de la variable cible.

        Returns:
            Liste des chemins des figures sauvegardees.
        """
        figures = []

        for model_name, res in results.items():
            preds_val = res.get("predictions_val", np.array([]))
            preds_test = res.get("predictions_test", np.array([]))

            if len(preds_val) == 0 and len(preds_test) == 0:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{model_name} — Predictions vs Reel ({target_name})")

            # Validation
            if len(preds_val) > 0:
                n = min(len(y_val), len(preds_val))
                axes[0].plot(range(n), y_val[:n], "b-o", label="Reel", markersize=4)
                axes[0].plot(range(n), preds_val[:n], "r--s", label="Predit", markersize=4)
                axes[0].set_title("Validation")
                axes[0].set_xlabel("Index temporel")
                axes[0].set_ylabel(target_name)
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

            # Test
            if len(preds_test) > 0:
                n = min(len(y_test), len(preds_test))
                axes[1].plot(range(n), y_test[:n], "b-o", label="Reel", markersize=4)
                axes[1].plot(range(n), preds_test[:n], "r--s", label="Predit", markersize=4)
                axes[1].set_title("Test")
                axes[1].set_xlabel("Index temporel")
                axes[1].set_ylabel(target_name)
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            path = self.figures_dir / f"predictions_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures.append(path)
            self.logger.info("  Figure sauvegardee → %s", path)

        return figures

    def plot_model_comparison(
        self, results: Dict[str, Any]
    ) -> Path:
        """Graphique comparatif des metriques par modele.

        Args:
            results: Resultats des modeles.

        Returns:
            Chemin de la figure sauvegardee.
        """
        df = self.compare_models(results)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Comparaison des modeles", fontsize=14)

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
            ax.set_xticklabels(df["Modele"], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # Ajouter les valeurs sur les barres
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
        self.logger.info("  Figure comparaison → %s", path)
        return path

    def plot_residuals(
        self,
        results: Dict[str, Any],
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Path:
        """Trace les residus pour chaque modele (diagnostic).

        Args:
            results: Resultats des modeles.
            y_val: Valeurs reelles validation.
            y_test: Valeurs reelles test.

        Returns:
            Chemin de la figure sauvegardee.
        """
        model_names = [
            name for name, res in results.items()
            if len(res.get("predictions_test", [])) > 0
        ]

        if not model_names:
            self.logger.warning("Aucune prediction disponible pour les residus")
            return self.figures_dir / "residuals.png"

        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle("Analyse des residus", fontsize=14)

        for i, name in enumerate(model_names):
            res = results[name]
            preds_test = res.get("predictions_test", np.array([]))
            n = min(len(y_test), len(preds_test))
            if n == 0:
                continue

            residuals = y_test[:n] - preds_test[:n]

            # Distribution des residus
            axes[i, 0].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
            axes[i, 0].axvline(0, color="red", linestyle="--")
            axes[i, 0].set_title(f"{name} — Distribution des residus")
            axes[i, 0].set_xlabel("Residu (reel - predit)")

            # Residus vs predictions
            axes[i, 1].scatter(preds_test[:n], residuals, alpha=0.6, s=30)
            axes[i, 1].axhline(0, color="red", linestyle="--")
            axes[i, 1].set_title(f"{name} — Residus vs Predictions")
            axes[i, 1].set_xlabel("Prediction")
            axes[i, 1].set_ylabel("Residu")
            axes[i, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = self.figures_dir / "residuals.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("  Figure residus → %s", path)
        return path

    def plot_feature_importance(
        self,
        results: Dict[str, Any],
        top_n: int = 20,
    ) -> List[Path]:
        """Trace l'importance des features pour les modeles qui le supportent.

        Args:
            results: Resultats des modeles.
            top_n: Nombre de features a afficher.

        Returns:
            Liste des chemins des figures.
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
        """Analyse SHAP : impact de chaque feature sur les predictions.

        Utilise TreeExplainer pour LightGBM (rapide) ou
        KernelExplainer pour les autres modeles (lent).

        Args:
            model: Modele entraine.
            X: Features (DataFrame avec noms de colonnes).
            model_name: Nom du modele (pour le titre et le fichier).
            max_display: Nombre max de features a afficher.

        Returns:
            Chemin de la figure SHAP, ou None si SHAP echoue.
        """
        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP non disponible. pip install shap")
            return None

        self.logger.info("  Analyse SHAP pour %s...", model_name)

        try:
            if model_name == "lightgbm":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                # KernelExplainer (lent, utiliser un sous-echantillon)
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
            self.logger.warning("  SHAP echoue pour %s : %s", model_name, e)
            return None

    def generate_full_report(
        self, results: Dict[str, Any], target: str = "nb_installations_pac"
    ) -> Path:
        """Genere un rapport textuel complet des resultats.

        Args:
            results: Resultats de tous les modeles.
            target: Variable cible utilisee.

        Returns:
            Chemin du fichier rapport.
        """
        report_lines = [
            "=" * 70,
            "RAPPORT D'EVALUATION — Phase 4 Modelisation ML",
            f"Variable cible : {target}",
            "=" * 70,
            "",
        ]

        # Tableau comparatif
        df_comp = self.compare_models(results)
        report_lines.append("COMPARAISON DES MODELES :")
        report_lines.append("-" * 70)
        report_lines.append(df_comp.to_string(index=False))
        report_lines.append("")

        # Details par modele
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

        # Recommandation
        report_lines.append("\n" + "=" * 70)
        report_lines.append("RECOMMANDATION :")

        # Trouver le meilleur modele sur val
        best_model = min(
            results.items(),
            key=lambda x: x[1].get("metrics_val", {}).get("rmse", float("inf")),
        )
        report_lines.append(
            f"  Meilleur modele (Val RMSE) : {best_model[0]} "
            f"(RMSE={best_model[1].get('metrics_val', {}).get('rmse', 0):.2f})"
        )
        report_lines.append("=" * 70)

        # Sauvegarder
        report = "\n".join(report_lines)
        path = Path("data/models") / "evaluation_report.txt"
        path.write_text(report, encoding="utf-8")
        self.logger.info("Rapport d'evaluation → %s", path)
        return path
