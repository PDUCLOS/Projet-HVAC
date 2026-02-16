# -*- coding: utf-8 -*-
"""
Études de corrélation — Phase 3.2.
====================================

Ce module analyse les corrélations entre features et variables cibles
pour guider la sélection de features en Phase 4 (modélisation ML).

Analyses produites :
    1. Matrice de corrélation complète (heatmap)
    2. Top corrélations avec chaque variable cible
    3. Corrélations par département (stabilité géographique)
    4. Corrélations par saison (variations saisonnières)
    5. Analyse de multicolinéarité (VIF)
    6. Matrice de corrélation des lags et rolling features

Usage :
    >>> from src.analysis.correlation import CorrelationAnalyzer
    >>> corr = CorrelationAnalyzer(config)
    >>> corr.run_full_correlation()

    # Ou via CLI
    python -m src.pipeline eda
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

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyseur de corrélations pour le dataset HVAC.

    Attributes:
        config: Configuration du projet.
        df: DataFrame features dataset.
        output_dir: Répertoire de sortie pour les graphiques.
    """

    DEPT_NAMES = {
        "01": "Ain", "07": "Ardèche", "26": "Drôme", "38": "Isère",
        "42": "Loire", "69": "Rhône", "73": "Savoie", "74": "Haute-Savoie",
    }

    # Features de base (hors lags/rolling) pour la matrice principale
    BASE_FEATURES = [
        "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
        "nb_dpe_classe_ab", "pct_pac", "pct_clim",
        "temp_mean", "temp_max", "temp_min", "hdd_sum", "cdd_sum",
        "precipitation_sum", "nb_jours_canicule", "nb_jours_gel",
        "confiance_menages", "climat_affaires_indus", "climat_affaires_bat",
        "ipi_manufacturing", "ipi_hvac_c28", "ipi_hvac_c2825",
        "month", "quarter", "is_heating", "is_cooling",
    ]

    TARGET_COL = "nb_installations_pac"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.output_dir = Path("data/analysis/figures")
        self.report_dir = Path("data/analysis")
        self.df: Optional[pd.DataFrame] = None

        sns.set_theme(style="whitegrid", font_scale=1.0)
        plt.rcParams.update({
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        })

    def _load_data(self) -> pd.DataFrame:
        """Charge le features dataset."""
        filepath = self.config.features_data_dir / "hvac_features_dataset.csv"
        if not filepath.exists():
            raise FileNotFoundError(f"Features dataset introuvable : {filepath}")

        df = pd.read_csv(filepath)
        df["dept"] = df["dept"].astype(str).str.zfill(2)
        df["date"] = pd.to_datetime(df["date_id"].astype(str), format="%Y%m")
        return df

    def _ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, name: str) -> Path:
        path = self.output_dir / f"{name}.png"
        plt.savefig(path)
        plt.close()
        logger.info("  → %s", path)
        return path

    # ------------------------------------------------------------------
    # 1. Matrice de corrélation complète
    # ------------------------------------------------------------------

    def plot_correlation_matrix(self) -> Path:
        """Matrice de corrélation des features de base (hors lags).

        Returns:
            Chemin du graphique sauvegardé.
        """
        df = self.df
        cols = [c for c in self.BASE_FEATURES if c in df.columns]
        corr_matrix = df[cols].corr()

        # Masquer la moitié supérieure (redondante)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={"size": 8},
        )
        ax.set_title("Matrice de corrélation — Features de base", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(fontsize=9)

        return self._save_fig("20_correlation_matrix_base")

    # ------------------------------------------------------------------
    # 2. Top corrélations avec la cible
    # ------------------------------------------------------------------

    def plot_top_correlations(self, target: str = None, n_top: int = 20) -> Path:
        """Bar chart des corrélations les plus fortes avec la cible.

        Args:
            target: Nom de la colonne cible (défaut: nb_installations_pac).
            n_top: Nombre de features à afficher.

        Returns:
            Chemin du graphique sauvegardé.
        """
        target = target or self.TARGET_COL
        df = self.df
        numeric = df.select_dtypes(include=[np.number])

        if target not in numeric.columns:
            logger.warning("Cible '%s' absente du dataset.", target)
            return Path()

        corr = numeric.corrwith(numeric[target])
        corr = corr.drop(target, errors="ignore")

        # Exclure les colonnes auxiliaires
        exclude = ["date_id", "year_trend", "n_valid_features", "pct_valid_features"]
        corr = corr.drop(exclude, errors="ignore")

        # Top N par valeur absolue
        top = corr.abs().sort_values(ascending=False).head(n_top)
        top_signed = corr.loc[top.index]

        fig, ax = plt.subplots(figsize=(12, max(6, n_top * 0.35)))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in top_signed.values]
        ax.barh(range(len(top_signed)), top_signed.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_signed)))
        ax.set_yticklabels(top_signed.index, fontsize=9)
        ax.set_xlabel("Coefficient de corrélation (Pearson)")
        ax.set_title(f"Top {n_top} corrélations avec {target}", fontsize=14)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        # Annotations
        for i, (val, feat) in enumerate(zip(top_signed.values, top_signed.index)):
            ax.text(val + 0.01 * np.sign(val), i, f"{val:.3f}",
                    va="center", fontsize=8)

        return self._save_fig(f"21_top_correlations_{target}")

    # ------------------------------------------------------------------
    # 3. Corrélations par département
    # ------------------------------------------------------------------

    def plot_correlation_by_dept(self) -> Path:
        """Heatmap des corrélations features→PAC par département.

        Permet de voir si les relations sont stables géographiquement.

        Returns:
            Chemin du graphique sauvegardé.
        """
        df = self.df
        target = self.TARGET_COL
        features = [
            "temp_mean", "hdd_sum", "cdd_sum", "precipitation_sum",
            "confiance_menages", "ipi_hvac_c28",
            "nb_jours_canicule", "nb_jours_gel", "month",
        ]
        features = [f for f in features if f in df.columns]

        # Corrélation par département
        corr_by_dept = {}
        for dept in sorted(df["dept"].unique()):
            subset = df[df["dept"] == dept]
            if len(subset) > 5 and target in subset.columns:
                corr = subset[features].corrwith(subset[target])
                corr_by_dept[f"{dept} — {self.DEPT_NAMES.get(dept, dept)}"] = corr

        if not corr_by_dept:
            logger.warning("Pas assez de données pour les corrélations par dept.")
            return Path()

        corr_df = pd.DataFrame(corr_by_dept).T

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 9},
        )
        ax.set_title(
            f"Corrélations features → {target} par département",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)

        return self._save_fig("22_correlation_by_dept")

    # ------------------------------------------------------------------
    # 4. Corrélations par saison
    # ------------------------------------------------------------------

    def plot_correlation_by_season(self) -> Path:
        """Heatmap des corrélations par saison (hiver/été/etc.).

        Returns:
            Chemin du graphique sauvegardé.
        """
        df = self.df.copy()
        target = self.TARGET_COL

        df["saison"] = df["month"].map(lambda m: (
            "1_Hiver" if m in [12, 1, 2] else
            "2_Printemps" if m in [3, 4, 5] else
            "3_Été" if m in [6, 7, 8] else "4_Automne"
        ))

        features = [
            "temp_mean", "hdd_sum", "cdd_sum", "precipitation_sum",
            "confiance_menages", "ipi_hvac_c28",
            "nb_jours_canicule", "nb_jours_gel",
        ]
        features = [f for f in features if f in df.columns]

        corr_by_season = {}
        for saison in sorted(df["saison"].unique()):
            subset = df[df["saison"] == saison]
            if len(subset) > 5 and target in subset.columns:
                corr = subset[features].corrwith(subset[target])
                corr_by_season[saison.split("_")[1]] = corr

        if not corr_by_season:
            return Path()

        corr_df = pd.DataFrame(corr_by_season).T

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            corr_df, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 10},
        )
        ax.set_title(
            f"Corrélations features → {target} par saison",
            fontsize=14,
        )
        plt.xticks(rotation=45, ha="right", fontsize=10)

        return self._save_fig("23_correlation_by_season")

    # ------------------------------------------------------------------
    # 5. Analyse de multicolinéarité
    # ------------------------------------------------------------------

    def plot_multicollinearity(self) -> Path:
        """Identifie les paires de features très corrélées (|r| > 0.8).

        Returns:
            Chemin du graphique sauvegardé.
        """
        df = self.df
        features = [c for c in self.BASE_FEATURES if c in df.columns]
        corr = df[features].corr()

        # Extraire les paires avec |r| > 0.8 (hors diagonale)
        pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                r = corr.iloc[i, j]
                if abs(r) > 0.8:
                    pairs.append((features[i], features[j], r))

        if not pairs:
            logger.info("  Aucune paire avec |r| > 0.8 trouvée.")
            return Path()

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        fig, ax = plt.subplots(figsize=(12, max(5, len(pairs) * 0.5)))
        labels = [f"{a} ↔ {b}" for a, b, _ in pairs]
        values = [r for _, _, r in pairs]
        colors = ["#e74c3c" if r > 0 else "#3498db" for r in values]

        ax.barh(range(len(pairs)), values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Coefficient de corrélation")
        ax.set_title("Paires multicolinéaires (|r| > 0.8)", fontsize=14)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()

        for i, val in enumerate(values):
            ax.text(val + 0.01 * np.sign(val), i, f"{val:.3f}",
                    va="center", fontsize=9)

        return self._save_fig("24_multicollinearity")

    # ------------------------------------------------------------------
    # 6. Corrélation des lags
    # ------------------------------------------------------------------

    def plot_lag_correlations(self) -> Path:
        """Heatmap des corrélations entre les différents lags et la cible.

        Permet de déterminer quels horizons de lag sont les plus prédictifs.

        Returns:
            Chemin du graphique sauvegardé.
        """
        df = self.df
        target = self.TARGET_COL

        # Trouver toutes les colonnes de lag
        lag_cols = [c for c in df.columns if "_lag_" in c]
        if not lag_cols:
            logger.info("  Aucune colonne lag trouvée.")
            return Path()

        # Organiser par variable et horizon
        lag_data = {}
        for col in lag_cols:
            # Format: variable_lag_Xm
            parts = col.rsplit("_lag_", 1)
            if len(parts) == 2:
                var = parts[0]
                horizon = parts[1]
                if var not in lag_data:
                    lag_data[var] = {}
                if target in df.columns:
                    lag_data[var][horizon] = df[col].corr(df[target])

        if not lag_data:
            return Path()

        # Créer un DataFrame pour la heatmap
        lag_df = pd.DataFrame(lag_data).T
        # Trier les colonnes par horizon
        col_order = sorted(lag_df.columns, key=lambda x: int(x.replace("m", "")))
        lag_df = lag_df[col_order]

        fig, ax = plt.subplots(figsize=(10, max(5, len(lag_df) * 0.6)))
        sns.heatmap(
            lag_df, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
        )
        ax.set_title(f"Corrélation des lags avec {target}", fontsize=14)
        ax.set_xlabel("Horizon du lag")
        ax.set_ylabel("Variable")

        return self._save_fig("25_lag_correlations")

    # ------------------------------------------------------------------
    # Rapport corrélations
    # ------------------------------------------------------------------

    def generate_correlation_report(self) -> Path:
        """Génère un rapport textuel des corrélations.

        Returns:
            Chemin du rapport sauvegardé.
        """
        df = self.df
        target = self.TARGET_COL
        report_path = self.report_dir / "correlation_report.txt"

        lines = []
        lines.append("=" * 70)
        lines.append("  RAPPORT CORRÉLATIONS — HVAC Market Analysis")
        lines.append("=" * 70)

        # Top 20 corrélations avec la cible
        numeric = df.select_dtypes(include=[np.number])
        if target in numeric.columns:
            corr = numeric.corrwith(numeric[target])
            corr = corr.drop(target, errors="ignore")
            top = corr.abs().sort_values(ascending=False).head(20)

            lines.append(f"\n\n1. TOP 20 CORRÉLATIONS AVEC {target}")
            lines.append("-" * 50)
            for feat in top.index:
                val = corr[feat]
                lines.append(f"  {'+' if val > 0 else '-'}{abs(val):.3f}  {feat}")

        # Paires multicolinéaires
        features = [c for c in self.BASE_FEATURES if c in df.columns]
        corr_matrix = df[features].corr()
        pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.8:
                    pairs.append((features[i], features[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        lines.append(f"\n\n2. PAIRES MULTICOLINÉAIRES (|r| > 0.8) — {len(pairs)} trouvées")
        lines.append("-" * 50)
        for a, b, r in pairs:
            lines.append(f"  {r:+.3f}  {a} ↔ {b}")

        # Recommandations
        lines.append("\n\n3. RECOMMANDATIONS POUR LA MODÉLISATION")
        lines.append("-" * 50)
        if pairs:
            lines.append("  ⚠ Multicolinéarité détectée :")
            seen = set()
            for a, b, r in pairs:
                if a not in seen and b not in seen:
                    lines.append(f"    → Garder {a}, considérer retirer {b} (r={r:.3f})")
                    seen.add(b)

        lines.append("  ✓ Utiliser Ridge ou Lasso pour gérer la colinéarité")
        lines.append("  ✓ LightGBM est naturellement robuste à la colinéarité")
        lines.append("  ✓ Vérifier la stabilité des corrélations par département")

        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("Rapport corrélations sauvegardé : %s", report_path)

        return report_path

    # ------------------------------------------------------------------
    # Orchestrateur
    # ------------------------------------------------------------------

    def run_full_correlation(self) -> Dict[str, Any]:
        """Exécute l'ensemble des analyses de corrélation.

        Returns:
            Dictionnaire avec les chemins des fichiers générés.
        """
        logger.info("=" * 60)
        logger.info("  Phase 3.2 — Analyse des corrélations")
        logger.info("=" * 60)

        self._ensure_dirs()
        self.df = self._load_data()

        results = {"figures": []}

        logger.info("Matrice de corrélation...")
        results["figures"].append(str(self.plot_correlation_matrix()))

        logger.info("Top corrélations vs cible...")
        results["figures"].append(str(self.plot_top_correlations()))

        logger.info("Corrélations par département...")
        fig = self.plot_correlation_by_dept()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Corrélations par saison...")
        fig = self.plot_correlation_by_season()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Analyse multicolinéarité...")
        fig = self.plot_multicollinearity()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Corrélations des lags...")
        fig = self.plot_lag_correlations()
        if fig.name:
            results["figures"].append(str(fig))

        logger.info("Rapport de corrélations...")
        report = self.generate_correlation_report()
        results["report"] = str(report)

        logger.info(
            "Analyse des corrélations terminée : %d graphiques.",
            len(results["figures"]),
        )
        return results
