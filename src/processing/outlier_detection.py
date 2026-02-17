# -*- coding: utf-8 -*-
"""
Detection et gestion des outliers — Phase 2.2.
================================================

Ce module fournit une detection multi-methode des valeurs aberrantes,
adaptee au petit dataset HVAC (~96-288 observations).

Methodes implementees :
    1. IQR (Tukey) — Robuste, non-parametrique. Seuil = 1.5 * IQR
    2. Z-score modifie — Utilise la mediane/MAD au lieu de mean/std
       (robuste aux outliers existants)
    3. Isolation Forest — Methode ML non-supervisee, detecte les
       anomalies multivariees

Strategies de traitement :
    - flag   : marquer les outliers (colonne booleenne) sans les modifier
    - clip   : ramener les outliers aux bornes IQR (winsorization)
    - remove : supprimer les lignes contenant des outliers

Usage :
    >>> from src.processing.outlier_detection import OutlierDetector
    >>> detector = OutlierDetector(config)
    >>> df_flagged, report = detector.detect_and_flag(df)
    >>> # Ou integre au pipeline :
    >>> df_clean = detector.detect_and_treat(df, strategy="clip")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import ProjectConfig


class OutlierDetector:
    """Detecteur d'outliers multi-methode pour le projet HVAC.

    Combine IQR, Z-score modifie et Isolation Forest pour identifier
    les valeurs aberrantes dans le dataset ML. Genere un rapport
    detaille avec statistiques et recommandations.

    Attributes:
        config: Configuration du projet.
        logger: Logger structure.
        iqr_factor: Facteur IQR pour la detection (defaut: 1.5).
        zscore_threshold: Seuil Z-score modifie (defaut: 3.5).
        contamination: Taux attendu d'anomalies pour Isolation Forest.
    """

    # Colonnes numeriques cles a analyser
    TARGET_COLS = [
        "nb_dpe_total", "nb_installations_pac",
        "nb_installations_clim", "nb_dpe_classe_ab",
    ]

    FEATURE_COLS = [
        "temp_mean", "hdd_sum", "cdd_sum",
        "precip_sum", "wind_max",
        "confiance_menages", "climat_affaires_industrie",
        "climat_affaires_bat", "ipi_hvac_c28", "ipi_hvac_c2825",
    ]

    def __init__(
        self,
        config: ProjectConfig,
        iqr_factor: float = 1.5,
        zscore_threshold: float = 3.5,
        contamination: float = 0.05,
    ) -> None:
        self.config = config
        self.logger = logging.getLogger("processing.outliers")
        self.iqr_factor = iqr_factor
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination

        # Repertoire pour les rapports
        self.report_dir = Path("data/analysis")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Detection IQR (Tukey)
    # ==================================================================

    def detect_iqr(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Detecte les outliers par la methode IQR (Tukey).

        Un point est outlier si : x < Q1 - factor*IQR  ou  x > Q3 + factor*IQR
        Avec IQR = Q3 - Q1 (ecart interquartile).

        Args:
            df: DataFrame a analyser.
            columns: Colonnes a verifier. Si None, utilise les colonnes par defaut.

        Returns:
            Dictionnaire {colonne: {n_outliers, pct, lower_bound, upper_bound, indices}}.
        """
        if columns is None:
            columns = self._get_numeric_columns(df)

        results = {}
        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr

            mask = (df[col] < lower) | (df[col] > upper)
            mask = mask & df[col].notna()

            n_outliers = mask.sum()
            if n_outliers > 0:
                results[col] = {
                    "n_outliers": int(n_outliers),
                    "pct": round(100 * n_outliers / len(series), 2),
                    "lower_bound": round(float(lower), 4),
                    "upper_bound": round(float(upper), 4),
                    "indices": df.index[mask].tolist(),
                    "values": df.loc[mask, col].tolist(),
                }

        return results

    # ==================================================================
    # Detection Z-score modifie (MAD-based)
    # ==================================================================

    def detect_zscore_modified(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Detecte les outliers par Z-score modifie (MAD).

        Le Z-score modifie utilise la mediane et la MAD (Median Absolute
        Deviation) au lieu de la moyenne et l'ecart-type, le rendant
        robuste aux outliers existants.

        Formule : M_i = 0.6745 * (x_i - median) / MAD
        Seuil par defaut : |M_i| > 3.5

        Args:
            df: DataFrame a analyser.
            columns: Colonnes a verifier.

        Returns:
            Dictionnaire {colonne: {n_outliers, pct, indices}}.
        """
        if columns is None:
            columns = self._get_numeric_columns(df)

        results = {}
        for col in columns:
            if col not in df.columns:
                continue

            series = df[col].dropna()
            if len(series) < 4:
                continue

            median = series.median()
            mad = np.median(np.abs(series - median))

            if mad == 0:
                # MAD=0 signifie que > 50% des valeurs sont identiques
                continue

            # Z-score modifie
            modified_z = 0.6745 * (df[col] - median) / mad
            mask = modified_z.abs() > self.zscore_threshold
            mask = mask & df[col].notna()

            n_outliers = mask.sum()
            if n_outliers > 0:
                results[col] = {
                    "n_outliers": int(n_outliers),
                    "pct": round(100 * n_outliers / len(series), 2),
                    "indices": df.index[mask].tolist(),
                    "z_scores": modified_z[mask].round(2).tolist(),
                }

        return results

    # ==================================================================
    # Detection Isolation Forest (multivariee)
    # ==================================================================

    def detect_isolation_forest(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Detecte les outliers multivaries par Isolation Forest.

        Contrairement a IQR et Z-score qui sont univaries, Isolation
        Forest detecte les anomalies dans l'espace multivarie (combinaisons
        inhabituelles de features).

        Args:
            df: DataFrame a analyser.
            columns: Colonnes a utiliser. Si None, utilise les features par defaut.

        Returns:
            Dictionnaire avec indices des outliers et scores d'anomalie.
        """
        from sklearn.ensemble import IsolationForest

        if columns is None:
            columns = self._get_numeric_columns(df)

        # Ne garder que les colonnes presentes et non-NaN
        available_cols = [c for c in columns if c in df.columns]
        if len(available_cols) < 2:
            self.logger.warning("Pas assez de colonnes pour Isolation Forest")
            return {"n_outliers": 0, "indices": [], "scores": []}

        # Imputer les NaN avec la mediane pour IF
        df_if = df[available_cols].copy()
        df_if = df_if.fillna(df_if.median())

        if len(df_if) < 10:
            self.logger.warning("Pas assez de donnees pour Isolation Forest")
            return {"n_outliers": 0, "indices": [], "scores": []}

        # Ajuster contamination au dataset
        contamination = min(self.contamination, 0.5)

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        predictions = model.fit_predict(df_if)
        scores = model.decision_function(df_if)

        # -1 = anomalie, 1 = normal
        mask = predictions == -1
        outlier_indices = df.index[mask].tolist()

        return {
            "n_outliers": int(mask.sum()),
            "pct": round(100 * mask.sum() / len(df), 2),
            "indices": outlier_indices,
            "scores": scores[mask].round(4).tolist(),
            "columns_used": available_cols,
        }

    # ==================================================================
    # Detection combinee + flagging
    # ==================================================================

    def detect_and_flag(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detecte les outliers par toutes les methodes et les flag.

        Ajoute des colonnes booleennes au DataFrame :
        - _outlier_iqr      : True si outlier IQR sur au moins une colonne
        - _outlier_zscore   : True si outlier Z-score modifie
        - _outlier_iforest  : True si outlier Isolation Forest
        - _outlier_consensus: True si detecte par >= 2 methodes

        Args:
            df: DataFrame a analyser.
            columns: Colonnes a verifier.

        Returns:
            Tuple (df_flagged, report) avec le DataFrame annote et
            le rapport de detection.
        """
        self.logger.info("=" * 60)
        self.logger.info("  DETECTION DES OUTLIERS")
        self.logger.info("=" * 60)

        df = df.copy()
        report: Dict[str, Any] = {"n_rows": len(df)}

        # 1. IQR
        self.logger.info("  Methode 1 : IQR (Tukey, factor=%.1f)...", self.iqr_factor)
        iqr_results = self.detect_iqr(df, columns)
        report["iqr"] = iqr_results

        iqr_indices = set()
        for col_info in iqr_results.values():
            iqr_indices.update(col_info["indices"])
        df["_outlier_iqr"] = df.index.isin(iqr_indices)

        n_iqr = df["_outlier_iqr"].sum()
        self.logger.info(
            "    IQR : %d outliers (%d colonnes affectees)",
            n_iqr, len(iqr_results),
        )

        # 2. Z-score modifie
        self.logger.info("  Methode 2 : Z-score modifie (seuil=%.1f)...", self.zscore_threshold)
        zscore_results = self.detect_zscore_modified(df, columns)
        report["zscore"] = zscore_results

        zscore_indices = set()
        for col_info in zscore_results.values():
            zscore_indices.update(col_info["indices"])
        df["_outlier_zscore"] = df.index.isin(zscore_indices)

        n_zscore = df["_outlier_zscore"].sum()
        self.logger.info(
            "    Z-score : %d outliers (%d colonnes affectees)",
            n_zscore, len(zscore_results),
        )

        # 3. Isolation Forest
        self.logger.info("  Methode 3 : Isolation Forest (contamination=%.2f)...", self.contamination)
        iforest_results = self.detect_isolation_forest(df, columns)
        report["isolation_forest"] = iforest_results

        df["_outlier_iforest"] = df.index.isin(iforest_results["indices"])

        n_iforest = df["_outlier_iforest"].sum()
        self.logger.info(
            "    Isolation Forest : %d outliers", n_iforest,
        )

        # 4. Consensus (detecte par >= 2 methodes)
        outlier_count = (
            df["_outlier_iqr"].astype(int)
            + df["_outlier_zscore"].astype(int)
            + df["_outlier_iforest"].astype(int)
        )
        df["_outlier_consensus"] = outlier_count >= 2
        df["_outlier_score"] = outlier_count

        n_consensus = df["_outlier_consensus"].sum()
        report["consensus"] = {
            "n_outliers": int(n_consensus),
            "pct": round(100 * n_consensus / len(df), 2),
        }

        self.logger.info("-" * 40)
        self.logger.info("  RESUME :")
        self.logger.info("    IQR              : %d outliers (%.1f%%)", n_iqr, 100 * n_iqr / len(df))
        self.logger.info("    Z-score modifie  : %d outliers (%.1f%%)", n_zscore, 100 * n_zscore / len(df))
        self.logger.info("    Isolation Forest : %d outliers (%.1f%%)", n_iforest, 100 * n_iforest / len(df))
        self.logger.info("    Consensus (>=2)  : %d outliers (%.1f%%)", n_consensus, 100 * n_consensus / len(df))
        self.logger.info("=" * 60)

        return df, report

    # ==================================================================
    # Traitement des outliers
    # ==================================================================

    def detect_and_treat(
        self,
        df: pd.DataFrame,
        strategy: str = "clip",
        columns: Optional[List[str]] = None,
        use_consensus: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detecte et traite les outliers selon la strategie choisie.

        Args:
            df: DataFrame a traiter.
            strategy: Strategie de traitement :
                - "flag"   : ajouter des colonnes booleennes (pas de modification)
                - "clip"   : winsorization aux bornes IQR
                - "remove" : supprimer les lignes outliers
            columns: Colonnes a traiter.
            use_consensus: Si True, ne traite que les outliers consensus (>= 2 methodes).

        Returns:
            Tuple (df_treated, report).
        """
        df_flagged, report = self.detect_and_flag(df, columns)
        report["strategy"] = strategy

        if strategy == "flag":
            return df_flagged, report

        if strategy == "clip":
            return self._apply_clip(df_flagged, columns, report)

        if strategy == "remove":
            return self._apply_remove(df_flagged, use_consensus, report)

        self.logger.error("Strategie inconnue : %s", strategy)
        return df_flagged, report

    def _apply_clip(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]],
        report: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applique le clipping (winsorization) aux bornes IQR.

        Les valeurs au-dela des bornes IQR sont ramenees aux bornes,
        preservant toutes les lignes mais attenuant l'impact des extremes.
        """
        if columns is None:
            columns = self._get_numeric_columns(df)

        n_clipped = 0
        clip_details = {}

        for col in columns:
            if col not in df.columns or col.startswith("_"):
                continue

            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_factor * iqr
            upper = q3 + self.iqr_factor * iqr

            mask = (df[col] < lower) | (df[col] > upper)
            mask = mask & df[col].notna()
            n_col = mask.sum()

            if n_col > 0:
                df[col] = df[col].clip(lower, upper)
                n_clipped += n_col
                clip_details[col] = {
                    "n_clipped": int(n_col),
                    "lower": round(float(lower), 4),
                    "upper": round(float(upper), 4),
                }

        report["clip_details"] = clip_details
        report["total_clipped"] = n_clipped
        self.logger.info("  Winsorization : %d valeurs clippees", n_clipped)

        return df, report

    def _apply_remove(
        self,
        df: pd.DataFrame,
        use_consensus: bool,
        report: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Supprime les lignes identifiees comme outliers."""
        col = "_outlier_consensus" if use_consensus else "_outlier_iqr"
        n_before = len(df)
        df = df[~df[col]].copy()
        n_removed = n_before - len(df)

        report["n_removed"] = n_removed
        self.logger.info(
            "  Suppression : %d lignes retirees (%s)",
            n_removed, col,
        )

        return df, report

    # ==================================================================
    # Analyse temporelle des anomalies
    # ==================================================================

    def detect_temporal_anomalies(
        self,
        df: pd.DataFrame,
        target_col: str = "nb_installations_pac",
        threshold_pct: float = 50.0,
    ) -> Dict[str, Any]:
        """Detecte les anomalies temporelles (pics/chutes soudains).

        Identifie les mois ou la variation relative depasse un seuil,
        ce qui peut indiquer des problemes de donnees ou des evenements
        exceptionnels.

        Args:
            df: DataFrame avec date_id, dept et la colonne cible.
            target_col: Colonne a analyser.
            threshold_pct: Seuil de variation en % pour flaguer une anomalie.

        Returns:
            Dictionnaire avec les anomalies detectees par departement.
        """
        if target_col not in df.columns or "dept" not in df.columns:
            return {"anomalies": []}

        anomalies = []
        for dept, group in df.groupby("dept"):
            group = group.sort_values("date_id")
            pct_change = group[target_col].pct_change() * 100

            for idx, val in pct_change.items():
                if pd.notna(val) and abs(val) > threshold_pct:
                    row = df.loc[idx]
                    anomalies.append({
                        "dept": dept,
                        "date_id": int(row["date_id"]),
                        "value": float(row[target_col]),
                        "pct_change": round(float(val), 1),
                        "type": "spike" if val > 0 else "drop",
                    })

        self.logger.info(
            "  Anomalies temporelles : %d detectees (seuil=%.0f%%)",
            len(anomalies), threshold_pct,
        )
        return {"anomalies": anomalies, "threshold_pct": threshold_pct}

    # ==================================================================
    # Rapport
    # ==================================================================

    def generate_report(
        self,
        df: pd.DataFrame,
        report: Dict[str, Any],
        temporal_report: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Genere un rapport textuel complet de la detection d'outliers.

        Args:
            df: DataFrame avec flags d'outliers.
            report: Rapport de detection.
            temporal_report: Rapport d'anomalies temporelles (optionnel).

        Returns:
            Chemin du fichier rapport.
        """
        lines = [
            "=" * 70,
            "RAPPORT DE DETECTION DES OUTLIERS",
            "=" * 70,
            f"Dataset : {report.get('n_rows', len(df))} lignes",
            f"Strategie appliquee : {report.get('strategy', 'flag')}",
            "",
        ]

        # IQR
        lines.append("--- METHODE 1 : IQR (Tukey, factor=%.1f) ---" % self.iqr_factor)
        iqr = report.get("iqr", {})
        if iqr:
            for col, info in iqr.items():
                lines.append(
                    "  %-35s : %3d outliers (%5.1f%%) | bornes [%.2f, %.2f]"
                    % (col, info["n_outliers"], info["pct"],
                       info["lower_bound"], info["upper_bound"])
                )
        else:
            lines.append("  Aucun outlier IQR detecte.")
        lines.append("")

        # Z-score
        lines.append("--- METHODE 2 : Z-score modifie (seuil=%.1f) ---" % self.zscore_threshold)
        zscore = report.get("zscore", {})
        if zscore:
            for col, info in zscore.items():
                lines.append(
                    "  %-35s : %3d outliers (%5.1f%%)"
                    % (col, info["n_outliers"], info["pct"])
                )
        else:
            lines.append("  Aucun outlier Z-score detecte.")
        lines.append("")

        # Isolation Forest
        lines.append("--- METHODE 3 : Isolation Forest ---")
        iforest = report.get("isolation_forest", {})
        lines.append(
            "  Outliers multivaries : %d (%.1f%%)"
            % (iforest.get("n_outliers", 0), iforest.get("pct", 0))
        )
        lines.append("")

        # Consensus
        lines.append("--- CONSENSUS (>= 2 methodes) ---")
        consensus = report.get("consensus", {})
        lines.append(
            "  Outliers confirmes : %d (%.1f%%)"
            % (consensus.get("n_outliers", 0), consensus.get("pct", 0))
        )
        lines.append("")

        # Clipping
        if "clip_details" in report:
            lines.append("--- WINSORIZATION APPLIQUEE ---")
            for col, info in report["clip_details"].items():
                lines.append(
                    "  %-35s : %3d valeurs clippees | bornes [%.2f, %.2f]"
                    % (col, info["n_clipped"], info["lower"], info["upper"])
                )
            lines.append(
                "  Total valeurs modifiees : %d" % report.get("total_clipped", 0)
            )
            lines.append("")

        # Anomalies temporelles
        if temporal_report and temporal_report.get("anomalies"):
            lines.append("--- ANOMALIES TEMPORELLES ---")
            lines.append(
                "  Seuil de variation : %.0f%%" % temporal_report["threshold_pct"]
            )
            for a in temporal_report["anomalies"][:20]:
                lines.append(
                    "  Dept %s | %d | %s %+.1f%% (valeur=%.0f)"
                    % (a["dept"], a["date_id"], a["type"], a["pct_change"], a["value"])
                )
            total = len(temporal_report["anomalies"])
            if total > 20:
                lines.append("  ... et %d autres anomalies" % (total - 20))
            lines.append("")

        lines.append("=" * 70)

        # Sauvegarder
        report_text = "\n".join(lines)
        path = self.report_dir / "outlier_report.txt"
        path.write_text(report_text, encoding="utf-8")
        self.logger.info("Rapport outliers sauvegarde → %s", path)

        return path

    # ==================================================================
    # Integration pipeline
    # ==================================================================

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        strategy: str = "clip",
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Path]:
        """Execute l'analyse complete des outliers + traitement + rapport.

        Methode de convenance qui enchaine detection, traitement,
        analyse temporelle et generation du rapport.

        Args:
            df: DataFrame a analyser (dataset features ou ML).
            strategy: Strategie de traitement ("flag", "clip", "remove").
            columns: Colonnes a analyser.

        Returns:
            Tuple (df_treated, report_path).
        """
        # Detection et traitement
        df_treated, report = self.detect_and_treat(df, strategy, columns)

        # Anomalies temporelles sur les cibles
        temporal_report = None
        for target in self.TARGET_COLS:
            if target in df_treated.columns:
                temporal_report = self.detect_temporal_anomalies(df_treated, target)
                break

        # Rapport
        report_path = self.generate_report(df_treated, report, temporal_report)

        return df_treated, report_path

    # ==================================================================
    # Utilitaires
    # ==================================================================

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Retourne les colonnes numeriques pertinentes du DataFrame.

        Exclut les identifiants, metadata et colonnes de flags existantes.
        """
        exclude = {
            "date_id", "year", "month", "quarter",
            "n_valid_features", "pct_valid_features",
        }
        # Exclure aussi les colonnes de flags internes
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [
            c for c in numeric_cols
            if c not in exclude and not c.startswith("_")
        ]
