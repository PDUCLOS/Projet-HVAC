# -*- coding: utf-8 -*-
"""Tests pour le module de detection des outliers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from config.settings import ProjectConfig
from src.processing.outlier_detection import OutlierDetector


@pytest.fixture
def detector(test_config: ProjectConfig) -> OutlierDetector:
    """Instance OutlierDetector pour les tests."""
    return OutlierDetector(test_config)


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame avec des outliers connus pour tester la detection."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "date_id": range(202201, 202201 + n),
        "dept": ["69"] * n,
        "nb_installations_pac": np.concatenate([
            np.random.normal(50, 10, n - 3),
            [200, 250, -10],  # 3 outliers evidents
        ]),
        "temp_mean": np.concatenate([
            np.random.normal(15, 5, n - 2),
            [60, -40],  # 2 outliers temperature
        ]),
        "hdd_sum": np.random.uniform(0, 300, n),
        "confiance_menages": np.concatenate([
            np.random.normal(95, 5, n - 1),
            [200],  # 1 outlier
        ]),
    })
    return df


@pytest.fixture
def df_clean() -> pd.DataFrame:
    """DataFrame sans outliers."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "date_id": range(202201, 202201 + n),
        "dept": ["69"] * n,
        "value_a": np.random.normal(100, 5, n),
        "value_b": np.random.normal(50, 3, n),
    })


# ==================================================================
# Tests IQR
# ==================================================================

class TestDetectIQR:
    """Tests pour la methode IQR."""

    def test_detects_known_outliers(self, detector, df_with_outliers):
        """L'IQR doit detecter les outliers evidents."""
        results = detector.detect_iqr(df_with_outliers)
        # La colonne nb_installations_pac a des outliers (200, 250, -10)
        assert "nb_installations_pac" in results
        assert results["nb_installations_pac"]["n_outliers"] >= 2

    def test_no_outliers_on_clean_data(self, detector, df_clean):
        """Pas d'outlier sur des donnees propres (distribution normale centree)."""
        results = detector.detect_iqr(df_clean, columns=["value_a", "value_b"])
        total = sum(info["n_outliers"] for info in results.values())
        # Distribution normale a tres peu d'outliers IQR (< 5%)
        assert total < 5

    def test_returns_bounds(self, detector, df_with_outliers):
        """Les bornes IQR doivent etre retournees."""
        results = detector.detect_iqr(df_with_outliers, columns=["nb_installations_pac"])
        if "nb_installations_pac" in results:
            info = results["nb_installations_pac"]
            assert "lower_bound" in info
            assert "upper_bound" in info
            assert info["lower_bound"] < info["upper_bound"]

    def test_returns_indices(self, detector, df_with_outliers):
        """Les indices des outliers doivent etre retournes."""
        results = detector.detect_iqr(df_with_outliers)
        for col, info in results.items():
            assert "indices" in info
            assert isinstance(info["indices"], list)

    def test_empty_dataframe(self, detector):
        """Ne plante pas sur un DataFrame vide."""
        df = pd.DataFrame({"val": []})
        results = detector.detect_iqr(df)
        assert results == {}

    def test_ignores_nan(self, detector):
        """Les NaN ne doivent pas etre comptes comme outliers."""
        df = pd.DataFrame({
            "val": [1, 2, 3, 4, 5, np.nan, np.nan, np.nan, 100]
        })
        results = detector.detect_iqr(df, columns=["val"])
        if "val" in results:
            for idx in results["val"]["indices"]:
                assert not pd.isna(df.loc[idx, "val"])

    def test_custom_factor(self, test_config):
        """Un facteur IQR plus strict (1.0) detecte plus d'outliers."""
        strict = OutlierDetector(test_config, iqr_factor=1.0)
        normal = OutlierDetector(test_config, iqr_factor=1.5)
        df = pd.DataFrame({"val": np.random.normal(0, 1, 200)})
        strict_res = strict.detect_iqr(df, columns=["val"])
        normal_res = normal.detect_iqr(df, columns=["val"])
        strict_n = strict_res.get("val", {}).get("n_outliers", 0)
        normal_n = normal_res.get("val", {}).get("n_outliers", 0)
        assert strict_n >= normal_n


# ==================================================================
# Tests Z-score modifie
# ==================================================================

class TestDetectZscoreModified:
    """Tests pour le Z-score modifie (MAD)."""

    def test_detects_extreme_values(self, detector, df_with_outliers):
        """Le Z-score modifie doit detecter les valeurs extremes."""
        results = detector.detect_zscore_modified(df_with_outliers)
        assert len(results) > 0

    def test_returns_z_scores(self, detector, df_with_outliers):
        """Les Z-scores doivent etre retournes."""
        results = detector.detect_zscore_modified(df_with_outliers)
        for col, info in results.items():
            assert "z_scores" in info
            # Tous les Z-scores doivent etre au-dessus du seuil
            for z in info["z_scores"]:
                assert abs(z) > detector.zscore_threshold

    def test_mad_zero_skipped(self, detector):
        """Si MAD=0 (>50% identiques), la colonne est ignoree."""
        df = pd.DataFrame({"val": [1, 1, 1, 1, 1, 1, 1, 1, 100]})
        results = detector.detect_zscore_modified(df, columns=["val"])
        assert "val" not in results

    def test_no_false_positives_normal(self, detector):
        """Distribution normale standard : < 1% d'outliers Z-score."""
        np.random.seed(42)
        df = pd.DataFrame({"val": np.random.normal(0, 1, 1000)})
        results = detector.detect_zscore_modified(df, columns=["val"])
        if "val" in results:
            assert results["val"]["pct"] < 1.0


# ==================================================================
# Tests Isolation Forest
# ==================================================================

class TestDetectIsolationForest:
    """Tests pour l'Isolation Forest."""

    def test_detects_multivariate_outliers(self, detector, df_with_outliers):
        """L'IF doit detecter des outliers multivaries."""
        results = detector.detect_isolation_forest(df_with_outliers)
        assert results["n_outliers"] > 0
        assert len(results["indices"]) == results["n_outliers"]

    def test_returns_scores(self, detector, df_with_outliers):
        """Les scores d'anomalie doivent etre retournes."""
        results = detector.detect_isolation_forest(df_with_outliers)
        assert "scores" in results

    def test_not_enough_data(self, detector):
        """Avec trop peu de donnees, retourne 0 outliers."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        results = detector.detect_isolation_forest(df)
        assert results["n_outliers"] == 0

    def test_single_column(self, detector):
        """Avec une seule colonne numerique, retourne 0 outliers."""
        df = pd.DataFrame({"a": range(50)})
        results = detector.detect_isolation_forest(df, columns=["a"])
        assert results["n_outliers"] == 0


# ==================================================================
# Tests detection combinee + flagging
# ==================================================================

class TestDetectAndFlag:
    """Tests pour la detection combinee."""

    def test_adds_flag_columns(self, detector, df_with_outliers):
        """Les colonnes de flag doivent etre ajoutees."""
        df_flagged, report = detector.detect_and_flag(df_with_outliers)
        assert "_outlier_iqr" in df_flagged.columns
        assert "_outlier_zscore" in df_flagged.columns
        assert "_outlier_iforest" in df_flagged.columns
        assert "_outlier_consensus" in df_flagged.columns
        assert "_outlier_score" in df_flagged.columns

    def test_consensus_requires_two_methods(self, detector, df_with_outliers):
        """Le consensus necessite au moins 2 methodes d'accord."""
        df_flagged, _ = detector.detect_and_flag(df_with_outliers)
        # Verifier que consensus => au moins 2 methodes
        consensus_rows = df_flagged[df_flagged["_outlier_consensus"]]
        for _, row in consensus_rows.iterrows():
            n_methods = (
                int(row["_outlier_iqr"])
                + int(row["_outlier_zscore"])
                + int(row["_outlier_iforest"])
            )
            assert n_methods >= 2

    def test_report_structure(self, detector, df_with_outliers):
        """Le rapport doit avoir la bonne structure."""
        _, report = detector.detect_and_flag(df_with_outliers)
        assert "n_rows" in report
        assert "iqr" in report
        assert "zscore" in report
        assert "isolation_forest" in report
        assert "consensus" in report

    def test_does_not_modify_data(self, detector, df_with_outliers):
        """La detection ne doit pas modifier les donnees originales."""
        original = df_with_outliers["nb_installations_pac"].copy()
        df_flagged, _ = detector.detect_and_flag(df_with_outliers)
        pd.testing.assert_series_equal(
            df_flagged["nb_installations_pac"], original,
        )


# ==================================================================
# Tests traitement
# ==================================================================

class TestDetectAndTreat:
    """Tests pour le traitement des outliers."""

    def test_strategy_flag(self, detector, df_with_outliers):
        """La strategie 'flag' ne modifie pas les valeurs."""
        original_sum = df_with_outliers["nb_installations_pac"].sum()
        df_treated, _ = detector.detect_and_treat(
            df_with_outliers, strategy="flag",
        )
        assert df_treated["nb_installations_pac"].sum() == original_sum

    def test_strategy_clip(self, detector, df_with_outliers):
        """La strategie 'clip' ramene les outliers aux bornes IQR."""
        df_treated, report = detector.detect_and_treat(
            df_with_outliers, strategy="clip",
        )
        assert report["strategy"] == "clip"
        # Les valeurs extremes doivent avoir ete attenuees
        assert df_treated["nb_installations_pac"].max() < 250

    def test_strategy_remove(self, detector, df_with_outliers):
        """La strategie 'remove' supprime des lignes."""
        original_len = len(df_with_outliers)
        df_treated, report = detector.detect_and_treat(
            df_with_outliers, strategy="remove",
        )
        assert len(df_treated) <= original_len

    def test_preserves_row_count_flag(self, detector, df_with_outliers):
        """Flag et clip preservent toutes les lignes."""
        for strategy in ["flag", "clip"]:
            df_treated, _ = detector.detect_and_treat(
                df_with_outliers, strategy=strategy,
            )
            assert len(df_treated) == len(df_with_outliers)


# ==================================================================
# Tests anomalies temporelles
# ==================================================================

class TestTemporalAnomalies:
    """Tests pour la detection d'anomalies temporelles."""

    def test_detects_spike(self, detector):
        """Detecte un pic soudain (doublement du mois precedent)."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303, 202304],
            "dept": ["69"] * 4,
            "nb_installations_pac": [100, 100, 100, 300],  # +200% au mois 4
        })
        result = detector.detect_temporal_anomalies(df, threshold_pct=50.0)
        anomalies = result["anomalies"]
        assert len(anomalies) >= 1
        assert any(a["type"] == "spike" for a in anomalies)

    def test_detects_drop(self, detector):
        """Detecte une chute soudaine."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303],
            "dept": ["69"] * 3,
            "nb_installations_pac": [100, 100, 20],  # -80% au mois 3
        })
        result = detector.detect_temporal_anomalies(df, threshold_pct=50.0)
        anomalies = result["anomalies"]
        assert len(anomalies) >= 1
        assert any(a["type"] == "drop" for a in anomalies)

    def test_no_anomaly_stable(self, detector):
        """Pas d'anomalie sur des donnees stables."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303, 202304],
            "dept": ["69"] * 4,
            "nb_installations_pac": [100, 105, 98, 103],  # < 10% variation
        })
        result = detector.detect_temporal_anomalies(df, threshold_pct=50.0)
        assert len(result["anomalies"]) == 0


# ==================================================================
# Tests rapport
# ==================================================================

class TestReport:
    """Tests pour la generation du rapport."""

    def test_generates_report_file(self, detector, df_with_outliers, tmp_path):
        """Le rapport doit etre genere en fichier texte."""
        detector.report_dir = tmp_path
        df_flagged, report = detector.detect_and_flag(df_with_outliers)
        path = detector.generate_report(df_flagged, report)
        assert path.exists()
        content = path.read_text()
        assert "RAPPORT DE DETECTION DES OUTLIERS" in content
        assert "IQR" in content
        assert "Z-score" in content
        assert "Isolation Forest" in content

    def test_report_includes_temporal(self, detector, df_with_outliers, tmp_path):
        """Le rapport inclut les anomalies temporelles si fournies."""
        detector.report_dir = tmp_path
        df_flagged, report = detector.detect_and_flag(df_with_outliers)
        temporal = {"anomalies": [{"dept": "69", "date_id": 202301,
                                   "value": 200, "pct_change": 150.0,
                                   "type": "spike"}],
                    "threshold_pct": 50.0}
        path = detector.generate_report(df_flagged, report, temporal)
        content = path.read_text()
        assert "ANOMALIES TEMPORELLES" in content


# ==================================================================
# Tests run_full_analysis
# ==================================================================

class TestRunFullAnalysis:
    """Tests pour l'analyse complete."""

    def test_returns_treated_df_and_report(self, detector, df_with_outliers, tmp_path):
        """L'analyse complete retourne le DataFrame traite et le rapport."""
        detector.report_dir = tmp_path
        df_treated, report_path = detector.run_full_analysis(
            df_with_outliers, strategy="clip",
        )
        assert isinstance(df_treated, pd.DataFrame)
        assert report_path.exists()
        assert len(df_treated) == len(df_with_outliers)


# ==================================================================
# Tests utilitaires
# ==================================================================

class TestUtilities:
    """Tests pour les methodes utilitaires."""

    def test_get_numeric_columns(self, detector):
        """Retourne uniquement les colonnes numeriques pertinentes."""
        df = pd.DataFrame({
            "date_id": [1, 2, 3],
            "dept": ["69", "38", "69"],
            "value": [10.0, 20.0, 30.0],
            "_flag": [True, False, True],
        })
        cols = detector._get_numeric_columns(df)
        assert "value" in cols
        assert "date_id" not in cols  # exclu
        assert "dept" not in cols      # non-numerique
        assert "_flag" not in cols     # prefixe _
