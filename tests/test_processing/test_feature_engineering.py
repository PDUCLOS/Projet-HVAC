# -*- coding: utf-8 -*-
"""Tests pour le module de feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processing.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Tests pour FeatureEngineer."""

    def test_lag_features_created(self, test_config, sample_ml_dataset):
        """Vérifie que les features de lag sont créées."""
        fe = FeatureEngineer(test_config)
        df = fe._add_lag_features(
            sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        )
        assert "nb_installations_pac_lag_1m" in df.columns
        assert "nb_installations_pac_lag_3m" in df.columns
        assert "temp_mean_lag_1m" in df.columns

    def test_lag_nan_at_start(self, test_config, sample_ml_dataset):
        """Vérifie que les lags créent des NaN en début de série."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_lag_features(df)

        # Première ligne de chaque département doit avoir lag_1m = NaN
        for dept in df["dept"].unique():
            dept_data = df[df["dept"] == dept]
            first_row = dept_data.iloc[0]
            assert pd.isna(first_row["nb_installations_pac_lag_1m"])

    def test_rolling_features_created(self, test_config, sample_ml_dataset):
        """Vérifie que les features rolling sont créées."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_rolling_features(df)

        assert "nb_installations_pac_rmean_3m" in df.columns
        assert "nb_installations_pac_rstd_3m" in df.columns

    def test_variation_features(self, test_config, sample_ml_dataset):
        """Vérifie les features de variation."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_variation_features(df)

        assert "nb_installations_pac_diff_1m" in df.columns
        assert "nb_installations_pac_pct_1m" in df.columns
        # Variations extrêmes doivent être clippées
        assert df["nb_installations_pac_pct_1m"].max() <= 500
        assert df["nb_installations_pac_pct_1m"].min() >= -200

    def test_interaction_features(self, test_config, sample_ml_dataset):
        """Vérifie les features d'interaction."""
        fe = FeatureEngineer(test_config)
        df = fe._add_interaction_features(sample_ml_dataset)

        assert "interact_hdd_confiance" in df.columns
        assert "interact_cdd_ipi" in df.columns
        assert "jours_extremes" in df.columns

    def test_trend_features(self, test_config, sample_ml_dataset):
        """Vérifie les features de tendance."""
        fe = FeatureEngineer(test_config)
        df = fe._add_trend_features(sample_ml_dataset)

        assert "year_trend" in df.columns
        assert df["year_trend"].min() == 0

    def test_completeness_flag(self, test_config, sample_ml_dataset):
        """Vérifie le flag de complétude."""
        fe = FeatureEngineer(test_config)
        df = fe._add_completeness_flag(sample_ml_dataset)

        assert "n_valid_features" in df.columns
        assert "pct_valid_features" in df.columns
        assert (df["pct_valid_features"] >= 0).all()
        assert (df["pct_valid_features"] <= 100).all()

    def test_full_engineer_pipeline(self, test_config, sample_ml_dataset, tmp_path):
        """Test complet du pipeline de feature engineering."""
        from dataclasses import replace
        config = replace(test_config, features_data_dir=tmp_path / "features")

        fe = FeatureEngineer(config)
        df = fe.engineer(sample_ml_dataset)

        # On doit avoir significativement plus de colonnes
        assert len(df.columns) > len(sample_ml_dataset.columns)
        # Le nombre de lignes ne doit pas changer
        assert len(df) == len(sample_ml_dataset)


class TestFeatureEngineerEdgeCases:
    """Tests des cas limites."""

    def test_missing_optional_columns(self, test_config):
        """Le FE doit fonctionner même sans colonnes optionnelles."""
        df = pd.DataFrame({
            "date_id": [202301, 202302, 202303],
            "dept": ["69", "69", "69"],
            "nb_dpe_total": [100, 150, 200],
            "nb_installations_pac": [10, 15, 20],
            "nb_installations_clim": [5, 8, 10],
            "year": [2023, 2023, 2023],
            "month": [1, 2, 3],
        })
        fe = FeatureEngineer(test_config)
        result = fe._add_interaction_features(df)
        # Pas d'erreur, les interactions non calculables sont ignorées
        assert len(result) == 3

    def test_single_department(self, test_config):
        """Le FE doit fonctionner avec un seul département."""
        np.random.seed(42)
        df = pd.DataFrame({
            "date_id": [202301 + i for i in range(12)],
            "dept": ["69"] * 12,
            "nb_dpe_total": np.random.randint(100, 500, 12),
            "nb_installations_pac": np.random.randint(10, 80, 12),
            "nb_installations_clim": np.random.randint(5, 30, 12),
            "temp_mean": np.random.uniform(-2, 25, 12),
            "hdd_sum": np.random.uniform(0, 500, 12),
            "cdd_sum": np.random.uniform(0, 200, 12),
            "year": [2023] * 12,
            "month": list(range(1, 13)),
        })
        fe = FeatureEngineer(test_config)
        df_sorted = df.sort_values(["dept", "date_id"]).reset_index(drop=True)
        result = fe._add_lag_features(df_sorted)
        assert "nb_installations_pac_lag_1m" in result.columns
