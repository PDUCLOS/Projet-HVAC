# -*- coding: utf-8 -*-
"""Tests for PAC (heat pump) efficiency features.

Covers:
- COP proxy estimation
- Mountain flag
- PAC viability score
- Altitude/frost interactions
- PAC inefficiency integration in weather collector
- Altitude in reference features
- Security and regression tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import PREFECTURE_ELEVATIONS, ThresholdsConfig
from src.processing.feature_engineering import FeatureEngineer


# =====================================================================
# COP proxy estimation
# =====================================================================

class TestCopProxy:
    """Tests for the COP proxy feature."""

    def test_cop_proxy_created(self, test_config, sample_ml_dataset):
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "cop_proxy" in df.columns

    def test_cop_proxy_range(self, test_config, sample_ml_dataset):
        """COP must be clipped between 1.0 and 5.0."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert df["cop_proxy"].min() >= 1.0
        assert df["cop_proxy"].max() <= 5.0

    def test_cop_lower_with_more_frost(self, test_config):
        """More frost days should produce a lower COP."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69", "69"],
            "date_id": [202301, 202302],
            "nb_jours_gel": [0, 20],
            "altitude": [175, 175],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df.loc[0, "cop_proxy"] > df.loc[1, "cop_proxy"]

    def test_cop_lower_with_higher_altitude(self, test_config):
        """Higher altitude should produce a lower COP."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69", "05"],
            "date_id": [202301, 202301],
            "nb_jours_gel": [5, 5],
            "altitude": [175, 735],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df.loc[0, "cop_proxy"] > df.loc[1, "cop_proxy"]

    def test_cop_without_altitude(self, test_config):
        """COP should still be computed with only frost days."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69"],
            "date_id": [202301],
            "nb_jours_gel": [10],
        })
        df = fe._add_pac_efficiency_features(df)
        assert "cop_proxy" in df.columns
        assert df["cop_proxy"].iloc[0] == pytest.approx(4.5 - 0.08 * 10, abs=0.01)

    def test_cop_without_frost(self, test_config):
        """COP should still be computed with only altitude."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["05"],
            "date_id": [202301],
            "altitude": [735],
        })
        df = fe._add_pac_efficiency_features(df)
        assert "cop_proxy" in df.columns
        assert df["cop_proxy"].iloc[0] == pytest.approx(4.5 - 0.0005 * 735, abs=0.01)


# =====================================================================
# Mountain flag
# =====================================================================

class TestMountainFlag:
    """Tests for the is_mountain feature."""

    def test_mountain_flag_created(self, test_config, sample_ml_dataset):
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "is_mountain" in df.columns

    def test_mountain_flag_binary(self, test_config, sample_ml_dataset):
        """Flag should be 0 or 1 only."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert set(df["is_mountain"].unique()).issubset({0, 1})

    def test_high_altitude_is_mountain(self, test_config):
        """Altitude > 800m should be flagged as mountain."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["05"],
            "date_id": [202301],
            "altitude": [1200],
            "nb_jours_gel": [15],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["is_mountain"].iloc[0] == 1

    def test_low_altitude_not_mountain(self, test_config):
        """Altitude <= 800m should NOT be flagged as mountain."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69"],
            "date_id": [202301],
            "altitude": [175],
            "nb_jours_gel": [5],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["is_mountain"].iloc[0] == 0


# =====================================================================
# PAC viability score
# =====================================================================

class TestPacViabilityScore:
    """Tests for the pac_viability_score composite feature."""

    def test_score_created(self, test_config, sample_ml_dataset):
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "pac_viability_score" in df.columns

    def test_score_range(self, test_config, sample_ml_dataset):
        """Score should be between 0 and 1."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert df["pac_viability_score"].min() >= 0
        assert df["pac_viability_score"].max() <= 1

    def test_warm_lowland_higher_score(self, test_config):
        """Warm lowland with houses should have higher score than cold mountain."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["13", "05"],
            "date_id": [202307, 202301],
            "nb_jours_gel": [0, 25],
            "altitude": [12, 735],
            "pct_maisons": [35, 55],
        })
        df = fe._add_pac_efficiency_features(df)
        score_marseille = df.loc[0, "pac_viability_score"]
        score_gap = df.loc[1, "pac_viability_score"]
        assert score_marseille > score_gap


# =====================================================================
# Interactions
# =====================================================================

class TestInteractions:
    """Tests for altitude/frost interaction features."""

    def test_altitude_frost_created(self, test_config, sample_ml_dataset):
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "interact_altitude_frost" in df.columns

    def test_maisons_altitude_created(self, test_config, sample_ml_dataset):
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "interact_maisons_altitude" in df.columns

    def test_altitude_frost_range(self, test_config, sample_ml_dataset):
        """Normalized interaction should be between 0 and 1."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert df["interact_altitude_frost"].min() >= 0
        assert df["interact_altitude_frost"].max() <= 1

    def test_cop_hdd_interaction(self, test_config, sample_ml_dataset):
        """COP x HDD interaction should be created within PAC features."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "interact_cop_hdd" in df.columns


# =====================================================================
# PAC inefficient days
# =====================================================================

class TestPacInefficient:
    """Tests for PAC inefficiency ratio feature."""

    def test_pct_jours_pac_inefficient(self, test_config, sample_ml_dataset):
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)
        df = fe._add_pac_efficiency_features(df)
        assert "pct_jours_pac_inefficient" in df.columns

    def test_pct_pac_inefficient_zero_days(self, test_config):
        """Zero inefficient days should give 0%."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69"],
            "date_id": [202307],
            "nb_jours_pac_inefficient": [0],
            "nb_jours_gel": [0],
            "altitude": [175],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["pct_jours_pac_inefficient"].iloc[0] == 0.0

    def test_pct_pac_inefficient_all_days(self, test_config):
        """30 inefficient days should give 100%."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["05"],
            "date_id": [202301],
            "nb_jours_pac_inefficient": [30],
            "nb_jours_gel": [25],
            "altitude": [735],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["pct_jours_pac_inefficient"].iloc[0] == 100.0


# =====================================================================
# Full pipeline integration
# =====================================================================

class TestFullPipelineIntegration:
    """Tests for full feature engineering pipeline with PAC features."""

    def test_full_engineer_adds_pac_features(self, test_config, sample_ml_dataset):
        """Full engineer() pipeline should produce PAC features."""
        fe = FeatureEngineer(test_config)
        df = fe.engineer(sample_ml_dataset)
        pac_features = [
            "cop_proxy", "is_mountain", "pac_viability_score",
            "interact_altitude_frost", "interact_maisons_altitude",
            "pct_jours_pac_inefficient", "interact_cop_hdd",
        ]
        for feat in pac_features:
            assert feat in df.columns, f"Missing PAC feature: {feat}"

    def test_pac_features_not_all_nan(self, test_config, sample_ml_dataset):
        """PAC features should not be all NaN."""
        fe = FeatureEngineer(test_config)
        df = fe.engineer(sample_ml_dataset)
        for feat in ["cop_proxy", "pac_viability_score"]:
            assert df[feat].notna().any(), f"{feat} is all NaN"

    def test_feature_count_increased(self, test_config, sample_ml_dataset):
        """Adding PAC features should increase total column count."""
        n_before = len(sample_ml_dataset.columns)
        fe = FeatureEngineer(test_config)
        df = fe.engineer(sample_ml_dataset)
        n_after = len(df.columns)
        # At least 7 PAC features + many others
        assert n_after > n_before + 7


# =====================================================================
# PREFECTURE_ELEVATIONS reference data
# =====================================================================

class TestPrefectureElevations:
    """Tests for the static elevation reference data."""

    def test_has_96_departments(self):
        assert len(PREFECTURE_ELEVATIONS) == 96

    def test_all_values_positive(self):
        for dept, alt in PREFECTURE_ELEVATIONS.items():
            assert alt >= 0, f"Negative altitude for dept {dept}: {alt}"

    def test_known_mountain_departments(self):
        """Known mountain prefectures should have high altitude."""
        # Gap (05), Mende (48), Aurillac (15)
        assert PREFECTURE_ELEVATIONS["05"] > 700   # Gap
        assert PREFECTURE_ELEVATIONS["48"] > 700   # Mende
        assert PREFECTURE_ELEVATIONS["15"] > 600   # Aurillac

    def test_known_lowland_departments(self):
        """Known lowland/coastal prefectures should have low altitude."""
        assert PREFECTURE_ELEVATIONS["75"] < 50    # Paris
        assert PREFECTURE_ELEVATIONS["13"] < 20    # Marseille
        assert PREFECTURE_ELEVATIONS["33"] < 20    # Bordeaux

    def test_corsica_included(self):
        """Corsica departments 2A and 2B should be present."""
        assert "2A" in PREFECTURE_ELEVATIONS
        assert "2B" in PREFECTURE_ELEVATIONS

    def test_all_dept_codes_valid(self):
        """All department codes should match expected format."""
        valid_codes = (
            [f"{i:02d}" for i in range(1, 20)]
            + ["2A", "2B"]
            + [f"{i}" for i in range(21, 96)]
        )
        for dept in PREFECTURE_ELEVATIONS:
            assert dept in valid_codes, f"Unexpected dept code: {dept}"


# =====================================================================
# ThresholdsConfig
# =====================================================================

class TestThresholdsConfig:
    """Tests for the PAC inefficiency threshold in config."""

    def test_pac_threshold_exists(self):
        t = ThresholdsConfig()
        assert hasattr(t, "pac_inefficiency_temp")

    def test_pac_threshold_default(self):
        t = ThresholdsConfig()
        assert t.pac_inefficiency_temp == -7.0

    def test_pac_threshold_negative(self):
        """PAC threshold should be negative (below zero)."""
        t = ThresholdsConfig()
        assert t.pac_inefficiency_temp < 0


# =====================================================================
# ML integration
# =====================================================================

class TestMLIntegration:
    """Tests that PAC features are correctly picked up by ML training."""

    def test_pac_features_not_excluded(self):
        """PAC features must NOT be in EXCLUDE_COLS."""
        from src.models.train import ModelTrainer
        pac_features = [
            "cop_proxy", "is_mountain", "pac_viability_score",
            "interact_altitude_frost", "interact_maisons_altitude",
            "pct_jours_pac_inefficient", "interact_cop_hdd",
            "altitude", "nb_jours_pac_inefficient",
        ]
        for feat in pac_features:
            assert feat not in ModelTrainer.EXCLUDE_COLS, (
                f"{feat} should NOT be excluded from training"
            )

    def test_prepare_features_includes_pac(self, test_config, sample_ml_dataset):
        """prepare_features() should include PAC features."""
        from src.models.train import ModelTrainer

        fe = FeatureEngineer(test_config)
        df = fe.engineer(sample_ml_dataset)

        trainer = ModelTrainer(test_config, target="nb_installations_pac")
        X, y = trainer.prepare_features(df)

        assert "cop_proxy" in X.columns
        assert "is_mountain" in X.columns
        assert "pac_viability_score" in X.columns


# =====================================================================
# Security tests
# =====================================================================

class TestSecurityPac:
    """Security tests for PAC features."""

    def test_nan_in_altitude_handled(self, test_config):
        """NaN altitude should not crash feature computation."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69", "38"],
            "date_id": [202301, 202301],
            "nb_jours_gel": [5, 10],
            "altitude": [175, np.nan],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["cop_proxy"].notna().all()

    def test_negative_altitude_handled(self, test_config):
        """Negative altitude (below sea level) should not crash."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69"],
            "date_id": [202301],
            "nb_jours_gel": [5],
            "altitude": [-10],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["cop_proxy"].iloc[0] >= 1.0
        assert df["cop_proxy"].iloc[0] <= 5.0

    def test_extreme_frost_clipped(self, test_config):
        """50 frost days (impossible but edge case) should produce COP = 1.0."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69"],
            "date_id": [202301],
            "nb_jours_gel": [50],
            "altitude": [0],
        })
        df = fe._add_pac_efficiency_features(df)
        assert df["cop_proxy"].iloc[0] == 1.0  # Clipped to minimum


# =====================================================================
# Regressions
# =====================================================================

class TestRegressionsPac:
    """Regression tests for PAC features."""

    def test_existing_features_unchanged(self, test_config, sample_ml_dataset):
        """Adding PAC features must not modify existing features."""
        fe = FeatureEngineer(test_config)
        df = sample_ml_dataset.sort_values(["dept", "date_id"]).reset_index(drop=True)

        # Save original values
        original_cols = list(df.columns)
        original_values = df[["nb_installations_pac", "temp_mean", "hdd_sum"]].copy()

        df = fe._add_pac_efficiency_features(df)

        # Original columns should still be present and unchanged
        for col in original_cols:
            assert col in df.columns, f"Original column {col} disappeared"

        pd.testing.assert_frame_equal(
            df[["nb_installations_pac", "temp_mean", "hdd_sum"]],
            original_values,
        )

    def test_no_feature_without_data(self, test_config):
        """If no altitude or frost data, no PAC features should be created."""
        fe = FeatureEngineer(test_config)
        df = pd.DataFrame({
            "dept": ["69"],
            "date_id": [202301],
            "nb_installations_pac": [100],
        })
        df = fe._add_pac_efficiency_features(df)
        assert "cop_proxy" not in df.columns
        assert "is_mountain" not in df.columns
