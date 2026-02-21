# -*- coding: utf-8 -*-
"""Tests for the data cleaning module (DataCleaner)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.processing.clean_data import DataCleaner


class TestDataCleanerWeather:
    """Tests for weather data cleaning."""

    def test_clean_weather_basic(self, test_config, sample_weather_df, tmp_path):
        """Test weather cleaning on valid data."""
        # Save the test data
        config = test_config
        raw_dir = tmp_path / "raw" / "weather"
        raw_dir.mkdir(parents=True)
        sample_weather_df.to_csv(raw_dir / "weather_france.csv", index=False)

        # Patch the paths
        config = _patch_config_dirs(config, tmp_path)
        cleaner = DataCleaner(config)
        df = cleaner.clean_weather()

        assert df is not None
        assert len(df) > 0
        assert "year" in df.columns
        assert "month" in df.columns
        assert "date_id" in df.columns
        assert "hdd" in df.columns
        assert "cdd" in df.columns

    def test_clean_weather_removes_duplicates(self, test_config, sample_weather_df, tmp_path):
        """Test that duplicates are removed."""
        # Duplicate some rows
        df_with_dups = pd.concat([sample_weather_df, sample_weather_df.head(5)])

        raw_dir = tmp_path / "raw" / "weather"
        raw_dir.mkdir(parents=True)
        df_with_dups.to_csv(raw_dir / "weather_france.csv", index=False)

        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        df = cleaner.clean_weather()

        assert df is not None
        assert len(df) == len(sample_weather_df)
        assert cleaner.stats["weather"]["duplicates_removed"] == 5

    def test_clean_weather_clips_outliers(self, test_config, sample_weather_df, tmp_path):
        """Test that outlier values are clipped."""
        sample_weather_df.loc[0, "temperature_2m_mean"] = 60.0  # > 50°C
        sample_weather_df.loc[1, "precipitation_sum"] = -10.0    # < 0

        raw_dir = tmp_path / "raw" / "weather"
        raw_dir.mkdir(parents=True)
        sample_weather_df.to_csv(raw_dir / "weather_france.csv", index=False)

        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        df = cleaner.clean_weather()

        assert df is not None
        assert df["temperature_2m_mean"].max() <= 50.0
        assert df["precipitation_sum"].min() >= 0.0

    def test_clean_weather_missing_file(self, test_config, tmp_path):
        """Test that None is returned if the file is missing."""
        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        result = cleaner.clean_weather()
        assert result is None


class TestDataCleanerINSEE:
    """Tests for INSEE data cleaning."""

    def test_clean_insee_basic(self, test_config, sample_insee_df, tmp_path):
        """Test INSEE cleaning on valid data."""
        raw_dir = tmp_path / "raw" / "insee"
        raw_dir.mkdir(parents=True)
        sample_insee_df.to_csv(raw_dir / "indicateurs_economiques.csv", index=False)

        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        df = cleaner.clean_insee()

        assert df is not None
        assert "date_id" in df.columns
        assert len(df) == 12

    def test_clean_insee_filters_non_monthly(self, test_config, tmp_path):
        """Test that non-monthly rows are filtered out."""
        df = pd.DataFrame({
            "period": ["2023-01", "2023-02", "2023Q1", "2023-03"],
            "confiance_menages": [95, 96, 97, 98],
            "climat_affaires_industrie": [100, 101, 102, 103],
        })

        raw_dir = tmp_path / "raw" / "insee"
        raw_dir.mkdir(parents=True)
        df.to_csv(raw_dir / "indicateurs_economiques.csv", index=False)

        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        result = cleaner.clean_insee()

        assert result is not None
        assert len(result) == 3  # "2023Q1" filtered out

    def test_clean_insee_interpolates_gaps(self, test_config, tmp_path):
        """Test that short gaps are interpolated."""
        df = pd.DataFrame({
            "period": [f"2023-{m:02d}" for m in range(1, 7)],
            "confiance_menages": [95, np.nan, np.nan, 98, 99, 100],
            "climat_affaires_industrie": [100, 101, 102, 103, 104, 105],
            "climat_affaires_batiment": [90, 91, 92, 93, 94, 95],
            "opinion_achats_importants": [-10, -9, -8, -7, -6, -5],
        })

        raw_dir = tmp_path / "raw" / "insee"
        raw_dir.mkdir(parents=True)
        df.to_csv(raw_dir / "indicateurs_economiques.csv", index=False)

        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        result = cleaner.clean_insee()

        assert result is not None
        # NaN values should have been interpolated
        assert result["confiance_menages"].isna().sum() == 0


class TestDataCleanerEurostat:
    """Tests for Eurostat data cleaning."""

    def test_clean_eurostat_basic(self, test_config, sample_eurostat_df, tmp_path):
        """Test Eurostat cleaning."""
        raw_dir = tmp_path / "raw" / "eurostat"
        raw_dir.mkdir(parents=True)
        sample_eurostat_df.to_csv(raw_dir / "ipi_hvac_france.csv", index=False)

        config = _patch_config_dirs(test_config, tmp_path)
        cleaner = DataCleaner(config)
        df = cleaner.clean_eurostat()

        assert df is not None
        assert "date_id" in df.columns
        assert "ipi_value" in df.columns


# ============================================================
# Utilities
# ============================================================

def _patch_config_dirs(config, tmp_path):
    """Create a copy of the config with temporary paths."""
    from dataclasses import replace
    return replace(
        config,
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        features_data_dir=tmp_path / "features",
    )
