# -*- coding: utf-8 -*-
"""Shared fixtures for the HVAC project tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    DatabaseConfig,
    GeoConfig,
    ModelConfig,
    NetworkConfig,
    ProjectConfig,
    TimeConfig,
)
from src.collectors.base import CollectorConfig


@pytest.fixture
def test_config() -> ProjectConfig:
    """Test configuration with temporary paths."""
    return ProjectConfig(
        geo=GeoConfig(
            region_code="84",
            departments=["69", "38"],
            cities={
                "Lyon": {"lat": 45.76, "lon": 4.84, "dept": "69"},
                "Grenoble": {"lat": 45.19, "lon": 5.72, "dept": "38"},
            },
        ),
        time=TimeConfig(
            start_date="2022-01-01",
            end_date="2023-12-31",
            dpe_start_date="2022-01-01",
            train_end="2023-06-30",
            val_end="2023-09-30",
        ),
        network=NetworkConfig(
            request_timeout=5,
            max_retries=1,
            retry_backoff_factor=0.1,
            rate_limit_delay=0.0,
        ),
        database=DatabaseConfig(db_type="sqlite", db_path=":memory:"),
        model=ModelConfig(max_lag_months=3, rolling_windows=[3]),
        raw_data_dir=Path("/tmp/hvac_test/raw"),
        processed_data_dir=Path("/tmp/hvac_test/processed"),
        features_data_dir=Path("/tmp/hvac_test/features"),
    )


@pytest.fixture
def collector_config() -> CollectorConfig:
    """Collector configuration for tests."""
    return CollectorConfig(
        raw_data_dir=Path("/tmp/hvac_test/raw"),
        processed_data_dir=Path("/tmp/hvac_test/processed"),
        start_date="2023-01-01",
        end_date="2023-12-31",
        departments=["69", "38"],
        region_code="84",
        request_timeout=5,
        max_retries=1,
        retry_backoff_factor=0.1,
        rate_limit_delay=0.0,
    )


@pytest.fixture
def sample_weather_df() -> pd.DataFrame:
    """Test weather DataFrame (2 cities, 30 days)."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    rows = []
    for city, dept in [("Lyon", "69"), ("Grenoble", "38")]:
        for d in dates:
            rows.append({
                "time": d,
                "city": city,
                "dept": dept,
                "temperature_2m_max": np.random.uniform(0, 20),
                "temperature_2m_min": np.random.uniform(-5, 10),
                "temperature_2m_mean": np.random.uniform(-2, 15),
                "precipitation_sum": np.random.uniform(0, 20),
                "wind_speed_10m_max": np.random.uniform(5, 50),
                "hdd": max(0, 18 - np.random.uniform(-2, 15)),
                "cdd": max(0, np.random.uniform(-2, 15) - 18),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_insee_df() -> pd.DataFrame:
    """Test INSEE DataFrame (12 months)."""
    periods = [f"2023-{m:02d}" for m in range(1, 13)]
    return pd.DataFrame({
        "period": periods,
        "confiance_menages": np.random.uniform(80, 110, 12),
        "climat_affaires_industrie": np.random.uniform(90, 110, 12),
        "climat_affaires_batiment": np.random.uniform(85, 105, 12),
        "opinion_achats_importants": np.random.uniform(-30, 10, 12),
        "situation_financiere_future": np.random.uniform(-20, 20, 12),
        "ipi_industrie_manuf": np.random.uniform(95, 110, 12),
    })


@pytest.fixture
def sample_eurostat_df() -> pd.DataFrame:
    """Test Eurostat DataFrame (2 NACE codes, 12 months)."""
    periods = [f"2023-{m:02d}" for m in range(1, 13)]
    rows = []
    for nace in ["C28", "C2825"]:
        for p in periods:
            rows.append({
                "period": p,
                "nace_r2": nace,
                "ipi_value": np.random.uniform(80, 120),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_ml_dataset() -> pd.DataFrame:
    """Minimal ML dataset for testing feature engineering."""
    np.random.seed(42)
    rows = []
    for dept in ["69", "38"]:
        for year in [2022, 2023]:
            for month in range(1, 13):
                date_id = year * 100 + month
                rows.append({
                    "date_id": date_id,
                    "dept": dept,
                    "nb_dpe_total": np.random.randint(100, 500),
                    "nb_installations_pac": np.random.randint(10, 80),
                    "nb_installations_clim": np.random.randint(5, 30),
                    "nb_dpe_classe_ab": np.random.randint(10, 100),
                    "pct_pac": np.random.uniform(5, 20),
                    "pct_clim": np.random.uniform(2, 10),
                    "temp_mean": np.random.uniform(-2, 25),
                    "hdd_sum": np.random.uniform(0, 500),
                    "cdd_sum": np.random.uniform(0, 200),
                    "precipitation_sum": np.random.uniform(20, 150),
                    "confiance_menages": np.random.uniform(80, 110),
                    "climat_affaires_bat": np.random.uniform(85, 105),
                    "ipi_hvac_c28": np.random.uniform(90, 115),
                    "year": year,
                    "month": month,
                    "quarter": (month - 1) // 3 + 1,
                    "is_heating": int(month in [1, 2, 3, 10, 11, 12]),
                    "is_cooling": int(month in [6, 7, 8, 9]),
                    "month_sin": np.sin(2 * np.pi * month / 12),
                    "month_cos": np.cos(2 * np.pi * month / 12),
                    "nb_jours_canicule": np.random.randint(0, 10),
                    "nb_jours_gel": np.random.randint(0, 15),
                })
    return pd.DataFrame(rows)
