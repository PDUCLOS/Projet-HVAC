# -*- coding: utf-8 -*-
"""Comprehensive tests for the multi-source merging module (DatasetMerger).

Tests cover all public and private methods:
    - __init__: instantiation with config
    - _prepare_weather_features: daily -> monthly aggregation
    - _prepare_economic_features: INSEE + Eurostat merge
    - _prepare_sitadel_features: SITADEL aggregation
    - _prepare_reference_features: static department data
    - _add_time_features: temporal enrichment (year, month, quarter, seasons, cyclic)
    - _add_geo_features: geographic metadata (dept_name, city_ref, lat/lon)
    - _dept_name: static method for department name lookup
    - build_ml_dataset: end-to-end pipeline with mock data
    - _save_ml_dataset: CSV persistence
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.settings import (
    GeoConfig,
    ProjectConfig,
    ThresholdsConfig,
    TimeConfig,
)
from src.processing.merge_datasets import DatasetMerger


# ============================================================
# Helpers
# ============================================================

def _make_test_config(tmp_path: Path, **overrides) -> ProjectConfig:
    """Build a ProjectConfig pointing at tmp_path directories.

    The config uses a small geo scope (2 departments) and a start date
    of 2022-07-01 so that test data from 2022-07+ is retained after
    the DPE v2 date filter.

    Args:
        tmp_path: Temporary directory for test data.
        **overrides: Additional keyword arguments forwarded to ProjectConfig.

    Returns:
        A ProjectConfig suitable for unit testing.
    """
    base = ProjectConfig(
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
            dpe_start_date="2022-07-01",
            train_end="2023-06-30",
            val_end="2023-09-30",
        ),
        thresholds=ThresholdsConfig(
            heatwave_temp=35.0,
            frost_temp=0.0,
        ),
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        features_data_dir=tmp_path / "features",
    )
    if overrides:
        base = replace(base, **overrides)
    return base


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Ensure parent dirs exist and write a DataFrame to CSV.

    Args:
        path: Target file path.
        df: DataFrame to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _make_dpe_csv(depts: list[str] = None, months: list[str] = None) -> pd.DataFrame:
    """Create a minimal DPE CSV with pre-computed columns.

    Args:
        depts: Department codes. Defaults to ["69", "38"].
        months: Months as YYYY-MM strings. Defaults to 2022-07..2022-12.

    Returns:
        DataFrame mimicking a cleaned DPE file.
    """
    depts = depts or ["69", "38"]
    months = months or [f"2022-{m:02d}" for m in range(7, 13)]
    rows = []
    for dept in depts:
        for m in months:
            year, month = m.split("-")
            date_id = int(year) * 100 + int(month)
            rows.append({
                "date_id": date_id,
                "code_departement_ban": dept,
                "is_pac": 1,
                "is_clim": 0,
                "is_classe_ab": 1,
            })
            # Add a second row per group so aggregation is non-trivial
            rows.append({
                "date_id": date_id,
                "code_departement_ban": dept,
                "is_pac": 0,
                "is_clim": 1,
                "is_classe_ab": 0,
            })
    return pd.DataFrame(rows)


def _make_weather_csv(
    depts: list[str] = None,
    n_days: int = 60,
    start_date: str = "2022-07-01",
) -> pd.DataFrame:
    """Create a minimal daily weather CSV.

    Args:
        depts: Department codes. Defaults to ["69", "38"].
        n_days: Number of daily records per department.
        start_date: First date.

    Returns:
        DataFrame mimicking a weather CSV.
    """
    depts = depts or ["69", "38"]
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    rows = []
    np.random.seed(42)
    for dept in depts:
        for d in dates:
            t_max = np.random.uniform(15, 40)
            t_min = np.random.uniform(-5, 10)
            t_mean = (t_max + t_min) / 2
            rows.append({
                "time": d.strftime("%Y-%m-%d"),
                "dept": dept,
                "temperature_2m_max": round(t_max, 2),
                "temperature_2m_min": round(t_min, 2),
                "temperature_2m_mean": round(t_mean, 2),
                "precipitation_sum": round(np.random.uniform(0, 20), 2),
                "wind_speed_10m_max": round(np.random.uniform(5, 40), 2),
                "hdd": round(max(0, 18 - t_mean), 2),
                "cdd": round(max(0, t_mean - 18), 2),
            })
    return pd.DataFrame(rows)


def _make_insee_csv(months: list[str] = None) -> pd.DataFrame:
    """Create a minimal INSEE indicators CSV.

    Args:
        months: Month strings in YYYY-MM format.

    Returns:
        DataFrame mimicking an INSEE indicators file.
    """
    months = months or [f"2022-{m:02d}" for m in range(7, 13)]
    return pd.DataFrame({
        "period": months,
        "confiance_menages": np.linspace(95, 100, len(months)),
        "climat_affaires_industrie": np.linspace(100, 105, len(months)),
        "climat_affaires_batiment": np.linspace(90, 95, len(months)),
        "opinion_achats_importants": np.linspace(-10, -5, len(months)),
        "situation_financiere_future": np.linspace(-5, 0, len(months)),
        "ipi_industrie_manuf": np.linspace(100, 106, len(months)),
    })


def _make_eurostat_csv(months: list[str] = None) -> pd.DataFrame:
    """Create a minimal Eurostat IPI CSV.

    Args:
        months: Month strings in YYYY-MM format.

    Returns:
        DataFrame mimicking an Eurostat IPI file.
    """
    months = months or [f"2022-{m:02d}" for m in range(7, 13)]
    rows = []
    for nace in ["C28", "C2825"]:
        for m in months:
            rows.append({
                "period": m,
                "nace_r2": nace,
                "ipi_value": round(np.random.uniform(90, 115), 2),
            })
    return pd.DataFrame(rows)


def _make_sitadel_csv(
    depts: list[str] = None,
    months: list[str] = None,
) -> pd.DataFrame:
    """Create a minimal SITADEL construction permits CSV.

    Args:
        depts: Department codes. Defaults to ["69", "38"].
        months: Months in YYYY-MM format.

    Returns:
        DataFrame mimicking a SITADEL file.
    """
    depts = depts or ["69", "38"]
    months = months or [f"2022-{m:02d}" for m in range(7, 13)]
    rows = []
    for dept in depts:
        for m in months:
            rows.append({
                "DATE_PRISE_EN_COMPTE": m,
                "DEP": dept,
                "NB_LGT_TOT_CREES": np.random.randint(50, 200),
                "NB_LGT_IND_CREES": np.random.randint(20, 100),
                "NB_LGT_COLL_CREES": np.random.randint(10, 80),
                "SURF_TOT_M2": np.random.randint(5000, 30000),
            })
    return pd.DataFrame(rows)


def _make_reference_csv(depts: list[str] = None) -> pd.DataFrame:
    """Create a minimal department reference CSV.

    Args:
        depts: Department codes. Defaults to ["69", "38"].

    Returns:
        DataFrame mimicking an INSEE reference file.
    """
    depts = depts or ["69", "38"]
    return pd.DataFrame({
        "dept": depts,
        "revenu_median": [23500, 22000],
        "prix_m2_median": [3200, 2800],
        "nb_logements_total": [800000, 550000],
        "pct_maisons": [35.2, 42.8],
    })


# ============================================================
# Tests — Initialization
# ============================================================

class TestDatasetMergerInit:
    """Tests for DatasetMerger.__init__."""

    def test_init_stores_config(self, tmp_path):
        """Merger stores the config reference."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        assert merger.config is cfg

    def test_init_creates_logger(self, tmp_path):
        """Merger creates a named logger."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        assert merger.logger.name == "processing.merge"

    def test_init_with_custom_thresholds(self, tmp_path):
        """Merger respects custom threshold values."""
        cfg = _make_test_config(
            tmp_path,
            thresholds=ThresholdsConfig(heatwave_temp=40.0, frost_temp=-5.0),
        )
        merger = DatasetMerger(cfg)
        assert merger.config.thresholds.heatwave_temp == 40.0
        assert merger.config.thresholds.frost_temp == -5.0


# ============================================================
# Tests — _prepare_weather_features
# ============================================================

class TestPrepareWeatherFeatures:
    """Tests for the daily-to-monthly weather aggregation."""

    def test_basic_aggregation(self, tmp_path):
        """Weather data is correctly aggregated from daily to monthly."""
        cfg = _make_test_config(tmp_path)
        weather_dir = tmp_path / "processed" / "weather"
        _write_csv(weather_dir / "weather_france.csv", _make_weather_csv())

        merger = DatasetMerger(cfg)
        df = merger._prepare_weather_features()

        assert df is not None
        assert "date_id" in df.columns
        assert "dept" in df.columns
        assert "temp_mean" in df.columns
        assert "temp_max" in df.columns
        assert "temp_min" in df.columns
        assert "precipitation_sum" in df.columns
        assert "wind_max" in df.columns
        assert "hdd_sum" in df.columns
        assert "cdd_sum" in df.columns
        # 2 depts, covering July and August = at least 2 months x 2 depts
        assert len(df) >= 4

    def test_heatwave_threshold_from_config(self, tmp_path):
        """Days above heatwave_temp are counted via nb_jours_canicule."""
        # Create weather data with known extreme temperatures
        df_weather = pd.DataFrame({
            "time": ["2022-07-01", "2022-07-02", "2022-07-03", "2022-07-04"],
            "dept": ["69", "69", "69", "69"],
            "temperature_2m_max": [36.0, 34.0, 38.0, 30.0],
            "temperature_2m_min": [20.0, 18.0, 22.0, 15.0],
            "temperature_2m_mean": [28.0, 26.0, 30.0, 22.5],
            "precipitation_sum": [0, 0, 0, 5],
            "wind_speed_10m_max": [10, 12, 8, 15],
            "hdd": [0, 0, 0, 0],
            "cdd": [10, 8, 12, 4.5],
        })
        cfg = _make_test_config(
            tmp_path,
            thresholds=ThresholdsConfig(heatwave_temp=35.0, frost_temp=0.0),
        )
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            df_weather,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()

        assert result is not None
        assert "nb_jours_canicule" in result.columns
        # 36 > 35 and 38 > 35 -> 2 heatwave days
        row = result[result["dept"] == "69"].iloc[0]
        assert row["nb_jours_canicule"] == 2

    def test_frost_threshold_from_config(self, tmp_path):
        """Days below frost_temp are counted via nb_jours_gel."""
        df_weather = pd.DataFrame({
            "time": ["2022-12-01", "2022-12-02", "2022-12-03"],
            "dept": ["38", "38", "38"],
            "temperature_2m_max": [5.0, 3.0, 8.0],
            "temperature_2m_min": [-2.0, -1.0, 2.0],
            "temperature_2m_mean": [1.5, 1.0, 5.0],
            "precipitation_sum": [10, 5, 0],
            "wind_speed_10m_max": [20, 25, 10],
            "hdd": [16.5, 17.0, 13.0],
            "cdd": [0, 0, 0],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            df_weather,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()

        assert result is not None
        assert "nb_jours_gel" in result.columns
        row = result[result["dept"] == "38"].iloc[0]
        # -2 < 0 and -1 < 0 -> 2 frost days; 2 >= 0 -> not frost
        assert row["nb_jours_gel"] == 2

    def test_custom_heatwave_threshold(self, tmp_path):
        """A higher heatwave threshold changes the count."""
        df_weather = pd.DataFrame({
            "time": ["2022-07-01", "2022-07-02", "2022-07-03"],
            "dept": ["69", "69", "69"],
            "temperature_2m_max": [36.0, 39.0, 41.0],
            "temperature_2m_min": [20.0, 22.0, 24.0],
            "temperature_2m_mean": [28.0, 30.5, 32.5],
            "precipitation_sum": [0, 0, 0],
            "wind_speed_10m_max": [10, 8, 12],
            "hdd": [0, 0, 0],
            "cdd": [10, 12.5, 14.5],
        })
        # Set heatwave threshold to 40 -> only 41 counts
        cfg = _make_test_config(
            tmp_path,
            thresholds=ThresholdsConfig(heatwave_temp=40.0, frost_temp=0.0),
        )
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            df_weather,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()

        row = result[result["dept"] == "69"].iloc[0]
        assert row["nb_jours_canicule"] == 1  # Only 41 > 40

    def test_missing_weather_file_returns_none(self, tmp_path):
        """Returns None when neither processed nor raw weather file exists."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        assert result is None

    def test_falls_back_to_raw(self, tmp_path):
        """Falls back to raw weather file if processed is missing."""
        cfg = _make_test_config(tmp_path)
        # Write only to raw, not processed
        _write_csv(
            tmp_path / "raw" / "weather" / "weather_france.csv",
            _make_weather_csv(),
        )
        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        assert result is not None
        assert len(result) > 0

    def test_date_column_fallback(self, tmp_path):
        """Handles 'date' column when 'time' is absent."""
        df_weather = _make_weather_csv()
        df_weather = df_weather.rename(columns={"time": "date"})

        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            df_weather,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        assert result is not None
        assert "date_id" in result.columns

    def test_dept_zero_padded(self, tmp_path):
        """Department codes are zero-padded to 2 chars."""
        df_weather = pd.DataFrame({
            "time": ["2022-07-01"],
            "dept": [1],  # integer, not zero-padded
            "temperature_2m_max": [30.0],
            "temperature_2m_min": [15.0],
            "temperature_2m_mean": [22.5],
            "precipitation_sum": [5.0],
            "wind_speed_10m_max": [12.0],
            "hdd": [0.0],
            "cdd": [4.5],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            df_weather,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        assert result is not None
        assert result["dept"].iloc[0] == "01"

    def test_missing_dept_column_returns_none(self, tmp_path):
        """Returns None if the 'dept' column is missing from weather data."""
        df_weather = pd.DataFrame({
            "time": ["2022-07-01"],
            "temperature_2m_max": [30.0],
            "temperature_2m_min": [15.0],
            "temperature_2m_mean": [22.5],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            df_weather,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        assert result is None

    def test_floats_are_rounded(self, tmp_path):
        """Float columns in the output are rounded to 2 decimals."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            _make_weather_csv(n_days=5),
        )
        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        assert result is not None
        float_cols = result.select_dtypes(include=["float64"]).columns
        for col in float_cols:
            values = result[col].dropna()
            # Check that values have at most 2 decimal places
            rounded = values.round(2)
            assert (values == rounded).all(), f"Column {col} not rounded to 2 decimals"


# ============================================================
# Tests — _prepare_economic_features
# ============================================================

class TestPrepareEconomicFeatures:
    """Tests for the INSEE + Eurostat economic feature merge."""

    def test_insee_only(self, tmp_path):
        """Returns economic data from INSEE alone if Eurostat is missing."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            _make_insee_csv(),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()

        assert result is not None
        assert "date_id" in result.columns
        assert "confiance_menages" in result.columns
        assert "ipi_hvac_c28" not in result.columns  # Eurostat missing

    def test_eurostat_only(self, tmp_path):
        """Returns economic data from Eurostat alone if INSEE is missing."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "eurostat" / "ipi_hvac_france.csv",
            _make_eurostat_csv(),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()

        assert result is not None
        assert "date_id" in result.columns
        assert "ipi_hvac_c28" in result.columns
        assert "ipi_hvac_c2825" in result.columns

    def test_combined_insee_eurostat(self, tmp_path):
        """Merge produces both INSEE and Eurostat columns."""
        cfg = _make_test_config(tmp_path)
        months = [f"2022-{m:02d}" for m in range(7, 13)]
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            _make_insee_csv(months),
        )
        _write_csv(
            tmp_path / "processed" / "eurostat" / "ipi_hvac_france.csv",
            _make_eurostat_csv(months),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()

        assert result is not None
        assert "confiance_menages" in result.columns
        assert "ipi_hvac_c28" in result.columns
        assert "ipi_hvac_c2825" in result.columns
        # Dates should cover the same months
        assert set(result["date_id"]) == {202207, 202208, 202209, 202210, 202211, 202212}

    def test_filters_non_monthly_periods(self, tmp_path):
        """Non-monthly period strings (e.g., quarterly) are filtered out."""
        cfg = _make_test_config(tmp_path)
        df_insee = pd.DataFrame({
            "period": ["2022-07", "2022-08", "2022Q3", "2022"],
            "confiance_menages": [95, 96, 97, 98],
            "climat_affaires_industrie": [100, 101, 102, 103],
        })
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            df_insee,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()

        assert result is not None
        # Only 2022-07 and 2022-08 should pass the monthly filter
        assert len(result) == 2

    def test_column_renaming(self, tmp_path):
        """INSEE columns are renamed according to the internal mapping."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            _make_insee_csv(),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()

        assert result is not None
        # Check renamed columns
        assert "climat_affaires_indus" in result.columns
        assert "climat_affaires_bat" in result.columns
        assert "opinion_achats" in result.columns
        assert "situation_fin_future" in result.columns
        assert "ipi_manufacturing" in result.columns

    def test_no_files_returns_none(self, tmp_path):
        """Returns None when neither INSEE nor Eurostat files exist."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()
        assert result is None

    def test_sorted_by_date_id(self, tmp_path):
        """Result is sorted by date_id."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            _make_insee_csv(),
        )
        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()
        assert result is not None
        assert list(result["date_id"]) == sorted(result["date_id"].tolist())

    def test_falls_back_to_raw_files(self, tmp_path):
        """Falls back to raw data dirs when processed files are missing."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "insee" / "indicateurs_economiques.csv",
            _make_insee_csv(),
        )
        _write_csv(
            tmp_path / "raw" / "eurostat" / "ipi_hvac_france.csv",
            _make_eurostat_csv(),
        )
        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()
        assert result is not None
        assert "confiance_menages" in result.columns
        assert "ipi_hvac_c28" in result.columns


# ============================================================
# Tests — _prepare_sitadel_features
# ============================================================

class TestPrepareSitadelFeatures:
    """Tests for SITADEL construction permits aggregation."""

    def test_basic_aggregation(self, tmp_path):
        """SITADEL data is aggregated per month x department."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            _make_sitadel_csv(),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()

        assert result is not None
        assert "date_id" in result.columns
        assert "dept" in result.columns
        assert "nb_logements_autorises" in result.columns
        assert "nb_logements_individuels" in result.columns
        assert "nb_logements_collectifs" in result.columns
        assert "surface_autorisee_m2" in result.columns
        # 2 depts x 6 months = 12 rows
        assert len(result) == 12

    def test_dept_zero_padded(self, tmp_path):
        """Department codes are zero-padded."""
        df_sitadel = pd.DataFrame({
            "DATE_PRISE_EN_COMPTE": ["2022-07"],
            "DEP": [1],
            "NB_LGT_TOT_CREES": [100],
            "NB_LGT_IND_CREES": [50],
            "NB_LGT_COLL_CREES": [30],
            "SURF_TOT_M2": [10000],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is not None
        assert result["dept"].iloc[0] == "01"

    def test_aggregation_sums_multiple_rows(self, tmp_path):
        """Multiple rows for the same (month, dept) are summed."""
        df_sitadel = pd.DataFrame({
            "DATE_PRISE_EN_COMPTE": ["2022-07", "2022-07"],
            "DEP": ["69", "69"],
            "NB_LGT_TOT_CREES": [100, 150],
            "NB_LGT_IND_CREES": [40, 60],
            "NB_LGT_COLL_CREES": [30, 50],
            "SURF_TOT_M2": [8000, 12000],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is not None
        assert len(result) == 1
        assert result["nb_logements_autorises"].iloc[0] == 250
        assert result["nb_logements_individuels"].iloc[0] == 100
        assert result["surface_autorisee_m2"].iloc[0] == 20000

    def test_missing_file_returns_none(self, tmp_path):
        """Returns None when the SITADEL file is missing."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is None

    def test_missing_date_column_returns_none(self, tmp_path):
        """Returns None if DATE_PRISE_EN_COMPTE is missing."""
        df_sitadel = pd.DataFrame({
            "DEP": ["69"],
            "NB_LGT_TOT_CREES": [100],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is None

    def test_missing_dep_column_returns_none(self, tmp_path):
        """Returns None if DEP column is missing."""
        df_sitadel = pd.DataFrame({
            "DATE_PRISE_EN_COMPTE": ["2022-07"],
            "NB_LGT_TOT_CREES": [100],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is None

    def test_partial_columns_handled(self, tmp_path):
        """Only available SITADEL columns are included in output."""
        df_sitadel = pd.DataFrame({
            "DATE_PRISE_EN_COMPTE": ["2022-07"],
            "DEP": ["69"],
            "NB_LGT_TOT_CREES": [100],
            # Missing NB_LGT_IND_CREES, NB_LGT_COLL_CREES, SURF_TOT_M2
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is not None
        assert "nb_logements_autorises" in result.columns
        assert "nb_logements_individuels" not in result.columns

    def test_yyyy_mm_dd_date_format(self, tmp_path):
        """YYYY-MM-DD dates (actual CSV format) are parsed correctly."""
        df_sitadel = pd.DataFrame({
            "DATE_PRISE_EN_COMPTE": [
                "2022-07-15", "2022-07-20", "2022-08-10",
            ],
            "DEP": ["69", "69", "69"],
            "NB_LGT_TOT_CREES": [50, 100, 75],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is not None
        # Two different months: 202207 and 202208
        assert set(result["date_id"].values) == {202207, 202208}
        # July rows should be summed
        july_row = result[result["date_id"] == 202207]
        assert july_row["nb_logements_autorises"].iloc[0] == 150

    def test_mixed_date_formats(self, tmp_path):
        """Handles mix of YYYY-MM and YYYY-MM-DD date formats."""
        df_sitadel = pd.DataFrame({
            "DATE_PRISE_EN_COMPTE": ["2022-07", "2022-08-15"],
            "DEP": ["69", "69"],
            "NB_LGT_TOT_CREES": [50, 75],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            df_sitadel,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_sitadel_features()
        assert result is not None
        assert set(result["date_id"].values) == {202207, 202208}


# ============================================================
# Tests — _prepare_reference_features
# ============================================================

class TestPrepareReferenceFeatures:
    """Tests for department-level reference data loading."""

    def test_basic_loading(self, tmp_path):
        """Reference features are loaded with correct columns."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "insee" / "reference_departements.csv",
            _make_reference_csv(),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_reference_features()

        assert result is not None
        assert "dept" in result.columns
        assert "revenu_median" in result.columns
        assert "prix_m2_median" in result.columns
        assert "nb_logements_total" in result.columns
        assert "pct_maisons" in result.columns
        assert len(result) == 2

    def test_dept_zero_padded(self, tmp_path):
        """Department codes are zero-padded."""
        df_ref = pd.DataFrame({
            "dept": [1, 7],
            "revenu_median": [21000, 22000],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "insee" / "reference_departements.csv",
            df_ref,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_reference_features()
        assert result is not None
        assert list(result["dept"]) == ["01", "07"]

    def test_missing_file_returns_none(self, tmp_path):
        """Returns None when the reference file is missing."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._prepare_reference_features()
        assert result is None

    def test_partial_columns(self, tmp_path):
        """Only available reference columns are kept."""
        df_ref = pd.DataFrame({
            "dept": ["69", "38"],
            "revenu_median": [23500, 22000],
            # Missing prix_m2_median, nb_logements_total, pct_maisons
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "insee" / "reference_departements.csv",
            df_ref,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_reference_features()
        assert result is not None
        assert "revenu_median" in result.columns
        assert "prix_m2_median" not in result.columns

    def test_extra_columns_are_dropped(self, tmp_path):
        """Columns beyond the expected set are not included."""
        df_ref = pd.DataFrame({
            "dept": ["69"],
            "revenu_median": [23500],
            "dept_name": ["Rhone"],
            "extra_column": [999],
        })
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "raw" / "insee" / "reference_departements.csv",
            df_ref,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_reference_features()
        assert result is not None
        assert "dept_name" not in result.columns
        assert "extra_column" not in result.columns


# ============================================================
# Tests — _add_time_features
# ============================================================

class TestAddTimeFeatures:
    """Tests for temporal feature enrichment."""

    @pytest.fixture
    def base_df(self) -> pd.DataFrame:
        """Minimal DataFrame with date_id for time feature tests."""
        return pd.DataFrame({
            "date_id": [202201, 202206, 202207, 202210, 202212],
            "dept": ["69", "69", "69", "69", "69"],
        })

    def test_year_month_quarter(self, tmp_path, base_df):
        """Year, month, and quarter are correctly extracted."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._add_time_features(base_df.copy())

        assert list(result["year"]) == [2022, 2022, 2022, 2022, 2022]
        assert list(result["month"]) == [1, 6, 7, 10, 12]
        assert list(result["quarter"]) == [1, 2, 3, 4, 4]

    def test_is_heating_season(self, tmp_path, base_df):
        """Heating season flag: Oct-Mar = 1, else 0."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._add_time_features(base_df.copy())

        # Month 1 (Jan)=heating, 6 (Jun)=not, 7 (Jul)=not, 10 (Oct)=heating, 12 (Dec)=heating
        assert list(result["is_heating"]) == [1, 0, 0, 1, 1]

    def test_is_cooling_season(self, tmp_path, base_df):
        """Cooling season flag: Jun-Sep = 1, else 0."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._add_time_features(base_df.copy())

        # Month 1=not, 6=cooling, 7=cooling, 10=not, 12=not
        assert list(result["is_cooling"]) == [0, 1, 1, 0, 0]

    def test_cyclic_encoding(self, tmp_path, base_df):
        """Month sin/cos encode the cyclical nature of months."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._add_time_features(base_df.copy())

        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

        # Verify the formula: sin(2*pi*month/12)
        for _, row in result.iterrows():
            expected_sin = round(np.sin(2 * np.pi * row["month"] / 12), 4)
            expected_cos = round(np.cos(2 * np.pi * row["month"] / 12), 4)
            assert row["month_sin"] == expected_sin
            assert row["month_cos"] == expected_cos

    def test_april_may_neither_heating_nor_cooling(self, tmp_path):
        """April and May are in neither heating nor cooling season."""
        df = pd.DataFrame({
            "date_id": [202204, 202205],
            "dept": ["69", "69"],
        })
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._add_time_features(df)

        assert list(result["is_heating"]) == [0, 0]
        assert list(result["is_cooling"]) == [0, 0]

    def test_all_12_months(self, tmp_path):
        """All 12 months produce valid quarter and season values."""
        df = pd.DataFrame({
            "date_id": [202200 + m for m in range(1, 13)],
            "dept": ["69"] * 12,
        })
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._add_time_features(df)

        assert list(result["quarter"]) == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
        # Heating: Jan Feb Mar Oct Nov Dec -> months 1,2,3,10,11,12
        expected_heating = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        assert list(result["is_heating"]) == expected_heating
        # Cooling: Jun Jul Aug Sep -> months 6,7,8,9
        expected_cooling = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        assert list(result["is_cooling"]) == expected_cooling


# ============================================================
# Tests — _add_geo_features
# ============================================================

class TestAddGeoFeatures:
    """Tests for geographic metadata enrichment."""

    def test_adds_geo_columns(self, tmp_path):
        """Geo features (dept_name, city_ref, lat, lon) are added."""
        cfg = _make_test_config(tmp_path)
        df = pd.DataFrame({
            "date_id": [202207, 202207],
            "dept": ["69", "38"],
        })

        merger = DatasetMerger(cfg)
        result = merger._add_geo_features(df)

        assert "dept_name" in result.columns
        assert "city_ref" in result.columns
        assert "latitude" in result.columns
        assert "longitude" in result.columns

    def test_lyon_metadata(self, tmp_path):
        """Lyon (dept 69) gets correct geographic metadata."""
        cfg = _make_test_config(tmp_path)
        df = pd.DataFrame({"date_id": [202207], "dept": ["69"]})

        merger = DatasetMerger(cfg)
        result = merger._add_geo_features(df)

        row = result.iloc[0]
        assert row["city_ref"] == "Lyon"
        assert row["latitude"] == 45.76
        assert row["longitude"] == 4.84
        assert row["dept_name"] == "Rhone"

    def test_grenoble_metadata(self, tmp_path):
        """Grenoble (dept 38) gets correct geographic metadata."""
        cfg = _make_test_config(tmp_path)
        df = pd.DataFrame({"date_id": [202207], "dept": ["38"]})

        merger = DatasetMerger(cfg)
        result = merger._add_geo_features(df)

        row = result.iloc[0]
        assert row["city_ref"] == "Grenoble"
        assert row["latitude"] == 45.19
        assert row["longitude"] == 5.72
        assert row["dept_name"] == "Isere"

    def test_unknown_dept_gets_nan(self, tmp_path):
        """A department not in config.geo.cities produces NaN geo values."""
        cfg = _make_test_config(tmp_path)
        df = pd.DataFrame({"date_id": [202207], "dept": ["75"]})

        merger = DatasetMerger(cfg)
        result = merger._add_geo_features(df)

        row = result.iloc[0]
        assert pd.isna(row["city_ref"])
        assert pd.isna(row["latitude"])


# ============================================================
# Tests — _dept_name (static method)
# ============================================================

class TestDeptName:
    """Tests for the static department name lookup."""

    def test_known_department(self):
        """Known department codes return correct names."""
        assert DatasetMerger._dept_name("69") == "Rhone"
        assert DatasetMerger._dept_name("75") == "Paris"
        assert DatasetMerger._dept_name("38") == "Isere"
        assert DatasetMerger._dept_name("2A") == "Corse-du-Sud"
        assert DatasetMerger._dept_name("2B") == "Haute-Corse"
        assert DatasetMerger._dept_name("01") == "Ain"

    def test_unknown_department(self):
        """Unknown department codes produce a fallback string."""
        result = DatasetMerger._dept_name("99")
        assert result == "Dept-99"

    def test_empty_code(self):
        """Empty string produces a fallback."""
        result = DatasetMerger._dept_name("")
        assert result == "Dept-"


# ============================================================
# Tests — _save_ml_dataset
# ============================================================

class TestSaveMlDataset:
    """Tests for ML dataset persistence."""

    def test_saves_csv(self, tmp_path):
        """Dataset is saved as CSV in the features directory."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        df = pd.DataFrame({
            "date_id": [202207, 202208],
            "dept": ["69", "38"],
            "nb_dpe_total": [100, 200],
        })

        output_path = merger._save_ml_dataset(df)

        assert output_path.exists()
        assert output_path.name == "hvac_ml_dataset.csv"
        assert output_path.parent == tmp_path / "features"

    def test_saved_content_matches(self, tmp_path):
        """The saved CSV can be re-loaded and matches the original."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        df = pd.DataFrame({
            "date_id": [202207, 202208, 202209],
            "dept": ["69", "38", "69"],
            "value": [1.5, 2.7, 3.9],
        })

        output_path = merger._save_ml_dataset(df)
        df_loaded = pd.read_csv(output_path)

        assert len(df_loaded) == 3
        assert list(df_loaded.columns) == ["date_id", "dept", "value"]
        assert list(df_loaded["date_id"]) == [202207, 202208, 202209]

    def test_creates_directory(self, tmp_path):
        """Creates the features directory if it does not exist."""
        cfg = _make_test_config(tmp_path)
        # Make sure features dir does not exist yet
        features_dir = tmp_path / "features"
        assert not features_dir.exists()

        merger = DatasetMerger(cfg)
        df = pd.DataFrame({"date_id": [202207], "dept": ["69"]})
        output_path = merger._save_ml_dataset(df)

        assert features_dir.exists()
        assert output_path.exists()

    def test_no_index_in_csv(self, tmp_path):
        """The saved CSV does not include the DataFrame index."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        df = pd.DataFrame({"date_id": [202207], "dept": ["69"]})

        output_path = merger._save_ml_dataset(df)
        # Read the raw CSV text: first line should be column headers only
        with open(output_path) as f:
            header = f.readline().strip()
        assert header == "date_id,dept"


# ============================================================
# Tests — _prepare_dpe_target
# ============================================================

class TestPrepareDpeTarget:
    """Tests for the DPE target variable preparation."""

    def test_basic_aggregation(self, tmp_path):
        """DPE data is aggregated per month x department."""
        cfg = _make_test_config(tmp_path)
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_dpe_target()

        assert result is not None
        assert "date_id" in result.columns
        assert "dept" in result.columns
        assert "nb_dpe_total" in result.columns
        assert "nb_installations_pac" in result.columns
        assert "nb_installations_clim" in result.columns
        assert "nb_dpe_classe_ab" in result.columns
        assert "pct_pac" in result.columns
        assert "pct_clim" in result.columns
        assert "pct_classe_ab" in result.columns

    def test_aggregation_values(self, tmp_path):
        """Aggregated counts match expected values."""
        cfg = _make_test_config(tmp_path)
        # Our helper creates 2 rows per (month, dept):
        # Row 1: is_pac=1, is_clim=0, is_classe_ab=1
        # Row 2: is_pac=0, is_clim=1, is_classe_ab=0
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(depts=["69"], months=["2022-07"]),
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_dpe_target()

        assert result is not None
        row = result[(result["dept"] == "69") & (result["date_id"] == 202207)].iloc[0]
        assert row["nb_dpe_total"] == 2
        assert row["nb_installations_pac"] == 1
        assert row["nb_installations_clim"] == 1
        assert row["nb_dpe_classe_ab"] == 1
        assert row["pct_pac"] == 50.0
        assert row["pct_clim"] == 50.0

    def test_missing_file_returns_none(self, tmp_path):
        """Returns None when DPE file is missing."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        result = merger._prepare_dpe_target()
        assert result is None

    def test_falls_back_to_raw(self, tmp_path):
        """Falls back to raw DPE file when processed is missing."""
        cfg = _make_test_config(tmp_path)
        # Write to raw, not processed
        df_raw = pd.DataFrame({
            "date_etablissement_dpe": ["2022-07-15", "2022-07-20"],
            "code_departement_ban": ["69", "69"],
            "type_generateur_chauffage_principal": ["PAC air/eau", "Chaudiere gaz"],
            "type_generateur_froid": ["", "Climatiseur"],
            "etiquette_dpe": ["A", "D"],
        })
        _write_csv(
            tmp_path / "raw" / "dpe" / "dpe_france_all.csv",
            df_raw,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_dpe_target()

        assert result is not None
        assert len(result) == 1  # 1 (month x dept) group
        row = result.iloc[0]
        assert row["nb_dpe_total"] == 2
        assert row["nb_installations_pac"] == 1  # PAC air/eau matches


# ============================================================
# Tests — build_ml_dataset (end-to-end)
# ============================================================

class TestBuildMlDataset:
    """End-to-end tests for the complete ML dataset build."""

    def _setup_all_sources(self, tmp_path: Path, cfg: ProjectConfig) -> None:
        """Write all mock CSVs needed for a complete build.

        Args:
            tmp_path: Temporary directory root.
            cfg: ProjectConfig (used to determine directory structure).
        """
        months = [f"2022-{m:02d}" for m in range(7, 13)]
        depts = ["69", "38"]

        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(depts=depts, months=months),
        )
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            _make_weather_csv(depts=depts, n_days=180, start_date="2022-07-01"),
        )
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            _make_insee_csv(months),
        )
        _write_csv(
            tmp_path / "processed" / "eurostat" / "ipi_hvac_france.csv",
            _make_eurostat_csv(months),
        )
        _write_csv(
            tmp_path / "raw" / "sitadel" / "permis_construire_france.csv",
            _make_sitadel_csv(depts=depts, months=months),
        )
        _write_csv(
            tmp_path / "raw" / "insee" / "reference_departements.csv",
            _make_reference_csv(depts=depts),
        )

    def test_full_pipeline(self, tmp_path):
        """End-to-end build produces a valid ML dataset with all columns."""
        cfg = _make_test_config(tmp_path)
        self._setup_all_sources(tmp_path, cfg)

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        assert not df.empty
        # DPE target columns
        assert "nb_dpe_total" in df.columns
        assert "nb_installations_pac" in df.columns
        assert "pct_pac" in df.columns
        # Weather columns
        assert "temp_mean" in df.columns
        assert "hdd_sum" in df.columns
        assert "nb_jours_canicule" in df.columns
        # Economic columns
        assert "confiance_menages" in df.columns
        assert "ipi_hvac_c28" in df.columns
        # SITADEL columns
        assert "nb_logements_autorises" in df.columns
        # Reference columns
        assert "revenu_median" in df.columns
        # Time features
        assert "year" in df.columns
        assert "month" in df.columns
        assert "quarter" in df.columns
        assert "is_heating" in df.columns
        assert "is_cooling" in df.columns
        assert "month_sin" in df.columns
        assert "month_cos" in df.columns
        # Geo features
        assert "dept_name" in df.columns
        assert "city_ref" in df.columns
        assert "latitude" in df.columns
        assert "longitude" in df.columns

    def test_dpe_date_filter_applied(self, tmp_path):
        """Only data from >= dpe_start_date is retained."""
        cfg = _make_test_config(tmp_path)
        # Include months before and after the DPE start date (2022-07)
        months_before = [f"2022-{m:02d}" for m in range(1, 7)]
        months_after = [f"2022-{m:02d}" for m in range(7, 13)]
        all_months = months_before + months_after

        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(depts=["69"], months=all_months),
        )
        # Provide weather covering the full range
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            _make_weather_csv(depts=["69"], n_days=365, start_date="2022-01-01"),
        )

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        # All rows should have date_id >= 202207
        assert (df["date_id"] >= 202207).all()

    def test_sorted_by_date_and_dept(self, tmp_path):
        """Output is sorted by date_id then dept."""
        cfg = _make_test_config(tmp_path)
        self._setup_all_sources(tmp_path, cfg)

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        # Verify sorting
        date_dept = list(zip(df["date_id"], df["dept"]))
        assert date_dept == sorted(date_dept)

    def test_output_saved_to_disk(self, tmp_path):
        """The build method saves the result to features dir."""
        cfg = _make_test_config(tmp_path)
        self._setup_all_sources(tmp_path, cfg)

        merger = DatasetMerger(cfg)
        merger.build_ml_dataset()

        output_path = tmp_path / "features" / "hvac_ml_dataset.csv"
        assert output_path.exists()

        df_loaded = pd.read_csv(output_path)
        assert len(df_loaded) > 0

    def test_empty_dpe_returns_empty(self, tmp_path):
        """When DPE data is missing, build returns an empty DataFrame."""
        cfg = _make_test_config(tmp_path)
        # No DPE file written
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            _make_weather_csv(),
        )

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()
        assert df.empty

    def test_weather_only_optional(self, tmp_path):
        """Build works even without weather data (columns just missing)."""
        cfg = _make_test_config(tmp_path)
        months = [f"2022-{m:02d}" for m in range(7, 13)]
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(months=months),
        )

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        assert not df.empty
        assert "nb_dpe_total" in df.columns
        # Weather columns should NOT be present
        assert "temp_mean" not in df.columns

    def test_economics_only_optional(self, tmp_path):
        """Build works even without economic data."""
        cfg = _make_test_config(tmp_path)
        months = [f"2022-{m:02d}" for m in range(7, 13)]
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(months=months),
        )

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        assert not df.empty
        assert "confiance_menages" not in df.columns

    def test_granularity_is_month_x_dept(self, tmp_path):
        """Each row is unique per (date_id, dept) pair."""
        cfg = _make_test_config(tmp_path)
        self._setup_all_sources(tmp_path, cfg)

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        # Check no duplicates on (date_id, dept)
        dup_count = df.duplicated(subset=["date_id", "dept"]).sum()
        assert dup_count == 0

    def test_two_departments_present(self, tmp_path):
        """Both departments from config appear in the result."""
        cfg = _make_test_config(tmp_path)
        self._setup_all_sources(tmp_path, cfg)

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        assert set(df["dept"].unique()) == {"69", "38"}


# ============================================================
# Tests — Security & Input Validation
# ============================================================

class TestSecurityValidation:
    """Security and edge-case tests."""

    def test_no_path_traversal_in_dept_name(self):
        """Department code with path-like characters returns safe fallback."""
        result = DatasetMerger._dept_name("../../etc")
        assert result == "Dept-../../etc"
        # No filesystem access; purely a dict lookup

    def test_special_chars_in_dept(self, tmp_path):
        """Department codes with special characters do not crash."""
        cfg = _make_test_config(tmp_path)
        df = pd.DataFrame({
            "date_id": [202207],
            "dept": ["<script>"],
        })
        merger = DatasetMerger(cfg)
        # Should not raise
        result = merger._add_time_features(df)
        assert len(result) == 1

    def test_empty_weather_csv(self, tmp_path):
        """An empty weather CSV (only headers) does not crash."""
        cfg = _make_test_config(tmp_path)
        empty_df = pd.DataFrame(columns=[
            "time", "dept", "temperature_2m_max", "temperature_2m_min",
            "temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max",
            "hdd", "cdd",
        ])
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            empty_df,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_weather_features()
        # Should return an empty DataFrame (not crash)
        assert result is not None
        assert len(result) == 0

    def test_empty_insee_csv(self, tmp_path):
        """An empty INSEE CSV (only headers) is handled gracefully."""
        cfg = _make_test_config(tmp_path)
        empty_df = pd.DataFrame(columns=[
            "period", "confiance_menages", "climat_affaires_industrie",
        ])
        _write_csv(
            tmp_path / "processed" / "insee" / "indicateurs_economiques.csv",
            empty_df,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_economic_features()
        # Should return an empty DataFrame or None, but not crash
        assert result is not None
        assert len(result) == 0


# ============================================================
# Tests — Regression: known behaviors that must be preserved
# ============================================================

class TestRegressions:
    """Regression tests for known edge cases and behaviors."""

    def test_pct_pac_division_by_zero(self, tmp_path):
        """pct_pac uses clip(lower=1) to prevent division by zero."""
        cfg = _make_test_config(tmp_path)
        # Create DPE where nb_dpe_total would be 0 for a group
        # (this is unlikely but guards against it)
        df_dpe = pd.DataFrame({
            "date_id": [202207, 202207],
            "code_departement_ban": ["69", "69"],
            "is_pac": [1, 0],
            "is_clim": [0, 1],
            "is_classe_ab": [0, 0],
        })
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            df_dpe,
        )

        merger = DatasetMerger(cfg)
        result = merger._prepare_dpe_target()

        assert result is not None
        # No NaN or Inf in percentage columns
        assert not result["pct_pac"].isna().any()
        assert not result["pct_clim"].isna().any()
        assert not np.isinf(result["pct_pac"]).any()

    def test_cyclic_encoding_december_january_continuity(self, tmp_path):
        """Cyclic encoding ensures December and January are close in feature space."""
        cfg = _make_test_config(tmp_path)
        merger = DatasetMerger(cfg)
        df = pd.DataFrame({
            "date_id": [202212, 202301],
            "dept": ["69", "69"],
        })
        result = merger._add_time_features(df)

        dec_sin = result.iloc[0]["month_sin"]
        jan_sin = result.iloc[1]["month_sin"]
        dec_cos = result.iloc[0]["month_cos"]
        jan_cos = result.iloc[1]["month_cos"]

        # Euclidean distance in (sin, cos) space should be small
        dist = np.sqrt((dec_sin - jan_sin) ** 2 + (dec_cos - jan_cos) ** 2)
        # December (12) and January (1) are 1 month apart out of 12
        # Distance should be around sin(2*pi/12) ~ 0.5, well under 1.0
        assert dist < 1.0

    def test_date_id_format_integer(self, tmp_path):
        """date_id is integer YYYYMM format throughout the pipeline."""
        cfg = _make_test_config(tmp_path)
        months = [f"2022-{m:02d}" for m in range(7, 13)]
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(depts=["69"], months=months),
        )
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            _make_weather_csv(depts=["69"], n_days=180, start_date="2022-07-01"),
        )

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        assert df["date_id"].dtype in [np.int64, np.int32, int]
        # All date_id values should be 6 digits
        assert (df["date_id"] >= 100000).all()
        assert (df["date_id"] <= 999999).all()

    def test_left_join_preserves_dpe_rows(self, tmp_path):
        """Weather merge is a LEFT JOIN — DPE rows with no weather are kept."""
        cfg = _make_test_config(tmp_path)
        months = [f"2022-{m:02d}" for m in range(7, 13)]
        _write_csv(
            tmp_path / "processed" / "dpe" / "dpe_france_clean.csv",
            _make_dpe_csv(depts=["69", "38"], months=months),
        )
        # Only provide weather for dept 69, not 38
        _write_csv(
            tmp_path / "processed" / "weather" / "weather_france.csv",
            _make_weather_csv(depts=["69"], n_days=180, start_date="2022-07-01"),
        )

        merger = DatasetMerger(cfg)
        df = merger.build_ml_dataset()

        # Dept 38 should still be present (from DPE), with NaN weather
        assert "38" in df["dept"].values
        assert "69" in df["dept"].values
