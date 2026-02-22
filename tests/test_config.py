# -*- coding: utf-8 -*-
"""Tests for the project configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config.settings import (
    DatabaseConfig,
    FRANCE_DEPARTMENTS,
    GeoConfig,
    ModelConfig,
    NetworkConfig,
    ProjectConfig,
    TimeConfig,
    _get_departments_for_scope,
    _get_cities_for_departments,
)


class TestGeoConfig:
    """Tests for GeoConfig."""

    def test_default_france_departments(self):
        """The default covers all metropolitan France (96 departments)."""
        geo = GeoConfig()
        assert len(geo.departments) == 96
        assert "69" in geo.departments  # Rhone
        assert "75" in geo.departments  # Paris
        assert "13" in geo.departments  # Bouches-du-Rhone
        assert "2A" in geo.departments  # Corse-du-Sud

    def test_default_cities(self):
        geo = GeoConfig()
        assert "Lyon" in geo.cities
        assert geo.cities["Lyon"]["dept"] == "69"
        assert "lat" in geo.cities["Lyon"]
        assert "lon" in geo.cities["Lyon"]
        assert "Paris" in geo.cities
        assert "Marseille" in geo.cities

    def test_one_city_per_department(self):
        """Verify there is exactly one city per department."""
        geo = GeoConfig()
        depts_from_cities = [info["dept"] for info in geo.cities.values()]
        assert len(depts_from_cities) == len(set(depts_from_cities))
        assert set(depts_from_cities) == set(geo.departments)

    def test_aura_scope(self):
        """AURA scope = 12 departments (legacy)."""
        depts = _get_departments_for_scope("84")
        assert len(depts) == 12
        assert "69" in depts  # Lyon
        assert "63" in depts  # Clermont-Ferrand

    def test_idf_scope(self):
        """Ile-de-France scope = 8 departments."""
        depts = _get_departments_for_scope("11")
        assert len(depts) == 8
        assert "75" in depts  # Paris

    def test_france_departments_reference(self):
        """The FRANCE_DEPARTMENTS reference covers all 96 departments."""
        depts = {info["dept"] for info in FRANCE_DEPARTMENTS.values()}
        assert len(depts) == 96

    def test_get_cities_for_departments(self):
        """Filter cities for a subset of departments."""
        cities = _get_cities_for_departments(["69", "38"])
        assert len(cities) == 2
        city_names = list(cities.keys())
        depts = [info["dept"] for info in cities.values()]
        assert "69" in depts
        assert "38" in depts


class TestTimeConfig:
    """Tests for TimeConfig."""

    def test_default_dates(self):
        time = TimeConfig()
        assert time.start_date == "2019-01-01"
        assert time.end_date == "2026-02-28"
        assert time.dpe_start_date == "2021-07-01"
        assert time.frequency == "MS"

    def test_split_chronological(self):
        """The temporal split must be chronological."""
        time = TimeConfig()
        assert time.start_date < time.train_end
        assert time.train_end < time.val_end
        assert time.val_end < time.end_date


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_sqlite_connection_string(self):
        db = DatabaseConfig(db_type="sqlite", db_path="test.db")
        assert db.connection_string == "sqlite:///test.db"

    def test_mssql_with_auth(self):
        db = DatabaseConfig(
            db_type="mssql",
            db_host="server",
            db_port=1433,
            db_name="mydb",
            db_user="user",
            db_password="pass",
            allow_non_local=True,
        )
        conn = db.connection_string
        assert "mssql+pyodbc://" in conn
        assert "user:pass" in conn

    def test_mssql_windows_auth(self):
        db = DatabaseConfig(
            db_type="mssql",
            db_host="server",
            db_name="mydb",
            allow_non_local=True,
        )
        conn = db.connection_string
        assert "Trusted_Connection=yes" in conn

    def test_postgresql_connection_string(self):
        db = DatabaseConfig(
            db_type="postgresql",
            db_host="pghost",
            db_port=5432,
            db_name="pgdb",
            db_user="pguser",
            db_password="pgpass",
            allow_non_local=True,
        )
        conn = db.connection_string
        assert "postgresql://" in conn
        assert "pguser:pgpass" in conn

    def test_non_local_requires_permission(self):
        db = DatabaseConfig(db_type="mssql", allow_non_local=False)
        with pytest.raises(ValueError, match="Non-local database.*disabled"):
            _ = db.connection_string

    def test_unknown_db_type(self):
        db = DatabaseConfig(db_type="oracle", allow_non_local=True)
        with pytest.raises(ValueError, match="Unknown database type"):
            _ = db.connection_string


class TestProjectConfig:
    """Tests for ProjectConfig."""

    def test_default_config(self):
        config = ProjectConfig()
        assert config.geo is not None
        assert config.time is not None
        assert config.network is not None
        assert config.database is not None
        assert config.model is not None

    def test_from_env(self):
        with patch.dict("os.environ", {
            "DB_TYPE": "sqlite",
            "DB_PATH": "test.db",
            "LOG_LEVEL": "DEBUG",
        }):
            config = ProjectConfig.from_env()
            assert config.log_level == "DEBUG"

    def test_frozen(self):
        config = ProjectConfig()
        with pytest.raises(AttributeError):
            config.log_level = "ERROR"


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        model = ModelConfig()
        assert model.max_lag_months == 6
        assert model.rolling_windows == [3, 6]
        assert model.hdd_base_temp == 18.0
        assert model.cdd_base_temp == 18.0

    def test_lightgbm_params_present(self):
        model = ModelConfig()
        params = model.lightgbm_params
        assert "max_depth" in params
        assert "num_leaves" in params
        assert "learning_rate" in params
        assert params["max_depth"] <= 6  # Regularized
