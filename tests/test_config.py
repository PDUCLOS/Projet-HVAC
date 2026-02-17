# -*- coding: utf-8 -*-
"""Tests pour la configuration du projet."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from config.settings import (
    DatabaseConfig,
    GeoConfig,
    ModelConfig,
    NetworkConfig,
    ProjectConfig,
    TimeConfig,
)


class TestGeoConfig:
    """Tests pour GeoConfig."""

    def test_default_departments(self):
        geo = GeoConfig()
        assert len(geo.departments) == 8
        assert "69" in geo.departments  # Rhône
        assert "38" in geo.departments  # Isère

    def test_default_cities(self):
        geo = GeoConfig()
        assert "Lyon" in geo.cities
        assert geo.cities["Lyon"]["dept"] == "69"
        assert "lat" in geo.cities["Lyon"]
        assert "lon" in geo.cities["Lyon"]

    def test_one_city_per_department(self):
        """Vérifie qu'il y a exactement une ville par département."""
        geo = GeoConfig()
        depts_from_cities = [info["dept"] for info in geo.cities.values()]
        assert len(depts_from_cities) == len(set(depts_from_cities))
        assert set(depts_from_cities) == set(geo.departments)


class TestTimeConfig:
    """Tests pour TimeConfig."""

    def test_default_dates(self):
        time = TimeConfig()
        assert time.start_date == "2019-01-01"
        assert time.end_date == "2026-02-28"
        assert time.dpe_start_date == "2021-07-01"
        assert time.frequency == "MS"

    def test_split_chronological(self):
        """Le split temporel doit être chronologique."""
        time = TimeConfig()
        assert time.start_date < time.train_end
        assert time.train_end < time.val_end
        assert time.val_end < time.end_date


class TestDatabaseConfig:
    """Tests pour DatabaseConfig."""

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
        with pytest.raises(ValueError, match="non locale désactivée"):
            _ = db.connection_string

    def test_unknown_db_type(self):
        db = DatabaseConfig(db_type="oracle", allow_non_local=True)
        with pytest.raises(ValueError, match="Type de BDD inconnu"):
            _ = db.connection_string


class TestProjectConfig:
    """Tests pour ProjectConfig."""

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
    """Tests pour ModelConfig."""

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
        assert params["max_depth"] <= 6  # Régularisé
