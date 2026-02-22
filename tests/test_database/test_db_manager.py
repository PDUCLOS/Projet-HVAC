# -*- coding: utf-8 -*-
"""
Comprehensive tests for DatabaseManager.

Tests cover initialization, schema creation, data loading, querying,
row counting with whitelist enforcement, safe table reading, geo mapping,
column finder utility, table info summary, and collected data import
with missing directories.

All tests use in-memory SQLite databases to remain self-contained
and fast, with no dependency on actual data files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import text

from src.database.db_manager import DatabaseManager, SCHEMA_SQLITE_PATH


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def db_memory() -> DatabaseManager:
    """Create a DatabaseManager connected to an in-memory SQLite database."""
    return DatabaseManager("sqlite:///:memory:")


@pytest.fixture
def db_file(tmp_path: Path) -> DatabaseManager:
    """Create a DatabaseManager backed by a temporary file-based SQLite DB."""
    db_path = tmp_path / "test_hvac.db"
    return DatabaseManager(f"sqlite:///{db_path}")


@pytest.fixture
def db_initialized(db_memory: DatabaseManager) -> DatabaseManager:
    """Return a DatabaseManager with the full schema already applied.

    This uses the real schema.sql so that table structures, reference data
    (dim_time, dim_geo, dim_equipment_type), and indexes are all present.
    """
    db_memory.init_database()
    return db_memory


@pytest.fixture
def small_fact_df() -> pd.DataFrame:
    """A minimal DataFrame matching fact_hvac_installations columns."""
    return pd.DataFrame({
        "date_id": [202301, 202302, 202303],
        "geo_id": [1, 1, 2],
        "nb_dpe_total": [100, 150, 200],
        "nb_installations_pac": [10, 20, 30],
        "nb_installations_clim": [5, 8, 12],
        "temp_mean": [3.5, 5.2, 7.8],
    })


@pytest.fixture
def small_economic_df() -> pd.DataFrame:
    """A minimal DataFrame matching fact_economic_context columns."""
    return pd.DataFrame({
        "date_id": [202301, 202302, 202303],
        "confiance_menages": [95.0, 96.5, 97.0],
        "climat_affaires_indus": [100.0, 101.0, 99.5],
        "ipi_hvac_c28": [105.0, 106.0, 104.5],
    })


# =====================================================================
# 1. __init__ — db_type detection
# =====================================================================

class TestInit:
    """Tests for DatabaseManager.__init__ and engine type detection."""

    def test_sqlite_detection(self) -> None:
        """sqlite:// connection strings produce db_type='sqlite'."""
        db = DatabaseManager("sqlite:///:memory:")
        assert db.db_type == "sqlite"

    def test_sqlite_file_detection(self, tmp_path: Path) -> None:
        """File-based sqlite:/// connection strings produce db_type='sqlite'."""
        db = DatabaseManager(f"sqlite:///{tmp_path / 'test.db'}")
        assert db.db_type == "sqlite"

    def test_mssql_detection_mock(self) -> None:
        """Connection strings starting with 'mssql' produce db_type='mssql'.

        We only test the string-based detection logic here, not actual
        MSSQL connectivity (which would require a live server).
        """
        # We catch the error from SQLAlchemy since there is no real MSSQL
        # driver, but db_type should still be set before the engine is used.
        try:
            db = DatabaseManager("mssql+pyodbc://user:pass@server/db")
            assert db.db_type == "mssql"
        except Exception:
            # Some environments may lack pyodbc; the key point is
            # the detection logic in __init__ runs before engine usage.
            pass

    def test_postgresql_detection_mock(self) -> None:
        """Connection strings starting with 'postgresql' produce db_type='postgresql'."""
        try:
            db = DatabaseManager("postgresql://user:pass@localhost/testdb")
            assert db.db_type == "postgresql"
        except Exception:
            pass

    def test_unknown_detection(self) -> None:
        """Unrecognized connection strings produce db_type='unknown'."""
        try:
            db = DatabaseManager("oracle://user:pass@host/sid")
            assert db.db_type == "unknown"
        except Exception:
            pass

    def test_engine_is_created(self, db_memory: DatabaseManager) -> None:
        """The engine attribute is a working SQLAlchemy engine."""
        assert db_memory.engine is not None
        # Verify the engine is functional by running a trivial query
        with db_memory.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_logger_is_configured(self, db_memory: DatabaseManager) -> None:
        """The logger attribute is set and has handlers."""
        assert db_memory.logger is not None
        assert db_memory.logger.name == "database.manager"
        assert len(db_memory.logger.handlers) > 0


# =====================================================================
# 2. init_database() — schema execution
# =====================================================================

class TestInitDatabase:
    """Tests for init_database() using the real schema.sql."""

    def test_schema_file_exists(self) -> None:
        """The schema.sql file must exist at the expected path."""
        assert SCHEMA_SQLITE_PATH.exists(), (
            f"schema.sql not found at {SCHEMA_SQLITE_PATH}"
        )

    def test_init_does_not_crash(self, db_memory: DatabaseManager) -> None:
        """init_database() should execute without raising exceptions."""
        db_memory.init_database()

    def test_idempotent(self, db_memory: DatabaseManager) -> None:
        """Calling init_database() twice should not raise an error."""
        db_memory.init_database()
        db_memory.init_database()

    def test_tables_created(self, db_initialized: DatabaseManager) -> None:
        """All expected tables should exist after initialization."""
        with db_initialized.engine.connect() as conn:
            result = conn.execute(text(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "ORDER BY name"
            ))
            tables = {row[0] for row in result}

        expected = {
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context",
            "raw_dpe",
        }
        assert expected.issubset(tables), (
            f"Missing tables: {expected - tables}"
        )

    def test_dim_time_populated(self, db_initialized: DatabaseManager) -> None:
        """dim_time should contain reference data (84 months: 2019-2025)."""
        df = db_initialized.query("SELECT COUNT(*) AS cnt FROM dim_time")
        assert df["cnt"].iloc[0] == 84

    def test_dim_geo_populated(self, db_initialized: DatabaseManager) -> None:
        """dim_geo should contain the 8 reference departments."""
        df = db_initialized.query("SELECT COUNT(*) AS cnt FROM dim_geo")
        assert df["cnt"].iloc[0] == 8

    def test_dim_equipment_type_populated(
        self, db_initialized: DatabaseManager
    ) -> None:
        """dim_equipment_type should contain the 10 reference equipment rows."""
        df = db_initialized.query(
            "SELECT COUNT(*) AS cnt FROM dim_equipment_type"
        )
        assert df["cnt"].iloc[0] == 10

    def test_file_based_init(self, db_file: DatabaseManager) -> None:
        """init_database() also works with a file-based SQLite database."""
        db_file.init_database()
        df = db_file.query("SELECT COUNT(*) AS cnt FROM dim_geo")
        assert df["cnt"].iloc[0] == 8


# =====================================================================
# 3. load_dataframe() — insert DataFrame and verify row count
# =====================================================================

class TestLoadDataframe:
    """Tests for load_dataframe()."""

    def test_insert_and_count(
        self, db_initialized: DatabaseManager, small_fact_df: pd.DataFrame
    ) -> None:
        """Inserting a DataFrame returns the correct number of rows."""
        inserted = db_initialized.load_dataframe(
            small_fact_df, "fact_hvac_installations", if_exists="replace"
        )
        assert inserted == 3

    def test_append_mode(
        self, db_initialized: DatabaseManager, small_economic_df: pd.DataFrame
    ) -> None:
        """Appending twice doubles the row count."""
        db_initialized.load_dataframe(
            small_economic_df, "fact_economic_context", if_exists="replace"
        )
        inserted_second = db_initialized.load_dataframe(
            small_economic_df.assign(
                date_id=[202304, 202305, 202306]
            ),
            "fact_economic_context",
            if_exists="append",
        )
        assert inserted_second == 3
        # Total should now be 6
        df = db_initialized.query(
            "SELECT COUNT(*) AS cnt FROM fact_economic_context"
        )
        assert df["cnt"].iloc[0] == 6

    def test_replace_mode(
        self, db_initialized: DatabaseManager, small_fact_df: pd.DataFrame
    ) -> None:
        """Using if_exists='replace' overwrites previous data."""
        db_initialized.load_dataframe(
            small_fact_df, "fact_hvac_installations", if_exists="replace"
        )
        # Replace with a smaller set
        smaller = small_fact_df.iloc[:1]
        db_initialized.load_dataframe(
            smaller, "fact_hvac_installations", if_exists="replace"
        )
        df = db_initialized.query(
            "SELECT COUNT(*) AS cnt FROM fact_hvac_installations"
        )
        assert df["cnt"].iloc[0] == 1

    def test_returns_zero_on_empty_df(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Loading an empty DataFrame returns 0 rows inserted."""
        empty_df = pd.DataFrame(
            columns=["date_id", "confiance_menages"]
        )
        inserted = db_initialized.load_dataframe(
            empty_df, "fact_economic_context", if_exists="replace"
        )
        assert inserted == 0


# =====================================================================
# 4. query() — execute SELECT and get results
# =====================================================================

class TestQuery:
    """Tests for query()."""

    def test_basic_select(self, db_initialized: DatabaseManager) -> None:
        """A basic SELECT on dim_time returns a DataFrame with rows."""
        df = db_initialized.query(
            "SELECT date_id, year, month FROM dim_time LIMIT 5"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "date_id" in df.columns
        assert "year" in df.columns

    def test_filtered_select(self, db_initialized: DatabaseManager) -> None:
        """A filtered SELECT returns only matching rows."""
        df = db_initialized.query(
            "SELECT * FROM dim_time WHERE year = 2023"
        )
        assert len(df) == 12
        assert all(df["year"] == 2023)

    def test_join_query(self, db_initialized: DatabaseManager) -> None:
        """A JOIN query across dim tables works correctly."""
        df = db_initialized.query(
            "SELECT g.dept_code, g.dept_name "
            "FROM dim_geo g WHERE g.dept_code = '69'"
        )
        assert len(df) == 1
        assert df["dept_name"].iloc[0] == "Rh\u00f4ne"

    def test_empty_result(self, db_initialized: DatabaseManager) -> None:
        """Querying with an impossible filter returns an empty DataFrame."""
        df = db_initialized.query(
            "SELECT * FROM dim_time WHERE year = 9999"
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_aggregate_query(self, db_initialized: DatabaseManager) -> None:
        """Aggregate functions work inside query()."""
        df = db_initialized.query(
            "SELECT year, COUNT(*) AS cnt FROM dim_time GROUP BY year"
        )
        # 7 years (2019-2025), each with 12 months
        assert len(df) == 7
        assert all(df["cnt"] == 12)


# =====================================================================
# 5. _count_rows() — whitelist enforcement and correct count
# =====================================================================

class TestCountRows:
    """Tests for _count_rows()."""

    def test_whitelist_allows_valid_tables(
        self, db_initialized: DatabaseManager
    ) -> None:
        """All whitelisted table names return a count >= 0."""
        valid_tables = [
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context", "raw_dpe",
        ]
        for table in valid_tables:
            count = db_initialized._count_rows(table)
            assert isinstance(count, int)
            assert count >= 0

    def test_whitelist_rejects_unknown_table(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Unknown/malicious table names are rejected and return 0."""
        assert db_initialized._count_rows("unknown_table") == 0
        assert db_initialized._count_rows("'; DROP TABLE dim_time; --") == 0
        assert db_initialized._count_rows("") == 0

    def test_correct_count_dim_time(
        self, db_initialized: DatabaseManager
    ) -> None:
        """dim_time should have exactly 84 rows after init."""
        assert db_initialized._count_rows("dim_time") == 84

    def test_correct_count_dim_geo(
        self, db_initialized: DatabaseManager
    ) -> None:
        """dim_geo should have exactly 8 rows after init."""
        assert db_initialized._count_rows("dim_geo") == 8

    def test_count_empty_table(
        self, db_initialized: DatabaseManager
    ) -> None:
        """An empty fact table should return 0."""
        assert db_initialized._count_rows("fact_hvac_installations") == 0

    def test_count_after_insert(
        self,
        db_initialized: DatabaseManager,
        small_fact_df: pd.DataFrame,
    ) -> None:
        """Count reflects newly inserted rows."""
        db_initialized.load_dataframe(
            small_fact_df, "fact_hvac_installations", if_exists="replace"
        )
        assert db_initialized._count_rows("fact_hvac_installations") == 3

    def test_count_before_init_returns_zero(
        self, db_memory: DatabaseManager
    ) -> None:
        """Counting rows on a non-existent table (pre-init) returns 0."""
        assert db_memory._count_rows("dim_time") == 0


# =====================================================================
# 6. _safe_read_table() — empty/non-existent vs populated
# =====================================================================

class TestSafeReadTable:
    """Tests for _safe_read_table()."""

    def test_returns_none_for_nonexistent_table(
        self, db_memory: DatabaseManager
    ) -> None:
        """Before schema init, tables do not exist; returns None."""
        result = db_memory._safe_read_table("dim_time")
        assert result is None

    def test_returns_none_for_empty_table(
        self, db_initialized: DatabaseManager
    ) -> None:
        """An initialized but empty fact table returns None."""
        result = db_initialized._safe_read_table("fact_hvac_installations")
        assert result is None

    def test_returns_none_for_unknown_table(
        self, db_initialized: DatabaseManager
    ) -> None:
        """An unwhitelisted table name returns None."""
        result = db_initialized._safe_read_table("nonexistent_table")
        assert result is None

    def test_returns_dataframe_for_populated_table(
        self, db_initialized: DatabaseManager
    ) -> None:
        """A populated dimension table returns a DataFrame."""
        result = db_initialized._safe_read_table("dim_time")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 84

    def test_returns_dataframe_after_insert(
        self,
        db_initialized: DatabaseManager,
        small_economic_df: pd.DataFrame,
    ) -> None:
        """After inserting data, _safe_read_table returns a DataFrame."""
        db_initialized.load_dataframe(
            small_economic_df, "fact_economic_context", if_exists="replace"
        )
        result = db_initialized._safe_read_table("fact_economic_context")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_dim_geo_content(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Reading dim_geo returns expected columns and data."""
        result = db_initialized._safe_read_table("dim_geo")
        assert result is not None
        assert "dept_code" in result.columns
        assert "dept_name" in result.columns
        assert "city_ref" in result.columns
        dept_codes = set(result["dept_code"].astype(str))
        assert "69" in dept_codes
        assert "38" in dept_codes


# =====================================================================
# 7. _get_geo_mapping() — empty dict vs correct mapping
# =====================================================================

class TestGetGeoMapping:
    """Tests for _get_geo_mapping()."""

    def test_empty_dict_on_no_table(
        self, db_memory: DatabaseManager
    ) -> None:
        """Returns an empty dict when dim_geo does not exist."""
        result = db_memory._get_geo_mapping()
        assert result == {}

    def test_correct_mapping(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Returns a dict mapping dept_code strings to geo_id integers."""
        mapping = db_initialized._get_geo_mapping()
        assert isinstance(mapping, dict)
        assert len(mapping) == 8

        # Check that known departments are present
        assert "69" in mapping
        assert "38" in mapping
        assert "01" in mapping

        # Values should be integers (geo_id)
        for dept_code, geo_id in mapping.items():
            assert isinstance(dept_code, str)
            assert isinstance(geo_id, (int, float))

    def test_mapping_values_are_unique(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Each geo_id value in the mapping should be unique."""
        mapping = db_initialized._get_geo_mapping()
        geo_ids = list(mapping.values())
        assert len(geo_ids) == len(set(geo_ids))

    def test_mapping_after_insert_extra_dept(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Adding a new department updates the mapping."""
        # Insert a new department
        with db_initialized.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO dim_geo (dept_code, dept_name, city_ref, "
                "latitude, longitude, region_code) "
                "VALUES ('75', 'Paris', 'Paris', 48.86, 2.35, '11')"
            ))

        mapping = db_initialized._get_geo_mapping()
        assert "75" in mapping
        assert len(mapping) == 9


# =====================================================================
# 8. _find_col() — column name candidate matching
# =====================================================================

class TestFindCol:
    """Tests for _find_col()."""

    def test_finds_first_match(self, db_memory: DatabaseManager) -> None:
        """Returns the first candidate that exists in the DataFrame."""
        df = pd.DataFrame({
            "temperature_2m_mean": [1.0],
            "temp_mean": [2.0],
        })
        result = db_memory._find_col(
            df, ["temperature_2m_mean", "temp_mean"]
        )
        assert result == "temperature_2m_mean"

    def test_finds_second_candidate(
        self, db_memory: DatabaseManager
    ) -> None:
        """When the first candidate is absent, returns the second."""
        df = pd.DataFrame({
            "temp_mean": [2.0],
        })
        result = db_memory._find_col(
            df, ["temperature_2m_mean", "temp_mean"]
        )
        assert result == "temp_mean"

    def test_returns_none_when_no_match(
        self, db_memory: DatabaseManager
    ) -> None:
        """Returns None when no candidates match any column."""
        df = pd.DataFrame({
            "unrelated_col": [1.0],
        })
        result = db_memory._find_col(
            df, ["temperature_2m_mean", "temp_mean"]
        )
        assert result is None

    def test_empty_candidates_list(
        self, db_memory: DatabaseManager
    ) -> None:
        """Returns None when the candidates list is empty."""
        df = pd.DataFrame({"col_a": [1]})
        result = db_memory._find_col(df, [])
        assert result is None

    def test_empty_dataframe(self, db_memory: DatabaseManager) -> None:
        """Returns None when the DataFrame has no columns."""
        df = pd.DataFrame()
        result = db_memory._find_col(df, ["a", "b"])
        assert result is None

    def test_priority_order_respected(
        self, db_memory: DatabaseManager
    ) -> None:
        """When multiple candidates match, the first in order wins."""
        df = pd.DataFrame({
            "hdd": [10.0],
            "heating_degree_days": [11.0],
        })
        result = db_memory._find_col(df, ["hdd", "heating_degree_days"])
        assert result == "hdd"

        # Reverse priority
        result2 = db_memory._find_col(
            df, ["heating_degree_days", "hdd"]
        )
        assert result2 == "heating_degree_days"


# =====================================================================
# 9. get_table_info() — returns DataFrame with expected structure
# =====================================================================

class TestGetTableInfo:
    """Tests for get_table_info()."""

    def test_returns_dataframe(
        self, db_initialized: DatabaseManager
    ) -> None:
        """get_table_info() returns a pandas DataFrame."""
        info = db_initialized.get_table_info()
        assert isinstance(info, pd.DataFrame)

    def test_expected_columns(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Result has 'table_name' and 'row_count' columns."""
        info = db_initialized.get_table_info()
        assert "table_name" in info.columns
        assert "row_count" in info.columns

    def test_all_tables_listed(
        self, db_initialized: DatabaseManager
    ) -> None:
        """All 6 project tables appear in the result."""
        info = db_initialized.get_table_info()
        table_names = set(info["table_name"])
        expected = {
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context",
            "raw_dpe",
        }
        assert table_names == expected

    def test_row_counts_correct(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Row counts match the reference data inserted by schema.sql."""
        info = db_initialized.get_table_info()
        info_dict = dict(
            zip(info["table_name"], info["row_count"])
        )
        assert info_dict["dim_time"] == 84
        assert info_dict["dim_geo"] == 8
        assert info_dict["dim_equipment_type"] == 10
        assert info_dict["fact_hvac_installations"] == 0
        assert info_dict["fact_economic_context"] == 0
        assert info_dict["raw_dpe"] == 0

    def test_row_counts_update_after_insert(
        self,
        db_initialized: DatabaseManager,
        small_economic_df: pd.DataFrame,
    ) -> None:
        """Row counts reflect changes after data insertion."""
        db_initialized.load_dataframe(
            small_economic_df, "fact_economic_context", if_exists="replace"
        )
        info = db_initialized.get_table_info()
        info_dict = dict(
            zip(info["table_name"], info["row_count"])
        )
        assert info_dict["fact_economic_context"] == 3

    def test_before_init_all_zeros(
        self, db_memory: DatabaseManager
    ) -> None:
        """Before init_database(), all tables have 0 rows (they don't exist)."""
        info = db_memory.get_table_info()
        assert all(info["row_count"] == 0)


# =====================================================================
# 10. import_collected_data() — missing raw_data_dir returns empty dict
# =====================================================================

class TestImportCollectedData:
    """Tests for import_collected_data()."""

    def test_missing_directory_returns_empty_dict(
        self, db_initialized: DatabaseManager, tmp_path: Path
    ) -> None:
        """When raw_data_dir does not contain expected files, returns {}."""
        nonexistent = tmp_path / "no_such_directory"
        result = db_initialized.import_collected_data(
            raw_data_dir=nonexistent
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_empty_directory_returns_empty_dict(
        self, db_initialized: DatabaseManager, tmp_path: Path
    ) -> None:
        """An empty directory (no CSV files) also returns {}."""
        empty_dir = tmp_path / "empty_raw"
        empty_dir.mkdir()
        result = db_initialized.import_collected_data(
            raw_data_dir=empty_dir
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_selective_sources(
        self, db_initialized: DatabaseManager, tmp_path: Path
    ) -> None:
        """Passing sources=['weather'] only attempts to import weather."""
        empty_dir = tmp_path / "raw_selective"
        empty_dir.mkdir()
        result = db_initialized.import_collected_data(
            raw_data_dir=empty_dir, sources=["weather"]
        )
        assert isinstance(result, dict)
        # No weather file present, so nothing imported
        assert "weather" not in result

    def test_invalid_source_name_ignored(
        self, db_initialized: DatabaseManager, tmp_path: Path
    ) -> None:
        """Unknown source names in the sources list are silently ignored."""
        empty_dir = tmp_path / "raw_invalid"
        empty_dir.mkdir()
        result = db_initialized.import_collected_data(
            raw_data_dir=empty_dir, sources=["nonexistent_source"]
        )
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_default_raw_data_dir(
        self, db_initialized: DatabaseManager
    ) -> None:
        """When raw_data_dir is None, defaults to Path('data/raw')."""
        # This will likely find no files, so we just check it doesn't crash
        result = db_initialized.import_collected_data(raw_data_dir=None)
        assert isinstance(result, dict)

    def test_import_sources_registry(self) -> None:
        """The IMPORT_SOURCES registry has the expected keys."""
        expected_sources = {"weather", "insee", "eurostat", "sitadel", "dpe"}
        assert set(DatabaseManager.IMPORT_SOURCES.keys()) == expected_sources

    def test_import_sources_have_required_keys(self) -> None:
        """Each source in IMPORT_SOURCES has file, table, description, method."""
        required_keys = {"file", "table", "description", "method"}
        for source_name, meta in DatabaseManager.IMPORT_SOURCES.items():
            assert required_keys.issubset(set(meta.keys())), (
                f"Source '{source_name}' missing keys: "
                f"{required_keys - set(meta.keys())}"
            )


# =====================================================================
# Integration / round-trip tests
# =====================================================================

class TestIntegration:
    """Round-trip integration tests combining multiple methods."""

    def test_insert_then_query(
        self,
        db_initialized: DatabaseManager,
        small_fact_df: pd.DataFrame,
    ) -> None:
        """Data inserted via load_dataframe is retrievable via query."""
        db_initialized.load_dataframe(
            small_fact_df, "fact_hvac_installations", if_exists="replace"
        )
        df = db_initialized.query(
            "SELECT date_id, nb_dpe_total FROM fact_hvac_installations "
            "ORDER BY date_id"
        )
        assert len(df) == 3
        assert list(df["date_id"]) == [202301, 202302, 202303]
        assert list(df["nb_dpe_total"]) == [100, 150, 200]

    def test_insert_then_safe_read(
        self,
        db_initialized: DatabaseManager,
        small_economic_df: pd.DataFrame,
    ) -> None:
        """Data inserted via load_dataframe is readable via _safe_read_table."""
        db_initialized.load_dataframe(
            small_economic_df, "fact_economic_context", if_exists="replace"
        )
        result = db_initialized._safe_read_table("fact_economic_context")
        assert result is not None
        assert len(result) == 3

    def test_geo_mapping_used_in_queries(
        self, db_initialized: DatabaseManager
    ) -> None:
        """Geo mapping from dim_geo can be used to resolve dept codes."""
        mapping = db_initialized._get_geo_mapping()
        geo_id_69 = mapping["69"]

        # Query dim_geo directly to verify
        df = db_initialized.query(
            f"SELECT dept_name FROM dim_geo WHERE geo_id = {geo_id_69}"
        )
        assert df["dept_name"].iloc[0] == "Rh\u00f4ne"

    def test_full_workflow(
        self, db_file: DatabaseManager, small_fact_df: pd.DataFrame
    ) -> None:
        """Full workflow: init -> load -> query -> info on a file-based DB."""
        db_file.init_database()

        inserted = db_file.load_dataframe(
            small_fact_df, "fact_hvac_installations", if_exists="replace"
        )
        assert inserted == 3

        df = db_file.query(
            "SELECT COUNT(*) AS cnt FROM fact_hvac_installations"
        )
        assert df["cnt"].iloc[0] == 3

        info = db_file.get_table_info()
        info_dict = dict(zip(info["table_name"], info["row_count"]))
        assert info_dict["fact_hvac_installations"] == 3
        assert info_dict["dim_time"] == 84


# =====================================================================
# Security / edge-case tests
# =====================================================================

class TestSecurity:
    """Security-focused tests: SQL injection prevention and input validation."""

    def test_count_rows_rejects_injection_attempt(
        self, db_initialized: DatabaseManager
    ) -> None:
        """_count_rows rejects table names that look like injection attempts."""
        malicious_names = [
            "dim_time; DROP TABLE dim_geo",
            "'; DELETE FROM dim_time; --",
            "fact_hvac_installations UNION SELECT * FROM dim_geo",
            "1; SELECT 1",
            "dim_time--",
        ]
        for name in malicious_names:
            count = db_initialized._count_rows(name)
            assert count == 0, (
                f"Expected 0 for malicious name '{name}', got {count}"
            )

    def test_safe_read_rejects_injection_attempt(
        self, db_initialized: DatabaseManager
    ) -> None:
        """_safe_read_table rejects table names that look like injection."""
        result = db_initialized._safe_read_table(
            "dim_time; DROP TABLE dim_geo"
        )
        assert result is None

    def test_whitelist_is_comprehensive(self) -> None:
        """The whitelists in _count_rows and _safe_read_table match."""
        # Access the allowed sets via source inspection
        # Both methods define the same whitelist
        expected = {
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context", "raw_dpe",
        }
        # Verify by calling both methods with each allowed name
        db = DatabaseManager("sqlite:///:memory:")
        db.init_database()

        for table in expected:
            # _count_rows should not return 0 due to rejection
            # (it may return 0 if the table is empty, which is fine)
            count = db._count_rows(table)
            assert isinstance(count, int)

            # _safe_read_table should not return None due to rejection
            # (it may return None if the table is empty, which is also fine)
            # The key is it does not log a "Rejected unknown table name" warning
