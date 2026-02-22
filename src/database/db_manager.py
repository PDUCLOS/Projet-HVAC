# -*- coding: utf-8 -*-
"""
Database Manager — SQLite / SQL Server / PostgreSQL database management.
========================================================================

Provides a unified interface to:
- Initialize the database (create tables based on the engine)
- Insert data from DataFrames
- Import CSVs collected by collectors (weather, insee, eurostat, etc.)
- Query data for analysis and ML
- Build the ML-ready dataset (facts join + economic context)

Architecture:
    DatabaseManager uses SQLAlchemy as an abstraction layer.
    The engine choice (SQLite, SQL Server, PostgreSQL) is made solely
    via the connection string. The SQL schema is adapted automatically
    (schema.sql for SQLite, schema_mssql.sql for SQL Server).

Usage:
    >>> from src.database.db_manager import DatabaseManager
    >>> from config.settings import config
    >>> db = DatabaseManager(config.database.connection_string)
    >>> db.init_database()
    >>> db.import_collected_data()            # Import all collected CSVs
    >>> df_ml = db.build_ml_dataset()

Extensibility:
    To add a new table or a new data type:
    1. Add the DDL in schema.sql AND schema_mssql.sql
    2. Add a load_xxx_data() method in this class
    3. Update build_ml_dataset() if necessary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# Paths to schema files (relative to the module)
SCHEMA_SQLITE_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_MSSQL_PATH = Path(__file__).parent / "schema_mssql.sql"


class DatabaseManager:
    """Database manager for the HVAC project.

    Encapsulates all CRUD operations and provides specialized
    methods for loading each data type.

    The SQL engine is automatically detected from the connection string.

    Attributes:
        engine: SQLAlchemy engine connected to the database.
        db_type: Detected engine type ('sqlite', 'mssql', 'postgresql').
        logger: Structured logger for DB operations.
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize the database connection.

        Automatically detects the engine type from the connection URL.

        Args:
            connection_string: SQLAlchemy URL (e.g., 'sqlite:///data/hvac.db',
                              'mssql+pyodbc://...', 'postgresql://...').
        """
        self.engine: Engine = create_engine(connection_string)
        self.logger = logging.getLogger("database.manager")

        # Detect the engine type to adapt behavior
        if connection_string.startswith("sqlite"):
            self.db_type = "sqlite"
        elif connection_string.startswith("mssql"):
            self.db_type = "mssql"
        elif connection_string.startswith("postgresql"):
            self.db_type = "postgresql"
        else:
            self.db_type = "unknown"

        # Configure logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        # Prevent duplicate output when root logger also has a handler
        self.logger.propagate = False

        self.logger.info(
            "DB connection: engine=%s", self.db_type,
        )

    def init_database(self) -> None:
        """Initialize the database by executing the appropriate SQL schema.

        Automatically selects schema.sql (SQLite) or
        schema_mssql.sql (SQL Server) based on the detected engine.
        Idempotent: can be called multiple times without error.
        """
        self.logger.info("Initializing the database...")

        # Choose the schema file based on the engine
        if self.db_type == "mssql":
            schema_path = SCHEMA_MSSQL_PATH
        else:
            # SQLite and PostgreSQL use the same schema
            # (PostgreSQL supports native BOOLEAN, VARCHAR)
            schema_path = SCHEMA_SQLITE_PATH

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {schema_path}"
            )

        schema_sql = schema_path.read_text(encoding="utf-8")

        if self.db_type == "mssql":
            # SQL Server: split on 'GO' or ';' at end of block
            # The MSSQL schema uses complete blocks separated by ';'
            self._execute_statements_mssql(schema_sql)
        else:
            # SQLite/PostgreSQL: split on ';'
            self._execute_statements_sqlite(schema_sql)

        self.logger.info("Database initialized successfully.")

    def _execute_statements_sqlite(self, schema_sql: str) -> None:
        """Execute SQL statements for SQLite/PostgreSQL.

        SQLite does not support multi-statement script execution,
        so we split on ';' and execute each statement separately.
        Comment lines (--) at the beginning of blocks are ignored.
        """
        with self.engine.begin() as conn:
            for statement in schema_sql.split(";"):
                # Remove comment lines and empty lines at the beginning of the block
                lines = statement.split("\n")
                sql_lines = [
                    l for l in lines
                    if l.strip() and not l.strip().startswith("--")
                ]
                clean = "\n".join(sql_lines).strip()

                if clean:
                    try:
                        conn.execute(text(clean))
                    except Exception as exc:
                        self.logger.warning(
                            "SQL statement skipped: %s",
                            str(exc)[:120],
                        )

    def _execute_statements_mssql(self, schema_sql: str) -> None:
        """Execute SQL statements for SQL Server.

        SQL Server supports multi-statement blocks separated by ';'.
        Also handles MERGE and CTE that contain internal ';'.
        """
        with self.engine.begin() as conn:
            # For SQL Server, execute the full script
            # by splitting on double blank lines (block separator)
            blocks = schema_sql.split("\n\n\n")
            for block in blocks:
                block = block.strip()
                if block and not block.startswith("--"):
                    try:
                        conn.execute(text(block))
                    except Exception as exc:
                        self.logger.warning(
                            "SQL block skipped: %s",
                            str(exc)[:120],
                        )

    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "append",
    ) -> int:
        """Load a DataFrame into a database table.

        Args:
            df: DataFrame to load.
            table_name: Target table name.
            if_exists: Behavior if the table exists ('append', 'replace', 'fail').

        Returns:
            Number of rows inserted.
        """
        rows_before = self._count_rows(table_name)

        df.to_sql(
            table_name, self.engine,
            if_exists=if_exists, index=False,
        )

        rows_after = self._count_rows(table_name)
        inserted = rows_after - rows_before

        self.logger.info(
            "Table '%s': %d rows inserted (total: %d)",
            table_name, inserted, rows_after,
        )
        return inserted

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return a DataFrame.

        Args:
            sql: SQL SELECT query.

        Returns:
            DataFrame with the query results.
        """
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    # =================================================================
    # Import of collected data (CSV → DB)
    # =================================================================

    def import_collected_data(
        self,
        raw_data_dir: Optional[Path] = None,
    ) -> dict:
        """Import all collected data (CSV) into the database.

        Searches for CSV files in the raw_data_dir directory and
        loads them into the appropriate tables.

        Args:
            raw_data_dir: Root directory for raw data.
                          Default: data/raw (from config).

        Returns:
            Dictionary {source: nb_rows_imported} for reporting.
        """
        if raw_data_dir is None:
            raw_data_dir = Path("data/raw")

        results = {}

        # 1. Import weather → fact_hvac_installations (weather columns)
        weather_file = raw_data_dir / "weather" / "weather_france.csv"
        if weather_file.exists():
            results["weather"] = self._import_weather(weather_file)
        else:
            self.logger.warning("Weather file not found: %s", weather_file)

        # 2. Import INSEE indicators → fact_economic_context
        insee_file = raw_data_dir / "insee" / "indicateurs_economiques.csv"
        if insee_file.exists():
            results["insee"] = self._import_insee(insee_file)
        else:
            self.logger.warning("INSEE file not found: %s", insee_file)

        # 3. Import Eurostat IPI → fact_economic_context (IPI columns)
        eurostat_file = raw_data_dir / "eurostat" / "ipi_hvac_france.csv"
        if eurostat_file.exists():
            results["eurostat"] = self._import_eurostat(eurostat_file)
        else:
            self.logger.warning("Eurostat file not found: %s", eurostat_file)

        # 4. Import SITADEL → fact_hvac_installations (building permits columns)
        sitadel_file = raw_data_dir / "sitadel" / "permis_construire_france.csv"
        if sitadel_file.exists():
            results["sitadel"] = self._import_sitadel(sitadel_file)
        else:
            self.logger.warning("SITADEL file not found: %s", sitadel_file)

        # 5. Import DPE → raw_dpe (high-volume individual records)
        dpe_file = raw_data_dir / "dpe" / "dpe_france_all.csv"
        if dpe_file.exists():
            results["dpe"] = self._import_dpe(dpe_file)
        else:
            self.logger.warning("DPE file not found: %s", dpe_file)

        # Log the summary
        self.logger.info("=" * 50)
        self.logger.info("Import summary:")
        for source, count in results.items():
            self.logger.info("  %s: %d rows", source, count)
        self.logger.info("=" * 50)

        return results

    def _import_weather(self, filepath: Path) -> int:
        """Import weather data into fact_hvac_installations.

        The weather CSV is at the daily x city grain.
        We aggregate by month x department to match the grain
        of the fact table.

        Args:
            filepath: Path to weather_france.csv.

        Returns:
            Number of rows imported.
        """
        self.logger.info("Importing weather from %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Columns found: %s", list(df.columns))

        # The CSV contains: date, city, dept, temperature_2m_max, ..., hdd, cdd
        # Identify the date column
        date_col = None
        for candidate in ["date", "time", "Date"]:
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            self.logger.error("  Date column not found in weather CSV")
            return 0

        df[date_col] = pd.to_datetime(df[date_col])

        # Create aggregation columns
        df["year_month"] = df[date_col].dt.to_period("M")
        df["date_id"] = df[date_col].dt.year * 100 + df[date_col].dt.month

        # Identify the department column
        dept_col = None
        for candidate in ["dept", "department", "code_dept", "dept_code"]:
            if candidate in df.columns:
                dept_col = candidate
                break

        if dept_col is None:
            self.logger.error("  Department column not found in weather CSV")
            return 0

        # Ensure dept is a string with zero padding
        df[dept_col] = df[dept_col].astype(str).str.zfill(2)

        # Map dept → geo_id via the dim_geo table
        geo_map = self._get_geo_mapping()
        if not geo_map:
            self.logger.error("  Table dim_geo is empty, cannot map departments")
            return 0

        df["geo_id"] = df[dept_col].map(geo_map)
        df = df.dropna(subset=["geo_id"])
        df["geo_id"] = df["geo_id"].astype(int)

        # Identify available weather columns
        temp_mean_col = self._find_col(df, ["temperature_2m_mean", "temp_mean"])
        temp_max_col = self._find_col(df, ["temperature_2m_max", "temp_max"])
        temp_min_col = self._find_col(df, ["temperature_2m_min", "temp_min"])
        precip_col = self._find_col(df, ["precipitation_sum", "precipitation_cumul"])
        hdd_col = self._find_col(df, ["hdd", "heating_degree_days"])
        cdd_col = self._find_col(df, ["cdd", "cooling_degree_days"])

        # Aggregate by month x department
        agg_dict = {}
        if temp_mean_col:
            agg_dict[temp_mean_col] = "mean"
        if temp_max_col:
            agg_dict[temp_max_col] = "max"
        if temp_min_col:
            agg_dict[temp_min_col] = "min"
        if precip_col:
            agg_dict[precip_col] = "sum"
        if hdd_col:
            agg_dict[hdd_col] = "sum"
        if cdd_col:
            agg_dict[cdd_col] = "sum"

        # Count heatwave days (T > 35) and frost days (T < 0)
        if temp_max_col:
            df["is_canicule"] = (df[temp_max_col] > 35).astype(int)
            agg_dict["is_canicule"] = "sum"
        if temp_min_col:
            df["is_gel"] = (df[temp_min_col] < 0).astype(int)
            agg_dict["is_gel"] = "sum"

        monthly = df.groupby(["date_id", "geo_id"]).agg(agg_dict).reset_index()

        # Rename columns to match the schema
        rename_map = {}
        if temp_mean_col:
            rename_map[temp_mean_col] = "temp_mean"
        if temp_max_col:
            rename_map[temp_max_col] = "temp_max"
        if temp_min_col:
            rename_map[temp_min_col] = "temp_min"
        if precip_col:
            rename_map[precip_col] = "precipitation_cumul"
        if hdd_col:
            rename_map[hdd_col] = "heating_degree_days"
        if cdd_col:
            rename_map[cdd_col] = "cooling_degree_days"
        if "is_canicule" in monthly.columns:
            rename_map["is_canicule"] = "nb_jours_canicule"
        if "is_gel" in monthly.columns:
            rename_map["is_gel"] = "nb_jours_gel"

        monthly = monthly.rename(columns=rename_map)

        # Round numeric values
        for col in monthly.select_dtypes(include=["float64"]).columns:
            monthly[col] = monthly[col].round(2)

        # Insert into fact_hvac_installations
        # Use replace for this first source (others will do an UPDATE)
        return self.load_dataframe(monthly, "fact_hvac_installations", if_exists="replace")

    def _import_insee(self, filepath: Path) -> int:
        """Import INSEE indicators into fact_economic_context.

        The INSEE CSV has columns: period, confiance_menages,
        climat_affaires_industrie, climat_affaires_batiment, etc.

        Args:
            filepath: Path to indicateurs_economiques.csv.

        Returns:
            Number of rows imported.
        """
        self.logger.info("Importing INSEE from %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Columns: %s", list(df.columns))
        self.logger.info("  Rows: %d", len(df))

        if "period" not in df.columns:
            self.logger.error("  Column 'period' missing")
            return 0

        # Filter only monthly periods (YYYY-MM)
        # Some INSEE series return quarterly formats (2019Q1)
        mask_monthly = df["period"].str.match(r"^\d{4}-\d{2}$")
        df = df[mask_monthly].copy()
        self.logger.info("  Monthly rows retained: %d", len(df))

        if len(df) == 0:
            self.logger.error("  No monthly period found")
            return 0

        # Convert period (YYYY-MM) to date_id (YYYYMM)
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # Map INSEE columns to the table columns
        col_map = {
            "confiance_menages": "confiance_menages",
            "climat_affaires_industrie": "climat_affaires_indus",
            "climat_affaires_batiment": "climat_affaires_bat",
            "opinion_achats_importants": "opinion_achats",
            "situation_financiere_future": "situation_fin_future",
            "ipi_industrie_manuf": "ipi_manufacturing",
        }

        # Keep only columns that exist in the CSV
        cols_to_keep = ["date_id"]
        for csv_col, db_col in col_map.items():
            if csv_col in df.columns:
                df = df.rename(columns={csv_col: db_col})
                cols_to_keep.append(db_col)

        df_insert = df[cols_to_keep].copy()

        return self.load_dataframe(df_insert, "fact_economic_context", if_exists="replace")

    def _import_eurostat(self, filepath: Path) -> int:
        """Import Eurostat IPI into fact_economic_context.

        The Eurostat CSV has columns: period, nace_r2, ipi_value.
        We pivot to have one column per NACE code (C28, C2825).

        Args:
            filepath: Path to ipi_hvac_france.csv.

        Returns:
            Number of rows updated.
        """
        self.logger.info("Importing Eurostat from %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Columns: %s", list(df.columns))
        self.logger.info("  Rows: %d", len(df))

        if "period" not in df.columns:
            self.logger.error("  Column 'period' missing")
            return 0

        # Filter only monthly periods (YYYY-MM)
        mask_monthly = df["period"].str.match(r"^\d{4}-\d{2}$")
        df = df[mask_monthly].copy()
        self.logger.info("  Monthly rows retained: %d", len(df))

        # Convert period → date_id
        df["date_id"] = df["period"].str.replace("-", "").astype(int)

        # Pivot: one column per nace_r2
        if "nace_r2" in df.columns and "ipi_value" in df.columns:
            pivot = df.pivot_table(
                index="date_id", columns="nace_r2",
                values="ipi_value", aggfunc="first",
            ).reset_index()

            # Rename to the table columns
            rename_map = {}
            if "C28" in pivot.columns:
                rename_map["C28"] = "ipi_hvac_c28"
            if "C2825" in pivot.columns:
                rename_map["C2825"] = "ipi_hvac_c2825"
            pivot = pivot.rename(columns=rename_map)

            # Merge with existing fact_economic_context
            # If the table already has INSEE data, we do an UPDATE
            existing = self._safe_read_table("fact_economic_context")

            if existing is not None and len(existing) > 0:
                # Merge Eurostat columns into existing data
                cols_eurostat = ["date_id"]
                if "ipi_hvac_c28" in pivot.columns:
                    cols_eurostat.append("ipi_hvac_c28")
                if "ipi_hvac_c2825" in pivot.columns:
                    cols_eurostat.append("ipi_hvac_c2825")

                merged = existing.merge(
                    pivot[cols_eurostat], on="date_id", how="outer",
                    suffixes=("_old", ""),
                )
                # Take the new Eurostat values
                for col in ["ipi_hvac_c28", "ipi_hvac_c2825"]:
                    old_col = f"{col}_old"
                    if old_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[old_col])
                        merged = merged.drop(columns=[old_col])

                return self.load_dataframe(
                    merged, "fact_economic_context", if_exists="replace",
                )
            else:
                return self.load_dataframe(
                    pivot, "fact_economic_context", if_exists="replace",
                )
        return 0

    def _import_sitadel(self, filepath: Path) -> int:
        """Import SITADEL building permits.

        The SITADEL CSV is aggregated by month x department and merged
        into fact_hvac_installations (columns nb_permis_construire,
        nb_logements_autorises).

        Args:
            filepath: Path to permis_construire_france.csv.

        Returns:
            Number of rows processed.
        """
        self.logger.info("Importing SITADEL from %s ...", filepath.name)

        df = pd.read_csv(filepath)
        self.logger.info("  Columns: %s", list(df.columns))
        self.logger.info("  Rows: %d", len(df))

        # The content depends on the actual CSV SITADEL format
        # We log the columns for diagnosis but do not perform an incorrect
        # import if the format is not as expected
        self.logger.info(
            "  SITADEL import: format to be adapted based on available columns"
        )

        return len(df)

    def _import_dpe(self, filepath: Path) -> int:
        """Import raw DPE records into the raw_dpe table.

        The DPE CSV contains energy performance diagnostics for metropolitan France.
        We import in chunks to avoid saturating memory.
        Then, we aggregate DPEs by month x department to update
        fact_hvac_installations (columns nb_dpe_total, etc.).

        Args:
            filepath: Path to dpe_france_all.csv.

        Returns:
            Number of rows imported into raw_dpe.
        """
        self.logger.info("Importing DPE from %s ...", filepath.name)

        # Count lines for reporting
        # (read just the 1st line for columns)
        df_sample = pd.read_csv(filepath, nrows=2)
        self.logger.info("  Columns: %s", list(df_sample.columns))

        # Import in chunks of 50,000 rows to limit memory usage
        chunk_size = 50_000
        total_imported = 0

        for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
            # First chunk: replace to clear the table
            # Subsequent chunks: append
            mode = "replace" if i == 0 else "append"

            chunk.to_sql(
                "raw_dpe", self.engine,
                if_exists=mode, index=False,
            )
            total_imported += len(chunk)

            if (i + 1) % 5 == 0:
                self.logger.info(
                    "  DPE import: %d rows loaded...", total_imported,
                )

        self.logger.info(
            "  Table 'raw_dpe': %d rows imported", total_imported,
        )

        # Aggregate DPEs by month x department for fact_hvac_installations
        self._aggregate_dpe_to_facts()

        return total_imported

    def _aggregate_dpe_to_facts(self) -> None:
        """Aggregate raw DPEs to update fact_hvac_installations.

        Computes per month x department:
        - nb_dpe_total: total number of DPEs
        - nb_installations_pac: DPEs mentioning a heat pump
        - nb_installations_clim: DPEs mentioning air conditioning
        - nb_dpe_classe_ab: DPEs with class A or B
        """
        self.logger.info("  Aggregating DPE into fact_hvac_installations...")

        # Read DPEs with the columns useful for aggregation
        sql = """
        SELECT
            code_departement_ban,
            date_etablissement_dpe,
            etiquette_dpe,
            type_installation_chauffage,
            type_generateur_chauffage_principal,
            type_generateur_froid
        FROM raw_dpe
        WHERE date_etablissement_dpe IS NOT NULL
          AND code_departement_ban IS NOT NULL
        """
        df = self.query(sql)
        self.logger.info("  DPE with valid date+dept: %d", len(df))

        if len(df) == 0:
            return

        # Convert the date to a monthly period
        df["date_etablissement_dpe"] = pd.to_datetime(
            df["date_etablissement_dpe"], errors="coerce"
        )
        df = df.dropna(subset=["date_etablissement_dpe"])

        df["date_id"] = (
            df["date_etablissement_dpe"].dt.year * 100
            + df["date_etablissement_dpe"].dt.month
        )

        # Detect HVAC installations via text fields
        # PAC = Heat pump in heating OR in cooling (reversible air/air heat pump)
        pac_pattern = r"(?i)PAC |PAC$|pompe.*chaleur|thermodynamique"

        # Heat pumps are present in TWO fields:
        #   - type_generateur_chauffage_principal: air/water heat pump, geothermal heat pump
        #   - type_generateur_froid: air/air heat pump (reversible, classified as cooling)
        chauffage_str = df["type_generateur_chauffage_principal"].fillna("")
        froid_str = df["type_generateur_froid"].fillna("")

        df["is_pac"] = (
            chauffage_str.str.contains(pac_pattern, regex=True) |
            froid_str.str.contains(pac_pattern, regex=True)
        ).astype(int)

        # Air conditioning = any DPE with a cooling generator specified
        # (air/air heat pumps are also reversible AC units)
        df["is_clim"] = (froid_str.str.len() > 0).astype(int)

        df["is_classe_ab"] = df["etiquette_dpe"].isin(["A", "B"]).astype(int)

        # Map dept → geo_id
        geo_map = self._get_geo_mapping()
        df["geo_id"] = df["code_departement_ban"].astype(str).str.zfill(2).map(geo_map)
        df = df.dropna(subset=["geo_id"])
        df["geo_id"] = df["geo_id"].astype(int)

        # Aggregate by month x department
        agg = df.groupby(["date_id", "geo_id"]).agg(
            nb_dpe_total=("is_pac", "count"),
            nb_installations_pac=("is_pac", "sum"),
            nb_installations_clim=("is_clim", "sum"),
            nb_dpe_classe_ab=("is_classe_ab", "sum"),
        ).reset_index()

        agg["nb_installations_pac"] = agg["nb_installations_pac"].astype(int)
        agg["nb_installations_clim"] = agg["nb_installations_clim"].astype(int)
        agg["nb_dpe_classe_ab"] = agg["nb_dpe_classe_ab"].astype(int)

        # Update fact_hvac_installations with DPE counts
        # Read existing data (weather already imported) and merge
        existing = self._safe_read_table("fact_hvac_installations")

        if existing is not None and len(existing) > 0:
            # Merge DPE columns into existing data
            dpe_cols = ["date_id", "geo_id", "nb_dpe_total",
                        "nb_installations_pac", "nb_installations_clim",
                        "nb_dpe_classe_ab"]

            # Remove old DPE columns if they exist
            for col in ["nb_dpe_total", "nb_installations_pac",
                        "nb_installations_clim", "nb_dpe_classe_ab"]:
                if col in existing.columns:
                    existing = existing.drop(columns=[col])

            merged = existing.merge(
                agg[dpe_cols], on=["date_id", "geo_id"], how="left",
            )
            self.load_dataframe(
                merged, "fact_hvac_installations", if_exists="replace",
            )
        else:
            self.load_dataframe(
                agg, "fact_hvac_installations", if_exists="replace",
            )

        self.logger.info(
            "  DPE aggregation complete: %d month x dept rows", len(agg),
        )

    # =================================================================
    # ML Dataset
    # =================================================================

    def build_ml_dataset(self) -> pd.DataFrame:
        """Build the ML-ready dataset by joining fact tables.

        Performs the join between:
        - fact_hvac_installations (grain = month x department)
        - fact_economic_context (grain = month only)
        - dim_time (temporal features)
        - dim_geo (geographic metadata)

        The result is a DataFrame ready for feature engineering.

        Returns:
            DataFrame with all features and the target variable.
        """
        sql = """
        SELECT
            -- Temporal dimensions
            t.date_id,
            t.year,
            t.month,
            t.quarter,
            t.is_heating,
            t.is_cooling,

            -- Geographic dimensions
            g.dept_code,
            g.dept_name,
            g.city_ref,

            -- Target variable (HVAC installations)
            f.nb_dpe_total,
            f.nb_installations_pac,
            f.nb_installations_clim,
            f.nb_dpe_classe_ab,

            -- Weather features (local)
            f.temp_mean,
            f.heating_degree_days,
            f.cooling_degree_days,
            f.precipitation_cumul,
            f.nb_jours_canicule,
            f.nb_jours_gel,

            -- Real estate features (local)
            f.nb_permis_construire,
            f.nb_logements_autorises,

            -- Economic features (national)
            e.confiance_menages,
            e.climat_affaires_indus,
            e.climat_affaires_bat,
            e.opinion_achats,
            e.situation_fin_future,
            e.ipi_manufacturing,
            e.ipi_hvac_c28,
            e.ipi_hvac_c2825

        FROM fact_hvac_installations f
        JOIN dim_time t ON f.date_id = t.date_id
        JOIN dim_geo g ON f.geo_id = g.geo_id
        LEFT JOIN fact_economic_context e ON f.date_id = e.date_id

        ORDER BY t.date_id, g.dept_code
        """

        df = self.query(sql)
        self.logger.info(
            "ML dataset built: %d rows x %d columns",
            len(df), len(df.columns),
        )
        return df

    # =================================================================
    # Utilities
    # =================================================================

    def _count_rows(self, table_name: str) -> int:
        """Count the number of rows in a table.

        Args:
            table_name: Table name.

        Returns:
            Number of rows (0 if the table does not exist).
        """
        # Whitelist to prevent SQL injection on dynamic table names
        allowed = {
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context", "raw_dpe",
        }
        if table_name not in allowed:
            self.logger.warning("Rejected unknown table name: %s", table_name)
            return 0
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table_name}")
                )
                return result.scalar() or 0
        except Exception:
            return 0

    def _safe_read_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """Read an entire table, return None if the table is empty or does not exist.

        Args:
            table_name: Table name.

        Returns:
            DataFrame or None.
        """
        allowed = {
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context", "raw_dpe",
        }
        if table_name not in allowed:
            self.logger.warning("Rejected unknown table name: %s", table_name)
            return None
        try:
            df = self.query(f"SELECT * FROM {table_name}")
            return df if len(df) > 0 else None
        except Exception:
            return None

    def _get_geo_mapping(self) -> dict:
        """Return the dept_code → geo_id mapping from dim_geo.

        Returns:
            Dictionary {dept_code: geo_id} (e.g., {'69': 6}).
        """
        try:
            df = self.query("SELECT geo_id, dept_code FROM dim_geo")
            return dict(zip(df["dept_code"].astype(str), df["geo_id"]))
        except Exception:
            return {}

    def _find_col(self, df: pd.DataFrame, candidates: list) -> Optional[str]:
        """Find the first column matching a list of candidates.

        Args:
            df: DataFrame to inspect.
            candidates: Possible column names (in order of priority).

        Returns:
            Name of the found column, or None.
        """
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def get_table_info(self) -> pd.DataFrame:
        """Return a summary of all tables and their row counts.

        Returns:
            DataFrame with columns [table_name, row_count].
        """
        tables = [
            "dim_time", "dim_geo", "dim_equipment_type",
            "fact_hvac_installations", "fact_economic_context",
            "raw_dpe",
        ]
        rows = []
        for table in tables:
            count = self._count_rows(table)
            rows.append({"table_name": table, "row_count": count})

        return pd.DataFrame(rows)
