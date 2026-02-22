# Database Architecture -- HVAC Project

> **Project** : HVAC market analysis in the Auvergne-Rhone-Alpes region
> **Default engine** : SQLite (compatible with PostgreSQL and SQL Server)
> **ORM** : SQLAlchemy | **Schema** : Star schema -- month x department grain

---

## 1. Documented Star Schema

### 1.1 Mermaid Diagram

```mermaid
erDiagram
    dim_time ||--o{ fact_hvac_installations : "date_id"
    dim_geo ||--o{ fact_hvac_installations : "geo_id"
    dim_time ||--o| fact_economic_context : "date_id"

    dim_time {
        INT date_id PK "YYYYMM"
        INT year "2019-2025"
        INT month "1-12"
        INT quarter "1-4"
        BOOLEAN is_heating "oct-mar"
        BOOLEAN is_cooling "jun-sep"
    }

    dim_geo {
        INT geo_id PK "Auto-increment"
        VARCHAR dept_code UK "Dept code"
        VARCHAR dept_name "Department name"
        VARCHAR city_ref "Weather city"
        DECIMAL latitude "Coordinate"
        DECIMAL longitude "Coordinate"
    }

    dim_equipment_type {
        INT equip_type_id PK "Auto-increment"
        VARCHAR label "Air-to-air HP etc"
        VARCHAR categorie "heating cooling mixed"
        VARCHAR energie_source "elec gas oil wood"
        BOOLEAN is_renewable "Renewable"
    }

    fact_hvac_installations {
        INT fact_id PK
        INT date_id FK "to dim_time"
        INT geo_id FK "to dim_geo"
        INT nb_dpe_total "Total DPE"
        INT nb_installations_pac "DPE with HP"
        INT nb_installations_clim "DPE with AC"
        DECIMAL temp_mean "Mean temp"
        DECIMAL heating_degree_days "HDD"
        DECIMAL cooling_degree_days "CDD"
        INT nb_permis_construire "Permits"
        DECIMAL conso_elec_mwh "Elec consumption"
    }

    fact_economic_context {
        INT date_id PK "FK to dim_time"
        DECIMAL confiance_menages "INSEE base 100"
        DECIMAL climat_affaires_bat "Building"
        DECIMAL ipi_hvac_c28 "Eurostat C28"
        DECIMAL ipi_hvac_c2825 "Eurostat C2825"
    }
```

### 1.2 Table Descriptions

#### dim_time (84 rows)

| Column | Type | Description |
|---------|------|-------------|
| `date_id` | INTEGER PK | Format YYYYMM (e.g.: 202401 = January 2024) |
| `year` | INTEGER NOT NULL | Year (2019-2025), CHECK BETWEEN 2015 AND 2030 |
| `month` | INTEGER NOT NULL | Month (1-12) |
| `quarter` | INTEGER NOT NULL | Quarter (1-4) |
| `is_heating` | BOOLEAN NOT NULL | Heating season (October to March) |
| `is_cooling` | BOOLEAN NOT NULL | Cooling season (June to September) |

#### dim_geo (8 rows -- AURA departments)

| Column | Type | Description |
|---------|------|-------------|
| `geo_id` | INTEGER PK AUTO | Unique identifier |
| `dept_code` | VARCHAR(3) UNIQUE | Department code (01, 07, 26, 38, 42, 69, 73, 74) |
| `dept_name` | VARCHAR(50) | Department name |
| `city_ref` | VARCHAR(50) | Weather reference city (Lyon, Grenoble...) |
| `latitude`, `longitude` | DECIMAL(7,4) | Coordinates of the reference city |
| `region_code` | VARCHAR(3) DEFAULT '84' | AURA region code |

#### dim_equipment_type (10 rows -- reference table)

Reference table of HVAC equipment (air-to-air heat pump, gas boiler, etc.). Not linked by FK to the fact table because facts are aggregated at the month x department grain, not by individual equipment.

#### fact_hvac_installations (672 rows -- main fact table)

| Column | Type | Description |
|---------|------|-------------|
| `fact_id` | INTEGER PK AUTO | Unique identifier |
| `date_id` | INTEGER FK NOT NULL | To dim_time |
| `geo_id` | INTEGER FK NOT NULL | To dim_geo |
| `nb_dpe_total` | INTEGER | Total DPE for the month/department |
| `nb_installations_pac` | INTEGER | DPE mentioning a heat pump |
| `nb_installations_clim` | INTEGER | DPE mentioning air conditioning |
| `nb_dpe_classe_ab` | INTEGER | DPE class A or B |
| `temp_mean/max/min` | DECIMAL(5,2) | Monthly temperatures (C) |
| `heating_degree_days` | DECIMAL(8,2) | Monthly HDD (base 18C) |
| `cooling_degree_days` | DECIMAL(8,2) | Monthly CDD (base 18C) |
| `nb_jours_canicule/gel` | INTEGER | Extreme weather days |
| `nb_permis_construire` | INTEGER | Granted building permits |
| `conso_elec/gaz_mwh` | DECIMAL(12,2) | Energy consumption (MWh) |

Constraint: `UNIQUE(date_id, geo_id)` -- one row per month x department.

#### fact_economic_context (84 rows -- national indicators)

| Column | Type | Description |
|---------|------|-------------|
| `date_id` | INTEGER PK, FK | To dim_time (YYYYMM) |
| `confiance_menages` | DECIMAL(6,2) | INSEE composite index (base 100) |
| `climat_affaires_indus/bat` | DECIMAL(6,2) | Business climate industry/building |
| `opinion_achats` | DECIMAL(6,2) | Major purchase opportunity (balance) |
| `ipi_manufacturing` | DECIMAL(6,2) | Manufacturing IPI |
| `ipi_hvac_c28/c2825` | DECIMAL(6,2) | HVAC machinery and equipment IPI (Eurostat) |

### 1.3 Grain and Justification

**Grain**: the main fact table has a grain of **month x department** (date_id + geo_id). The economic table has a grain of **month only** (national data, not duplicated across the 8 departments).

**Why a star schema for HVAC analysis?**

1. **Simple joins**: facts + 2-3 dimensions, no snowflake cascade.
2. **OLAP performance**: direct aggregations (SUM, AVG, COUNT) with GROUP BY.
3. **Local/national separation**: avoids artificial duplication of INSEE indicators across 8 departments.
4. **ML compatibility**: `build_ml_dataset()` produces an ML-ready DataFrame through natural join.
5. **Scalability**: adding regions = adding rows to `dim_geo`.

---

## 2. SQLite to PostgreSQL Migration

### 2.1 Type Differences

| Concept | SQLite | PostgreSQL |
|---------|--------|------------|
| Boolean | BOOLEAN (stores 0/1) | Native BOOLEAN |
| Auto-increment | AUTOINCREMENT | SERIAL |
| VARCHAR | No enforced limit | Strict limit |
| DECIMAL | Stored as REAL | Exact precision |
| Date | ISO 8601 text | Native DATE |
| Upsert | INSERT OR IGNORE | INSERT ... ON CONFLICT DO NOTHING |
| Concurrency | File locking | MVCC, concurrent access |

The `schema.sql` file is already PostgreSQL-compatible (native BOOLEAN, VARCHAR, DECIMAL). SQLAlchemy handles the translation from AUTOINCREMENT to SERIAL via `to_sql()`.

### 2.2 Steps to Switch

```bash
# 1. Install the driver
pip install psycopg2-binary

# 2. Create the PostgreSQL database
psql -U postgres -c "CREATE DATABASE hvac_market;"
psql -U postgres -c "CREATE USER hvac_user WITH PASSWORD 'mdp_securise';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE hvac_market TO hvac_user;"
```

```env
# 3. Update .env
DB_TYPE=postgresql
DATABASE_URL=postgresql://hvac_user:mdp_securise@localhost:5432/hvac_market
```

```python
# 4. Initialize and import
from src.database.db_manager import DatabaseManager
db = DatabaseManager("postgresql://hvac_user:mdp@localhost:5432/hvac_market")
db.init_database()          # Uses schema.sql (PostgreSQL-compatible)
db.import_collected_data()  # Imports all collected CSVs
```

### 2.3 Existing Data Migration Script

```python
"""SQLite to PostgreSQL migration."""
import pandas as pd
from sqlalchemy import create_engine

src = create_engine("sqlite:///data/hvac_market.db")
dst = create_engine("postgresql://hvac_user:mdp@localhost:5432/hvac_market")

# Order respecting FKs
for table in ["dim_time", "dim_geo", "dim_equipment_type",
              "fact_hvac_installations", "fact_economic_context", "raw_dpe"]:
    df = pd.read_sql(f"SELECT * FROM {table}", src)
    df.to_sql(table, dst, if_exists="append", index=False)
    print(f"{table} : {len(df)} rows migrated")
```

---

## 3. Advanced OLAP Queries

Queries with **window functions**, compatible with SQLite 3.25+ and PostgreSQL.

### 3.1 Department Ranking by Heat Pump Installations

```sql
SELECT t.year, t.month, g.dept_name, f.nb_installations_pac,
    RANK() OVER (
        PARTITION BY f.date_id ORDER BY f.nb_installations_pac DESC
    ) AS rang_pac
FROM fact_hvac_installations f
JOIN dim_time t ON f.date_id = t.date_id
JOIN dim_geo g ON f.geo_id = g.geo_id
WHERE f.nb_installations_pac IS NOT NULL
ORDER BY t.year, t.month, rang_pac;
```

**Explanation**: `RANK()` partitions by month and ranks departments by decreasing number of heat pumps. Rhone (69) and Isere (38) generally dominate due to their population density.

### 3.2 Three-Month Moving Average by Department

```sql
SELECT t.year, t.month, g.dept_name, f.nb_installations_pac,
    ROUND(AVG(f.nb_installations_pac) OVER (
        PARTITION BY f.geo_id ORDER BY f.date_id
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 1) AS moyenne_mobile_3m
FROM fact_hvac_installations f
JOIN dim_time t ON f.date_id = t.date_id
JOIN dim_geo g ON f.geo_id = g.geo_id
WHERE f.nb_installations_pac IS NOT NULL
ORDER BY g.dept_name, t.year, t.month;
```

**Explanation**: the 3-row sliding window smooths seasonal peaks to reveal the underlying heat pump market trend.

### 3.3 Year-over-Year (YoY) Variation

```sql
SELECT t.year, t.month, g.dept_name,
    f.nb_installations_pac AS pac_courant,
    LAG(f.nb_installations_pac, 12) OVER (
        PARTITION BY f.geo_id ORDER BY f.date_id
    ) AS pac_annee_prec,
    ROUND(
        (CAST(f.nb_installations_pac AS FLOAT)
         - LAG(f.nb_installations_pac, 12) OVER (
             PARTITION BY f.geo_id ORDER BY f.date_id))
        / NULLIF(LAG(f.nb_installations_pac, 12) OVER (
            PARTITION BY f.geo_id ORDER BY f.date_id), 0)
        * 100, 1
    ) AS variation_yoy_pct
FROM fact_hvac_installations f
JOIN dim_time t ON f.date_id = t.date_id
JOIN dim_geo g ON f.geo_id = g.geo_id
WHERE f.nb_installations_pac IS NOT NULL
ORDER BY g.dept_name, t.year, t.month;
```

**Explanation**: `LAG(..., 12)` retrieves the value from the same month of the previous year. `NULLIF(..., 0)` prevents division by zero. The years 2021-2023 show significant growth (MaPrimeRenov', energy prices).

### 3.4 Market Share by Equipment Type

```sql
SELECT t.year, g.dept_name,
    SUM(f.nb_installations_pac) AS total_pac,
    SUM(f.nb_dpe_total) AS total_dpe,
    ROUND(CAST(SUM(f.nb_installations_pac) AS FLOAT)
        / NULLIF(SUM(f.nb_dpe_total), 0) * 100, 2) AS part_pac_pct,
    ROUND(CAST(SUM(f.nb_installations_clim) AS FLOAT)
        / NULLIF(SUM(f.nb_dpe_total), 0) * 100, 2) AS part_clim_pct,
    ROUND(CAST(SUM(f.nb_installations_pac) AS FLOAT)
        / NULLIF(SUM(SUM(f.nb_installations_pac)) OVER (PARTITION BY t.year), 0)
        * 100, 2) AS part_pac_regionale_pct
FROM fact_hvac_installations f
JOIN dim_time t ON f.date_id = t.date_id
JOIN dim_geo g ON f.geo_id = g.geo_id
WHERE f.nb_dpe_total IS NOT NULL
GROUP BY t.year, g.dept_name
ORDER BY t.year, part_pac_pct DESC;
```

**Explanation**: combines GROUP BY and window function (`SUM() OVER`) to calculate each department's share in the annual regional total. Southern departments (Drome, Ardeche) have a higher air conditioning share.

### 3.5 Weather/Installations Correlation with CTE

```sql
WITH stats_dept AS (
    SELECT g.dept_name,
        AVG(f.heating_degree_days) AS hdd_moyen,
        AVG(f.cooling_degree_days) AS cdd_moyen,
        AVG(f.nb_installations_pac) AS pac_moyen,
        AVG(f.nb_installations_clim) AS clim_moyen
    FROM fact_hvac_installations f
    JOIN dim_geo g ON f.geo_id = g.geo_id
    WHERE f.nb_installations_pac IS NOT NULL
      AND f.heating_degree_days IS NOT NULL
    GROUP BY g.dept_name
),
classement AS (
    SELECT dept_name, hdd_moyen, cdd_moyen, pac_moyen, clim_moyen,
        RANK() OVER (ORDER BY hdd_moyen DESC) AS rang_hdd,
        RANK() OVER (ORDER BY cdd_moyen DESC) AS rang_cdd,
        RANK() OVER (ORDER BY pac_moyen DESC) AS rang_pac,
        RANK() OVER (ORDER BY clim_moyen DESC) AS rang_clim
    FROM stats_dept
)
SELECT dept_name,
    ROUND(hdd_moyen, 0) AS hdd_moyen, ROUND(cdd_moyen, 0) AS cdd_moyen,
    ROUND(pac_moyen, 0) AS pac_moyen, ROUND(clim_moyen, 0) AS clim_moyen,
    rang_hdd, rang_cdd, rang_pac, rang_clim,
    ABS(rang_hdd - rang_pac) AS ecart_hdd_pac,
    ABS(rang_cdd - rang_clim) AS ecart_cdd_clim
FROM classement ORDER BY rang_pac;
```

**Explanation**: two CTEs structure the analysis. The first computes averages by department, the second assigns ranks. If `ecart_hdd_pac` is low, there is a correlation between heating demand and heat pump installations. Mountain departments (Savoie, Haute-Savoie) with high HDD have more heat pumps; southern departments with high CDD dominate in air conditioning.

---

## 4. OLTP vs OLAP vs NoSQL Comparison

### 4.1 Comparative Table for the HVAC Case

| Criterion | OLTP (PostgreSQL 3NF) | OLAP (Star schema) | NoSQL (MongoDB) |
|-----------|----------------------|---------------------|-----------------|
| **Usage** | Real-time transactions | Analysis, reporting, ML | Raw JSON storage |
| **Schema** | Normalized 3NF | Denormalized star | Schema-less |
| **Queries** | Individual INSERT/UPDATE | SELECT + GROUP BY | find(), aggregate() |
| **Joins** | Numerous (10+) | Few (2-3 tables) | None (embedded) |
| **Read perf.** | Average | Excellent | Good |
| **Write perf.** | Excellent | Adequate (batch) | Excellent |
| **Integrity** | Strong (FK, constraints) | Medium (dimension FKs) | None |
| **HVAC case** | Real-time DPE entry | Market analysis + ML | API response cache |

### 4.2 Justification of Current Choice

The **OLAP star schema** is appropriate because the project is oriented toward analysis/prediction, not transactional data entry. Data is collected in batches from APIs (ADEME, INSEE, Eurostat, Open-Meteo). The volume is manageable (672 fact rows). SQLite requires no server; migration to PostgreSQL is trivial.

### 4.3 MongoDB Example: API Response Cache

In a hybrid architecture, MongoDB would store raw JSON API responses before transformation, enabling reprocessing without calling the APIs again.

```json
{
    "_id": "ObjectId(...)",
    "source": "open_meteo",
    "endpoint": "/v1/forecast",
    "collected_at": "2024-06-15T14:30:00Z",
    "parameters": {
        "latitude": 45.76, "longitude": 4.84,
        "start_date": "2024-06-01", "end_date": "2024-06-30"
    },
    "response": {
        "daily": {
            "time": ["2024-06-01", "2024-06-02"],
            "temperature_2m_max": [28.3, 31.1],
            "temperature_2m_min": [15.2, 17.8]
        }
    },
    "http_status": 200,
    "processing_status": "imported"
}
```

### 4.4 MongoDB Python Script

```python
"""MongoDB example for HVAC API response cache."""
from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["hvac_raw_cache"]
collection = db["api_responses"]

# --- Insert ---
doc = {
    "source": "open_meteo",
    "collected_at": datetime.utcnow(),
    "parameters": {"latitude": 45.76, "longitude": 4.84,
                    "start_date": "2024-01-01", "end_date": "2024-01-31"},
    "response": {"daily": {"time": ["2024-01-01"], "temperature_2m_max": [5.3]}},
    "http_status": 200,
    "processing_status": "pending",
}
result = collection.insert_one(doc)
print(f"Document inserted: {result.inserted_id}")

# --- Query: unprocessed responses ---
for doc in collection.find({"source": "open_meteo", "processing_status": "pending"}):
    print(f"To process: {doc['parameters']['start_date']}")
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"processing_status": "imported"}},
    )

# --- Aggregation: count by source ---
pipeline = [
    {"$group": {"_id": "$source", "total": {"$sum": 1},
                "last_collection": {"$max": "$collected_at"}}},
    {"$sort": {"total": -1}},
]
for r in collection.aggregate(pipeline):
    print(f"{r['_id']} : {r['total']} responses")

# --- Indexes ---
collection.create_index([("source", 1), ("processing_status", 1)])
collection.create_index([("collected_at", -1)])
client.close()
```

---

## 5. Optimization and Indexing

### 5.1 Existing Indexes

```sql
-- Main fact table
CREATE INDEX idx_fact_hvac_date     ON fact_hvac_installations(date_id);
CREATE INDEX idx_fact_hvac_geo      ON fact_hvac_installations(geo_id);
CREATE INDEX idx_fact_hvac_date_geo ON fact_hvac_installations(date_id, geo_id);
CREATE INDEX idx_dim_geo_dept       ON dim_geo(dept_code);

-- Raw DPE table
CREATE INDEX idx_raw_dpe_date       ON raw_dpe(date_etablissement_dpe);
CREATE INDEX idx_raw_dpe_dept       ON raw_dpe(code_departement_ban);
CREATE INDEX idx_raw_dpe_date_dept  ON raw_dpe(date_etablissement_dpe, code_departement_ban);
CREATE INDEX idx_raw_dpe_etiquette  ON raw_dpe(etiquette_dpe);
```

### 5.2 Recommended Additional Indexes

```sql
-- Covering index for ML queries
CREATE INDEX idx_fact_hvac_ml ON fact_hvac_installations(
    date_id, geo_id, nb_installations_pac, nb_installations_clim,
    temp_mean, heating_degree_days, cooling_degree_days);

-- Index on raw_dpe for heat pump detection
CREATE INDEX idx_raw_dpe_generateur ON raw_dpe(type_generateur_chauffage_principal);
CREATE INDEX idx_raw_dpe_froid      ON raw_dpe(type_generateur_froid);

-- Partial PostgreSQL index for recent DPEs
CREATE INDEX idx_raw_dpe_recent ON raw_dpe(date_etablissement_dpe)
    WHERE date_etablissement_dpe >= '2022-01-01';
```

### 5.3 Date Partitioning (PostgreSQL)

For `raw_dpe` (~1.4M rows), yearly partitioning reduces the scan:

```sql
CREATE TABLE raw_dpe_partitioned (
    numero_dpe VARCHAR(20) NOT NULL,
    date_etablissement_dpe DATE NOT NULL,
    code_departement_ban VARCHAR(3),
    etiquette_dpe VARCHAR(1),
    type_generateur_chauffage_principal VARCHAR(80),
    type_generateur_froid VARCHAR(80)
) PARTITION BY RANGE (date_etablissement_dpe);

CREATE TABLE raw_dpe_2019 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2019-01-01') TO ('2020-01-01');
CREATE TABLE raw_dpe_2020 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');
CREATE TABLE raw_dpe_2021 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');
CREATE TABLE raw_dpe_2022 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
CREATE TABLE raw_dpe_2023 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
CREATE TABLE raw_dpe_2024 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE raw_dpe_2025 PARTITION OF raw_dpe_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
```

A query filtering on 2023 scans only ~200K rows instead of 1.4M.

### 5.4 Volume Estimates

| Table | Rows | SQLite | PostgreSQL |
|-------|--------|--------|------------|
| `dim_time` | 84 | < 1 KB | < 1 KB |
| `dim_geo` | 8 | < 1 KB | < 1 KB |
| `dim_equipment_type` | 10 | < 1 KB | < 1 KB |
| `fact_hvac_installations` | 672 | ~50 KB | ~80 KB |
| `fact_economic_context` | 84 | ~5 KB | ~10 KB |
| `raw_dpe` | ~1 400 000 | ~500 MB | ~700 MB |
| **Total** | **~1 400 850** | **~500 MB** | **~700 MB** |

The `raw_dpe` table represents 99.9% of the volume. It is the only table where partitioning and advanced indexing have a real impact. The star schema tables (< 100 KB) perform well without extensive optimization. Plan for 1 GB of total disk space (data + indexes) in PostgreSQL production.

---

> **HVAC Market Analysis Project -- AURA Region**
> Star schema | SQLAlchemy | SQLite / PostgreSQL / SQL Server
