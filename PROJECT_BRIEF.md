# Portfolio Project: HVAC Market Analysis in Rhône-Alpes
## Sales Prediction via ML/Deep Learning

**Author**: Patrice — Senior Data Analyst
**Stack**: Python, SQL (local SQLite / SQL Server), Power BI
**Objective**: Portfolio project for Data Analyst job search

---

## 1. Project Vision

Build a complete data pipeline (collection → cleaning → analysis → prediction → dashboard) on the HVAC market in the Auvergne-Rhône-Alpes region, by combining energy, weather, economic, and household confidence data to predict sales/installation volumes of heating and air conditioning equipment.

**Design Principles:**
- Local storage only, CSV outputs and deduplication to avoid duplicates
- Systematic dual backup: `data/raw/` (raw) + `data/processed/` (deduplicated)
- Local SQLite database by default (zero installation required)
- Easy migration to SQL Server for a professional environment
- Portable project: transferable from one machine to another in a few commands

## 2. Technical Architecture

```
hvac-market-analysis/
├── README.md                    # GitHub showcase
├── setup_project.py             # Setup script (init on new machine)
├── requirements.txt             # Core Python dependencies
├── requirements-dl.txt          # Deep Learning dependencies (optional)
├── .env.example                 # Configuration template
├── .env                         # Local configuration (not versioned)
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration (dataclasses)
├── data/
│   ├── raw/                     # Downloaded raw data
│   │   ├── weather/             # Open-Meteo
│   │   ├── insee/               # INSEE BDM
│   │   ├── eurostat/            # Eurostat IPI
│   │   ├── sitadel/             # Building permits
│   │   ├── dpe/                 # DPE ADEME
│   │   └── energy/              # SDES energy
│   ├── processed/               # Cleaned + deduplicated data
│   ├── features/                # Engineered features for ML
│   └── hvac_market.db           # Local SQLite database
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # CLI orchestrator
│   ├── collectors/              # Extensible plugin-based architecture
│   │   ├── __init__.py
│   │   ├── base.py              # BaseCollector + CollectorRegistry
│   │   ├── weather.py           # Open-Meteo (WeatherCollector)
│   │   ├── insee.py             # INSEE BDM (InseeCollector)
│   │   ├── eurostat_col.py      # Eurostat (EurostatCollector)
│   │   ├── sitadel.py           # SITADEL via DiDo API (SitadelCollector)
│   │   └── dpe.py               # DPE ADEME (DpeCollector)
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── clean_data.py        # Source-by-source cleaning (DataCleaner)
│   │   ├── merge_datasets.py    # Multi-source merge → ML dataset (DatasetMerger)
│   │   └── feature_engineering.py  # Advanced features: lags, rolling, interactions (FeatureEngineer)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql           # Star schema SQLite (+ raw_dpe)
│   │   ├── schema_mssql.sql     # SQL Server schema (IDENTITY, BIT, MERGE)
│   │   └── db_manager.py        # CRUD + CSV import + DPE aggregation
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py               # Exploratory analysis
│   │   └── correlation.py       # Correlation studies
│   └── models/
│       ├── __init__.py
│       ├── baseline.py          # Classic ML models
│       ├── deep_learning.py     # LSTM (educational exploration)
│       ├── train.py             # Training pipeline
│       └── evaluate.py          # Metrics & comparison
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_results_analysis.ipynb
├── dashboards/                  # Power BI exports
│   └── screenshots/
├── tests/
│   └── test_collectors/
└── docs/
    ├── data_dictionary.md
    └── methodology.md
```

## 3. Database — Multi-engine Strategy

### 3.1 Principle
The project uses **SQLAlchemy** as an abstraction layer. The database engine is configurable via the `.env` file without modifying any code.

### 3.2 SQLite (default — local development)
```env
DB_TYPE=sqlite
DB_PATH=data/hvac_market.db
```
- Zero installation required (built into Python)
- Single file, portable, copyable
- Perfect for development and demonstration

### 3.3 SQL Server (production / enterprise)
```env
DB_TYPE=mssql
DB_HOST=localhost
DB_PORT=1433
DB_NAME=hvac_market
DB_USER=sa
DB_PASSWORD=YourPassword
```
- Requires `pip install pyodbc` and a SQL Server ODBC driver
- Compatible SQL schema (ANSI standard syntax)
- Migration in 3 steps: install the driver, modify `.env`, rerun `python -m src.pipeline init_db`

### 3.4 PostgreSQL (optional)
```env
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=hvac_market
DB_USER=patrice
DB_PASSWORD=YourPassword
```

## 4. Project Portability

### 4.1 Moving the project to another machine

**Option A — Download data from pCloud (fast, ~5 min)**

Pre-collected data (~1.5 GB) is available for public download:
> **pCloud link**: https://e.pcloud.link/publink/show?code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy

```bash
# 1. Clone the repo
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Dependencies + config
pip install -r requirements.txt
copy .env.example .env         # Windows

# 4. Copy the downloaded data from pCloud into data/
#    (hvac_market.db + raw/ + processed/ + features/)
```

**Option B — Regenerate from APIs (complete, ~1h)**

```bash
# 1. Clone the repo
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Dependencies + config
pip install -r requirements.txt
copy .env.example .env         # Windows

# 4. Collect and process (~1h for DPE ADEME)
python -m src.pipeline all
```

### 4.2 What is versioned (Git) vs what is not
| Versioned | Not versioned (.gitignore) |
|-----------|---------------------------|
| Source code (src/, config/, tests/) | Raw data (data/raw/) |
| requirements.txt, .env.example | Processed data (data/processed/) |
| schema.sql, notebooks/ | .env file (secrets) |
| Default configurations | SQLite database (data/*.db) |
| | Trained models (*.pkl, *.pt) |

### 4.3 Locations and backups

| Location | Content | Size |
|----------|---------|------|
| `E:\Projet Linkedin` | Full project (daily work) | ~1.5 GB |
| `L:\Projet-HVAC-Backup` | Full backup (code + data) | ~1.5 GB |
| `P:\Projet LinkedIn` | Data on pCloud (cloud sync) | ~1.5 GB |
| GitHub (PDUCLOS/Projet-HVAC) | Source code only | ~200 KB |

**Public pCloud link** (pre-collected data):
https://e.pcloud.link/publink/show?code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy

### 4.4 Regenerate all data
```bash
# Rebuild everything from scratch (data + DB + processing)
python -m src.pipeline all

# Or step by step
python -m src.pipeline init_db                        # Create tables
python -m src.pipeline collect --sources weather,insee # Specific sources
python -m src.pipeline collect                         # All sources
python -m src.pipeline import_data                     # CSV → DB
python -m src.pipeline clean                           # Cleaning → data/processed/
python -m src.pipeline merge                           # Merge → data/features/hvac_ml_dataset.csv
python -m src.pipeline features                        # Features → data/features/hvac_features_dataset.csv
python -m src.pipeline process                         # clean + merge + features in one command
```

## 5. Data Sources — Technical Details

### 5.1 DPE (Energy Performance Diagnostic) — ADEME
- **URL (CORRECTED)**: https://data.ademe.fr/datasets/dpe03existant
- **API**: `https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines`
- **Format**: Cursor-paginated JSON (~14M total entries)
- **Scope**: DPE since July 2021
- **Filtering**: departments 01, 07, 26, 38, 42, 69, 73, 74
- **Collected volume**: 1,377,781 DPE (30 columns, 300 MB CSV, 316 MB SQLite)
- **Usage**: Proxy for "sales" — each DPE mentioning heat pump/air conditioning = recent installation
- **Heat pump detection**: field `type_generateur_chauffage_principal` (air/water heat pump) + `type_generateur_froid` (reversible air/air heat pump)
- **Collector**: `src/collectors/dpe.py` (DpeCollector)
- **DB table**: `raw_dpe` (individual records) → aggregated into `fact_hvac_installations`

### 5.2 Historical Weather — Open-Meteo
- **API**: `https://archive-api.open-meteo.com/v1/archive`
- **Auth**: None (free, fair use ~10,000 calls/day)
- **Variables**: temperature_2m_max/min/mean, precipitation_sum, wind_speed_10m_max
- **Computed**: HDD = max(0, 18 - temp_mean), CDD = max(0, temp_mean - 18)
- **8 cities**: Lyon, Grenoble, Saint-Etienne, Annecy, Valence, Chambery, Bourg-en-Bresse, Privas
- **Collector**: `src/collectors/weather.py` (WeatherCollector)

### 5.3 INSEE BDM — Household Confidence & Business Climate
- **SDMX API**: `https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}`
- **Format**: XML SDMX 2.1 (StructureSpecific)
- **Series**:

  | Indicator | idbank | Status |
  |-----------|--------|--------|
  | Household confidence (composite) | 001759970 | OK |
  | Business climate (all sectors) | 001565530 | OK (CORRECTED) |
  | Business climate (construction) | 001586808 | OK |
  | IPI Manufacturing industry | 010768261 | OK |

- **Collector**: `src/collectors/insee.py` (InseeCollector)

### 5.4 Building Permits — SITADEL
- **Source**: SDES DiDo API (MIGRATION — direct ZIP downloads no longer available)
- **API**: `https://data.statistiques.developpement-durable.gouv.fr/dido/api/v1/datafiles/{rid}/csv`
- **Filtering**: DEP in [01, 07, 26, 38, 42, 69, 73, 74]
- **Collector**: `src/collectors/sitadel.py` (SitadelCollector)

### 5.5 Eurostat — HVAC Industrial Production
- **Python package**: `pip install eurostat`
- **Dataset**: sts_inpr_m (Short-term statistics, Industrial Production, Monthly)
- **Filters**: geo=FR, nace_r2=[C28, C2825], unit=I21, s_adj=SCA
- **Collector**: `src/collectors/eurostat_col.py` (EurostatCollector)

### 5.6 Local Energy Consumption — SDES
- **Source**: DiDo API (`data.statistiques.developpement-durable.gouv.fr/dido/api/v1/`)
- **Granularity**: Annual (not monthly — used in EDA only)

## 6. Data Model (Star Schema — corrected)

```
                    ┌──────────────┐
                    │  dim_time    │
                    │──────────────│
                    │ date_id (PK) │ ← YYYYMM format
                    │ year         │
                    │ month        │
                    │ quarter      │
                    │ is_heating   │
                    │ is_cooling   │
                    └──────┬───────┘
                           │
┌──────────────┐   ┌──────┴────────────────────┐
│ dim_geo      │   │  fact_hvac_installations  │
│──────────────│   │───────────────────────────│
│ geo_id (PK)  ├──►│ date_id (FK)             │
│ dept_code    │   │ geo_id (FK)              │
│ dept_name    │   │ nb_dpe_total             │ ← Target variable
│ city_ref     │   │ nb_installations_pac     │ ← Target variable
│ latitude     │   │ nb_installations_clim    │
│ longitude    │   │ temp_mean, HDD, CDD      │ ← Local weather
│ region_code  │   │ nb_permis_construire     │ ← Local real estate
└──────────────┘   │ UNIQUE(date_id, geo_id)  │
                   └──────┬────────────────────┘
                          │
                   ┌──────┴────────────────────┐
                   │ fact_economic_context      │ ← Grain = month only
                   │───────────────────────────│   (national data)
                   │ date_id (PK, FK)          │
                   │ confiance_menages          │
                   │ climat_affaires_indus      │
                   │ climat_affaires_bat        │
                   │ ipi_manufacturing          │
                   │ ipi_hvac_c28, ipi_hvac_c2825│
                   └───────────────────────────┘
```

**Additional raw table:**
```
┌─────────────────────────┐
│ raw_dpe (1.38M rows)    │   ← Grain = 1 individual DPE
│─────────────────────────│
│ numero_dpe (PK)         │
│ date_etablissement_dpe  │
│ code_departement_ban    │
│ etiquette_dpe (A-G)     │
│ type_generateur_chauffage│   → Heat pump detection
│ type_generateur_froid    │   → Air/air heat pump + AC detection
│ surface, isolation, coûts│   → 30 columns total
│ INDEX(date, dept)        │
└─────────────────────────┘
```
**Aggregation**: `raw_dpe` → `fact_hvac_installations` (nb_dpe_total, nb_installations_pac, nb_installations_clim, nb_dpe_classe_ab) per month x department.

**Audit correction**: economic indicators (national) are separated into their own fact table. No more duplication across 8 departments.

## 7. Feature Engineering for ML (IMPLEMENTED ✓)

### ML-ready dataset (`data/features/hvac_ml_dataset.csv`)
- **Grain**: month x department (448 rows x 35 columns)
- **Pipeline**: `python -m src.pipeline merge`
- **Module**: `src/processing/merge_datasets.py` (DatasetMerger)

### Features dataset (`data/features/hvac_features_dataset.csv`)
- **Grain**: month x department (448 rows x 90 columns)
- **Pipeline**: `python -m src.pipeline features`
- **Module**: `src/processing/feature_engineering.py` (FeatureEngineer)

### Feature categories (90 columns)

**Temporal (7)**
- month, quarter, year, is_heating, is_cooling
- month_sin, month_cos (cyclical encoding)
- year_trend (normalized linear trend)

**Temporal lags (21)** — per department
- lag_1m, lag_3m, lag_6m on: nb_dpe_total, nb_installations_pac,
  nb_installations_clim, temp_mean, hdd_sum, cdd_sum, confiance_menages

**Rolling windows (20)** — per department
- rolling_mean_3m, rolling_mean_6m, rolling_std_3m, rolling_std_6m
  on: nb_dpe_total, nb_installations_pac, temp_mean, hdd_sum, cdd_sum

**Variations (6)**
- diff_1m, pct_change_1m on: nb_dpe_total, nb_installations_pac, nb_installations_clim

**Interactions (4)**
- interact_hdd_confiance: HDD x household confidence
- interact_cdd_ipi: CDD x industrial HVAC IPI
- interact_confiance_bat: confidence x construction climate
- jours_extremes: heatwave + frost

**Weather (8)** — local per department
- temp_mean, temp_max, temp_min
- hdd_sum (base 18°C), cdd_sum
- precipitation_sum, nb_jours_canicule, nb_jours_gel
- delta_temp_vs_mean (deviation from historical average)

**Economic (6)** — national
- confiance_menages, climat_affaires_indus, climat_affaires_bat
- ipi_manufacturing, ipi_hvac_c28, ipi_hvac_c2825

### Target features (Y)
- nb_installations_pac: DPE with heat pump per month/department
- nb_installations_clim: DPE with air conditioning
- nb_dpe_total: total DPE volume
- nb_dpe_classe_ab: DPE class A-B (high-performance building)
- pct_pac, pct_clim, pct_classe_ab: derived percentages

## 8. ML / Deep Learning Modeling

### 8.1 Model hierarchy (audit: adapted to data volume)

| Tier | Model | Usage | Viability |
|------|-------|-------|-----------|
| 1 (robust) | Ridge Regression | Baseline | ~288 rows OK |
| 1 (robust) | Prophet + regressors | Time series | 36 pts/dept OK |
| 2 (feasible) | LightGBM (regularized) | Non-linearities | With tuning |
| 3 (educational) | Minimalist LSTM | DL exploration | Honesty required |

**Transformer removed** from scope (insufficient data).

### 8.2 Metrics
- RMSE, MAE, MAPE, R²
- SHAP values (feature importance)
- Temporal cross-validation (TimeSeriesSplit)

### 8.3 Temporal split
```
Train  : 2021-07 → 2024-06 (36 months)
Val    : 2024-07 → 2024-12 (6 months)
Test   : 2025-01 → 2025-12 (12 months)
```

## 9. Final Deliverables

1. **GitHub repo** clean with comprehensive README
2. **Jupyter notebooks** with commentary (EDA + Modeling)
3. **Power BI dashboard** with KPIs and predictions
4. **Medium article** telling the project story (data storytelling)
5. **PDF report** summarizing ML results

## 10. Execution Order

```
Phase 1 — Setup & Collection (COMPLETED ✓)
  ├── 1.1 Init repo, venv, requirements, centralized config    ✓
  ├── 1.2 Extensible architecture (BaseCollector + Registry)   ✓
  ├── 1.3 Collect Open-Meteo (20,456 rows, 8 cities)          ✓
  ├── 1.4 Collect INSEE BDM (85 monthly rows, 4 series)       ✓
  ├── 1.5 Collect Eurostat IPI (168 rows, 2 NACE)             ✓
  ├── 1.6 Collect SITADEL (DiDo migration in progress)        ~
  ├── 1.7 DPE ADEME (1,377,781 DPE, 300 MB CSV)              ✓
  ├── 1.8 SQLite DB initialized + full import (316 MB)        ✓
  ├── 1.9 Dual support SQLite / SQL Server / PostgreSQL        ✓
  ├── 1.10 Portability script (setup_project.py)               ✓
  └── 1.11 DPE aggregation → fact_hvac_installations           ✓
      Total data/ volume: 1.2 GB

Phase 2 — Processing (COMPLETED ✓)
  ├── 2.1 Clean each source (clean_data.py)                     ✓
  │       Weather: 20,456 rows, 0 duplicates, HDD/CDD recalculated
  │       INSEE: 114 → 85 (monthly filter), 2 interpolations
  │       Eurostat: 168 rows, 4 variations > 30% flagged
  │       DPE: 1,377,781 rows, 14,105 values clipped, 6.9% heat pump
  ├── 2.2 Create SQL database (corrected star schema)           ✓ (done in Phase 1)
  ├── 2.3 Load data                                             ✓ (done in Phase 1)
  ├── 2.4 Merge into ML-ready dataset (merge_datasets.py)       ✓
  │       448 rows x 35 columns (month x department)
  │       Period: 2021-07 → 2026-02 (55 months x 8 depts)
  │       → data/features/hvac_ml_dataset.csv
  └── 2.5 Feature engineering (feature_engineering.py)           ✓
          448 rows x 90 columns (+55 advanced features)
          Lags: 1m, 3m, 6m | Rolling: 3m, 6m (mean + std)
          Variations: diff + pct_change | Interactions: 4
          Completeness: 95.3% (NaN = lags at series start)
          → data/features/hvac_features_dataset.csv

Phase 3 — Analysis
  ├── 3.1 EDA notebook
  └── 3.2 Correlations

Phase 4 — Modeling
  ├── 4.1 Baseline models (Ridge, LightGBM, Prophet)
  ├── 4.2 Exploratory LSTM
  ├── 4.3 Comparative evaluation
  └── 4.4 SHAP analysis

Phase 5 — Delivery
  ├── 5.1 Power BI dashboard
  ├── 5.2 GitHub README
  └── 5.3 Medium article
```

## 11. Extensibility

The project is designed to easily add:

### New data source
Create a file in `src/collectors/` with a class inheriting from `BaseCollector` (~50 lines). Auto-registered in the registry.

### New ML model
Add a class in `src/models/` following the common interface.

### Change database engine
Modify only the `.env` file (DB_TYPE, DB_HOST, etc.). No code changes required.
