# Portfolio Project: HVAC Market Analysis — Metropolitan France
## Sales Prediction via ML/Deep Learning

**Author**: Patrice DUCLOS — Senior Data Analyst (20 years of experience)
**Stack**: Python, SQL (local SQLite / SQL Server / PostgreSQL), FastAPI, Streamlit, Docker
**Objective**: Portfolio project for the Data Science Lead certification (Jedha Bootcamp, Bac+5 RNCP Level 7)

---

## 1. Project Vision

Build a complete data pipeline (collection → cleaning → analysis → prediction → dashboard) on the HVAC market across **96 metropolitan French departments**, by combining energy, weather, economic, socioeconomic, and construction data to predict sales/installation volumes of heating and air conditioning equipment.

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
│   └── settings.py              # Centralized configuration (96 depts, names, coordinates)
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
│   ├── pipeline.py              # CLI orchestrator (16 commands, interactive mode -i)
│   ├── collectors/              # Extensible plugin-based architecture
│   │   ├── __init__.py
│   │   ├── base.py              # BaseCollector + CollectorRegistry
│   │   ├── weather.py           # Open-Meteo (WeatherCollector)
│   │   ├── insee.py             # INSEE BDM (InseeCollector)
│   │   ├── eurostat_col.py      # Eurostat (EurostatCollector)
│   │   ├── sitadel.py           # SITADEL via DiDo API (SitadelCollector)
│   │   ├── dpe.py               # DPE ADEME (DpeCollector)
│   │   └── pcloud_sync.py       # pCloud synchronization (data + features)
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── clean_data.py        # Source-by-source cleaning (skip rules + preview mode)
│   │   ├── merge_datasets.py    # Multi-source merge (DPE+weather+SITADEL+INSEE ref)
│   │   ├── feature_engineering.py  # Advanced features: lags, rolling, interactions, PAC efficiency
│   │   └── outlier_detection.py # IQR + Z-score + Isolation Forest (consensus 2/3)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql           # Star schema SQLite (+ raw_dpe)
│   │   ├── schema_mssql.sql     # SQL Server schema (IDENTITY, BIT, MERGE)
│   │   └── db_manager.py        # CRUD + CSV import + DPE aggregation (interactive source selection)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py               # Exploratory analysis
│   │   └── correlation.py       # Correlation studies
│   └── models/
│       ├── __init__.py
│       ├── baseline.py          # Classic ML models (Ridge, LightGBM, Prophet)
│       ├── deep_learning.py     # LSTM (PyTorch)
│       ├── train.py             # Training pipeline (outlier leakage prevention)
│       ├── evaluate.py          # Metrics, SHAP, visualizations
│       └── reinforcement_learning_demo.py  # RL demo (Gymnasium)
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
python -m src.pipeline import_data -i                  # Interactive: choose sources to import
python -m src.pipeline clean                           # Cleaning → data/processed/
python -m src.pipeline clean -i                        # Interactive: preview & skip rules
python -m src.pipeline merge                           # Merge → data/features/hvac_ml_dataset.csv
python -m src.pipeline features                        # Features → data/features/hvac_features_dataset.csv
python -m src.pipeline process                         # clean + merge + features in one command
python -m src.pipeline process -i                      # Interactive cleaning during process
python -m src.pipeline train                           # Train ML models
python -m src.pipeline evaluate                        # Evaluate models
python -m src.pipeline upload_pcloud                   # Upload results to pCloud
python -m src.pipeline update_all                      # Full pipeline (collect + process + train + upload)
```

## 5. Data Sources — Technical Details

### 5.1 DPE (Energy Performance Diagnostic) — ADEME
- **URL (CORRECTED)**: https://data.ademe.fr/datasets/dpe03existant
- **API**: `https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines`
- **Format**: Cursor-paginated JSON (~14M total entries)
- **Scope**: DPE since July 2021
- **Filtering**: 96 metropolitan departments (01-95, 2A, 2B)
- **Collected volume**: Varies by deployment (up to ~5-8M DPE for 96 departments)
- **Usage**: Proxy for "sales" — each DPE mentioning heat pump/air conditioning = recent installation
- **Heat pump detection**: field `type_generateur_chauffage_principal` (air/water heat pump) + `type_generateur_froid` (reversible air/air heat pump)
- **Collector**: `src/collectors/dpe.py` (DpeCollector)
- **DB table**: `raw_dpe` (individual records) → aggregated into `fact_hvac_installations`

### 5.2 Historical Weather — Open-Meteo
- **API**: `https://archive-api.open-meteo.com/v1/archive` (weather)
          `https://api.open-meteo.com/v1/elevation` (elevation, Copernicus DEM GLO-90)
- **Auth**: None (free, fair use ~10,000 calls/day)
- **Variables**: temperature_2m_max/min/mean, precipitation_sum, wind_speed_10m_max
- **Computed**:
  - HDD = max(0, 18 - temp_mean) — heating demand indicator
  - CDD = max(0, temp_mean - 18) — cooling demand indicator
  - pac_inefficient = 1 if temp_min < -7°C — heat pump COP critical flag
  - elevation = altitude in meters from Elevation API (fallback: static PREFECTURE_ELEVATIONS)
- **96 prefectures**: One per metropolitan department (configured in settings.py)
- **Collector**: `src/collectors/weather.py` (WeatherCollector)
- **PAC domain logic**: Below -7°C, air-source heat pump COP drops below ~2.0,
  making it economically unviable. This threshold is configurable via
  `ThresholdsConfig.pac_inefficiency_temp` in `config/settings.py`

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
- **Filtering**: 96 metropolitan departments
- **Collector**: `src/collectors/sitadel.py` (SitadelCollector)
- **Integration**: SITADEL features (nb_logements_autorises, nb_logements_individuels, nb_logements_collectifs, surface_autorisee_m2) are merged into the ML dataset via `merge_datasets.py`

### 5.7 INSEE Filosofi — Reference Data (Static)
- **Source**: `data/raw/insee/reference_departements.csv` (pre-built from INSEE open data)
- **Granularity**: Per department (static, no temporal dimension)
- **Features**: revenu_median, prix_m2_median, nb_logements_total, pct_maisons
- **Usage**: Socioeconomic context per department, linked to aid eligibility (MaPrimeRenov, CEE) and housing market activity
- **Integration**: Merged as static features in `merge_datasets.py`

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
│ raw_dpe                 │   ← Grain = 1 individual DPE
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

**Audit corrections**:
- Economic indicators (national) are separated into their own fact table. No more duplication across departments.
- SITADEL building permits are now integrated into the ML merge (nb_logements_autorises, surface_autorisee_m2).
- INSEE Filosofi reference data (revenu_median, prix_m2_median, nb_logements_total, pct_maisons) merged as static department-level features.
- Outlier detection flags (IQR, Z-score, Isolation Forest) are excluded from training to prevent data leakage.

## 7. Feature Engineering for ML (IMPLEMENTED ✓)

### ML-ready dataset (`data/features/hvac_ml_dataset.csv`)
- **Grain**: month x department (96 departments x N months)
- **Pipeline**: `python -m src.pipeline merge`
- **Module**: `src/processing/merge_datasets.py` (DatasetMerger)
- **Sources merged**: DPE + Weather + INSEE BDM + Eurostat + SITADEL + INSEE Filosofi reference

### Features dataset (`data/features/hvac_features_dataset.csv`)
- **Grain**: month x department (96 departments x N months, ~100+ columns)
- **Pipeline**: `python -m src.pipeline features`
- **Module**: `src/processing/feature_engineering.py` (FeatureEngineer)
- **Steps**: lags → rolling → variations → interactions → PAC efficiency → trends → completeness

### Feature categories (~100+ columns)

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

**PAC (heat pump) efficiency (7)** — domain knowledge features
- cop_proxy: estimated COP (Coefficient of Performance) from temperature and altitude
  Formula: COP = 4.5 - 0.08 x frost_days - 0.0005 x altitude, clipped [1.0, 5.0]
- is_mountain: binary flag for high-altitude departments (altitude > 800m)
- pac_viability_score: composite [0,1] score (0.5 x COP_norm + 0.3 x housing + 0.2 x frost_penalty)
- interact_altitude_frost: normalized altitude x frost days interaction (cold mountain penalty)
- interact_maisons_altitude: pct_maisons x normalized altitude (houses in mountains)
- pct_jours_pac_inefficient: % of days with T_min < -7°C (PAC COP drops below ~2.0)
- interact_cop_hdd: COP/5 x HDD/max (heating demand vs heat pump efficiency)

> **Domain logic**: Air-source heat pumps (PAC air-air/air-eau) lose efficiency in cold
> weather. Below -7°C, COP drops below ~2.0, making PAC economically uncompetitive vs
> gas/oil heating. Mountain departments (>800m altitude) have structurally colder base
> temperatures and different HVAC adoption patterns. These features encode this domain
> knowledge so the ML model can learn geographically-differentiated PAC viability.

**Weather (9)** — local per department
- temp_mean, temp_max, temp_min
- hdd_sum (base 18°C), cdd_sum
- precipitation_sum, nb_jours_canicule, nb_jours_gel
- nb_jours_pac_inefficient (days with T_min < -7°C — PAC COP critical threshold)
- delta_temp_vs_mean (deviation from historical average)

**Economic (6)** — national
- confiance_menages, climat_affaires_indus, climat_affaires_bat
- ipi_manufacturing, ipi_hvac_c28, ipi_hvac_c2825

**Construction / SITADEL (4)** — per department, monthly
- nb_logements_autorises, nb_logements_individuels, nb_logements_collectifs
- surface_autorisee_m2

**Socioeconomic & geographic reference (5)** — per department, static
- revenu_median (INSEE Filosofi — linked to aid eligibility)
- prix_m2_median (proxy for real estate market activity)
- nb_logements_total (housing stock, enables normalization)
- pct_maisons (structural differences: houses → more heat pumps)
- altitude (prefecture elevation in meters, from PREFECTURE_ELEVATIONS / Open-Meteo Elevation API)

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
3. **Streamlit dashboard** with 6 interactive pages (map, metrics, predictions, model comparison)
4. **FastAPI REST API** with 6 endpoints (predictions, metrics, departments)
5. **Docker deployment** (docker-compose with API + Dashboard + optional PostgreSQL)
6. **Documentation** (governance, database architecture, pipeline — `docs/`)
7. **Medium article** telling the project story (data storytelling)
8. **PDF report** summarizing ML results

## 10. Execution Order

```
Phase 1 — Setup & Collection (COMPLETED ✓)
  ├── 1.1 Init repo, venv, requirements, centralized config    ✓
  ├── 1.2 Extensible architecture (BaseCollector + Registry)   ✓
  ├── 1.3 Collect Open-Meteo (96 prefectures)                 ✓
  ├── 1.4 Collect INSEE BDM (85 monthly rows, 4 series)       ✓
  ├── 1.5 Collect Eurostat IPI (168 rows, 2 NACE)             ✓
  ├── 1.6 Collect SITADEL (DiDo API, 96 departments)          ✓
  ├── 1.7 DPE ADEME (96 departments)                          ✓
  ├── 1.8 SQLite DB initialized + full import                  ✓
  ├── 1.9 Dual support SQLite / SQL Server / PostgreSQL        ✓
  ├── 1.10 Portability script (setup_project.py)               ✓
  ├── 1.11 DPE aggregation → fact_hvac_installations           ✓
  ├── 1.12 INSEE Filosofi reference data (96 departments)      ✓
  └── 1.13 Interactive import menu (source selection)           ✓

Phase 2 — Processing (COMPLETED ✓)
  ├── 2.1 Clean each source (clean_data.py)                     ✓
  │       Weather, INSEE, Eurostat, DPE — with skip rules + preview
  ├── 2.2 Interactive cleaning menu (preview impact + skip)     ✓
  ├── 2.3 Create SQL database (corrected star schema)           ✓
  ├── 2.4 Merge into ML-ready dataset (merge_datasets.py)       ✓
  │       96 departments x N months
  │       Sources: DPE + Weather + SITADEL + INSEE ref + Economic
  │       → data/features/hvac_ml_dataset.csv
  ├── 2.5 Feature engineering (feature_engineering.py)           ✓
  │       ~90+ columns (lags, rolling, interactions, static)
  │       → data/features/hvac_features_dataset.csv
  └── 2.6 Outlier detection (IQR + Z-score + Isolation Forest)  ✓
          Consensus 2/3, outlier flags excluded from training

Phase 3 — Analysis
  ├── 3.1 EDA notebook
  └── 3.2 Correlations

Phase 4 — Modeling (COMPLETED ✓)
  ├── 4.1 Baseline models (Ridge, LightGBM, Prophet)           ✓
  ├── 4.2 Exploratory LSTM (PyTorch)                           ✓
  ├── 4.3 Comparative evaluation                                ✓
  ├── 4.4 SHAP analysis                                         ✓
  └── 4.5 Outlier leakage prevention in training                ✓

Phase 5 — Deployment & Delivery (IN PROGRESS)
  ├── 5.1 FastAPI REST API (6 endpoints)                        ✓
  ├── 5.2 Streamlit dashboard (6 interactive pages)             ✓
  ├── 5.3 Docker + docker-compose                               ✓
  ├── 5.4 Kubernetes manifests                                  ✓
  ├── 5.5 Airflow DAG                                           ✓
  ├── 5.6 pCloud synchronization (data + features)              ✓
  ├── 5.7 GitHub README                                         ✓
  └── 5.8 Documentation (governance, architecture, pipeline)    ✓
```

## 11. Extensibility

The project is designed to easily add:

### New data source
Create a file in `src/collectors/` with a class inheriting from `BaseCollector` (~50 lines). Auto-registered in the registry.

### New ML model
Add a class in `src/models/` following the common interface.

### Change database engine
Modify only the `.env` file (DB_TYPE, DB_HOST, etc.). No code changes required.
