# HVAC Market Analysis — Metropolitan France

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Tests](https://img.shields.io/badge/tests-814%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)
![Security](https://img.shields.io/badge/security-bandit%20%2B%20pip--audit-green)
![ML](https://img.shields.io/badge/ML-Ridge%20R%C2%B2%3D0.9996-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions)

End-to-end data science platform that collects, processes, and models HVAC (heating, ventilation, air conditioning) installation data across **96 metropolitan French departments**. Cross-references energy performance diagnostics, historical weather, economic indicators, building permits, and socioeconomic data to predict heat pump and air conditioning installations with **R² = 0.9996**.

> **Portfolio project** for the Data Science Lead certification (Jedha Bootcamp, Bac+5 RNCP Level 7) demonstrating the complete data science lifecycle: collection, ETL, feature engineering, ML/DL modeling, REST API, interactive dashboard, Docker deployment, and CI/CD.

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/PDUCLOS/Projet-HVAC.git && cd Projet-HVAC

# 2. Install and generate demo data
make setup && make demo

# 3. Run pipeline and launch dashboard
make pipeline && make serve-dashboard
```

> **API:** http://localhost:8000/docs | **Dashboard:** http://localhost:8501

---

## Quick Access

| Resource | Link |
|----------|------|
| **REST API** (Swagger UI) | `http://localhost:8000/docs` |
| **Dashboard** (Streamlit) | `http://localhost:8501` |
| **Pre-collected Data** | [pCloud Download](https://e.pcloud.link/publink/show?code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy) |
| **Architecture Docs** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| **Pipeline Docs** | [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) |
| **Database Docs** | [`docs/DATABASE_ARCHITECTURE.md`](docs/DATABASE_ARCHITECTURE.md) |

---

## Architecture Overview

```
                    +--------------------------------------------------+
                    |           HVAC Market Analysis Platform           |
                    +--------------------------------------------------+
                    |                                                    |
  +-----------+     |  +-----------+    +----------+    +------------+  |
  | 6 Open    |---->|  | Pipeline  |--->| ML Models|--->| FastAPI    |  |
  | Data APIs |     |  | (Python)  |    | Ridge,   |    | REST API   |  |
  +-----------+     |  +-----------+    | LightGBM |    | :8000      |  |
                    |       |           +----------+    +------+-----+  |
                    |       v                                  |        |
                    |  +-----------+                    +------v-----+  |
                    |  | Data Lake |                    | Streamlit  |  |
                    |  | CSV/SQLite|                    | Dashboard  |  |
                    |  +-----------+                    | :8501      |  |
                    |                                   +------------+  |
                    +--------------------------------------------------+

  Collection (5 APIs) --> Clean --> Merge --> Features (109) --> Train --> Serve
```

For detailed C4 diagrams (Context, Container, Component, Deployment), see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Data Sources

| Source | API | Coverage | Data | Update |
|--------|-----|----------|------|--------|
| **DPE ADEME** | data.ademe.fr | 96 departments | Energy performance diagnostics (heat pump detection) | Continuous |
| **Open-Meteo** | archive-api.open-meteo.com | 96 prefectures | Temperature, precipitation, HDD/CDD, elevation | Daily |
| **INSEE BDM** | bdm.insee.fr (SDMX) | France (national) | Household confidence, business climate, IPI | Monthly |
| **INSEE Filosofi** | Reference CSV (curated) | 96 departments | Median income, price/m², housing stock, % houses | Static |
| **Eurostat** | Python `eurostat` package | France (national) | Industrial Production Index (C28, C2825) | Monthly |
| **SITADEL** | DiDo API (SDES) | 96 departments | Building permits (housing authorized) | Monthly |

> All sources are **Open Data** — no API key required.

---

## ML Results

| Model | Val RMSE | Val R² | Test RMSE | Test R² | Training Time |
|-------|----------|--------|-----------|---------|---------------|
| **Ridge** | 3.76 | 0.9999 | **9.18** | **0.9996** | < 1 sec |
| Ridge (exogenous) | 10.1 | 0.9994 | 23.1 | 0.9982 | < 1 sec |
| LightGBM | 63.5 | 0.9737 | 105.2 | 0.9452 | ~5 sec |
| Prophet | 65.1 | 0.7870 | 221.8 | 0.7564 | ~2 min |
| LSTM (PyTorch) | 393.2 | 0.1193 | 449.9 | 0.0560 | ~3 min |

**Target:** `nb_installations_pac` (heat pump installations per department per month)

**Top features** (SHAP / LightGBM importance): `nb_installations_pac_rmean_3m`, `nb_installations_pac_diff_1m`, `nb_installations_pac_lag_1m`, `nb_installations_pac_pct_1m`, `nb_dpe_classe_ab_lag_6m`

**Temporal split:** Train 2021-07/2024-06 | Val 2024-07/2024-12 | Test 2025-01/2025-12

---

## REST API (FastAPI)

```bash
# Start the API
make serve-api
# Swagger UI: http://localhost:8000/docs
```

| Endpoint | Method | Description | Example |
|----------|--------|-------------|---------|
| `/health` | GET | API status, version, loaded model info | `curl localhost:8000/health` |
| `/predictions` | GET | Predictions by department and horizon | `curl "localhost:8000/predictions?departement=69&horizon=6"` |
| `/predict` | POST | Custom prediction with JSON parameters | `curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"departement":"69","horizon":6}'` |
| `/data/summary` | GET | Summary of available data (rows, columns, date range) | `curl localhost:8000/data/summary` |
| `/model/metrics` | GET | ML metrics (RMSE, MAE, R², MAPE) | `curl localhost:8000/model/metrics` |
| `/departments` | GET | List of the 96 metropolitan departments | `curl localhost:8000/departments` |

---

## Dashboard (Streamlit)

```bash
make serve-dashboard
# Open http://localhost:8501
```

| Page | Description |
|------|-------------|
| **Home** | Overview, key metrics, Top 10 departments, model performance, architecture diagram |
| **Exploration** | Interactive stats, distributions, correlation matrices |
| **Map of France** | Choropleth map with metric selector (PAC, DPE, income, price/m²), department ranking |
| **ML Predictions** | Predictions vs actual values, residuals analysis, feature importance |
| **Model Comparison** | Comparison table across all models, radar chart |
| **Pipeline** | Data status, one-click pipeline execution, pCloud sync |

> **Screenshots**: see `dashboards/screenshots/` for dashboard visuals.

---

## CLI Commands

```bash
# Collection
python -m src.pipeline collect                        # All sources
python -m src.pipeline collect --sources weather,insee # Specific sources

# Processing
python -m src.pipeline process                        # clean + merge + features + outliers
python -m src.pipeline clean -i                       # Interactive: preview rules before applying

# ML
python -m src.pipeline train                          # Train all models
python -m src.pipeline evaluate                       # Evaluate + SHAP analysis

# All-in-one
python -m src.pipeline update_all                     # Collect + process + train + upload
```

See `make help` for a full list of Makefile targets.

---

## Deployment

### Local (Python)

```bash
make setup      # Create venv, install deps, init DB
make demo       # Generate demonstration data
make pipeline   # Process + train + evaluate
make serve-api  # Start FastAPI on :8000
```

### Docker

```bash
docker compose up                   # API + Dashboard
docker compose --profile db up     # With PostgreSQL
docker compose run --rm pipeline   # Run pipeline in container
```

### Cloud (Render.com)

The project includes a `render.yaml` for one-click deployment on [Render.com](https://render.com):
- **hvac-api**: FastAPI web service with `/health` endpoint monitoring
- **hvac-dashboard**: Streamlit web service

Push to `main` triggers automatic deployment.

---

## Project Structure

```
Projet-HVAC/
├── config/settings.py              # Centralized configuration (96 departments)
├── src/
│   ├── pipeline.py                 # CLI orchestrator (16 commands, interactive mode)
│   ├── collectors/                 # Data collection (plugin architecture)
│   │   ├── base.py                 # BaseCollector + CollectorRegistry
│   │   ├── weather.py              # Open-Meteo (96 prefectures)
│   │   ├── insee.py                # INSEE BDM (SDMX)
│   │   ├── eurostat_col.py         # Eurostat IPI
│   │   ├── sitadel.py              # SITADEL building permits
│   │   ├── dpe.py                  # DPE ADEME
│   │   └── pcloud_sync.py          # pCloud synchronization
│   ├── processing/                 # Data transformation
│   │   ├── clean_data.py           # Cleaning (skip rules + preview mode)
│   │   ├── merge_datasets.py       # Multi-source merge
│   │   ├── feature_engineering.py  # 109 features (lags, rolling, PAC efficiency)
│   │   └── outlier_detection.py    # IQR + Z-score + Isolation Forest
│   ├── models/                     # ML / Deep Learning
│   │   ├── baseline.py             # Ridge, LightGBM, Prophet
│   │   ├── deep_learning.py        # LSTM (PyTorch)
│   │   ├── train.py                # Training orchestrator
│   │   ├── evaluate.py             # Metrics, SHAP, visualizations
│   │   └── reinforcement_learning_demo.py  # RL demo (Gymnasium)
│   ├── analysis/                   # EDA + correlations
│   └── database/                   # Star schema, SQLAlchemy
├── api/                            # FastAPI REST API (6 endpoints)
├── app/                            # Streamlit dashboard (6 pages)
├── airflow/dags/                   # Airflow DAG (orchestration)
├── kubernetes/                     # K8s manifests
├── docs/                           # Architecture, pipeline, governance docs
├── tests/                          # 561 tests (pytest)
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml              # API + Dashboard + PostgreSQL
├── Makefile                        # Project commands (make help)
├── render.yaml                     # Render.com deployment
└── requirements.txt                # Pinned dependencies
```

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **Data** | pandas, numpy, SQLAlchemy |
| **ML** | scikit-learn, LightGBM, XGBoost, Prophet, SHAP |
| **Deep Learning** | PyTorch (LSTM), Gymnasium (RL) |
| **API** | FastAPI, Pydantic, uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Database** | SQLite, PostgreSQL, SQL Server |
| **DevOps** | Docker, Kubernetes, Airflow, Makefile |
| **CI/CD** | GitHub Actions (test + lint + Docker build) |
| **Cloud** | Render.com (deploy), pCloud (data sync) |
| **Quality** | pytest, ruff, bandit, pip-audit |

---

## Certification Coverage

This project covers the 6 modules of the Bac+5 Data Science Lead certification:

| Module | Subject | Implementation |
|--------|---------|----------------|
| **M1** | Data Governance | [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) — GDPR, AI Act, lineage |
| **M2** | Deployment & Distributed ML | `Dockerfile`, `docker-compose.yml`, `kubernetes/`, `render.yaml` |
| **M3** | Database Architecture | [`docs/DATABASE_ARCHITECTURE.md`](docs/DATABASE_ARCHITECTURE.md) — Star schema, OLAP, NoSQL |
| **M4** | Data Pipelines | [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) — ELT, monitoring, Airbyte comparison |
| **M5** | Automation & Workflow | [`airflow/dags/hvac_pipeline_dag.py`](airflow/dags/hvac_pipeline_dag.py) — Airflow DAG |
| **M6** | Reinforcement Learning | [`src/models/reinforcement_learning_demo.py`](src/models/reinforcement_learning_demo.py) — Gymnasium, Q-Learning |

---

## Tests

```bash
make test          # Run 561 tests
make test-cov      # With coverage report
make lint          # Ruff linter
make security      # pip-audit + bandit
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`make test`)
5. Run linting and security checks (`make lint && make security`)
6. Commit with a clear message and open a Pull Request

All code comments and docstrings must be in English.

---

## Author

**Patrice DUCLOS** — Data Analyst (20 years of experience)

Portfolio project for the **Data Science Lead** program (Jedha Bootcamp, Bac+5 RNCP Level 7 certification).

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — Free for non-commercial use (research, learning, portfolio review, personal projects). Commercial use requires explicit written permission from the author.
