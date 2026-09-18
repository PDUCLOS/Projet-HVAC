# Architecture — HVAC Market Analysis

> Comprehensive architecture documentation for the HVAC Market Analysis platform.
> Diagrams follow the C4 model (Context, Container, Component) using Mermaid.

---

## 1. C4 Context Diagram

High-level view: the HVAC Market Analysis system and its external actors and data sources.

```mermaid
graph TB
    subgraph External Actors
        USER["Data Scientist / Analyst<br/>(human user)"]
        REVIEWER["Portfolio Reviewer<br/>(certification jury)"]
    end

    subgraph External Data Sources
        ADEME["ADEME DPE API<br/>(Energy Performance Diagnostics)"]
        OPENMETEO["Open-Meteo API<br/>(Historical Weather)"]
        INSEE["INSEE BDM API<br/>(Economic Indicators, SDMX)"]
        EUROSTAT["Eurostat API<br/>(Industrial Production Index)"]
        SITADEL["SITADEL / DiDo API<br/>(Building Permits)"]
        PCLOUD["pCloud<br/>(Data Backup & Sync)"]
    end

    HVAC["HVAC Market Analysis<br/>Platform<br/><br/>Collects, processes, and models<br/>HVAC installation data across<br/>96 French metropolitan departments"]

    USER -->|"Runs pipeline,<br/>views dashboard,<br/>queries API"| HVAC
    REVIEWER -->|"Reviews dashboard,<br/>reads documentation"| HVAC
    ADEME -->|"DPE records<br/>(JSON, paginated)"| HVAC
    OPENMETEO -->|"Daily weather<br/>(JSON REST)"| HVAC
    INSEE -->|"Monthly indicators<br/>(XML SDMX)"| HVAC
    EUROSTAT -->|"IPI data<br/>(Python package)"| HVAC
    SITADEL -->|"Building permits<br/>(CSV via DiDo)"| HVAC
    HVAC -->|"Backup data<br/>& features"| PCLOUD

    style HVAC fill:#1565c0,color:#fff,stroke:#0d47a1
    style ADEME fill:#e8f5e9,stroke:#2e7d32
    style OPENMETEO fill:#e3f2fd,stroke:#1565c0
    style INSEE fill:#fff3e0,stroke:#ef6c00
    style EUROSTAT fill:#f3e5f5,stroke:#7b1fa2
    style SITADEL fill:#fce4ec,stroke:#c62828
```

---

## 2. C4 Container Diagram

The system is composed of four main containers: Pipeline, API, Dashboard, and Database.

```mermaid
graph TB
    subgraph HVAC Platform
        PIPELINE["Pipeline<br/>(Python CLI)<br/><br/>src/pipeline.py<br/>16 commands<br/>Orchestrates collection,<br/>processing, training"]

        API["FastAPI REST API<br/>(Python / uvicorn)<br/><br/>api/main.py<br/>6 endpoints<br/>Predictions, metrics,<br/>department data"]

        DASHBOARD["Streamlit Dashboard<br/>(Python)<br/><br/>app/app.py<br/>6 interactive pages<br/>Maps, charts,<br/>model comparison"]

        DB["Database<br/>(SQLite / PostgreSQL)<br/><br/>Star schema<br/>5 tables<br/>Month x Department grain"]

        MODELS["Trained Models<br/>(Pickle / PyTorch)<br/><br/>data/models/<br/>Ridge, LightGBM,<br/>Prophet, LSTM"]

        DATALAKE["Data Lake<br/>(CSV files)<br/><br/>data/raw/<br/>data/processed/<br/>data/features/"]
    end

    USER["User"] -->|"CLI commands"| PIPELINE
    USER -->|"HTTP :8000"| API
    USER -->|"HTTP :8501"| DASHBOARD

    PIPELINE -->|"Reads/writes"| DATALAKE
    PIPELINE -->|"Imports data"| DB
    PIPELINE -->|"Saves models"| MODELS

    API -->|"Reads"| DB
    API -->|"Loads"| MODELS

    DASHBOARD -->|"Queries"| API
    DASHBOARD -->|"Reads"| DATALAKE

    style PIPELINE fill:#fff3e0,stroke:#ef6c00
    style API fill:#e8f5e9,stroke:#2e7d32
    style DASHBOARD fill:#e3f2fd,stroke:#1565c0
    style DB fill:#f3e5f5,stroke:#7b1fa2
    style MODELS fill:#fff8e1,stroke:#f9a825
    style DATALAKE fill:#fce4ec,stroke:#c62828
```

---

## 3. C4 Component Diagram — Pipeline

Detailed view of the pipeline's internal components: collectors, processing modules, and model trainers.

```mermaid
graph TB
    subgraph Collectors ["src/collectors/ — Data Collection"]
        BASE["BaseCollector<br/>(Abstract base class)"]
        REG["CollectorRegistry<br/>(Plugin auto-registration)"]
        WC["WeatherCollector<br/>(Open-Meteo)"]
        IC["InseeCollector<br/>(INSEE BDM SDMX)"]
        EC["EurostatCollector<br/>(Eurostat IPI)"]
        SC["SitadelCollector<br/>(DiDo API)"]
        DC["DpeCollector<br/>(ADEME)"]
        PC["pCloudSync<br/>(Upload/Download)"]
    end

    subgraph Processing ["src/processing/ — Data Transformation"]
        CLEAN["clean_data.py<br/>Source-by-source cleaning<br/>Skip rules + preview mode"]
        MERGE["merge_datasets.py<br/>Multi-source merge<br/>DPE + Weather + SITADEL + INSEE"]
        FEAT["feature_engineering.py<br/>109 features total<br/>Lags, rolling, interactions,<br/>PAC efficiency"]
        OUT["outlier_detection.py<br/>IQR + Z-score +<br/>Isolation Forest<br/>(consensus 2/3)"]
    end

    subgraph Models ["src/models/ — ML & Deep Learning"]
        BL["baseline.py<br/>Ridge, LightGBM, Prophet"]
        DL["deep_learning.py<br/>LSTM (PyTorch)"]
        TR["train.py<br/>Training orchestrator<br/>Temporal split"]
        EV["evaluate.py<br/>Metrics, SHAP,<br/>visualizations"]
        RL["reinforcement_learning_demo.py<br/>Gymnasium Q-Learning"]
    end

    ORCH["pipeline.py<br/>(CLI Orchestrator)<br/>16 commands"]

    ORCH --> REG
    REG --> WC & IC & EC & SC & DC
    BASE --> WC & IC & EC & SC & DC
    ORCH --> CLEAN --> MERGE --> FEAT --> OUT
    ORCH --> TR --> EV
    TR --> BL & DL

    style Collectors fill:#fff3e0,stroke:#ef6c00
    style Processing fill:#fce4ec,stroke:#c62828
    style Models fill:#fff8e1,stroke:#f9a825
    style ORCH fill:#e3f2fd,stroke:#1565c0
```

---

## 4. Data Flow Diagram

End-to-end data flow from raw sources through processing to predictions.

```mermaid
flowchart LR
    subgraph Sources ["External APIs"]
        S1["ADEME DPE<br/>~1.4M records"]
        S2["Open-Meteo<br/>~250K rows"]
        S3["INSEE BDM<br/>~500 rows"]
        S4["Eurostat<br/>~200 rows"]
        S5["SITADEL<br/>~5K rows"]
        S6["INSEE Filosofi<br/>96 depts static"]
    end

    subgraph Raw ["data/raw/ (Data Lake)"]
        R1["dpe_france_all.csv"]
        R2["weather_france.csv"]
        R3["indicateurs_economiques.csv"]
        R4["ipi_hvac_france.csv"]
        R5["sitadel.csv"]
        R6["reference_departements.csv"]
    end

    subgraph Processed ["data/processed/"]
        P1["Cleaned CSVs<br/>(deduplicated,<br/>validated, typed)"]
    end

    subgraph Features ["data/features/"]
        F1["hvac_ml_dataset.csv<br/>(merged, 36 cols)"]
        F2["hvac_features_dataset.csv<br/>(engineered, 109 cols)"]
    end

    subgraph Models ["data/models/"]
        M1["ridge_model.pkl"]
        M2["lightgbm_model.pkl"]
        M3["prophet_model.pkl"]
        M4["lstm_model.pt"]
        M5["Evaluation figures<br/>+ SHAP analysis"]
    end

    subgraph Serving ["User-Facing"]
        API["FastAPI<br/>:8000/docs"]
        DASH["Streamlit<br/>:8501"]
    end

    S1 --> R1
    S2 --> R2
    S3 --> R3
    S4 --> R4
    S5 --> R5
    S6 --> R6

    R1 & R2 & R3 & R4 & R5 --> P1
    P1 --> F1
    R6 --> F1
    F1 --> F2
    F2 --> M1 & M2 & M3 & M4
    M1 & M2 --> M5

    M1 & M2 --> API
    F2 --> DASH
    API --> DASH

    style Sources fill:#e1f5fe,stroke:#0288d1
    style Raw fill:#f3e5f5,stroke:#7b1fa2
    style Processed fill:#fce4ec,stroke:#c62828
    style Features fill:#e8f5e9,stroke:#2e7d32
    style Models fill:#fff8e1,stroke:#f9a825
    style Serving fill:#e3f2fd,stroke:#1565c0
```

---

## 5. Deployment Diagram

Multiple deployment options from local development to cloud hosting.

```mermaid
graph TB
    subgraph Local ["Local Development"]
        VENV["Python venv<br/>+ SQLite"]
        CLI["CLI: python -m src.pipeline"]
        LOCAL_API["uvicorn api.main:app<br/>:8000"]
        LOCAL_DASH["streamlit run app/app.py<br/>:8501"]
    end

    subgraph Docker ["Docker Compose"]
        D_API["hvac-api<br/>(FastAPI container)<br/>:8000"]
        D_DASH["hvac-dashboard<br/>(Streamlit container)<br/>:8501"]
        D_PIPE["hvac-pipeline<br/>(One-shot container)"]
        D_PG["PostgreSQL 16<br/>(Optional, profile: db)<br/>:5432"]
        D_VOL["postgres_data<br/>(Named volume)"]
    end

    subgraph Cloud ["Cloud (Render.com)"]
        R_API["hvac-api<br/>(Web Service)<br/>Free tier"]
        R_DASH["hvac-dashboard<br/>(Web Service)<br/>Free tier"]
    end

    subgraph Orchestration ["Orchestration (Optional)"]
        AIRFLOW["Apache Airflow<br/>DAG: hvac_pipeline_dag"]
        K8S["Kubernetes<br/>Deployment + Service + Ingress"]
    end

    CLI --> VENV
    LOCAL_API --> VENV
    LOCAL_DASH --> VENV

    D_API --> D_PG
    D_DASH --> D_API
    D_PIPE --> D_PG
    D_PG --> D_VOL

    AIRFLOW -->|"Triggers"| D_PIPE

    style Local fill:#e8f5e9,stroke:#2e7d32
    style Docker fill:#e3f2fd,stroke:#1565c0
    style Cloud fill:#fff3e0,stroke:#ef6c00
    style Orchestration fill:#f3e5f5,stroke:#7b1fa2
```

---

## 6. Database Schema (Star Schema)

See [`docs/DATABASE_ARCHITECTURE.md`](DATABASE_ARCHITECTURE.md) for full details.

```mermaid
erDiagram
    dim_time ||--o{ fact_hvac_installations : "date_id"
    dim_geo ||--o{ fact_hvac_installations : "geo_id"
    dim_time ||--o| fact_economic_context : "date_id"

    dim_time {
        INT date_id PK "YYYYMM"
        INT year
        INT month
        INT quarter
        BOOLEAN is_heating
        BOOLEAN is_cooling
    }

    dim_geo {
        INT geo_id PK
        VARCHAR dept_code UK
        VARCHAR dept_name
        VARCHAR city_ref
        DECIMAL latitude
        DECIMAL longitude
    }

    fact_hvac_installations {
        INT fact_id PK
        INT date_id FK
        INT geo_id FK
        INT nb_dpe_total
        INT nb_installations_pac
        INT nb_installations_clim
        DECIMAL temp_mean
        DECIMAL heating_degree_days
        DECIMAL cooling_degree_days
    }

    fact_economic_context {
        INT date_id PK
        DECIMAL confiance_menages
        DECIMAL climat_affaires_bat
        DECIMAL ipi_hvac_c28
        DECIMAL ipi_hvac_c2825
    }

    raw_dpe {
        VARCHAR numero_dpe PK
        DATE date_etablissement_dpe
        VARCHAR code_departement_ban
        VARCHAR etiquette_dpe
        VARCHAR type_generateur_chauffage
        VARCHAR type_generateur_froid
    }
```

---

## 7. Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core development language |
| **Data** | pandas, numpy, SQLAlchemy | Data manipulation and ORM |
| **ML** | scikit-learn, LightGBM, Prophet | Classical ML models |
| **Deep Learning** | PyTorch | LSTM time series model |
| **Reinforcement Learning** | Gymnasium | Q-Learning demo |
| **Feature Analysis** | SHAP, XGBoost | Feature importance and explainability |
| **API** | FastAPI, Pydantic, uvicorn | REST API with auto-documentation |
| **Dashboard** | Streamlit, Plotly | Interactive data visualization |
| **Database** | SQLite, PostgreSQL, SQL Server | Multi-engine via SQLAlchemy |
| **DevOps** | Docker, Kubernetes, Airflow | Containerization and orchestration |
| **CI/CD** | GitHub Actions | Automated testing and builds |
| **Cloud** | Render.com | Deployment hosting |
| **Quality** | pytest, ruff, bandit, pip-audit | Testing, linting, security |

---

## 8. Key Design Decisions

### 8.1 ELT over ETL

Raw data is loaded first into `data/raw/`, then transformed. This preserves the source of truth and enables reprocessing without re-collection. Each step is idempotent.

### 8.2 Plugin Architecture for Collectors

New data sources are added by creating a class inheriting from `BaseCollector`. Auto-registration via `__init_subclass__` eliminates boilerplate. No existing code needs modification.

### 8.3 Star Schema for Analytics

The OLAP star schema separates dimensions (time, geography) from facts (installations, economics). This enables efficient aggregation queries with minimal joins, directly compatible with ML feature extraction.

### 8.4 Local-First Design

SQLite is the default database, requiring zero infrastructure. The entire project runs on a single machine with `python -m src.pipeline all`. Cloud and Docker deployments are optional enhancements.

### 8.5 Temporal Train/Val/Test Split

Data is split chronologically (Train: 2021-07 to 2024-06, Val: 2024-07 to 2024-12, Test: 2025-01 to 2025-12) to prevent temporal data leakage and reflect real-world forecasting conditions.

### 8.6 Domain-Driven Feature Engineering

PAC efficiency features encode heat pump physics (COP degradation below -7 degrees C, altitude impact, mountain zone classification). These features capture domain knowledge that pure statistical features cannot.

---

*Architecture documentation for the HVAC Market Analysis project*
*Data Science Lead certification — Jedha Bootcamp (Bac+5 RNCP Level 7)*
