# Pipeline de Donnees — HVAC Market Analysis

> Documentation technique du pipeline ELT complet, de la collecte des sources
> externes jusqu'a la production de modeles ML.

---

## 1. Architecture du pipeline ELT

### 1.1 Diagramme du flux complet

L'orchestrateur central `src/pipeline.py` pilote 10 etapes sequentielles via CLI.

```mermaid
flowchart TD
    subgraph Sources["Sources API externes"]
        S1["Open-Meteo\n(meteo historique)"]
        S2["INSEE BDM\n(indicateurs eco)"]
        S3["Eurostat\n(IPI industriel)"]
        S4["SITADEL\n(permis construire)"]
        S5["ADEME DPE\n(diagnostics energie)"]
    end

    subgraph Collect["1. Collecte (EL)"]
        C1["WeatherCollector"]
        C2["InseeCollector"]
        C3["EurostatCollector"]
        C4["SitadelCollector"]
        C5["DpeCollector"]
    end

    subgraph Raw["Stockage brut — data/raw/"]
        R1["weather_france.csv"]
        R2["indicateurs_economiques.csv"]
        R3["ipi_hvac_france.csv"]
        R4["sitadel.csv"]
        R5["dpe_france_all.csv"]
    end

    subgraph Process["4. Traitement (T)"]
        P1["clean\n(doublons, NaN, clipping)"]
        P2["merge\n(jointure multi-sources)"]
        P3["features\n(lags, rolling, interactions)"]
        P4["outliers\n(IQR + Z-score + IF)"]
    end

    subgraph ML["5-6. Modelisation"]
        M1["train\n(Ridge, LightGBM, Prophet, LSTM)"]
        M2["evaluate\n(RMSE, MAE, SHAP)"]
    end

    S1 --> C1 --> R1
    S2 --> C2 --> R2
    S3 --> C3 --> R3
    S4 --> C4 --> R4
    S5 --> C5 --> R5

    R1 & R2 & R3 & R5 --> P1 --> P2 --> P3 --> P4
    P4 --> M1 --> M2

    M2 --> O2["data/models/\nfigures + rapports"]

    style Sources fill:#e1f5fe,stroke:#0288d1
    style Collect fill:#fff3e0,stroke:#f57c00
    style Raw fill:#f3e5f5,stroke:#7b1fa2
    style Process fill:#fce4ec,stroke:#c62828
    style ML fill:#fff8e1,stroke:#f9a825
```

### 1.2 Pattern ELT : justification

| Aspect | ETL classique | ELT (notre choix) |
|--------|--------------|-------------------|
| Ordre | Transform avant Load | Load brut, puis Transform |
| Stockage brut | Non | Oui (`data/raw/`) |
| Rejouabilite | Limitee | Totale (raw conserve) |
| Debug | Difficile | Facile (comparer raw vs processed) |

**Pourquoi ELT ?** Les CSV bruts dans `data/raw/` sont la source de verite. Si une regle
de nettoyage change, on relance `clean` sans re-collecter. Chaque etape est idempotente.
Pour un projet portfolio, la transparence raw/processed est essentielle.

---

## 2. Detail de chaque etape

### 2.1 Collect — Collecte des donnees brutes

| | |
|---|---|
| **Entrees** | APIs externes (Open-Meteo, INSEE, Eurostat, SITADEL, ADEME) |
| **Sorties** | `data/raw/{source}/{source}.csv` |
| **CLI** | `python -m src.pipeline collect` ou `--sources weather,insee` |
| **Temps** | 15-45 min (96 depts, 5 APIs, rate limiting) |
| **Idempotent** | Oui (ecrase le raw precedent) |

| Source | API | Volume | Grain |
|--------|-----|--------|-------|
| weather | Open-Meteo Archive + Elevation | ~250K lignes | jour x departement |
| insee | INSEE BDM (SDMX) | ~500 lignes | mois (national) |
| eurostat | Eurostat SDMX | ~200 lignes | mois x code NACE |
| sitadel | SITADEL2 | ~5K lignes | mois x departement |
| dpe | ADEME Data Fair | ~1.4M lignes | DPE individuel |

### 2.2-2.3 Init DB + Import Data

| | Init DB | Import Data |
|---|---------|-------------|
| **Sorties** | Schema BDD (tables, index) | Tables peuplees |
| **CLI** | `python -m src.pipeline init_db` | `python -m src.pipeline import_data` |
| **Temps** | < 5 sec | 1-5 min |
| **Idempotent** | Oui (CREATE IF NOT EXISTS) | Oui (deduplication cle primaire) |

### 2.4 Clean — Nettoyage des donnees brutes

| | |
|---|---|
| **Entrees** | `data/raw/{source}/*.csv` |
| **Sorties** | `data/processed/{source}/*.csv` |
| **CLI** | `python -m src.pipeline clean` |
| **Temps** | 30 sec - 2 min |
| **Idempotent** | Oui (lit raw, ecrit processed, deterministe) |

```mermaid
flowchart LR
    subgraph Weather["Meteo"]
        W1["Dates"] --> W2["Dedup\ndate x ville"] --> W3["Clip\ntemp -30/+50"] --> W4["HDD/CDD"]
    end
    subgraph INSEE["INSEE"]
        I1["Filtrage\nYYYY-MM"] --> I2["Dedup"] --> I3["Interpolation\ngaps <= 3 mois"]
    end
    subgraph DPE["DPE ADEME"]
        D1["Dedup\nnumero_dpe"] --> D2["Dates\n>= 2021-07"] --> D3["Etiquettes\nA-G"] --> D4["Detection PAC"]
    end
    style Weather fill:#e3f2fd
    style INSEE fill:#f1f8e9
    style DPE fill:#fce4ec
```

### 2.5 Merge — Fusion multi-sources

| | |
|---|---|
| **Entrees** | `data/processed/` (dpe, weather, insee, eurostat) |
| **Sorties** | `data/features/hvac_ml_dataset.csv` |
| **CLI** | `python -m src.pipeline merge` |
| **Temps** | 30 sec - 1 min |
| **Idempotent** | Oui |

```mermaid
flowchart TD
    DPE["DPE agrege\nmois x dept\n(variable cible)"] -->|"LEFT JOIN\nON date_id + dept"| J1[" "]
    METEO["Meteo agregee\nmois x dept"] --> J1
    J1 -->|"LEFT JOIN\nON date_id"| J2[" "]
    ECO["INSEE + Eurostat\nmois (national)"] --> J2
    J2 -->|"Filtre >= 202107"| DATASET["hvac_ml_dataset.csv\ngrain: mois x departement"]
    style DATASET fill:#c8e6c9,stroke:#2e7d32
```

### 2.6 Features — Feature engineering

| | |
|---|---|
| **Entrees** | `data/features/hvac_ml_dataset.csv` |
| **Sorties** | `data/features/hvac_features_dataset.csv` |
| **CLI** | `python -m src.pipeline features` |
| **Temps** | < 30 sec |
| **Idempotent** | Oui |

| Categorie | Exemples | Justification |
|-----------|----------|---------------|
| Lags (1, 3, 6 mois) | `nb_pac_lag_1m` | Auto-correlation |
| Rolling (3, 6 mois) | `nb_pac_rmean_3m` | Lissage, volatilite |
| Variations | `nb_pac_diff_1m` | Acceleration marche |
| Interactions | `interact_hdd_confiance` | Hypotheses metier croisees |
| **Efficacite PAC** | `cop_proxy`, `pac_viability_score` | **Viabilite pompe a chaleur** |
| **Geo/Altitude** | `altitude_mean`, `pct_zone_montagne`, `densite_pop` | **Profil geographique** |
| Tendance | `year_trend`, `delta_temp_vs_mean` | Croissance, anomalies meteo |

#### Focus : PAC Efficiency Features (Step 5)

Domain knowledge: air-source heat pumps (PAC air-air/air-eau) lose efficiency in cold
weather. Below -7°C, the COP (Coefficient of Performance) drops below ~2.0, making the
PAC economically uncompetitive vs gas/oil. Altitude is a structural proxy for colder base
temperatures.

**Altitude data**: Two levels of granularity are available:
- `altitude` (prefecture): Point elevation of the reference city (Open-Meteo Elevation API)
- `altitude_mean` (department): Average altitude of the entire department territory (IGN BD ALTI)
- `pct_zone_montagne`: % of territory classified as mountain zone (loi montagne)
- `densite_pop`: Population density in hab/km² (INSEE Recensement)

The COP proxy uses `altitude_mean` when available (more representative than prefecture
altitude alone), and `is_mountain` combines both altitude and mountain zone classification.

```mermaid
flowchart LR
    subgraph Inputs["Input Signals"]
        T["nb_jours_gel\n(frost days)"]
        A["altitude_mean\n(dept mean altitude)"]
        H["pct_maisons\n(housing type)"]
        P["nb_jours_pac_inefficient\n(T_min < -7°C)"]
        HDD["hdd_sum\n(heating demand)"]
        MT["pct_zone_montagne\n(% mountain territory)"]
        D["densite_pop\n(hab/km²)"]
    end
    subgraph Features["PAC Efficiency Features"]
        COP["cop_proxy\n4.5 - 0.08×frost - 0.0005×alt_mean"]
        MTN["is_mountain\nalt>800 OR montagne>50%"]
        VIA["pac_viability_score\n0.5×COP + 0.3×housing + 0.2×frost"]
        IAF["interact_altitude_frost\nalt_norm × frost_norm"]
        IMA["interact_maisons_altitude\npct_maisons × alt_norm"]
        PCT["pct_jours_pac_inefficient\n% days T_min < -7°C"]
        ICH["interact_cop_hdd\nCOP/5 × HDD/max"]
        IMD["interact_montagne_densite\nmontagne × inv_density"]
    end
    T --> COP & VIA & IAF
    A --> COP & MTN & IAF & IMA
    H --> VIA & IMA
    P --> PCT
    MT --> MTN & IMD
    D --> IMD
    COP --> VIA & ICH
    HDD --> ICH
    style Features fill:#e8f5e9,stroke:#2e7d32
```

| Feature | Formula | Range | Business Meaning |
|---------|---------|-------|-----------------|
| `cop_proxy` | 4.5 - 0.08×frost - 0.0005×alt_mean | [1.0, 5.0] | Estimated heat pump efficiency |
| `is_mountain` | alt_mean>800 OR montagne>50% | 0/1 | Mountain department flag |
| `pac_viability_score` | 0.5×COP_norm + 0.3×housing + 0.2×frost_penalty | [0, 1] | Composite PAC suitability |
| `interact_altitude_frost` | alt_norm × frost_norm | [0, 1] | Cold mountain penalty |
| `interact_maisons_altitude` | pct_maisons/100 × alt_norm | [0, 1] | Mountain houses potential |
| `pct_jours_pac_inefficient` | nb_jours_pac_ineff / 30 × 100 | [0, 100] | % days with critical COP |
| `interact_cop_hdd` | COP/5 × HDD/max | [0, 1] | Heating demand vs efficiency |
| `interact_montagne_densite` | montagne/100 × (1 - density/max) | [0, 1] | Sparse mountain constraint |

**Data sources for PAC features:**
- `altitude`: Static reference (`PREFECTURE_ELEVATIONS` in `config/settings.py`, 96 departments)
- `altitude_mean`: Department mean altitude (IGN BD ALTI, in `reference_departements.csv`)
- `pct_zone_montagne`: % territory classified mountain (loi montagne, in `reference_departements.csv`)
- `densite_pop`: Population density hab/km² (INSEE Recensement, in `reference_departements.csv`)
- `nb_jours_gel`: Weather aggregation (days with T_min < 0°C)
- `nb_jours_pac_inefficient`: Weather aggregation (days with T_min < -7°C)
- `pct_maisons`: INSEE Filosofi reference data (% houses vs apartments)
- `elevation` (raw): Open-Meteo Elevation API (Copernicus DEM GLO-90) with static fallback

### 2.7 Outliers — Detection multi-methode

| | |
|---|---|
| **Entrees** | `data/features/hvac_features_dataset.csv` |
| **Sorties** | Dataset traite (ecrase) + `data/analysis/outlier_report.txt` |
| **CLI** | `python -m src.pipeline outliers` |
| **Temps** | < 30 sec |
| **Idempotent** | Oui |

```mermaid
flowchart LR
    DATA["Dataset"] --> IQR["IQR\nfactor=1.5"]
    DATA --> Z["Z-score\nMAD seuil=3.5"]
    DATA --> IF["Isolation\nForest 5%"]
    IQR & Z & IF --> VOTE["Consensus\n>= 2 methodes"]
    VOTE --> CLIP["Winsorization\n(defaut)"]
    style VOTE fill:#fff9c4,stroke:#f9a825
```

### 2.8-2.9 Train + Evaluate

| | Train | Evaluate |
|---|-------|---------|
| **Entrees** | Dataset features | Modeles entraines |
| **Sorties** | `data/models/*.pkl` | Figures, rapport, SHAP |
| **CLI** | `python -m src.pipeline train` | `python -m src.pipeline evaluate` |
| **Temps** | 1-5 min | 1-3 min |
| **Idempotent** | Oui (random_state fixe) | Oui |

**Modeles** : Ridge (baseline), LightGBM (principal), Prophet (saisonnalite), LSTM (exploratoire).
**Split temporel** : Train 2021-07/2024-06 | Val 2024-07/2024-12 | Test 2025-01/2025-12.

### 2.10 Commandes composites

| Commande | Etapes | Usage |
|----------|--------|-------|
| `process` | clean + merge + features + outliers | Retraitement complet |
| `all` | init_db + collect + import + process + eda + train + evaluate | Pipeline bout en bout |
| `update_all` | collect + init_db + import + process + eda + train + evaluate + upload | Mise a jour + cloud |

---

## 3. Modularite et reproductibilite

### 3.1 Configuration centralisee

`config/settings.py` utilise des **dataclasses immuables** (`frozen=True`). Aucun chemin en dur.

```mermaid
classDiagram
    class ProjectConfig {
        +GeoConfig geo
        +TimeConfig time
        +NetworkConfig network
        +DatabaseConfig database
        +ModelConfig model
        +ThresholdsConfig thresholds
        +Path raw_data_dir
        +Path processed_data_dir
        +Path features_data_dir
        +from_env()$ ProjectConfig
    }
    class GeoConfig {
        +str region_code
        +list departments
        +dict cities
    }
    class TimeConfig {
        +str start_date / end_date
        +str train_end / val_end
    }
    class NetworkConfig {
        +int request_timeout
        +int max_retries
    }
    class DatabaseConfig {
        +str db_type
        +connection_string() str
    }
    class ModelConfig {
        +int max_lag_months
        +list rolling_windows
        +dict lightgbm_params
    }
    class ThresholdsConfig {
        +float heatwave_temp = 35.0
        +float frost_temp = 0.0
        +float pac_inefficiency_temp = -7.0
        +float cost_outlier_max = 50000
    }
    ProjectConfig *-- GeoConfig
    ProjectConfig *-- TimeConfig
    ProjectConfig *-- NetworkConfig
    ProjectConfig *-- DatabaseConfig
    ProjectConfig *-- ModelConfig
    ProjectConfig *-- ThresholdsConfig
```

### 3.2 Variables d'environnement (.env)

| Variable | Defaut | Description |
|----------|--------|-------------|
| `TARGET_REGION` | `FR` | `FR` = 96 depts, `84` = AURA |
| `TARGET_DEPARTMENTS` | (vide) | Override : `69,38,42` |
| `DATA_START_DATE` | `2019-01-01` | Debut collecte |
| `DB_TYPE` | `sqlite` | `sqlite`, `mssql`, `postgresql` |
| `REQUEST_TIMEOUT` | `30` | Timeout HTTP (sec) |
| `MAX_RETRIES` | `3` | Retries reseau |
| `RAW_DATA_DIR` | `data/raw` | Repertoire brut |
| `LOG_LEVEL` | `INFO` | Niveau logging |

### 3.3 Architecture plugin des collecteurs

```mermaid
classDiagram
    class BaseCollector {
        <<abstract>>
        +str source_name*
        +run() CollectorResult
        +collect()* DataFrame
        +validate(df)* DataFrame
        +fetch_json() / fetch_xml() / fetch_bytes()
        -__init_subclass__() auto-enregistrement
    }
    class CollectorRegistry {
        -dict _collectors$
        +available()$ list
        +run(name, config)$ CollectorResult
        +run_all(config)$ list
    }
    BaseCollector <|-- WeatherCollector : source_name="weather"
    BaseCollector <|-- InseeCollector : source_name="insee"
    BaseCollector <|-- EurostatCollector : source_name="eurostat"
    BaseCollector <|-- SitadelCollector : source_name="sitadel"
    BaseCollector <|-- DpeCollector : source_name="dpe"
    BaseCollector ..> CollectorRegistry : auto-enregistrement
    style BaseCollector fill:#e3f2fd,stroke:#1565c0
    style CollectorRegistry fill:#fff3e0,stroke:#ef6c00
```

**Ajout d'une source** : creer un fichier dans `src/collectors/`, heriter de `BaseCollector`,
implementer `collect()` et `validate()`. L'auto-enregistrement via `__init_subclass__` rend la
source disponible sans modifier aucun fichier existant.

---

## 4. Logs structures

### 4.1 Format et niveaux

```
%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s
```

| Niveau | Usage |
|--------|-------|
| `DEBUG` | Requetes HTTP, URLs, parametres internes |
| `INFO` | Demarrage/fin d'etape, volumes, chemins de sortie |
| `WARNING` | Fichier manquant (non bloquant), NaN restants |
| `ERROR` | Echec collecte, dataset vide, source introuvable |

### 4.2 Exemple de sortie (run complet)

```
2026-02-21 14:30:00 | pipeline                  | INFO     | ============================================================
2026-02-21 14:30:00 | pipeline                  | INFO     |   HVAC Market Analysis — Pipeline
2026-02-21 14:30:00 | pipeline                  | INFO     |   Etape : all
2026-02-21 14:30:00 | pipeline                  | INFO     | ============================================================
2026-02-21 14:30:01 | collectors.weather        | INFO     |   COLLECTE : WEATHER
2026-02-21 14:30:01 | collectors.weather        | INFO     |   Periode  : 2019-01-01 -> 2026-02-28
2026-02-21 14:35:24 | collectors.weather        | INFO     | Collecte reussie : 254016 lignes
...
2026-02-21 14:48:00 | processing.clean          | INFO     |   PHASE 2.1 — Nettoyage des donnees brutes
2026-02-21 14:48:01 | processing.clean          | INFO     |   Lignes brutes : 254016
2026-02-21 14:48:02 | processing.clean          | INFO     |   Meteo nettoyee : 254016 -> 254016 lignes
...
2026-02-21 14:52:00 | processing.merge          | INFO     | Dataset ML : 4032 lignes x 35 colonnes
2026-02-21 14:52:30 | processing.features       | INFO     | Features ajoutees : 52 nouvelles colonnes
2026-02-21 14:53:00 | processing.outliers       | INFO     | Winsorization : 127 valeurs clippees
...
2026-02-21 14:58:00 | models.train              | INFO     | Phase 4 terminee : 4 modeles entraines.
2026-02-21 15:01:00 | models.evaluate           | INFO     | Phase 4 (evaluate) terminee.
2026-02-21 15:01:00 | pipeline                  | INFO     | Pipeline termine.
```

---

## 5. Alternative Airbyte

### 5.1 Mapping des sources vers Airbyte

| Source | Collecteur custom | Connecteur Airbyte |
|--------|-------------------|-------------------|
| Open-Meteo | `WeatherCollector` | HTTP API Source (generique) |
| INSEE BDM | `InseeCollector` | HTTP API Source (SDMX/XML) |
| Eurostat | `EurostatCollector` | HTTP API Source (SDMX) |
| SITADEL | `SitadelCollector` | HTTP API Source ou File Source |
| ADEME DPE | `DpeCollector` | HTTP API Source (paginee, a configurer) |

### 5.2 Architecture alternative

```mermaid
flowchart TD
    subgraph Sources["Sources API"]
        S1["Open-Meteo"] & S2["INSEE"] & S3["Eurostat"] & S4["SITADEL"] & S5["ADEME"]
    end
    subgraph Airbyte["Airbyte (EL)"]
        CONN["5 connecteurs HTTP"] --> ORCH["Orchestrateur"] --> NORM["Normalisation"]
    end
    subgraph Dest["Destination"]
        D1[("PostgreSQL\n_airbyte_raw")]
    end
    subgraph Transform["dbt (T)"]
        T1["staging"] --> T2["intermediate"] --> T3["mart"]
    end
    subgraph ML["Pipeline ML"]
        M1["Features"] --> M2["Train"] --> M3["Evaluate"]
    end
    S1 & S2 & S3 & S4 & S5 --> CONN
    NORM --> D1 --> T1
    T3 --> M1
    style Airbyte fill:#e8eaf6,stroke:#3f51b5
    style Transform fill:#f3e5f5,stroke:#7b1fa2
```

### 5.3 Comparaison

| Critere | Collecteurs custom | Airbyte |
|---------|-------------------|---------|
| **Controle** | Total (Python) | Config UI/YAML |
| **Complexite** | Faible (pip install) | Moyenne (Docker) |
| **Maintenance** | Manuelle (si API change) | Communaute |
| **Monitoring** | Logs custom | Interface web native |
| **Scheduling** | Cron manuel | Natif |
| **Incremental sync** | A implementer | Natif (CDC) |
| **Portfolio** | Demontre Python | Demontre Data Eng |

**Recommandation** : pour un projet portfolio Data Analyst, les collecteurs custom sont
preferes car ils demontrent la maitrise Python, la gestion d'erreurs et la conception
d'architecture extensible. Airbyte serait pertinent en production avec plus de volume.

---

## 6. Monitoring et alerting

### 6.1 Trois axes de surveillance

```mermaid
flowchart TD
    subgraph F["Fraicheur"]
        F1["Age des fichiers raw"]
        F2["Ecart date_max vs aujourd'hui"]
    end
    subgraph V["Volume"]
        V1["Nb lignes par source"]
        V2["Nb departements couverts"]
        V3["Mois manquants"]
    end
    subgraph Q["Qualite"]
        Q1["Taux NaN par colonne"]
        Q2["Nb outliers"]
        Q3["Metriques ML (R2, RMSE)"]
    end
    F & V & Q --> ALERT["Alerte"]
    ALERT --> LOG["Log WARNING/ERROR"]
    ALERT --> NOTIF["Email / Slack"]
    style F fill:#e3f2fd,stroke:#1565c0
    style V fill:#e8f5e9,stroke:#2e7d32
    style Q fill:#fce4ec,stroke:#c62828
```

### 6.2 Seuils d'alerte recommandes

| Metrique | WARNING | CRITICAL |
|----------|---------|----------|
| Age donnees raw | > 7 jours | > 30 jours |
| Lignes meteo (raw) | < 200 000 | < 100 000 |
| Lignes DPE (raw) | < 500 000 | < 100 000 |
| Lignes dataset ML | < 2 000 | < 500 |
| Departements dans ML | < 90 | < 50 |
| Taux NaN (cible) | > 5% | > 20% |
| Taux NaN (features) | > 15% | > 40% |
| Outliers consensus | > 10% | > 25% |
| R2 meilleur modele | < 0.5 | < 0.2 |
| MAPE meilleur modele | > 30% | > 50% |
| Duree run complet | > 60 min | > 120 min |
| Collecteurs en echec | 1 source | >= 3 sources |

### 6.3 Outils complementaires envisageables

| Outil | Usage | Complexite |
|-------|-------|-----------|
| **Great Expectations** | Tests qualite donnees automatises | Moyenne |
| **Soda Core** | Verification qualite (SodaCL) | Faible |
| **Prometheus + Grafana** | Monitoring temps reel | Elevee |
| **dbt tests** | Tests integres aux transformations SQL | Faible |

---

*Pipeline orchestre par `src/pipeline.py` — CLI : `python -m src.pipeline <etape>`*
