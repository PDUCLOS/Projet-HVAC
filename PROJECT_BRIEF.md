# Projet Portfolio : Analyse du Marché HVAC en Rhône-Alpes
## Prédiction des ventes par ML/Deep Learning

**Auteur** : Patrice — Data Analyst Senior
**Stack** : Python, SQL (SQLite local / SQL Server), Power BI
**Objectif** : Projet portfolio pour recherche d'emploi Data Analyst

---

## 1. Vision du projet

Construire un pipeline data complet (collecte → nettoyage → analyse → prédiction → dashboard) sur le marché HVAC en région Auvergne-Rhône-Alpes, en croisant des données énergétiques, météorologiques, économiques et de confiance des ménages pour prédire les volumes de ventes/installations d'équipements de chauffage et climatisation.

**Principes de conception :**
- Stockage local uniquement, sorties CSV et déduplication pour éviter les doublons
- Double sauvegarde systématique : `data/raw/` (brut) + `data/processed/` (dédupliqué)
- Base de données SQLite en local par défaut (zéro installation requise)
- Migration facile vers SQL Server pour un environnement professionnel
- Projet portable : déplaçable d'une machine à l'autre en quelques commandes

## 2. Architecture technique

```
hvac-market-analysis/
├── README.md                    # Vitrine GitHub
├── setup_project.py             # Script de setup (init sur nouvelle machine)
├── requirements.txt             # Dépendances Python core
├── requirements-dl.txt          # Dépendances Deep Learning (optionnel)
├── .env.example                 # Template de configuration
├── .env                         # Configuration locale (non versionné)
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration centralisée (dataclasses)
├── data/
│   ├── raw/                     # Données brutes téléchargées
│   │   ├── weather/             # Open-Meteo
│   │   ├── insee/               # INSEE BDM
│   │   ├── eurostat/            # Eurostat IPI
│   │   ├── sitadel/             # Permis de construire
│   │   ├── dpe/                 # DPE ADEME
│   │   └── energy/              # SDES énergie
│   ├── processed/               # Données nettoyées + dédupliquées
│   ├── features/                # Features engineerées pour ML
│   └── hvac_market.db           # Base SQLite locale
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # Orchestrateur CLI
│   ├── collectors/              # Architecture extensible par plugins
│   │   ├── __init__.py
│   │   ├── base.py              # BaseCollector + CollectorRegistry
│   │   ├── weather.py           # Open-Meteo (WeatherCollector)
│   │   ├── insee.py             # INSEE BDM (InseeCollector)
│   │   ├── eurostat_col.py      # Eurostat (EurostatCollector)
│   │   ├── sitadel.py           # SITADEL via API DiDo (SitadelCollector)
│   │   └── dpe.py               # DPE ADEME (DpeCollector)
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── clean_data.py        # Nettoyage source par source (DataCleaner)
│   │   ├── merge_datasets.py    # Fusion multi-sources → dataset ML (DatasetMerger)
│   │   └── feature_engineering.py  # Features avancées : lags, rolling, interactions (FeatureEngineer)
│   ├── database/
│   │   ├── __init__.py
│   │   ├── schema.sql           # Schéma en étoile SQLite (+ raw_dpe)
│   │   ├── schema_mssql.sql     # Schéma SQL Server (IDENTITY, BIT, MERGE)
│   │   └── db_manager.py        # CRUD + import CSV + agrégation DPE
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── eda.py               # Analyse exploratoire
│   │   └── correlation.py       # Études de corrélation
│   └── models/
│       ├── __init__.py
│       ├── baseline.py          # Modèles ML classiques
│       ├── deep_learning.py     # LSTM (exploration pédagogique)
│       ├── train.py             # Pipeline d'entraînement
│       └── evaluate.py          # Métriques & comparaison
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_results_analysis.ipynb
├── dashboards/                  # Exports Power BI
│   └── screenshots/
├── tests/
│   └── test_collectors/
└── docs/
    ├── data_dictionary.md
    └── methodology.md
```

## 3. Base de données — Stratégie multi-moteur

### 3.1 Principe
Le projet utilise **SQLAlchemy** comme couche d'abstraction. Le moteur de BDD est configurable via le fichier `.env` sans toucher au code.

### 3.2 SQLite (défaut — développement local)
```env
DB_TYPE=sqlite
DB_PATH=data/hvac_market.db
```
- Zéro installation requise (intégré à Python)
- Fichier unique, portable, copiable
- Parfait pour le développement et la démonstration

### 3.3 SQL Server (production / entreprise)
```env
DB_TYPE=mssql
DB_HOST=localhost
DB_PORT=1433
DB_NAME=hvac_market
DB_USER=sa
DB_PASSWORD=VotreMotDePasse
```
- Nécessite `pip install pyodbc` et un driver ODBC SQL Server
- Schéma SQL compatible (syntaxe ANSI standard)
- Migration en 3 étapes : installer le driver, modifier `.env`, relancer `python -m src.pipeline init_db`

### 3.4 PostgreSQL (optionnel)
```env
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=hvac_market
DB_USER=patrice
DB_PASSWORD=VotreMotDePasse
```

## 4. Portabilité du projet

### 4.1 Déplacer le projet sur une autre machine
```bash
# 1. Copier le dossier complet (ou git clone)
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer l'environnement
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac
# Éditer .env si nécessaire (BDD, chemins, etc.)

# 5. Initialiser la BDD et collecter les données
python -m src.pipeline init_db
python -m src.pipeline collect
```

### 4.2 Ce qui est versionné (Git) vs ce qui ne l'est pas
| Versionné | Non versionné (.gitignore) |
|-----------|---------------------------|
| Code source (src/, config/, tests/) | Données brutes (data/raw/) |
| requirements.txt, .env.example | Données traitées (data/processed/) |
| schema.sql, notebooks/ | Fichier .env (secrets) |
| Configurations par défaut | Base SQLite (data/*.db) |
| | Modèles entraînés (*.pkl, *.pt) |

### 4.3 Régénérer toutes les données
```bash
# Tout reconstruire depuis zéro (données + BDD + traitement)
python -m src.pipeline all

# Ou étape par étape
python -m src.pipeline init_db                        # Créer les tables
python -m src.pipeline collect --sources weather,insee # Sources spécifiques
python -m src.pipeline collect                         # Toutes les sources
python -m src.pipeline import_data                     # CSV → BDD
python -m src.pipeline clean                           # Nettoyage → data/processed/
python -m src.pipeline merge                           # Fusion → data/features/hvac_ml_dataset.csv
python -m src.pipeline features                        # Features → data/features/hvac_features_dataset.csv
python -m src.pipeline process                         # clean + merge + features en une commande
```

## 5. Sources de données — Détails techniques

### 5.1 DPE (Diagnostic Performance Énergétique) — ADEME
- **URL (CORRIGÉE)** : https://data.ademe.fr/datasets/dpe03existant
- **API** : `https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines`
- **Format** : JSON paginé par curseur (~14M entrées total)
- **Scope** : DPE depuis juillet 2021
- **Filtrage** : départements 01, 07, 26, 38, 42, 69, 73, 74
- **Volume collecté** : 1 377 781 DPE (30 colonnes, 300 Mo CSV, 316 Mo SQLite)
- **Usage** : Proxy des "ventes" — chaque DPE mentionnant PAC/climatisation = installation récente
- **Détection PAC** : champ `type_generateur_chauffage_principal` (PAC air/eau) + `type_generateur_froid` (PAC air/air réversible)
- **Collecteur** : `src/collectors/dpe.py` (DpeCollector)
- **Table BDD** : `raw_dpe` (données unitaires) → agrégé dans `fact_hvac_installations`

### 5.2 Météo historique — Open-Meteo
- **API** : `https://archive-api.open-meteo.com/v1/archive`
- **Auth** : Aucune (gratuit, fair use ~10 000 appels/jour)
- **Variables** : temperature_2m_max/min/mean, precipitation_sum, wind_speed_10m_max
- **Calculées** : HDD = max(0, 18 - temp_mean), CDD = max(0, temp_mean - 18)
- **8 villes** : Lyon, Grenoble, Saint-Etienne, Annecy, Valence, Chambery, Bourg-en-Bresse, Privas
- **Collecteur** : `src/collectors/weather.py` (WeatherCollector)

### 5.3 INSEE BDM — Confiance ménages & Climat des affaires
- **API SDMX** : `https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}`
- **Format** : XML SDMX 2.1 (StructureSpecific)
- **Séries** :

  | Indicateur | idbank | Statut |
  |-----------|--------|--------|
  | Confiance des ménages (synthétique) | 001759970 | OK |
  | Climat des affaires (tous secteurs) | 001565530 | OK (CORRIGE) |
  | Climat des affaires (bâtiment) | 001586808 | OK |
  | IPI Industrie manufacturière | 010768261 | OK |

- **Collecteur** : `src/collectors/insee.py` (InseeCollector)

### 5.4 Permis de construire — SITADEL
- **Source** : API DiDo du SDES (MIGRATION — les ZIP directs n'existent plus)
- **API** : `https://data.statistiques.developpement-durable.gouv.fr/dido/api/v1/datafiles/{rid}/csv`
- **Filtrage** : DEP in [01, 07, 26, 38, 42, 69, 73, 74]
- **Collecteur** : `src/collectors/sitadel.py` (SitadelCollector)

### 5.5 Eurostat — Production industrielle HVAC
- **Package Python** : `pip install eurostat`
- **Dataset** : sts_inpr_m (Short-term statistics, Industrial Production, Monthly)
- **Filtres** : geo=FR, nace_r2=[C28, C2825], unit=I21, s_adj=SCA
- **Collecteur** : `src/collectors/eurostat_col.py` (EurostatCollector)

### 5.6 Consommation énergie locale — SDES
- **Source** : API DiDo (`data.statistiques.developpement-durable.gouv.fr/dido/api/v1/`)
- **Granularité** : Annuelle (pas mensuelle — usage en EDA uniquement)

## 6. Modèle de données (Schéma en étoile — corrigé)

```
                    ┌──────────────┐
                    │  dim_time    │
                    │──────────────│
                    │ date_id (PK) │ ← Format YYYYMM
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
│ dept_name    │   │ nb_dpe_total             │ ← Variable cible
│ city_ref     │   │ nb_installations_pac     │ ← Variable cible
│ latitude     │   │ nb_installations_clim    │
│ longitude    │   │ temp_mean, HDD, CDD      │ ← Météo locale
│ region_code  │   │ nb_permis_construire     │ ← Immobilier local
└──────────────┘   │ UNIQUE(date_id, geo_id)  │
                   └──────┬────────────────────┘
                          │
                   ┌──────┴────────────────────┐
                   │ fact_economic_context      │ ← Grain = mois seul
                   │───────────────────────────│   (données nationales)
                   │ date_id (PK, FK)          │
                   │ confiance_menages          │
                   │ climat_affaires_indus      │
                   │ climat_affaires_bat        │
                   │ ipi_manufacturing          │
                   │ ipi_hvac_c28, ipi_hvac_c2825│
                   └───────────────────────────┘
```

**Table brute additionnelle :**
```
┌─────────────────────────┐
│ raw_dpe (1.38M lignes)  │   ← Grain = 1 DPE individuel
│─────────────────────────│
│ numero_dpe (PK)         │
│ date_etablissement_dpe  │
│ code_departement_ban    │
│ etiquette_dpe (A-G)     │
│ type_generateur_chauffage│   → Détection PAC
│ type_generateur_froid    │   → Détection PAC air/air + clim
│ surface, isolation, coûts│   → 30 colonnes au total
│ INDEX(date, dept)        │
└─────────────────────────┘
```
**Agrégation** : `raw_dpe` → `fact_hvac_installations` (nb_dpe_total, nb_installations_pac, nb_installations_clim, nb_dpe_classe_ab) par mois × département.

**Correction audit** : les indicateurs économiques (nationaux) sont séparés dans leur propre table de faits. Plus de duplication sur 8 départements.

## 7. Feature Engineering pour ML (IMPLÉMENTÉ ✓)

### Dataset ML-ready (`data/features/hvac_ml_dataset.csv`)
- **Grain** : mois × département (448 lignes × 35 colonnes)
- **Pipeline** : `python -m src.pipeline merge`
- **Module** : `src/processing/merge_datasets.py` (DatasetMerger)

### Features dataset (`data/features/hvac_features_dataset.csv`)
- **Grain** : mois × département (448 lignes × 90 colonnes)
- **Pipeline** : `python -m src.pipeline features`
- **Module** : `src/processing/feature_engineering.py` (FeatureEngineer)

### Catégories de features (90 colonnes)

**Temporelles (7)**
- month, quarter, year, is_heating, is_cooling
- month_sin, month_cos (encoding cyclique)
- year_trend (tendance linéaire normalisée)

**Lags temporels (21)** — par département
- lag_1m, lag_3m, lag_6m sur : nb_dpe_total, nb_installations_pac,
  nb_installations_clim, temp_mean, hdd_sum, cdd_sum, confiance_menages

**Rolling windows (20)** — par département
- rolling_mean_3m, rolling_mean_6m, rolling_std_3m, rolling_std_6m
  sur : nb_dpe_total, nb_installations_pac, temp_mean, hdd_sum, cdd_sum

**Variations (6)**
- diff_1m, pct_change_1m sur : nb_dpe_total, nb_installations_pac, nb_installations_clim

**Interactions (4)**
- interact_hdd_confiance : HDD × confiance ménages
- interact_cdd_ipi : CDD × IPI HVAC industriel
- interact_confiance_bat : confiance × climat bâtiment
- jours_extremes : canicule + gel

**Météo (8)** — locales par département
- temp_mean, temp_max, temp_min
- hdd_sum (base 18°C), cdd_sum
- precipitation_sum, nb_jours_canicule, nb_jours_gel
- delta_temp_vs_mean (écart à la moyenne historique)

**Économiques (6)** — nationales
- confiance_menages, climat_affaires_indus, climat_affaires_bat
- ipi_manufacturing, ipi_hvac_c28, ipi_hvac_c2825

### Features cibles (Y)
- nb_installations_pac : DPE avec PAC par mois/département
- nb_installations_clim : DPE avec climatisation
- nb_dpe_total : volume total de DPE
- nb_dpe_classe_ab : DPE classe A-B (bâtiment performant)
- pct_pac, pct_clim, pct_classe_ab : pourcentages dérivés

## 8. Modélisation ML / Deep Learning

### 8.1 Hiérarchie des modèles (audit : adapté au volume de données)

| Tier | Modèle | Usage | Viabilité |
|------|--------|-------|-----------|
| 1 (robuste) | Ridge Regression | Baseline | ~288 lignes OK |
| 1 (robuste) | Prophet + régresseurs | Séries temporelles | 36 pts/dept OK |
| 2 (faisable) | LightGBM (régularisé) | Non-linéarités | Avec tuning |
| 3 (pédagogique) | LSTM minimaliste | Exploration DL | Honnêteté requise |

**Transformer retiré** du scope (insuffisance de données).

### 8.2 Métriques
- RMSE, MAE, MAPE, R²
- SHAP values (feature importance)
- Cross-validation temporelle (TimeSeriesSplit)

### 8.3 Split temporel
```
Train  : 2021-07 → 2024-06 (36 mois)
Val    : 2024-07 → 2024-12 (6 mois)
Test   : 2025-01 → 2025-12 (12 mois)
```

## 9. Livrables finaux

1. **Repo GitHub** propre avec README complet
2. **Notebooks Jupyter** commentés (EDA + Modélisation)
3. **Dashboard Power BI** avec KPIs et prédictions
4. **Article Medium** racontant le projet (storytelling data)
5. **Rapport PDF** de synthèse avec résultats ML

## 10. Ordre d'exécution

```
Phase 1 — Setup & Collecte (TERMINÉE ✓)
  ├── 1.1 Init repo, venv, requirements, config centralisée   ✓
  ├── 1.2 Architecture extensible (BaseCollector + Registry)   ✓
  ├── 1.3 Collecter Open-Meteo (20 456 lignes, 8 villes)      ✓
  ├── 1.4 Collecter INSEE BDM (85 lignes mensuelles, 4 séries)✓
  ├── 1.5 Collecter Eurostat IPI (168 lignes, 2 NACE)         ✓
  ├── 1.6 Collecter SITADEL (migration DiDo en cours)         ~
  ├── 1.7 DPE ADEME (1 377 781 DPE, 300 Mo CSV)              ✓
  ├── 1.8 BDD SQLite initialisée + import complet (316 Mo)    ✓
  ├── 1.9 Support dual SQLite / SQL Server / PostgreSQL        ✓
  ├── 1.10 Script portabilité (setup_project.py)               ✓
  └── 1.11 Agrégation DPE → fact_hvac_installations            ✓
      Volume total data/ : 1.2 Go

Phase 2 — Traitement (TERMINÉE ✓)
  ├── 2.1 Nettoyage de chaque source (clean_data.py)            ✓
  │       Météo: 20 456 lignes, 0 doublons, HDD/CDD recalculés
  │       INSEE: 114 → 85 (filtre mensuel), 2 interpolations
  │       Eurostat: 168 lignes, 4 variations > 30% flaggées
  │       DPE: 1 377 781 lignes, 14 105 valeurs clippées, 6.9% PAC
  ├── 2.2 Créer la base SQL (schéma en étoile corrigé)          ✓ (fait en Phase 1)
  ├── 2.3 Charger les données                                   ✓ (fait en Phase 1)
  ├── 2.4 Fusionner en dataset ML-ready (merge_datasets.py)     ✓
  │       448 lignes × 35 colonnes (mois × département)
  │       Période: 2021-07 → 2026-02 (55 mois × 8 depts)
  │       → data/features/hvac_ml_dataset.csv
  └── 2.5 Feature engineering (feature_engineering.py)           ✓
          448 lignes × 90 colonnes (+55 features avancées)
          Lags: 1m, 3m, 6m | Rolling: 3m, 6m (mean + std)
          Variations: diff + pct_change | Interactions: 4
          Complétude: 95.3% (NaN = lags début série)
          → data/features/hvac_features_dataset.csv

Phase 3 — Analyse
  ├── 3.1 EDA notebook
  └── 3.2 Corrélations

Phase 4 — Modélisation
  ├── 4.1 Modèles baseline (Ridge, LightGBM, Prophet)
  ├── 4.2 LSTM exploratoire
  ├── 4.3 Évaluation comparative
  └── 4.4 SHAP analysis

Phase 5 — Restitution
  ├── 5.1 Dashboard Power BI
  ├── 5.2 README GitHub
  └── 5.3 Article Medium
```

## 11. Extensibilité

Le projet est conçu pour ajouter facilement :

### Nouvelle source de données
Créer un fichier dans `src/collectors/` avec une classe héritant de `BaseCollector` (~50 lignes). Auto-enregistré dans le registry.

### Nouveau modèle ML
Ajouter une classe dans `src/models/` suivant l'interface commune.

### Changer de moteur BDD
Modifier uniquement le fichier `.env` (DB_TYPE, DB_HOST, etc.). Aucun changement de code requis.
