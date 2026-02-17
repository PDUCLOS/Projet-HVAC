# HVAC Market Analysis — Auvergne-Rhone-Alpes

**Analyse predictive du marche HVAC (chauffage, ventilation, climatisation) en region Auvergne-Rhone-Alpes par croisement de donnees energetiques, meteorologiques et economiques.**

> Projet portfolio Data Analyst — Pipeline complet de la collecte a la prediction ML.

---

## Vue d'ensemble

Ce projet construit un pipeline data de bout en bout pour analyser et predire les installations d'equipements HVAC (pompes a chaleur, climatisation) dans les 8 departements de la region AURA, en croisant :

- **1.4M diagnostics energetiques** (DPE ADEME) comme proxy des installations
- **Donnees meteo historiques** (Open-Meteo, 8 villes, 7 ans)
- **Indicateurs economiques** (INSEE, Eurostat)
- **Feature engineering avance** (90 features : lags, rolling, interactions)

### Resultats cles

| Metrique | Valeur |
|----------|--------|
| DPE collectes | 1 377 781 |
| Pompes a chaleur detectees | 95 140 (6.9%) |
| Dataset ML | 448 lignes x 90 features |
| Periode couverte | Juillet 2021 - Fevrier 2026 |
| Volume total des donnees | ~1.2 Go |

---

## Architecture technique

```
hvac-market-analysis/
├── config/settings.py          # Configuration centralisee (dataclasses)
├── src/
│   ├── pipeline.py             # Orchestrateur CLI (10 etapes + all)
│   ├── collectors/             # Collecte de donnees (architecture plugin)
│   │   ├── base.py             # BaseCollector + Registry auto-enregistrement
│   │   ├── weather.py          # Open-Meteo
│   │   ├── insee.py            # INSEE BDM (SDMX)
│   │   ├── eurostat_col.py     # Eurostat IPI
│   │   ├── sitadel.py          # Permis de construire (DiDo)
│   │   └── dpe.py              # DPE ADEME (1.4M lignes)
│   ├── processing/             # Traitement des donnees
│   │   ├── clean_data.py       # Nettoyage source par source
│   │   ├── merge_datasets.py   # Fusion multi-sources
│   │   └── feature_engineering.py  # Features ML avancees
│   ├── models/                 # Modelisation ML (Phase 4)
│   │   ├── baseline.py         # Ridge, LightGBM, Prophet
│   │   ├── deep_learning.py    # LSTM (PyTorch, exploratoire)
│   │   ├── train.py            # Orchestrateur d'entrainement
│   │   └── evaluate.py         # Metriques, SHAP, visualisations
│   ├── analysis/               # Analyse exploratoire (Phase 3)
│   │   ├── eda.py              # EDA automatisee
│   │   └── correlation.py      # Matrice de correlations
│   └── database/               # Persistance
│       ├── schema.sql          # Schema en etoile (SQLite)
│       ├── schema_mssql.sql    # Schema SQL Server
│       └── db_manager.py       # Import CSV + aggregation DPE
├── tests/                      # Tests unitaires (57 tests)
│   ├── test_config.py          # Tests configuration
│   ├── test_collectors/        # Tests collecteurs
│   └── test_processing/        # Tests nettoyage + features
├── notebooks/                  # Jupyter notebooks (EDA, ML)
├── setup_project.py            # Script d'initialisation (nouvelle machine)
├── requirements.txt            # Dependances Python
└── .env.example                # Template de configuration
```

### Points techniques notables

- **Architecture plugin** : ajouter une source = creer un fichier, zero config
- **Multi-BDD** : SQLite (defaut) / SQL Server / PostgreSQL via SQLAlchemy
- **Pipeline CLI** : `python -m src.pipeline <commande>`
- **Idempotent** : chaque etape peut etre relancee sans risque

---

## Installation rapide

### Prerequis

- Python 3.10+
- Git

### Setup en 5 commandes

```bash
# 1. Cloner le repo
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd hvac-market-analysis

# 2. Creer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Installer les dependances
pip install -r requirements.txt

# 4. Configurer l'environnement
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac

# 5. Initialiser et collecter (tout-en-un)
python setup_project.py
```

### Collecter les donnees et traiter

```bash
# Pipeline complet (collecte + BDD + traitement)
python -m src.pipeline all

# Ou etape par etape
python -m src.pipeline init_db          # Creer les tables SQLite
python -m src.pipeline collect          # Collecter toutes les sources (~1h pour DPE)
python -m src.pipeline import_data      # Importer CSV dans la BDD
python -m src.pipeline process          # Nettoyage + fusion + features
```

> **Note** : la collecte DPE ADEME prend ~30-60 minutes (1.4M lignes via API paginee).

---

## Travailler sur plusieurs machines

Le projet est concu pour etre portable. Seul le **code** est versionne dans Git. Les **donnees** sont regenerees localement.

### Ce qui est dans Git vs ce qui ne l'est pas

| Dans Git | PAS dans Git (.gitignore) |
|----------|--------------------------|
| Code source (src/, config/) | Donnees brutes (data/raw/) ~600 Mo |
| Configuration (.env.example) | Donnees traitees (data/processed/) ~325 Mo |
| Schema SQL | Features ML (data/features/) |
| setup_project.py | Base SQLite (data/*.db) ~316 Mo |
| requirements.txt | Fichier .env (secrets) |
| Notebooks (structure) | Modeles entraines (*.pkl, *.pt) |

### Procedure sur une nouvelle machine

**Option A — Telecharger les donnees depuis pCloud (rapide, ~5 min)**

Les donnees pre-collectees (~1.5 Go) sont disponibles en telechargement :

> **[Telecharger les donnees depuis pCloud](https://e.pcloud.link/publink/show?code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy)**

```bash
# 1. Cloner
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd hvac-market-analysis

# 2. Environnement
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 3. Configuration
copy .env.example .env

# 4. Copier les donnees telecharges depuis pCloud dans data/
#    (base SQLite + raw/ + processed/ + features/)

# 5. Verifier que tout fonctionne
python -m src.pipeline list
```

**Option B — Regenerer les donnees depuis les APIs (complet, ~1h)**

```bash
# 1. Cloner
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd hvac-market-analysis

# 2. Environnement
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 3. Configuration
copy .env.example .env

# 4. Regenerer toutes les donnees (~1h)
python -m src.pipeline all
```

Apres l'une ou l'autre option, vous aurez exactement le meme etat qu'une machine existante :
- Base SQLite initialisee avec le schema en etoile
- ~1.4M DPE collectes et importes
- Meteo, INSEE, Eurostat collectes
- Dataset ML de 448 lignes x 90 features pret pour la modelisation

### Synchroniser le travail entre machines

```bash
# Machine A — apres avoir modifie du code
git add -A && git commit -m "description" && git push

# Machine B — recuperer les modifications
git pull
# Si les donnees doivent etre re-traitees :
python -m src.pipeline process
```

---

## Sources de donnees

| Source | API | Volume | Auth |
|--------|-----|--------|------|
| **DPE ADEME** | data.ademe.fr (JSON pagine) | 1.4M lignes | Aucune (Open Data) |
| **Open-Meteo** | archive-api.open-meteo.com | 20 456 lignes | Aucune |
| **INSEE BDM** | bdm.insee.fr (SDMX XML) | 85 lignes/mois | Aucune |
| **Eurostat** | via package `eurostat` | 168 lignes | Aucune |
| **SITADEL** | DiDo API (SDES) | En cours | Aucune |

> Toutes les sources sont **Open Data** — aucune cle API requise.

---

## Schema de donnees

```
dim_time (84 mois)          fact_hvac_installations (mois x dept)
  date_id (PK)       ------>  date_id (FK)
  year, month                  geo_id (FK)
  is_heating                   nb_installations_pac    <- Variable cible
  is_cooling                   temp_mean, HDD, CDD     <- Features meteo
                               nb_permis_construire

dim_geo (8 depts)           fact_economic_context (mois)
  geo_id (PK)        ------>  date_id (FK)
  dept_code, dept_name         confiance_menages
  city_ref, lat, lon           ipi_hvac_c28, c2825

raw_dpe (1.38M lignes) ---- Donnees unitaires DPE, agregees dans les faits
```

---

## Stack technique

| Categorie | Technologies |
|-----------|-------------|
| Langage | Python 3.10+ |
| Data | pandas, numpy, SQLAlchemy |
| ML | scikit-learn, LightGBM, Prophet, SHAP |
| Deep Learning | PyTorch (LSTM exploratoire) |
| Visualisation | matplotlib, seaborn, plotly |
| BDD | SQLite (defaut), SQL Server, PostgreSQL |
| Dashboard | Power BI |

---

## Pipeline du projet

```
Phase 1 — Collecte            [TERMINEE]
Phase 2 — Traitement          [TERMINEE]
Phase 3 — Analyse (EDA)       [TERMINEE]
Phase 4 — Modelisation ML     [TERMINEE]
  - Ridge Regression           R2 test = 0.998 (avec lags cible)
  - Ridge exogenes             Evalue sans auto-correlation
  - LightGBM                   R2 test = 0.865
  - Prophet                    R2 test = 0.719
  - LSTM (exploratoire)
Phase 5 — Dashboard & Article [A FAIRE]
```

### Tests unitaires

```bash
# 57 tests couvrant config, collecteurs, nettoyage et feature engineering
python -m pytest tests/ -v
```

---

## Auteur

**Patrice DUCLOS** — Data Analyst Senior (20 ans d'experience)

Projet portfolio pour recherche d'emploi Data Analyst.

---

## Licence

Ce projet est a usage personnel et pedagogique.
