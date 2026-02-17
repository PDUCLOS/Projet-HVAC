# HVAC Market Analysis — France metropolitaine

**Analyse predictive du marche HVAC (chauffage, ventilation, climatisation) sur les 96 departements metropolitains francais par croisement de donnees energetiques, meteorologiques et economiques.**

> Projet portfolio Data Analyst — Pipeline complet de la collecte a la prediction ML avec dashboard interactif Streamlit.

---

## Vue d'ensemble

Ce projet construit un pipeline data de bout en bout pour analyser et predire les installations d'equipements HVAC (pompes a chaleur, climatisation) dans les **96 departements** metropolitains, en croisant :

- **Diagnostics energetiques DPE** (ADEME) comme proxy des installations
- **Donnees meteo historiques** (Open-Meteo, 96 prefectures, 7 ans)
- **Indicateurs economiques** (INSEE, Eurostat)
- **Permis de construire** (SITADEL/DiDo)
- **Feature engineering avance** (~90 features : lags, rolling, interactions)
- **Detection d'outliers multi-methode** (IQR, Z-score modifie, Isolation Forest)

### Resultats cles

| Metrique | Valeur |
|----------|--------|
| Couverture geographique | 96 departements (France metro.) |
| Periode | 2019 - 2025 |
| Meilleur modele | Ridge (R2 test = 0.989) |
| Modeles entraines | Ridge, LightGBM, Ridge exogenes, Prophet (optionnel), LSTM (optionnel) |
| Tests unitaires | 119 tests |
| Dashboard | Streamlit (6 pages interactives) |

---

## Architecture technique

```
Projet-HVAC/
├── config/settings.py              # Configuration centralisee (dataclasses, 96 departements)
├── src/
│   ├── pipeline.py                 # Orchestrateur CLI (16 commandes)
│   ├── collectors/                 # Collecte de donnees (architecture plugin)
│   │   ├── base.py                 # BaseCollector + Registry + CollectorConfig
│   │   ├── weather.py              # Open-Meteo (96 prefectures)
│   │   ├── insee.py                # INSEE BDM (SDMX)
│   │   ├── eurostat_col.py         # Eurostat IPI
│   │   ├── sitadel.py              # Permis de construire (DiDo)
│   │   ├── dpe.py                  # DPE ADEME
│   │   └── pcloud_sync.py          # Synchronisation pCloud (download + upload)
│   ├── processing/                 # Traitement des donnees
│   │   ├── clean_data.py           # Nettoyage source par source
│   │   ├── merge_datasets.py       # Fusion multi-sources
│   │   ├── feature_engineering.py  # Features ML avancees
│   │   └── outlier_detection.py    # IQR + Z-score + Isolation Forest
│   ├── models/                     # Modelisation ML
│   │   ├── baseline.py             # Ridge, LightGBM, Prophet
│   │   ├── deep_learning.py        # LSTM (PyTorch, exploratoire)
│   │   ├── train.py                # Orchestrateur d'entrainement
│   │   └── evaluate.py             # Metriques, SHAP, visualisations
│   ├── analysis/                   # Analyse exploratoire
│   │   ├── eda.py                  # EDA automatisee
│   │   └── correlation.py          # Matrice de correlations
│   └── database/                   # Persistance
│       ├── schema.sql              # Schema en etoile (SQLite)
│       ├── schema_mssql.sql        # Schema SQL Server
│       └── db_manager.py           # Import CSV + aggregation DPE
├── app/                            # Dashboard Streamlit
│   ├── app.py                      # Point d'entree (streamlit run app/app.py)
│   └── pages/
│       ├── home.py                 # Accueil, metriques, architecture
│       ├── exploration.py          # Exploration interactive des donnees
│       ├── carte.py                # Carte de France interactive (96 depts)
│       ├── predictions.py          # Predictions ML, residus, feature importance
│       ├── models.py               # Comparaison des modeles, radar chart
│       └── pipeline_page.py        # Etat des donnees, lancement pipeline
├── scripts/
│   └── generate_demo_data.py       # Generateur de donnees de demo (offline)
├── tests/                          # Tests unitaires (119 tests)
├── setup_project.py                # Script d'initialisation
├── Makefile                        # Commandes raccourcies
├── deploy.sh                       # Script de deploiement one-click
├── requirements.txt                # Dependances Python
├── requirements-dl.txt             # Dependances Deep Learning (optionnel)
└── .env.example                    # Template de configuration
```

### Points techniques notables

- **Architecture plugin** : ajouter une source = creer un fichier, zero config
- **Multi-BDD** : SQLite (defaut) / SQL Server / PostgreSQL via SQLAlchemy
- **Pipeline CLI** : `python -m src.pipeline <commande>` (16 commandes)
- **Idempotent** : chaque etape peut etre relancee sans risque
- **Detection d'outliers** : triple methode (IQR + Z-score modifie + Isolation Forest)
- **ML robuste** : RobustScaler, gestion des NaN par imputation
- **Dashboard Streamlit** : 6 pages interactives avec Plotly
- **Synchronisation pCloud** : upload/download automatique des donnees

---

## Deploiement local rapide

### Prerequis

- Python 3.10+
- Git
- ~2 Go d'espace disque (donnees + modeles)

### Option 1 — Deploiement one-click (recommande)

```bash
# Cloner le repo
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC

# Lancer le deploiement automatique
chmod +x deploy.sh
./deploy.sh
```

Le script `deploy.sh` execute automatiquement :
1. Creation de l'environnement virtuel Python
2. Installation des dependances
3. Configuration du .env
4. Creation des repertoires de donnees
5. Generation des donnees de demonstration
6. Pipeline complet (clean → merge → features → outliers → train → evaluate)
7. Lancement du dashboard Streamlit

### Option 2 — Avec Make

```bash
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC

make install       # Creer venv + installer dependances
make demo          # Generer les donnees de demo
make pipeline      # Executer le pipeline complet
make dashboard     # Lancer le dashboard Streamlit
```

### Option 3 — Manuel, etape par etape

```bash
# 1. Cloner le repo
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC

# 2. Creer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dependances
pip install -r requirements.txt

# 4. Configurer l'environnement
cp .env.example .env            # Linux/Mac
# copy .env.example .env        # Windows

# 5. Generer les donnees de demonstration
python scripts/generate_demo_data.py

# 6. Executer le pipeline complet
python -m src.pipeline process     # clean + merge + features + outliers
python -m src.pipeline train       # Entrainer les modeles
python -m src.pipeline evaluate    # Evaluer et comparer

# 7. Lancer le dashboard
streamlit run app/app.py
```

### Avec des donnees reelles (APIs)

```bash
# Collecter depuis les APIs (necessite acces internet, ~1h pour DPE)
python -m src.pipeline collect

# Ou mise a jour complete (collect → process → train → upload)
python -m src.pipeline update_all
```

---

## Dashboard Streamlit

Lancer le dashboard :

```bash
streamlit run app/app.py
```

Le dashboard comprend 6 pages :

| Page | Description |
|------|-------------|
| **Accueil** | Vue d'ensemble, metriques cles, architecture du pipeline |
| **Exploration** | Exploration interactive des datasets (stats, distributions, correlations) |
| **Carte de France** | Carte interactive des 96 departements avec metriques HVAC |
| **Predictions ML** | Predictions vs realite, distribution des residus, feature importance |
| **Comparaison modeles** | Tableau comparatif, radar chart, classement par metrique |
| **Pipeline** | Etat des donnees, lancement des etapes, synchronisation pCloud |

---

## Commandes du pipeline

```bash
# === Collecte de donnees ===
python -m src.pipeline collect                        # Toutes les sources
python -m src.pipeline collect --sources weather,insee # Sources specifiques
python -m src.pipeline list                           # Lister les collecteurs

# === Base de donnees ===
python -m src.pipeline init_db                        # Creer les tables
python -m src.pipeline import_data                    # Importer les CSV

# === Traitement ===
python -m src.pipeline clean                          # Nettoyage
python -m src.pipeline merge                          # Fusion multi-sources
python -m src.pipeline features                       # Feature engineering
python -m src.pipeline outliers                       # Detection d'outliers
python -m src.pipeline process                        # clean+merge+features+outliers

# === Analyse ===
python -m src.pipeline eda                            # EDA + correlations

# === Machine Learning ===
python -m src.pipeline train                          # Entrainer les modeles
python -m src.pipeline train --target nb_dpe_total    # Autre variable cible
python -m src.pipeline evaluate                       # Evaluer et comparer

# === Synchronisation ===
python -m src.pipeline sync_pcloud                    # Telecharger depuis pCloud
python -m src.pipeline upload_pcloud                  # Upload vers pCloud
python -m src.pipeline update_all                     # Pipeline complet bout en bout

# === Tout-en-un ===
python -m src.pipeline all                            # Toutes les etapes
```

---

## Travailler sur plusieurs machines

### Ce qui est dans Git vs ce qui ne l'est pas

| Dans Git | PAS dans Git (.gitignore) |
|----------|--------------------------|
| Code source (src/, config/, app/) | Donnees brutes (data/raw/) |
| Configuration (.env.example) | Donnees traitees (data/processed/) |
| Schema SQL | Features ML (data/features/) |
| Scripts (scripts/, setup_project.py) | Base SQLite (data/*.db) |
| requirements.txt | Fichier .env (secrets) |
| Makefile, deploy.sh | Modeles entraines (*.pkl, *.pt) |

### Synchroniser via pCloud

Les donnees pre-collectees sont disponibles sur pCloud :

> **[Telecharger les donnees depuis pCloud](https://e.pcloud.link/publink/show?code=kZbQQ3Zg1slD5WfRgh42fH5rRpDDYWyBEsy)**

```bash
# Telecharger automatiquement depuis pCloud
python -m src.pipeline sync_pcloud

# Ou uploader apres collecte
python -m src.pipeline upload_pcloud
```

### Procedure sur une nouvelle machine

```bash
git clone https://github.com/PDUCLOS/Projet-HVAC.git
cd Projet-HVAC
./deploy.sh   # Setup complet avec donnees de demo

# OU avec donnees pCloud
make install
python -m src.pipeline sync_pcloud
python -m src.pipeline process
```

---

## Sources de donnees

| Source | API | Couverture | Auth |
|--------|-----|-----------|------|
| **DPE ADEME** | data.ademe.fr (JSON pagine) | 96 departements | Open Data |
| **Open-Meteo** | archive-api.open-meteo.com | 96 prefectures, 7 ans | Open Data |
| **INSEE BDM** | bdm.insee.fr (SDMX XML) | France, mensuel | Open Data |
| **Eurostat** | via package `eurostat` | France, mensuel | Open Data |
| **SITADEL** | DiDo API (SDES) | 96 departements, mensuel | Open Data |

> Toutes les sources sont **Open Data** — aucune cle API requise.

---

## Stack technique

| Categorie | Technologies |
|-----------|-------------|
| Langage | Python 3.10+ |
| Data | pandas, numpy, SQLAlchemy |
| ML | scikit-learn, LightGBM, XGBoost, Prophet, SHAP |
| Deep Learning | PyTorch (LSTM exploratoire, optionnel) |
| Outliers | IQR, Z-score modifie, Isolation Forest |
| Visualisation | matplotlib, seaborn, plotly |
| Dashboard | Streamlit (6 pages interactives) |
| BDD | SQLite (defaut), SQL Server, PostgreSQL |
| Sync | pCloud (upload/download automatique) |

---

## Resultats ML

```
Modele             Val RMSE   Val R2    Test RMSE   Test R2
ridge              1.178      0.9798    0.929       0.9885
lightgbm           1.456      0.9691    1.283       0.9781
ridge_exogenes     1.535      0.9657    1.339       0.9762
prophet            (optionnel — necessite prophet)
lstm               (optionnel — necessite pytorch)
```

Top features (Ridge) : `nb_installations_pac_lag_1m`, `nb_installations_pac_diff_1m`, `nb_dpe_total_rmean_3m`, `temp_mean_rmean_6m`, `hdd_sum_rmean_6m`

---

## Tests

```bash
# 119 tests couvrant config, collecteurs, nettoyage, features, outliers et ML
python -m pytest tests/ -v

# Avec couverture
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Auteur

**Patrice DUCLOS** — Data Analyst Senior (20 ans d'experience)

Projet portfolio pour recherche d'emploi Data Analyst.

---

## Licence

Ce projet est a usage personnel et pedagogique.
