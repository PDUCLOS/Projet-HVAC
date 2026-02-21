# Architecture de la Base de Donnees -- Projet HVAC

> **Projet** : Analyse du marche HVAC en region Auvergne-Rhone-Alpes
> **Moteur par defaut** : SQLite (compatible PostgreSQL et SQL Server)
> **ORM** : SQLAlchemy | **Schema** : Etoile (star schema) -- grain mois x departement

---

## 1. Schema en etoile documente

### 1.1 Diagramme Mermaid

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
        BOOLEAN is_heating "oct-mars"
        BOOLEAN is_cooling "juin-sept"
    }

    dim_geo {
        INT geo_id PK "Auto-increment"
        VARCHAR dept_code UK "Code dept"
        VARCHAR dept_name "Nom departement"
        VARCHAR city_ref "Ville meteo"
        DECIMAL latitude "Coordonnee"
        DECIMAL longitude "Coordonnee"
    }

    dim_equipment_type {
        INT equip_type_id PK "Auto-increment"
        VARCHAR label "PAC air-air etc"
        VARCHAR categorie "chauffage clim mixte"
        VARCHAR energie_source "elec gaz fioul bois"
        BOOLEAN is_renewable "Renouvelable"
    }

    fact_hvac_installations {
        INT fact_id PK
        INT date_id FK "vers dim_time"
        INT geo_id FK "vers dim_geo"
        INT nb_dpe_total "Total DPE"
        INT nb_installations_pac "DPE avec PAC"
        INT nb_installations_clim "DPE avec clim"
        DECIMAL temp_mean "Temp moyenne"
        DECIMAL heating_degree_days "HDD"
        DECIMAL cooling_degree_days "CDD"
        INT nb_permis_construire "Permis"
        DECIMAL conso_elec_mwh "Conso elec"
    }

    fact_economic_context {
        INT date_id PK "FK vers dim_time"
        DECIMAL confiance_menages "INSEE base 100"
        DECIMAL climat_affaires_bat "Batiment"
        DECIMAL ipi_hvac_c28 "Eurostat C28"
        DECIMAL ipi_hvac_c2825 "Eurostat C2825"
    }
```

### 1.2 Description des tables

#### dim_time (84 lignes)

| Colonne | Type | Description |
|---------|------|-------------|
| `date_id` | INTEGER PK | Format YYYYMM (ex: 202401 = janvier 2024) |
| `year` | INTEGER NOT NULL | Annee (2019-2025), CHECK BETWEEN 2015 AND 2030 |
| `month` | INTEGER NOT NULL | Mois (1-12) |
| `quarter` | INTEGER NOT NULL | Trimestre (1-4) |
| `is_heating` | BOOLEAN NOT NULL | Saison de chauffage (octobre a mars) |
| `is_cooling` | BOOLEAN NOT NULL | Saison de climatisation (juin a septembre) |

#### dim_geo (8 lignes -- departements AURA)

| Colonne | Type | Description |
|---------|------|-------------|
| `geo_id` | INTEGER PK AUTO | Identifiant unique |
| `dept_code` | VARCHAR(3) UNIQUE | Code departement (01, 07, 26, 38, 42, 69, 73, 74) |
| `dept_name` | VARCHAR(50) | Nom du departement |
| `city_ref` | VARCHAR(50) | Ville de reference meteo (Lyon, Grenoble...) |
| `latitude`, `longitude` | DECIMAL(7,4) | Coordonnees de la ville de reference |
| `region_code` | VARCHAR(3) DEFAULT '84' | Code region AURA |

#### dim_equipment_type (10 lignes -- table de reference)

Table de reference des equipements HVAC (PAC air-air, Chaudiere gaz, etc.). Non reliee par FK a la table de faits car les faits sont agreges au grain mois x departement, pas par equipement individuel.

#### fact_hvac_installations (672 lignes -- table de faits principale)

| Colonne | Type | Description |
|---------|------|-------------|
| `fact_id` | INTEGER PK AUTO | Identifiant unique |
| `date_id` | INTEGER FK NOT NULL | Vers dim_time |
| `geo_id` | INTEGER FK NOT NULL | Vers dim_geo |
| `nb_dpe_total` | INTEGER | Total DPE du mois/departement |
| `nb_installations_pac` | INTEGER | DPE mentionnant une PAC |
| `nb_installations_clim` | INTEGER | DPE mentionnant une climatisation |
| `nb_dpe_classe_ab` | INTEGER | DPE classe A ou B |
| `temp_mean/max/min` | DECIMAL(5,2) | Temperatures mensuelles (C) |
| `heating_degree_days` | DECIMAL(8,2) | HDD mensuel (base 18C) |
| `cooling_degree_days` | DECIMAL(8,2) | CDD mensuel (base 18C) |
| `nb_jours_canicule/gel` | INTEGER | Jours extremes |
| `nb_permis_construire` | INTEGER | Permis de construire accordes |
| `conso_elec/gaz_mwh` | DECIMAL(12,2) | Consommation energie (MWh) |

Contrainte : `UNIQUE(date_id, geo_id)` -- une seule ligne par mois x departement.

#### fact_economic_context (84 lignes -- indicateurs nationaux)

| Colonne | Type | Description |
|---------|------|-------------|
| `date_id` | INTEGER PK, FK | Vers dim_time (YYYYMM) |
| `confiance_menages` | DECIMAL(6,2) | Indice synthetique INSEE (base 100) |
| `climat_affaires_indus/bat` | DECIMAL(6,2) | Climat affaires industrie/batiment |
| `opinion_achats` | DECIMAL(6,2) | Opportunite achats importants (solde) |
| `ipi_manufacturing` | DECIMAL(6,2) | IPI industrie manufacturiere |
| `ipi_hvac_c28/c2825` | DECIMAL(6,2) | IPI machines et equipements clim (Eurostat) |

### 1.3 Grain et justification

**Grain** : la table de faits principale a un grain de **mois x departement** (date_id + geo_id). La table economique a un grain de **mois seulement** (donnees nationales, non dupliquees sur les 8 departements).

**Pourquoi un schema en etoile pour l'analyse HVAC ?**

1. **Jointures simples** : faits + 2-3 dimensions, pas de cascade flocon.
2. **Performance OLAP** : agregations (SUM, AVG, COUNT) directes avec GROUP BY.
3. **Separation local/national** : evite la duplication artificielle des indicateurs INSEE sur 8 departements.
4. **Compatibilite ML** : `build_ml_dataset()` produit un DataFrame ML-ready par jointure naturelle.
5. **Scalabilite** : ajouter des regions = ajouter des lignes dans `dim_geo`.

---

## 2. Migration SQLite vers PostgreSQL

### 2.1 Differences de types

| Concept | SQLite | PostgreSQL |
|---------|--------|------------|
| Boolean | BOOLEAN (stocke 0/1) | BOOLEAN natif |
| Auto-increment | AUTOINCREMENT | SERIAL |
| VARCHAR | Pas de limite enforced | Limite stricte |
| DECIMAL | Stocke comme REAL | Precision exacte |
| Date | Texte ISO 8601 | DATE natif |
| Upsert | INSERT OR IGNORE | INSERT ... ON CONFLICT DO NOTHING |
| Concurrence | Verrouillage fichier | MVCC, acces concurrent |

Le fichier `schema.sql` est deja compatible PostgreSQL (BOOLEAN, VARCHAR, DECIMAL natifs). SQLAlchemy gere la traduction de AUTOINCREMENT vers SERIAL via `to_sql()`.

### 2.2 Etapes pour basculer

```bash
# 1. Installer le driver
pip install psycopg2-binary

# 2. Creer la base PostgreSQL
psql -U postgres -c "CREATE DATABASE hvac_market;"
psql -U postgres -c "CREATE USER hvac_user WITH PASSWORD 'mdp_securise';"
psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE hvac_market TO hvac_user;"
```

```env
# 3. Modifier .env
DB_TYPE=postgresql
DATABASE_URL=postgresql://hvac_user:mdp_securise@localhost:5432/hvac_market
```

```python
# 4. Initialiser et importer
from src.database.db_manager import DatabaseManager
db = DatabaseManager("postgresql://hvac_user:mdp@localhost:5432/hvac_market")
db.init_database()          # Utilise schema.sql (compatible PostgreSQL)
db.import_collected_data()  # Importe tous les CSV collectes
```

### 2.3 Script de migration des donnees existantes

```python
"""Migration SQLite vers PostgreSQL."""
import pandas as pd
from sqlalchemy import create_engine

src = create_engine("sqlite:///data/hvac_market.db")
dst = create_engine("postgresql://hvac_user:mdp@localhost:5432/hvac_market")

# Ordre respectant les FK
for table in ["dim_time", "dim_geo", "dim_equipment_type",
              "fact_hvac_installations", "fact_economic_context", "raw_dpe"]:
    df = pd.read_sql(f"SELECT * FROM {table}", src)
    df.to_sql(table, dst, if_exists="append", index=False)
    print(f"{table} : {len(df)} lignes migrees")
```

---

## 3. Requetes OLAP avancees

Requetes avec **window functions**, compatibles SQLite 3.25+ et PostgreSQL.

### 3.1 Classement des departements par installations PAC

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

**Explication** : `RANK()` partitionne par mois et classe les departements par nombre decroissant de PAC. Le Rhone (69) et l'Isere (38) dominent generalement grace a leur densite de population.

### 3.2 Moyenne mobile 3 mois par departement

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

**Explication** : la fenetre glissante de 3 lignes lisse les pics saisonniers pour reveler la tendance sous-jacente du marche PAC.

### 3.3 Variation year-over-year (YoY)

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

**Explication** : `LAG(..., 12)` recupere la valeur du meme mois N-1. `NULLIF(..., 0)` evite la division par zero. Les annees 2021-2023 montrent une croissance significative (MaPrimeRenov', prix de l'energie).

### 3.4 Part de marche par type d'equipement

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

**Explication** : combine GROUP BY et window function (`SUM() OVER`) pour calculer la part de chaque departement dans le total regional annuel. Les departements meridionaux (Drome, Ardeche) ont une part de climatisation plus elevee.

### 3.5 Correlation meteo/installations avec CTE

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

**Explication** : deux CTE structurent l'analyse. La premiere calcule les moyennes par departement, la seconde attribue des rangs. Si `ecart_hdd_pac` est faible, il y a correlation entre besoin de chauffage et installations PAC. Les departements montagnards (Savoie, Haute-Savoie) avec HDD eleves ont plus de PAC ; les departements meridionaux avec CDD eleves dominent en climatisation.

---

## 4. Comparaison OLTP vs OLAP vs NoSQL

### 4.1 Tableau comparatif pour le cas HVAC

| Critere | OLTP (PostgreSQL 3NF) | OLAP (Star schema) | NoSQL (MongoDB) |
|---------|----------------------|---------------------|-----------------|
| **Usage** | Transactions temps reel | Analyse, reporting, ML | Stockage JSON brut |
| **Schema** | 3NF normalise | Etoile denormalise | Schema-less |
| **Requetes** | INSERT/UPDATE unitaires | SELECT + GROUP BY | find(), aggregate() |
| **Jointures** | Nombreuses (10+) | Peu (2-3 tables) | Aucune (embarque) |
| **Perf. lecture** | Moyenne | Excellente | Bonne |
| **Perf. ecriture** | Excellente | Correcte (batch) | Excellente |
| **Integrite** | Forte (FK, contraintes) | Moyenne (FK dimensions) | Aucune |
| **Cas HVAC** | Saisie temps reel DPE | Analyse marche + ML | Cache reponses API |

### 4.2 Justification du choix actuel

Le schema **OLAP en etoile** est adapte car le projet est oriente analyse/prediction, pas saisie transactionnelle. Les donnees sont collectees en batch depuis des APIs (ADEME, INSEE, Eurostat, Open-Meteo). Le volume est maitrise (672 lignes de faits). SQLite ne necessite aucun serveur ; la migration vers PostgreSQL est triviale.

### 4.3 Exemple MongoDB : cache des reponses API

Dans une architecture hybride, MongoDB stockerait les reponses JSON brutes des APIs avant transformation, permettant un retraitement sans rappeler les APIs.

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

### 4.4 Script Python MongoDB

```python
"""Exemple MongoDB pour le cache des reponses API HVAC."""
from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["hvac_raw_cache"]
collection = db["api_responses"]

# --- Insertion ---
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
print(f"Document insere : {result.inserted_id}")

# --- Requete : reponses non traitees ---
for doc in collection.find({"source": "open_meteo", "processing_status": "pending"}):
    print(f"A traiter : {doc['parameters']['start_date']}")
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"processing_status": "imported"}},
    )

# --- Agregation : comptage par source ---
pipeline = [
    {"$group": {"_id": "$source", "total": {"$sum": 1},
                "derniere_collecte": {"$max": "$collected_at"}}},
    {"$sort": {"total": -1}},
]
for r in collection.aggregate(pipeline):
    print(f"{r['_id']} : {r['total']} reponses")

# --- Index ---
collection.create_index([("source", 1), ("processing_status", 1)])
collection.create_index([("collected_at", -1)])
client.close()
```

---

## 5. Optimisation et indexation

### 5.1 Index existants

```sql
-- Table de faits principale
CREATE INDEX idx_fact_hvac_date     ON fact_hvac_installations(date_id);
CREATE INDEX idx_fact_hvac_geo      ON fact_hvac_installations(geo_id);
CREATE INDEX idx_fact_hvac_date_geo ON fact_hvac_installations(date_id, geo_id);
CREATE INDEX idx_dim_geo_dept       ON dim_geo(dept_code);

-- Table brute DPE
CREATE INDEX idx_raw_dpe_date       ON raw_dpe(date_etablissement_dpe);
CREATE INDEX idx_raw_dpe_dept       ON raw_dpe(code_departement_ban);
CREATE INDEX idx_raw_dpe_date_dept  ON raw_dpe(date_etablissement_dpe, code_departement_ban);
CREATE INDEX idx_raw_dpe_etiquette  ON raw_dpe(etiquette_dpe);
```

### 5.2 Index supplementaires recommandes

```sql
-- Index couvrant pour les requetes ML
CREATE INDEX idx_fact_hvac_ml ON fact_hvac_installations(
    date_id, geo_id, nb_installations_pac, nb_installations_clim,
    temp_mean, heating_degree_days, cooling_degree_days);

-- Index sur raw_dpe pour la detection des PAC
CREATE INDEX idx_raw_dpe_generateur ON raw_dpe(type_generateur_chauffage_principal);
CREATE INDEX idx_raw_dpe_froid      ON raw_dpe(type_generateur_froid);

-- Index partiel PostgreSQL pour les DPE recents
CREATE INDEX idx_raw_dpe_recent ON raw_dpe(date_etablissement_dpe)
    WHERE date_etablissement_dpe >= '2022-01-01';
```

### 5.3 Partitionnement par date (PostgreSQL)

Pour `raw_dpe` (~1.4M lignes), le partitionnement par annee reduit le scan :

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

Une requete filtrant sur 2023 ne scanne que ~200K lignes au lieu de 1.4M.

### 5.4 Estimation des volumes

| Table | Lignes | SQLite | PostgreSQL |
|-------|--------|--------|------------|
| `dim_time` | 84 | < 1 Ko | < 1 Ko |
| `dim_geo` | 8 | < 1 Ko | < 1 Ko |
| `dim_equipment_type` | 10 | < 1 Ko | < 1 Ko |
| `fact_hvac_installations` | 672 | ~50 Ko | ~80 Ko |
| `fact_economic_context` | 84 | ~5 Ko | ~10 Ko |
| `raw_dpe` | ~1 400 000 | ~500 Mo | ~700 Mo |
| **Total** | **~1 400 850** | **~500 Mo** | **~700 Mo** |

La table `raw_dpe` represente 99.9% du volume. C'est la seule table ou le partitionnement et l'indexation avancee ont un impact reel. Les tables du schema en etoile (< 100 Ko) sont performantes sans optimisation poussee. Prevoir 1 Go d'espace disque total (donnees + index) en production PostgreSQL.

---

> **Projet HVAC Market Analysis -- Region AURA**
> Schema en etoile | SQLAlchemy | SQLite / PostgreSQL / SQL Server
