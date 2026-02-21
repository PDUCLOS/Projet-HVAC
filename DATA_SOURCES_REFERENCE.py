# -*- coding: utf-8 -*-
"""
Data Sources Reference — HVAC Market Analysis
===============================================

This file documents ALL sources with exact URLs,
collection parameters, and corrections from the audit.

STATUS : This file is a DOCUMENTATION REFERENCE.
The actual collection code is located in src/collectors/.

LAST UPDATE : February 2026 (full audit)

APPLIED CORRECTIONS :
  - [CRITICAL] DPE ADEME URL corrected : dpe-v2-logements-existants -> dpe03existant
  - [CRITICAL] INSEE business climate idbank corrected : 001586763 -> 001565530
  - [HIGH]     Villeurbanne and Clermont-Ferrand removed from cities (duplicates/out of scope)
  - [HIGH]     SITADEL domain : mandatory addition of 'www.' (TLS certificate)
  - [MEDIUM]   SDES : migration to DiDo API (data.gouv.fr archived)
"""

# =============================================================================
# SOURCE 1 : OPEN-METEO — Historical Weather
# =============================================================================
# Free API, no key, no registration
# Doc : https://open-meteo.com/en/docs/historical-weather-api
# Limit : fair use, no more than 10,000 calls/day
# Collector : src/collectors/weather.py (WeatherCollector)

# Reference cities — ONE single city per department (historical AURA region)
# AUDIT CORRECTION : Villeurbanne (duplicate dept 69) and
#                    Clermont-Ferrand (dept 63, out of scope) REMOVED
CITIES_AURA = {
    "Lyon":            {"lat": 45.76, "lon": 4.84, "dept": "69"},
    "Grenoble":        {"lat": 45.19, "lon": 5.72, "dept": "38"},
    "Saint-Etienne":   {"lat": 45.44, "lon": 4.39, "dept": "42"},
    "Annecy":          {"lat": 45.90, "lon": 6.13, "dept": "74"},
    "Valence":         {"lat": 44.93, "lon": 4.89, "dept": "26"},
    "Chambery":        {"lat": 45.57, "lon": 5.92, "dept": "73"},
    "Bourg-en-Bresse": {"lat": 46.21, "lon": 5.23, "dept": "01"},
    "Privas":          {"lat": 44.74, "lon": 4.60, "dept": "07"},
}

OPENMETEO_BASE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_PARAMS = {
    "start_date": "2019-01-01",
    "end_date": "2025-12-31",
    "daily": ",".join([
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]),
    "timezone": "Europe/Paris",
}
# Note: HDD and CDD are computed in the collector :
# HDD = max(0, 18 - temp_mean)  (base 18C, heating demand)
# CDD = max(0, temp_mean - 18)  (base 18C, cooling demand)


# =============================================================================
# SOURCE 2 : INSEE BDM — Household Confidence & Business Climate
# =============================================================================
# Open SDMX API, no key
# Doc : https://www.bdm.insee.fr/series/sdmx
# Format : XML SDMX 2.1 -> parse with lxml
# Collector : src/collectors/insee.py (InseeCollector)

# AUDIT CORRECTION : industry business climate idbank corrected
# The old 001586763 was specific to industry only
# The new 001565530 covers all sectors (more relevant)
INSEE_SERIES = {
    "confiance_menages": {
        "idbank": "001759970",
        "desc": "Synthetic household confidence indicator (seasonally adjusted)",
        "freq": "monthly",
        "base": 100,
    },
    "climat_affaires_industrie": {
        "idbank": "001565530",  # CORRECTED : all sectors
        "desc": "Business climate indicator - All sectors (seasonally adjusted)",
        "freq": "monthly",
        "base": 100,
    },
    "climat_affaires_batiment": {
        "idbank": "001586808",
        "desc": "Business climate indicator in construction (seasonally adjusted)",
        "freq": "monthly",
        "base": 100,
    },
    "opinion_achats_importants": {
        "idbank": "001759974",
        "desc": "Opportunity to make major purchases (seasonally adjusted balance)",
        "freq": "monthly",
    },
    "situation_financiere_future": {
        "idbank": "001759972",
        "desc": "Future household financial situation (seasonally adjusted balance)",
        "freq": "monthly",
    },
    "ipi_industrie_manuf": {
        "idbank": "010768261",
        "desc": "Manufacturing IPI (seasonally and calendar adjusted, base 2021)",
        "freq": "monthly",
    },
}

INSEE_BDM_URL = "https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}"


# =============================================================================
# SOURCE 3 : DPE ADEME — HVAC Installations (sales proxy)
# =============================================================================
# AUDIT CORRECTION : old URL (dpe-v2-logements-existants) BROKEN
# New URL : dpe03existant (verified February 2026, ~14M DPE records)
# Collector : src/collectors/dpe.py (DpeCollector)

DEPTS_RHONE_ALPES = ["01", "07", "26", "38", "42", "69", "73", "74"]

# CORRECTED URL (February 2026)
DPE_API_BASE = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines"
DPE_API_PARAMS = {
    "size": 10000,
    "select": ",".join([
        "numero_dpe",
        "date_etablissement_dpe",
        "etiquette_dpe",
        "etiquette_ges",
        "type_energie_principale_chauffage",
        "type_installation_chauffage",
        "type_energie_principale_ecs",
        "type_generateur_climatisation",
        "surface_habitable_logement",
        "annee_construction",
        "code_postal_ban",
        "code_departement_ban",
    ]),
}


# =============================================================================
# SOURCE 4 : SITADEL — Building Permits
# =============================================================================
# AUDIT CORRECTION : always use 'www.' in the domain (TLS certificate)
# The ZIP URL changes with each monthly update
# Recommended alternative : DiDo API
# Collector : src/collectors/sitadel.py (SitadelCollector)

SITADEL_URL = (
    "https://www.statistiques.developpement-durable.gouv.fr/"
    "sites/default/files/2025-01/"
    "pc-dp-logement-depuis-2017-janv2025.csv.zip"
)


# =============================================================================
# SOURCE 5 : EUROSTAT — HVAC Industrial Production
# =============================================================================
# Python package : pip install eurostat
# Doc : https://pypi.org/project/eurostat/
# Collector : src/collectors/eurostat_col.py (EurostatCollector)

EUROSTAT_DATASETS = {
    "production_industrielle": {
        "code": "sts_inpr_m",
        "desc": "Monthly industrial production index",
        "nace_filter": ["C28", "C2825"],
        "geo": "FR",
        "unit": "I21",       # Index base 2021
        "s_adj": "SCA",      # Seasonally and calendar adjusted
    },
}


# =============================================================================
# SOURCE 6 : SDES — Local Energy Consumption
# =============================================================================
# AUDIT CORRECTION : data.gouv.fr archived since 2018
# Primary source : SDES DiDo API
# API : https://data.statistiques.developpement-durable.gouv.fr/dido/api/v1/
# NOTE : annual data (lower granularity than other sources)
# Recommendation : use in EDA only, not as ML feature (annual vs monthly)

SDES_DIDO_API = "https://data.statistiques.developpement-durable.gouv.fr/dido/api/v1/"


# =============================================================================
# TECHNICAL NOTES
# =============================================================================
#
# 1. COLLECTION ORDER : weather -> insee -> eurostat -> sitadel -> dpe
#    (from simplest/fastest to most voluminous)
#
# 2. ALL COLLECTION CODE IS IN src/collectors/
#    This file is a REFERENCE for URLs and parameters.
#
# 3. EXTENSIBILITY :
#    To add a new source :
#    a) Document the URLs/parameters in this file
#    b) Create a collector in src/collectors/new_source.py
#       inheriting from BaseCollector
#    c) That's it ! Auto-registered in the registry.
#
# 4. DATA VOLUME :
#    - Open-Meteo  : ~a few MB (daily, 8 cities, 7 years)
#    - INSEE       : ~a few KB (monthly, 6 series)
#    - Eurostat    : ~50 MB raw -> a few KB filtered
#    - SITADEL     : ~50-100 MB ZIP -> a few MB filtered
#    - DPE         : ~14M total rows -> filtered by departments
#    - SDES        : ~100 MB (annual, optional)
#
# 5. FINAL ML DATASET GRANULARITY :
#    Monthly x Department = ~528 rows
#    (8 depts x 66 months, Jul 2021 - Dec 2026)
#    -> Sufficient for Ridge, LightGBM, Prophet
#    -> Insufficient for LSTM/Transformer (educational exploration only)
