# -*- coding: utf-8 -*-
"""
Référence des Sources de Données — HVAC Market Analysis
========================================================

Ce fichier documente TOUTES les sources avec les URLs exactes,
les paramètres de collecte, et les corrections issues de l'audit.

STATUT : Ce fichier est une RÉFÉRENCE DOCUMENTAIRE.
Le code de collecte réel se trouve dans src/collectors/.

DERNIÈRE MISE À JOUR : Février 2026 (audit complet)

CORRECTIONS APPLIQUÉES :
  - [CRITIQUE] URL DPE ADEME corrigée : dpe-v2-logements-existants → dpe03existant
  - [CRITIQUE] Idbank INSEE climat affaires corrigé : 001586763 → 001565530
  - [ÉLEVÉ]   Villeurbanne et Clermont-Ferrand retirés des villes (doublons/hors périmètre)
  - [ÉLEVÉ]   Domaine SITADEL : ajout obligatoire de 'www.' (certificat TLS)
  - [MOYEN]   SDES : migration vers API DiDo (data.gouv.fr archivé)
"""

# =============================================================================
# SOURCE 1 : OPEN-METEO — Météo historique
# =============================================================================
# API gratuite, pas de clé, pas d'inscription
# Doc : https://open-meteo.com/en/docs/historical-weather-api
# Limite : fair use, pas plus de 10 000 appels/jour
# Collecteur : src/collectors/weather.py (WeatherCollector)

# Villes de référence AURA — UNE seule ville par département
# CORRECTION AUDIT : Villeurbanne (doublon dept 69) et
#                    Clermont-Ferrand (dept 63, hors périmètre) RETIRÉS
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
# Note: HDD et CDD calculés dans le collecteur :
# HDD = max(0, 18 - temp_mean)  (base 18°C, demande chauffage)
# CDD = max(0, temp_mean - 18)  (base 18°C, demande climatisation)


# =============================================================================
# SOURCE 2 : INSEE BDM — Confiance ménages & Climat affaires
# =============================================================================
# API SDMX ouverte, pas de clé
# Doc : https://www.bdm.insee.fr/series/sdmx
# Format : XML SDMX 2.1 → parser avec lxml
# Collecteur : src/collectors/insee.py (InseeCollector)

# CORRECTION AUDIT : idbank climat affaires industrie corrigé
# L'ancien 001586763 était spécifique à l'industrie uniquement
# Le nouveau 001565530 couvre tous les secteurs (plus pertinent)
INSEE_SERIES = {
    "confiance_menages": {
        "idbank": "001759970",
        "desc": "Indicateur synthétique de confiance des ménages (CVS)",
        "freq": "mensuel",
        "base": 100,
    },
    "climat_affaires_industrie": {
        "idbank": "001565530",  # CORRIGÉ : tous secteurs
        "desc": "Indicateur du climat des affaires - Tous secteurs (CVS)",
        "freq": "mensuel",
        "base": 100,
    },
    "climat_affaires_batiment": {
        "idbank": "001586808",
        "desc": "Indicateur climat des affaires dans le bâtiment (CVS)",
        "freq": "mensuel",
        "base": 100,
    },
    "opinion_achats_importants": {
        "idbank": "001759974",
        "desc": "Opportunité de faire des achats importants (solde CVS)",
        "freq": "mensuel",
    },
    "situation_financiere_future": {
        "idbank": "001759972",
        "desc": "Situation financière future des ménages (solde CVS)",
        "freq": "mensuel",
    },
    "ipi_industrie_manuf": {
        "idbank": "010768261",
        "desc": "IPI Industrie manufacturière (CVS-CJO, base 2021)",
        "freq": "mensuel",
    },
}

INSEE_BDM_URL = "https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{idbank}"


# =============================================================================
# SOURCE 3 : DPE ADEME — Installations HVAC (proxy ventes)
# =============================================================================
# CORRECTION AUDIT : ancienne URL (dpe-v2-logements-existants) CASSÉE
# Nouvelle URL : dpe03existant (vérifiée février 2026, ~14M de DPE)
# Collecteur : src/collectors/dpe.py (DpeCollector)

DEPTS_RHONE_ALPES = ["01", "07", "26", "38", "42", "69", "73", "74"]

# URL CORRIGÉE (février 2026)
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
# SOURCE 4 : SITADEL — Permis de construire
# =============================================================================
# CORRECTION AUDIT : toujours utiliser 'www.' dans le domaine (certificat TLS)
# L'URL du ZIP change à chaque mise à jour mensuelle
# Alternative recommandée : API DiDo
# Collecteur : src/collectors/sitadel.py (SitadelCollector)

SITADEL_URL = (
    "https://www.statistiques.developpement-durable.gouv.fr/"
    "sites/default/files/2025-01/"
    "pc-dp-logement-depuis-2017-janv2025.csv.zip"
)


# =============================================================================
# SOURCE 5 : EUROSTAT — Production industrielle HVAC
# =============================================================================
# Package Python : pip install eurostat
# Doc : https://pypi.org/project/eurostat/
# Collecteur : src/collectors/eurostat_col.py (EurostatCollector)

EUROSTAT_DATASETS = {
    "production_industrielle": {
        "code": "sts_inpr_m",
        "desc": "Indice de production industrielle mensuel",
        "nace_filter": ["C28", "C2825"],
        "geo": "FR",
        "unit": "I21",       # Indice base 2021
        "s_adj": "SCA",      # CVS-CJO
    },
}


# =============================================================================
# SOURCE 6 : SDES — Consommation énergie locale
# =============================================================================
# CORRECTION AUDIT : data.gouv.fr archivé depuis 2018
# Source primaire : API DiDo du SDES
# API : https://data.statistiques.developpement-durable.gouv.fr/dido/api/v1/
# NOTE : données annuelles (granularité plus faible que les autres sources)
# Recommandation : utiliser en EDA uniquement, pas en feature ML (annuel vs mensuel)

SDES_DIDO_API = "https://data.statistiques.developpement-durable.gouv.fr/dido/api/v1/"


# =============================================================================
# NOTES TECHNIQUES
# =============================================================================
#
# 1. ORDRE DE COLLECTE : weather → insee → eurostat → sitadel → dpe
#    (du plus simple/rapide au plus volumineux)
#
# 2. TOUT LE CODE DE COLLECTE EST DANS src/collectors/
#    Ce fichier est une RÉFÉRENCE pour les URLs et paramètres.
#
# 3. EXTENSIBILITÉ :
#    Pour ajouter une nouvelle source :
#    a) Documenter les URLs/paramètres dans ce fichier
#    b) Créer un collecteur dans src/collectors/nouvelle_source.py
#       héritant de BaseCollector
#    c) C'est tout ! Auto-enregistré dans le registry.
#
# 4. VOLUME DE DONNÉES :
#    - Open-Meteo  : ~quelques Mo (quotidien, 8 villes, 7 ans)
#    - INSEE       : ~quelques Ko (mensuel, 6 séries)
#    - Eurostat    : ~50 Mo brut → quelques Ko filtrés
#    - SITADEL     : ~50-100 Mo ZIP → quelques Mo filtrés AURA
#    - DPE         : ~14M lignes total → ~500K-1M lignes AURA
#    - SDES        : ~100 Mo (annuel, optionnel)
#
# 5. GRANULARITÉ FINALE DU DATASET ML :
#    Mensuel × Département = ~528 lignes
#    (8 depts × 66 mois, juil 2021 - dec 2026)
#    → Suffisant pour Ridge, LightGBM, Prophet
#    → Insuffisant pour LSTM/Transformer (exploration pédagogique uniquement)
