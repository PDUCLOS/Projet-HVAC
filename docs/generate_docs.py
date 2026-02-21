# -*- coding: utf-8 -*-
"""
Word documentation generator for the HVAC Market Analysis project.
===================================================================

Generates two comprehensive documents:
1. documentation_technique.docx — Full technical architecture & methodology
2. guide_utilisation.docx — Installation, configuration, accounts & deployment

Usage:
    python docs/generate_docs.py
"""

from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import os


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def set_doc_styles(doc):
    """Configure document styles (Calibri 11, blue headings)."""
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    for level in range(1, 5):
        doc.styles[f"Heading {level}"].font.color.rgb = RGBColor(0, 51, 102)


def add_cover_page(doc, title, subtitle):
    """Add a formatted cover page with project info."""
    for _ in range(6):
        doc.add_paragraph("")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.font.size = Pt(28)
    run.font.color.rgb = RGBColor(0, 51, 102)
    run.bold = True

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(subtitle)
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Projet HVAC Market Analysis\nFrance metropolitaine")
    run.font.size = Pt(14)

    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(
        "Patrice DUCLOS\nData Analyst Senior\n"
        "Formation Data Science Lead — Jedha Bootcamp (Bac+5)\n"
        "Fevrier 2026"
    )
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_page_break()


def add_table(doc, headers, rows):
    """Add a formatted table with headers and data rows."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            table.rows[r + 1].cells[c].text = str(val)

    doc.add_paragraph("")
    return table


def add_bullets(doc, items):
    """Add a bulleted list."""
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def add_numbered(doc, items):
    """Add a numbered list."""
    for i, item in enumerate(items, 1):
        doc.add_paragraph(f"{i}. {item}")


# ================================================================
# DOCUMENT 1 : Documentation Technique
# ================================================================

def generate_technical_doc():
    """Generate the complete technical documentation."""
    doc = Document()
    set_doc_styles(doc)
    add_cover_page(
        doc,
        "Documentation Technique",
        "Architecture, Pipeline de Donnees et Methodologie ML"
    )

    # --- TABLE OF CONTENTS ---
    doc.add_heading("Table des matieres", level=1)
    toc = [
        "1. Vue d'ensemble du projet",
        "2. Architecture technique",
        "3. Sources de donnees",
        "4. Pipeline de collecte",
        "5. Traitement des donnees",
        "6. Feature Engineering",
        "7. Schema de base de donnees",
        "8. Modelisation ML",
        "9. Analyse de l'auto-correlation Ridge",
        "10. API REST (FastAPI)",
        "11. Dashboard Streamlit",
        "12. Deploiement (Docker / Kubernetes)",
        "13. Orchestration (Airflow)",
        "14. Reinforcement Learning (demo)",
        "15. Tests unitaires",
        "16. Couverture des modules de formation",
    ]
    for item in toc:
        doc.add_paragraph(item, style="List Number")
    doc.add_page_break()

    # =================================================================
    # 1. PROJECT OVERVIEW
    # =================================================================
    doc.add_heading("1. Vue d'ensemble du projet", level=1)

    doc.add_heading("1.1 Objectif", level=2)
    doc.add_paragraph(
        "Ce projet construit un pipeline data complet pour analyser et predire "
        "les installations d'equipements HVAC (pompes a chaleur, climatisation) "
        "sur l'ensemble des 96 departements de France metropolitaine."
    )
    doc.add_paragraph(
        "Il s'agit d'un projet portfolio Data Science Lead (Bac+5) demonstrant "
        "la maitrise de la chaine complete : collecte de donnees ouvertes, "
        "nettoyage, modelisation en etoile, feature engineering, modelisation ML, "
        "API REST, dashboard interactif, deploiement Docker/Kubernetes et "
        "orchestration Airflow."
    )

    doc.add_heading("1.2 Strategie proxy", level=2)
    doc.add_paragraph(
        "Aucune donnee directe de ventes HVAC n'est disponible en open data. "
        "Le projet utilise les Diagnostics de Performance Energetique (DPE) "
        "de l'ADEME comme proxy : chaque DPE mentionne le type de chauffage "
        "et climatisation installe. En filtrant les DPE mentionnant une pompe "
        "a chaleur (PAC) ou une climatisation, on obtient un indicateur fiable "
        "du volume d'installations."
    )

    doc.add_heading("1.3 Perimetre", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Perimetre", "France metropolitaine (96 departements)"],
            ["Periode", "Juillet 2021 — Fevrier 2026 (~55 mois)"],
            ["Grain", "Mois x Departement"],
            ["Volume DPE", "Plusieurs millions de diagnostics"],
            ["Features ML", "~90 features engineerees"],
            ["API", "6 endpoints REST (FastAPI)"],
            ["Dashboard", "6 pages interactives (Streamlit)"],
            ["Tests", "119 tests unitaires (pytest)"],
        ]
    )

    doc.add_heading("1.4 Resultats cles", level=2)
    add_table(doc,
        ["Modele", "Val RMSE", "Val R2", "Test RMSE", "Test R2"],
        [
            ["Ridge (avec lags)", "1.09", "0.9975", "0.96", "0.9982"],
            ["Ridge (exogenes)", "—", "—", "—", "—"],
            ["LightGBM", "5.45", "0.937", "8.37", "0.865"],
            ["Prophet", "13.75", "0.599", "12.05", "0.719"],
        ]
    )
    doc.add_paragraph(
        "Note : Le R2 tres eleve de Ridge est du a l'auto-correlation "
        "via les features lag. Le modele 'Ridge exogenes' evalue la vraie "
        "contribution des features meteo et economiques (voir section 9)."
    )
    doc.add_page_break()

    # =================================================================
    # 2. TECHNICAL ARCHITECTURE
    # =================================================================
    doc.add_heading("2. Architecture technique", level=1)

    doc.add_heading("2.1 Structure du projet", level=2)
    add_table(doc,
        ["Couche", "Repertoire", "Responsabilite"],
        [
            ["Configuration", "config/settings.py", "Parametres centralises (dataclasses frozen)"],
            ["Collecte", "src/collectors/", "5 collecteurs plugins (auto-enregistrement)"],
            ["Traitement", "src/processing/", "Nettoyage, fusion, feature engineering, outliers"],
            ["Base de donnees", "src/database/", "Schema en etoile, import CSV → SQLite/PostgreSQL"],
            ["Analyse", "src/analysis/", "EDA automatisee, correlations"],
            ["Modelisation", "src/models/", "Ridge, LightGBM, Prophet, LSTM, RL demo"],
            ["API REST", "api/", "FastAPI — 6 endpoints prediction/monitoring"],
            ["Dashboard", "app/", "Streamlit — 6 pages interactives"],
            ["Orchestration", "airflow/dags/", "DAG Airflow pipeline complet"],
            ["Deploiement", "kubernetes/, Dockerfile", "Docker multi-stage, K8s, Render"],
            ["Tests", "tests/", "119 tests unitaires (pytest)"],
        ]
    )

    doc.add_heading("2.2 Patterns architecturaux", level=2)

    doc.add_heading("Plugin Registry (Collecteurs)", level=3)
    doc.add_paragraph(
        "Les collecteurs utilisent un systeme de plugins base sur __init_subclass__. "
        "Chaque sous-classe concrete de BaseCollector est automatiquement enregistree "
        "dans le CollectorRegistry. Pour ajouter une nouvelle source de donnees, "
        "il suffit de creer un fichier dans src/collectors/ avec une classe heritant "
        "de BaseCollector — aucune configuration manuelle n'est requise."
    )

    doc.add_heading("Dataclasses frozen", level=3)
    doc.add_paragraph(
        "Toute la configuration est definie via des dataclasses frozen (immutables). "
        "Les valeurs sont chargees depuis le fichier .env via python-dotenv."
    )

    doc.add_heading("Pipeline idempotent", level=3)
    doc.add_paragraph(
        "Chaque etape du pipeline peut etre relancee sans risque. Les fichiers "
        "de sortie sont ecrases. Les doublons sont geres en amont."
    )

    doc.add_heading("2.3 Technologies", level=2)
    add_table(doc,
        ["Categorie", "Technologies"],
        [
            ["Langage", "Python 3.10+"],
            ["Data", "pandas, numpy, SQLAlchemy"],
            ["ML", "scikit-learn (Ridge, StandardScaler, TimeSeriesSplit)"],
            ["Gradient Boosting", "LightGBM"],
            ["Series temporelles", "Prophet (Meta)"],
            ["Deep Learning", "PyTorch (LSTM, exploratoire)"],
            ["Reinforcement Learning", "Gymnasium, Q-Learning"],
            ["Interpretabilite", "SHAP"],
            ["Visualisation", "matplotlib, seaborn, plotly"],
            ["API", "FastAPI, Pydantic v2, uvicorn"],
            ["Dashboard", "Streamlit"],
            ["BDD", "SQLite (defaut), PostgreSQL, SQL Server"],
            ["Conteneurisation", "Docker (multi-stage), Docker Compose"],
            ["Orchestration K8s", "Kubernetes (Deployment, Ingress, CronJob)"],
            ["CI/CD", "Render (PaaS), deploy.sh"],
            ["Tests", "pytest"],
            ["HTTP", "requests (retry automatique)"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 3. DATA SOURCES
    # =================================================================
    doc.add_heading("3. Sources de donnees", level=1)
    doc.add_paragraph(
        "Toutes les sources sont Open Data — aucune cle API n'est requise."
    )

    doc.add_heading("3.1 DPE ADEME (variable cible)", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Source", "data.ademe.fr (API JSON paginee)"],
            ["Perimetre", "96 departements de France metropolitaine"],
            ["Periode", "Juillet 2021+ (DPE v2)"],
            ["Grain", "1 ligne par diagnostic"],
            ["Detection PAC", "Regex sur chauffage/froid : PAC, pompe a chaleur, thermodynamique"],
            ["Collecteur", "DpeCollector (src/collectors/dpe.py)"],
        ]
    )

    doc.add_heading("3.2 Open-Meteo (features meteo)", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Source", "archive-api.open-meteo.com"],
            ["Perimetre", "96 villes de reference (prefectures)"],
            ["Variables", "temperature max/min/mean, precipitations, vent"],
            ["Derivees", "HDD (Heating Degree Days), CDD (Cooling Degree Days)"],
            ["Collecteur", "WeatherCollector (src/collectors/weather.py)"],
        ]
    )

    doc.add_heading("3.3 INSEE BDM (features economiques)", level=2)
    add_table(doc,
        ["Serie", "idbank", "Description"],
        [
            ["confiance_menages", "001759970", "Indicateur synthetique de confiance"],
            ["climat_affaires_industrie", "001565530", "Climat des affaires tous secteurs"],
            ["climat_affaires_batiment", "001586808", "Climat des affaires batiment"],
            ["opinion_achats_importants", "001759974", "Opportunite achats importants"],
            ["situation_financiere_future", "001759972", "Situation financiere future"],
            ["ipi_industrie_manuf", "010768261", "IPI industrie manufacturiere"],
        ]
    )

    doc.add_heading("3.4 Eurostat (IPI HVAC)", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Dataset", "sts_inpr_m (Industrial Production)"],
            ["Codes NACE", "C28 (machines), C2825 (equipements clim)"],
            ["Filtre geo", "France (FR)"],
            ["Collecteur", "EurostatCollector (src/collectors/eurostat_col.py)"],
        ]
    )

    doc.add_heading("3.5 SITADEL (permis de construire)", level=2)
    doc.add_paragraph(
        "Source en cours de migration vers l'API DiDo du SDES. "
        "Le collecteur SitadelCollector existe mais n'est pas pleinement fonctionnel."
    )
    doc.add_page_break()

    # =================================================================
    # 4. COLLECTION PIPELINE
    # =================================================================
    doc.add_heading("4. Pipeline de collecte", level=1)

    doc.add_heading("4.1 Architecture BaseCollector", level=2)
    doc.add_paragraph("Tous les collecteurs heritent de BaseCollector :")
    add_bullets(doc, [
        "Session HTTP avec retry automatique (codes 429, 500-504)",
        "Helpers fetch_json(), fetch_xml(), fetch_bytes()",
        "Cycle de vie : collect() → validate() → save()",
        "Auto-enregistrement dans CollectorRegistry via __init_subclass__",
        "Pause de politesse entre appels API (rate_limit_delay)",
    ])

    doc.add_heading("4.2 Cycle de vie d'une collecte", level=2)
    add_numbered(doc, [
        "Log de demarrage avec parametres",
        "collect() — Recuperation des donnees brutes",
        "validate() — Verification qualite (colonnes, types)",
        "save() — Persistance CSV dans data/raw/",
        "Log de fin avec bilan (lignes, duree, statut)",
    ])

    doc.add_heading("4.3 Resilience", level=2)
    add_bullets(doc, [
        "Si une ville echoue (meteo), les autres sont collectees",
        "Si une serie echoue (INSEE), les autres sont fusionnees",
        "Retry HTTP automatique avec backoff exponentiel",
        "Statut PARTIAL si certains elements ont echoue",
    ])
    doc.add_page_break()

    # =================================================================
    # 5. DATA PROCESSING
    # =================================================================
    doc.add_heading("5. Traitement des donnees", level=1)

    doc.add_heading("5.1 Nettoyage (DataCleaner)", level=2)
    doc.add_paragraph(
        "Le module clean_data.py applique des regles de nettoyage specifiques "
        "a chaque source. Le nettoyage est idempotent."
    )

    doc.add_heading("Meteo", level=3)
    add_bullets(doc, [
        "Conversion dates en datetime",
        "Suppression doublons (date x ville)",
        "Clipping : temperature [-30, 50], precipitations [0, 300], vent [0, 200]",
        "Recalcul HDD/CDD (base 18 degres)",
    ])

    doc.add_heading("DPE", level=3)
    add_bullets(doc, [
        "Traitement par chunks de 200 000 lignes (gestion memoire)",
        "Deduplication sur numero_dpe",
        "Validation dates (>= 2021-07-01)",
        "Validation etiquettes DPE (A-G)",
        "Clipping : surface [5, 1000], conso [0, 1000], cout [0, 50000]",
        "Detection PAC : regex sur type_generateur_chauffage et froid",
        "Detection climatisation : generateur froid renseigne",
    ])

    doc.add_heading("5.2 Fusion (DatasetMerger)", level=2)
    doc.add_paragraph(
        "DPE (agrege mois x dept) "
        "LEFT JOIN Meteo (agrege mois x dept) ON date_id + dept "
        "LEFT JOIN INSEE (mois) ON date_id "
        "LEFT JOIN Eurostat (mois) ON date_id"
    )
    doc.add_paragraph(
        "Les indicateurs economiques (INSEE, Eurostat) sont nationaux — "
        "ils sont broadcast sur tous les departements lors de la jointure."
    )

    doc.add_heading("5.3 Detection d'outliers", level=2)
    doc.add_paragraph(
        "Le module outlier_detection.py fournit plusieurs methodes : "
        "IQR, Z-score, Isolation Forest, LOF. Les outliers sont "
        "reportes mais pas supprimes par defaut (approche conservative)."
    )
    doc.add_page_break()

    # =================================================================
    # 6. FEATURE ENGINEERING
    # =================================================================
    doc.add_heading("6. Feature Engineering", level=1)
    doc.add_paragraph(
        "Le FeatureEngineer transforme le dataset en ~90 features. "
        "Toutes les operations temporelles sont faites PAR DEPARTEMENT "
        "(groupby dept) pour eviter les fuites inter-departements."
    )

    doc.add_heading("6.1 Lags temporels", level=2)
    add_table(doc,
        ["Feature", "Description"],
        [
            ["*_lag_1m", "Valeur du mois precedent"],
            ["*_lag_3m", "Valeur de 3 mois avant"],
            ["*_lag_6m", "Valeur de 6 mois avant"],
        ]
    )

    doc.add_heading("6.2 Rolling windows", level=2)
    add_table(doc,
        ["Feature", "Description"],
        [
            ["*_rmean_3m / *_rmean_6m", "Moyenne glissante sur 3/6 mois"],
            ["*_rstd_3m / *_rstd_6m", "Ecart-type glissant sur 3/6 mois"],
        ]
    )

    doc.add_heading("6.3 Variations", level=2)
    add_table(doc,
        ["Feature", "Description"],
        [
            ["*_diff_1m", "Difference absolue avec le mois precedent"],
            ["*_pct_1m", "Variation relative en % (clippee [-200, +500])"],
        ]
    )

    doc.add_heading("6.4 Features d'interaction", level=2)
    add_table(doc,
        ["Feature", "Hypothese metier"],
        [
            ["interact_hdd_confiance", "Hiver froid + confiance = investissement chauffage"],
            ["interact_cdd_ipi", "Ete chaud + activite industrielle = installations clim"],
            ["interact_confiance_bat", "Double proxy d'intention d'investissement"],
            ["jours_extremes", "Meteo extreme = declencheur d'achat"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 7. DATABASE SCHEMA
    # =================================================================
    doc.add_heading("7. Schema de base de donnees", level=1)
    doc.add_paragraph(
        "Schema en etoile (Star Schema) avec SQLAlchemy. "
        "Trois moteurs supportes : SQLite (defaut), PostgreSQL, SQL Server."
    )

    doc.add_heading("7.1 Tables dimensionnelles", level=2)
    add_table(doc,
        ["Table", "Grain", "Description"],
        [
            ["dim_time", "Mois", "Dimension temporelle (YYYYMM, saisons HVAC)"],
            ["dim_geo", "Departement", "96 departements avec ville de reference"],
            ["dim_equipment_type", "Type", "Catalogue des equipements HVAC"],
        ]
    )

    doc.add_heading("7.2 Tables de faits", level=2)
    add_table(doc,
        ["Table", "Grain", "Description"],
        [
            ["fact_hvac_installations", "Mois x Dept", "Variable cible + meteo + permis"],
            ["fact_economic_context", "Mois", "Indicateurs economiques nationaux"],
            ["raw_dpe", "DPE unitaire", "Diagnostics individuels"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 8. ML MODELING
    # =================================================================
    doc.add_heading("8. Modelisation ML", level=1)

    doc.add_heading("8.1 Split temporel", level=2)
    doc.add_paragraph("Le split respecte la chronologie (pas de fuite temporelle) :")
    add_table(doc,
        ["Split", "Periode", "Duree"],
        [
            ["Train", "2021-07 a 2024-06", "36 mois"],
            ["Validation", "2024-07 a 2024-12", "6 mois"],
            ["Test", "2025-01 a 2025-12+", "12+ mois"],
        ]
    )

    doc.add_heading("8.2 Preprocessing", level=2)
    add_numbered(doc, [
        "Separation features (X) / cible (y)",
        "Suppression des identifiants (date_id, dept, city_ref)",
        "Imputation median des NaN (SimpleImputer)",
        "Standardisation (StandardScaler) pour Ridge et LSTM",
    ])

    doc.add_heading("8.3 Modeles", level=2)

    doc.add_heading("Ridge Regression (robuste)", level=3)
    doc.add_paragraph(
        "Regression lineaire regularisee L2. Alpha selectionne par cross-validation "
        "temporelle (TimeSeriesSplit, 3 folds). Predictions negatives clippees a 0."
    )

    doc.add_heading("LightGBM (gradient boosting)", level=3)
    doc.add_paragraph(
        "Gradient boosting regularise pour petit dataset : "
        "max_depth=4, num_leaves=15, min_child_samples=20. "
        "Early stopping sur la validation (20 rounds)."
    )

    doc.add_heading("Prophet (series temporelles)", level=3)
    doc.add_paragraph(
        "Modele additif de Meta entraine par departement. "
        "Regresseurs externes : temp_mean, hdd_sum, cdd_sum, confiance_menages."
    )

    doc.add_heading("LSTM (exploratoire)", level=3)
    doc.add_paragraph(
        "Reseau de neurones recurrent (PyTorch). Exploratoire uniquement — "
        "le volume de donnees est trop faible pour un DL robuste."
    )

    doc.add_heading("8.4 Evaluation", level=2)
    add_table(doc,
        ["Metrique", "Interpretation"],
        [
            ["RMSE", "Erreur moyenne en unites de la cible"],
            ["MAE", "Erreur absolue moyenne"],
            ["MAPE", "Erreur relative en pourcentage"],
            ["R2", "Part de variance expliquee (1 = parfait)"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 9. RIDGE AUTOCORRELATION ANALYSIS
    # =================================================================
    doc.add_heading("9. Analyse de l'auto-correlation Ridge", level=1)

    doc.add_heading("9.1 Constat", level=2)
    doc.add_paragraph(
        "Le modele Ridge atteint un R2 de 0.998 sur le jeu de test. "
        "L'analyse des coefficients revele que les features dominantes "
        "sont les lags de la variable cible (auto-regression)."
    )

    doc.add_heading("9.2 Explication", level=2)
    doc.add_paragraph(
        "Ce n'est pas une fuite de donnees (les lags utilisent des valeurs du passe), "
        "mais le modele fait essentiellement de l'auto-regression. "
        "Il predit le mois M a partir du mois M-1."
    )

    doc.add_heading("9.3 Solution", level=2)
    doc.add_paragraph(
        "Le ModelTrainer dispose d'un parametre exclude_target_lags=True qui "
        "exclut toutes les features derivees de la variable cible. "
        "Un modele 'ridge_exogenes' est entraine automatiquement pour comparer."
    )
    doc.add_page_break()

    # =================================================================
    # 10. REST API (FastAPI)
    # =================================================================
    doc.add_heading("10. API REST (FastAPI)", level=1)

    doc.add_paragraph(
        "L'API expose les modeles ML via 6 endpoints REST. "
        "Elle est construite avec FastAPI et Pydantic v2."
    )

    doc.add_heading("10.1 Architecture", level=2)
    add_table(doc,
        ["Fichier", "Responsabilite"],
        [
            ["api/main.py", "Endpoints, CORS middleware, lifespan (chargement modeles)"],
            ["api/models.py", "Schemas Pydantic v2 (request/response)"],
            ["api/dependencies.py", "AppState singleton, chargement modeles .pkl"],
        ]
    )

    doc.add_heading("10.2 Endpoints", level=2)
    add_table(doc,
        ["Methode", "Endpoint", "Description"],
        [
            ["GET", "/health", "Health check + version + statut modeles"],
            ["GET", "/predictions", "Predictions stockees (avec filtres dept/date)"],
            ["POST", "/predict", "Prediction custom (valeurs manuelles)"],
            ["GET", "/data/summary", "Resume du dataset (lignes, colonnes, sources)"],
            ["GET", "/model/metrics", "Metriques d'evaluation des modeles"],
            ["GET", "/departments", "Liste des 96 departements"],
        ]
    )

    doc.add_heading("10.3 Demarrage", level=2)
    doc.add_paragraph("uvicorn api.main:app --reload", style="No Spacing")
    doc.add_paragraph("Documentation interactive : http://localhost:8000/docs")
    doc.add_page_break()

    # =================================================================
    # 11. STREAMLIT DASHBOARD
    # =================================================================
    doc.add_heading("11. Dashboard Streamlit", level=1)

    doc.add_paragraph(
        "Le dashboard Streamlit fournit 6 pages interactives pour "
        "explorer les donnees et les resultats des modeles."
    )

    add_table(doc,
        ["Page", "Fichier", "Contenu"],
        [
            ["Accueil", "app/pages/home.py", "Presentation du projet et KPIs"],
            ["Exploration", "app/pages/exploration.py", "Graphiques interactifs, filtres par dept/date"],
            ["Carte", "app/pages/carte.py", "Carte choropleth des departements"],
            ["Modeles", "app/pages/models.py", "Comparaison des modeles, metriques"],
            ["Predictions", "app/pages/predictions.py", "Predictions futures, graphiques"],
            ["Pipeline", "app/pages/pipeline_page.py", "Execution du pipeline depuis le dashboard"],
        ]
    )

    doc.add_paragraph("")
    doc.add_paragraph(
        "Le dashboard utilise @st.cache_data(ttl=300) pour le caching "
        "et un systeme de chargement dynamique des pages."
    )
    doc.add_paragraph("Demarrage : streamlit run app/app.py", style="No Spacing")
    doc.add_page_break()

    # =================================================================
    # 12. DEPLOYMENT (Docker / Kubernetes)
    # =================================================================
    doc.add_heading("12. Deploiement (Docker / Kubernetes)", level=1)

    doc.add_heading("12.1 Docker", level=2)
    doc.add_paragraph("Le Dockerfile utilise un build multi-stage (3 etapes) :")
    add_numbered(doc, [
        "base — Python 3.11-slim, dependances systeme",
        "dependencies — Installation pip (couche cacheee)",
        "app — Copie du code, entrypoint",
    ])

    doc.add_heading("12.2 Docker Compose", level=2)
    add_table(doc,
        ["Service", "Port", "Description"],
        [
            ["api", "8000", "FastAPI (uvicorn)"],
            ["dashboard", "8501", "Streamlit"],
            ["pipeline", "—", "Execution du pipeline (profile: tools)"],
            ["postgres", "5432", "PostgreSQL 16 (profile: db, optionnel)"],
        ]
    )

    doc.add_heading("12.3 Modes d'execution", level=2)
    doc.add_paragraph("Le docker-entrypoint.sh supporte 5 modes :")
    add_table(doc,
        ["Mode", "Commande", "Description"],
        [
            ["all", "docker run -e MODE=all ...", "API + Dashboard (defaut)"],
            ["api", "docker run -e MODE=api ...", "API seule"],
            ["dashboard", "docker run -e MODE=dashboard ...", "Dashboard seul"],
            ["pipeline", "docker run -e MODE=pipeline ...", "Execution pipeline"],
            ["demo", "docker run -e MODE=demo ...", "Donnees demo → pipeline → services"],
        ]
    )

    doc.add_heading("12.4 Kubernetes", level=2)
    doc.add_paragraph("6 manifestes dans le repertoire kubernetes/ :")
    add_table(doc,
        ["Fichier", "Ressource", "Description"],
        [
            ["namespace.yaml", "Namespace", "hvac-market"],
            ["api-deployment.yaml", "Deployment + Service", "2 replicas, probes /health"],
            ["dashboard-deployment.yaml", "Deployment + Service", "1 replica, probes /_stcore/health"],
            ["ingress.yaml", "Ingress", "nginx, /api/* → API, /* → Dashboard"],
            ["pvc.yaml", "PersistentVolumeClaim", "5Gi ReadWriteOnce"],
            ["cronjob-pipeline.yaml", "CronJob", "Pipeline mensuel (1er du mois, 6h)"],
        ]
    )

    doc.add_heading("12.5 Render (PaaS)", level=2)
    doc.add_paragraph(
        "Le fichier render.yaml definit 2 web services (API + Dashboard) "
        "deployables en un clic sur Render.com (free tier)."
    )
    doc.add_page_break()

    # =================================================================
    # 13. ORCHESTRATION (Airflow)
    # =================================================================
    doc.add_heading("13. Orchestration (Airflow)", level=1)

    doc.add_paragraph(
        "Le DAG Airflow (airflow/dags/hvac_pipeline_dag.py) orchestre "
        "le pipeline complet avec 3 groupes de taches :"
    )

    add_table(doc,
        ["TaskGroup", "Taches", "Mode"],
        [
            ["collect_group", "5 collecteurs (weather, insee, eurostat, dpe, sitadel)", "Parallele"],
            ["process_group", "clean → merge → features → eda (6 etapes)", "Sequentiel"],
            ["ml_group", "train + evaluate", "Parallele"],
        ]
    )

    doc.add_paragraph("")
    add_bullets(doc, [
        "Schedule : 0 6 1 * * (1er du mois a 6h)",
        "Quality gates entre chaque groupe (PythonOperator)",
        "SLAs et callbacks d'echec",
        "Execution_timeout pour eviter les blocages",
    ])
    doc.add_page_break()

    # =================================================================
    # 14. REINFORCEMENT LEARNING
    # =================================================================
    doc.add_heading("14. Reinforcement Learning (demo)", level=1)

    doc.add_paragraph(
        "Le module src/models/reinforcement_learning_demo.py implemente "
        "un environnement Gymnasium simulant la maintenance HVAC."
    )

    doc.add_heading("14.1 Environnement (HVACMaintenanceEnv)", level=2)
    add_table(doc,
        ["Element", "Description"],
        [
            ["Etat (6 dim)", "age_equipement, efficacite, temperature, humidite, "
             "nb_pannes, cout_cumule"],
            ["Actions (4)", "0: rien, 1: maintenance legere, 2: maintenance complete, "
             "3: remplacement"],
            ["Reward", "Efficacite x bonus — cout_action — penalite_panne"],
            ["Episode", "12 mois (1 an de maintenance)"],
        ]
    )

    doc.add_heading("14.2 Agent Q-Learning", level=2)
    add_bullets(doc, [
        "Discretisation de l'etat continu en bins",
        "Politique epsilon-greedy (exploration → exploitation)",
        "Comparaison avec politique aleatoire",
        "Visualisations : courbe d'apprentissage, heatmap Q-values",
    ])
    doc.add_page_break()

    # =================================================================
    # 15. UNIT TESTS
    # =================================================================
    doc.add_heading("15. Tests unitaires", level=1)

    doc.add_paragraph("Le projet dispose de 119 tests unitaires :")
    add_table(doc,
        ["Fichier", "Tests", "Couverture"],
        [
            ["tests/test_config.py", "11+", "GeoConfig, TimeConfig, DatabaseConfig, ModelConfig"],
            ["tests/test_collectors/test_base.py", "12", "CollectorStatus, Config, Result, Registry"],
            ["tests/test_collectors/test_weather.py", "6", "Validation, collecte mockee, resilience"],
            ["tests/test_collectors/test_insee.py", "5", "Validation, structure des series"],
            ["tests/test_collectors/test_pcloud_sync.py", "~20", "Sync, upload, download, retrocompat"],
            ["tests/test_processing/test_clean_data.py", "7", "Nettoyage meteo, INSEE, Eurostat"],
            ["tests/test_processing/test_feature_engineering.py", "16", "Lags, rolling, interactions"],
            ["tests/test_processing/test_outlier_detection.py", "~20", "IQR, Z-score, IF, LOF"],
            ["tests/test_models/test_robust_scaling.py", "~5", "RobustScaler"],
        ]
    )

    doc.add_paragraph(
        "Les tests utilisent des fixtures partagees (tests/conftest.py) "
        "et les appels HTTP sont mockes (pas de dependance reseau)."
    )
    doc.add_page_break()

    # =================================================================
    # 16. FORMATION MODULES COVERAGE
    # =================================================================
    doc.add_heading("16. Couverture des modules de formation", level=1)

    doc.add_paragraph(
        "Le projet couvre les 6 modules de la certification "
        "Data Science Lead (RNCP Bac+5, Jedha Bootcamp) :"
    )

    add_table(doc,
        ["Module", "Sujet", "Implementation dans le projet"],
        [
            ["M1", "Data Governance",
             "docs/DATA_GOVERNANCE.md — RGPD, AI Act, data lineage, maturity audit"],
            ["M2", "Deploiement & ML Distribue",
             "Dockerfile, docker-compose.yml, kubernetes/, render.yaml"],
            ["M3", "Architecture BDD",
             "docs/DATABASE_ARCHITECTURE.md — Star schema, OLAP, MongoDB"],
            ["M4", "Data Pipelines",
             "docs/DATA_PIPELINE.md — ETL architecture, monitoring, Airbyte"],
            ["M5", "Automatisation & Workflow",
             "airflow/dags/hvac_pipeline_dag.py — DAG complet, TaskGroups"],
            ["M6", "Reinforcement Learning",
             "src/models/reinforcement_learning_demo.py — Gymnasium + Q-Learning"],
        ]
    )

    # Save
    path = "docs/documentation_technique.docx"
    doc.save(path)
    print(f"Technical documentation generated: {path}")
    return path


# ================================================================
# DOCUMENT 2 : User Guide & Configuration
# ================================================================

def generate_user_guide():
    """Generate the user guide with installation, config, and accounts."""
    doc = Document()
    set_doc_styles(doc)
    add_cover_page(
        doc,
        "Guide d'Utilisation",
        "Installation, Configuration, Comptes et Deploiement"
    )

    # --- TABLE OF CONTENTS ---
    doc.add_heading("Table des matieres", level=1)
    toc = [
        "1. Prerequis",
        "2. Installation",
        "3. Configuration (.env)",
        "4. Commandes du pipeline",
        "5. Comptes et services externes",
        "6. Deploiement Docker",
        "7. Deploiement Kubernetes",
        "8. Deploiement Render (PaaS)",
        "9. Dashboard & API",
        "10. Tests",
        "11. Structure des donnees",
        "12. Depannage",
    ]
    for item in toc:
        doc.add_paragraph(item, style="List Number")
    doc.add_page_break()

    # =================================================================
    # 1. PREREQUISITES
    # =================================================================
    doc.add_heading("1. Prerequis", level=1)
    add_table(doc,
        ["Composant", "Version", "Notes"],
        [
            ["Python", "3.10+", "Requis"],
            ["Git", "2.x+", "Versionnement"],
            ["pip", "Derniere", "Gestionnaire de packages"],
            ["Docker", "20.x+", "Optionnel (deploiement conteneurise)"],
            ["Docker Compose", "2.x+", "Optionnel (multi-services)"],
            ["kubectl", "1.25+", "Optionnel (deploiement Kubernetes)"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 2. INSTALLATION
    # =================================================================
    doc.add_heading("2. Installation", level=1)

    doc.add_heading("2.1 Clonage du depot", level=2)
    doc.add_paragraph(
        "git clone https://github.com/PDUCLOS/Projet-HVAC.git\n"
        "cd Projet-HVAC",
        style="No Spacing"
    )

    doc.add_heading("2.2 Environnement virtuel", level=2)
    doc.add_paragraph("Windows :", style="List Bullet")
    doc.add_paragraph(
        "python -m venv venv\nvenv\\Scripts\\activate",
        style="No Spacing"
    )
    doc.add_paragraph("Linux / Mac :", style="List Bullet")
    doc.add_paragraph(
        "python -m venv venv\nsource venv/bin/activate",
        style="No Spacing"
    )

    doc.add_heading("2.3 Dependances", level=2)
    doc.add_paragraph(
        "# Core (requis)\n"
        "pip install -r requirements.txt\n\n"
        "# API (si utilisation FastAPI)\n"
        "pip install -r requirements-api.txt\n\n"
        "# Deep Learning (optionnel)\n"
        "pip install -r requirements-dl.txt",
        style="No Spacing"
    )

    doc.add_heading("2.4 Initialisation rapide", level=2)
    doc.add_paragraph("python setup_project.py", style="No Spacing")
    doc.add_paragraph(
        "Ce script cree les repertoires data/, initialise la BDD SQLite "
        "et verifie que les dependances sont installees."
    )
    doc.add_page_break()

    # =================================================================
    # 3. CONFIGURATION
    # =================================================================
    doc.add_heading("3. Configuration (.env)", level=1)

    doc.add_heading("3.1 Creation du fichier", level=2)
    doc.add_paragraph("cp .env.example .env", style="No Spacing")

    doc.add_heading("3.2 Variables principales", level=2)
    add_table(doc,
        ["Variable", "Defaut", "Description"],
        [
            ["DB_TYPE", "sqlite", "Type de BDD (sqlite, mssql, postgresql)"],
            ["DB_PATH", "data/hvac_market.db", "Chemin fichier SQLite"],
            ["DATA_START_DATE", "2019-01-01", "Debut de la collecte"],
            ["DATA_END_DATE", "2026-02-28", "Fin de la collecte"],
            ["TARGET_REGION", "FR", "Code region (FR = France metro, 96 depts)"],
            ["REQUEST_TIMEOUT", "30", "Timeout HTTP en secondes"],
            ["MAX_RETRIES", "3", "Nombre max de retries"],
            ["LOG_LEVEL", "INFO", "Niveau de logging"],
        ]
    )

    doc.add_heading("3.3 Configuration base de donnees", level=2)
    doc.add_paragraph(
        "# SQLite (defaut — aucune installation)\n"
        "DB_TYPE=sqlite\n"
        "DB_PATH=data/hvac_market.db\n\n"
        "# PostgreSQL (production/cloud)\n"
        "DB_TYPE=postgresql\n"
        "DB_HOST=localhost\n"
        "DB_PORT=5432\n"
        "DB_NAME=hvac\n"
        "DB_USER=hvac_user\n"
        "DB_PASSWORD=your_password\n"
        "ALLOW_NON_LOCAL=true",
        style="No Spacing"
    )
    doc.add_page_break()

    # =================================================================
    # 4. PIPELINE COMMANDS
    # =================================================================
    doc.add_heading("4. Commandes du pipeline", level=1)

    add_table(doc,
        ["Commande", "Description", "Duree"],
        [
            ["python -m src.pipeline list", "Lister les collecteurs", "< 1s"],
            ["python -m src.pipeline init_db", "Initialiser la BDD", "< 5s"],
            ["python -m src.pipeline collect", "Collecter toutes les sources", "30-90 min"],
            ["python -m src.pipeline collect --sources weather,insee",
             "Collecter des sources specifiques", "< 5 min"],
            ["python -m src.pipeline clean", "Nettoyer les donnees brutes", "1-3 min"],
            ["python -m src.pipeline merge", "Fusionner en dataset ML", "< 30s"],
            ["python -m src.pipeline features", "Feature engineering", "< 10s"],
            ["python -m src.pipeline process", "clean + merge + features", "2-5 min"],
            ["python -m src.pipeline eda", "Analyse exploratoire", "1-2 min"],
            ["python -m src.pipeline train", "Entrainer les modeles ML", "2-5 min"],
            ["python -m src.pipeline evaluate", "Evaluer les modeles", "3-5 min"],
            ["python -m src.pipeline all", "Pipeline complet", "40-90 min"],
        ]
    )

    doc.add_heading("Raccourcis Makefile", level=2)
    add_table(doc,
        ["Commande", "Action"],
        [
            ["make install", "Creer venv + installer dependances"],
            ["make collect", "Collecter toutes les sources"],
            ["make process", "Traiter les donnees"],
            ["make train", "Entrainer les modeles"],
            ["make all", "Pipeline complet"],
            ["make test", "Lancer les 119 tests"],
            ["make api", "Demarrer l'API FastAPI"],
            ["make dashboard", "Demarrer le dashboard Streamlit"],
            ["make docker-build", "Build de l'image Docker"],
            ["make docker-run", "Lancer via Docker"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 5. ACCOUNTS AND EXTERNAL SERVICES
    # =================================================================
    doc.add_heading("5. Comptes et services externes", level=1)

    doc.add_paragraph(
        "Voici les comptes et configurations necessaires selon votre usage :"
    )

    doc.add_heading("5.1 Aucun compte requis (usage local)", level=2)
    doc.add_paragraph(
        "Pour utiliser le projet en local, AUCUN compte n'est necessaire. "
        "Toutes les sources de donnees sont Open Data sans cle API."
    )
    add_bullets(doc, [
        "DPE ADEME : acces libre (data.ademe.fr)",
        "Open-Meteo : acces libre (archive-api.open-meteo.com)",
        "INSEE BDM : acces libre (bdm.insee.fr/sdmx)",
        "Eurostat : acces libre (ec.europa.eu/eurostat)",
    ])

    doc.add_heading("5.2 GitHub (versionning + CI/CD)", level=2)
    add_table(doc,
        ["Element", "Action"],
        [
            ["Compte", "Creer un compte sur github.com (gratuit)"],
            ["Repository", "Fork ou cloner PDUCLOS/Projet-HVAC"],
            ["Visibilite", "Mettre le repo en PUBLIC pour le portfolio"],
            ["GitHub Pages", "Optionnel : activer pour heberger la doc"],
        ]
    )

    doc.add_heading("5.3 Render.com (deploiement PaaS)", level=2)
    add_table(doc,
        ["Element", "Action"],
        [
            ["Compte", "Creer un compte sur render.com (gratuit)"],
            ["Connexion", "Lier avec votre compte GitHub"],
            ["Deploiement", "New > Blueprint > selectionner le repo"],
            ["Blueprint", "render.yaml configure 2 services automatiquement"],
            ["Tier", "Free tier (suffisant pour demo/portfolio)"],
            ["Variables", "Ajouter DB_TYPE=sqlite dans Environment"],
        ]
    )

    doc.add_heading("5.4 Docker Hub (images Docker)", level=2)
    add_table(doc,
        ["Element", "Action"],
        [
            ["Compte", "Creer un compte sur hub.docker.com (gratuit)"],
            ["Login", "docker login"],
            ["Build", "docker build -t votre-user/hvac-market ."],
            ["Push", "docker push votre-user/hvac-market"],
        ]
    )
    doc.add_paragraph(
        "Note : Docker Hub n'est necessaire que si vous voulez distribuer "
        "l'image. Pour un usage local, 'docker build' suffit."
    )

    doc.add_heading("5.5 Cloud Kubernetes (optionnel)", level=2)
    doc.add_paragraph(
        "Pour deployer sur un cluster Kubernetes :"
    )
    add_table(doc,
        ["Provider", "Service", "Tier gratuit"],
        [
            ["Google Cloud", "GKE Autopilot", "300$ credits (90 jours)"],
            ["AWS", "EKS", "Pas de free tier pour EKS"],
            ["Azure", "AKS", "Free tier control plane"],
            ["OVHcloud", "Managed Kubernetes", "Gratuit (pay-as-you-go nodes)"],
        ]
    )
    doc.add_paragraph(
        "Pour le portfolio, Render.com (gratuit) est recommande. "
        "Kubernetes est documente pour demontrer la competence."
    )

    doc.add_heading("5.6 pCloud (stockage donnees)", level=2)
    add_table(doc,
        ["Element", "Action"],
        [
            ["Compte", "Creer un compte sur pcloud.com (gratuit, 10 Go)"],
            ["Configuration", "Ajouter PCLOUD_USER et PCLOUD_PASS dans .env"],
            ["Usage", "python -m src.pipeline sync — synchronise data/"],
            ["Optionnel", "Uniquement si vous travaillez sur plusieurs machines"],
        ]
    )

    doc.add_heading("5.7 Streamlit Cloud (dashboard en ligne)", level=2)
    add_table(doc,
        ["Element", "Action"],
        [
            ["Compte", "Creer un compte sur streamlit.io (gratuit)"],
            ["Connexion", "Lier avec votre compte GitHub"],
            ["Deploiement", "New app > selectionner repo > app/app.py"],
            ["Donnees", "Utiliser data/sample/ pour le mode demo"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 6. DOCKER DEPLOYMENT
    # =================================================================
    doc.add_heading("6. Deploiement Docker", level=1)

    doc.add_heading("6.1 Build", level=2)
    doc.add_paragraph("docker build -t hvac-market .", style="No Spacing")

    doc.add_heading("6.2 Run (mode complet)", level=2)
    doc.add_paragraph(
        "docker compose up -d\n"
        "# API : http://localhost:8000\n"
        "# Dashboard : http://localhost:8501",
        style="No Spacing"
    )

    doc.add_heading("6.3 Run (mode demo)", level=2)
    doc.add_paragraph(
        "docker run -e MODE=demo -p 8000:8000 -p 8501:8501 hvac-market",
        style="No Spacing"
    )
    doc.add_paragraph(
        "Le mode demo genere des donnees synthetiques, execute le pipeline, "
        "puis lance l'API et le dashboard automatiquement."
    )
    doc.add_page_break()

    # =================================================================
    # 7. KUBERNETES DEPLOYMENT
    # =================================================================
    doc.add_heading("7. Deploiement Kubernetes", level=1)

    doc.add_paragraph("Appliquer les manifestes dans l'ordre :")
    doc.add_paragraph(
        "kubectl apply -f kubernetes/namespace.yaml\n"
        "kubectl apply -f kubernetes/pvc.yaml\n"
        "kubectl apply -f kubernetes/api-deployment.yaml\n"
        "kubectl apply -f kubernetes/dashboard-deployment.yaml\n"
        "kubectl apply -f kubernetes/ingress.yaml\n"
        "kubectl apply -f kubernetes/cronjob-pipeline.yaml",
        style="No Spacing"
    )
    doc.add_paragraph("")
    doc.add_paragraph(
        "Verifier le deploiement :\n"
        "kubectl get pods -n hvac-market\n"
        "kubectl get svc -n hvac-market",
        style="No Spacing"
    )
    doc.add_page_break()

    # =================================================================
    # 8. RENDER DEPLOYMENT
    # =================================================================
    doc.add_heading("8. Deploiement Render (PaaS)", level=1)

    doc.add_heading("Etapes", level=2)
    add_numbered(doc, [
        "Creer un compte sur render.com et lier GitHub",
        "Cliquer 'New > Blueprint'",
        "Selectionner le repository PDUCLOS/Projet-HVAC",
        "Render detecte automatiquement render.yaml",
        "2 services sont crees : hvac-api (FastAPI) et hvac-dashboard (Streamlit)",
        "Ajouter les variables d'environnement si besoin (DB_TYPE=sqlite)",
        "Cliquer Deploy — les URLs sont generees automatiquement",
    ])
    doc.add_page_break()

    # =================================================================
    # 9. DASHBOARD & API
    # =================================================================
    doc.add_heading("9. Dashboard & API", level=1)

    doc.add_heading("9.1 Lancement local", level=2)
    doc.add_paragraph(
        "# API (terminal 1)\n"
        "uvicorn api.main:app --reload --port 8000\n\n"
        "# Dashboard (terminal 2)\n"
        "streamlit run app/app.py --server.port 8501",
        style="No Spacing"
    )

    doc.add_heading("9.2 Documentation API", level=2)
    add_bullets(doc, [
        "Swagger UI : http://localhost:8000/docs",
        "ReDoc : http://localhost:8000/redoc",
        "OpenAPI JSON : http://localhost:8000/openapi.json",
    ])
    doc.add_page_break()

    # =================================================================
    # 10. TESTS
    # =================================================================
    doc.add_heading("10. Tests", level=1)

    doc.add_paragraph("python -m pytest tests/ -v", style="No Spacing")
    doc.add_paragraph("")
    doc.add_paragraph(
        "# Avec couverture\n"
        "pip install pytest-cov\n"
        "python -m pytest tests/ --cov=src --cov=config --cov-report=term-missing",
        style="No Spacing"
    )
    doc.add_page_break()

    # =================================================================
    # 11. DATA STRUCTURE
    # =================================================================
    doc.add_heading("11. Structure des donnees", level=1)

    add_table(doc,
        ["Repertoire", "Contenu"],
        [
            ["data/raw/weather/", "weather_france.csv — donnees meteo quotidiennes"],
            ["data/raw/insee/", "indicateurs_economiques.csv — series INSEE"],
            ["data/raw/eurostat/", "ipi_hvac_france.csv — indices de production"],
            ["data/raw/dpe/", "dpe_france_all.csv — DPE bruts"],
            ["data/processed/", "Donnees nettoyees"],
            ["data/features/", "hvac_ml_dataset.csv et hvac_features_dataset.csv"],
            ["data/models/", "Modeles .pkl, metriques, rapports"],
            ["data/models/figures/", "Graphiques de visualisation"],
            ["data/analysis/", "Resultats EDA et correlations"],
            ["data/sample/", "Donnees echantillon pour demo (200 lignes)"],
        ]
    )
    doc.add_page_break()

    # =================================================================
    # 12. TROUBLESHOOTING
    # =================================================================
    doc.add_heading("12. Depannage", level=1)

    problems = [
        (
            "ModuleNotFoundError",
            "pip install -r requirements.txt\n"
            "Verifier que l'environnement virtuel est active."
        ),
        (
            "La collecte DPE echoue ou est lente",
            "L'API ADEME peut etre surchargee. Relancer la commande. "
            "Le collecteur est idempotent."
        ),
        (
            "FileNotFoundError: Dataset introuvable",
            "Les etapes doivent etre executees dans l'ordre : "
            "collect → process → train. "
            "Lancer : python -m src.pipeline process"
        ),
        (
            "Ridge R2 > 0.99",
            "Normal avec les features lag (auto-regression). "
            "Voir le modele 'ridge_exogenes' pour les performances "
            "sans auto-correlation."
        ),
        (
            "Docker build echoue",
            "Verifier que Docker est installe et demarre. "
            "Verifier l'espace disque (>5 Go recommandes)."
        ),
        (
            "API ne demarre pas",
            "Verifier que les modeles .pkl existent dans data/models/. "
            "Sinon, lancer : python -m src.pipeline train"
        ),
    ]

    for problem, solution in problems:
        doc.add_heading(problem, level=3)
        doc.add_paragraph(solution)

    doc.add_heading("Contact", level=2)
    doc.add_paragraph(
        "Auteur : Patrice DUCLOS — Data Analyst Senior\n"
        "Depot GitHub : https://github.com/PDUCLOS/Projet-HVAC\n"
        "Formation : Data Science Lead — Jedha Bootcamp"
    )

    # Save
    path = "docs/guide_utilisation.docx"
    doc.save(path)
    print(f"User guide generated: {path}")
    return path


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_technical_doc()
    generate_user_guide()
    print("\nDocumentation generated successfully in docs/")
