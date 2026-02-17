# -*- coding: utf-8 -*-
"""
Generateur de documentation Word pour le projet HVAC Market Analysis.
=====================================================================

Genere deux documents :
1. documentation_technique.docx — Architecture, processus, methodologie
2. guide_utilisation.docx — Installation, commandes, utilisation quotidienne

Usage :
    python docs/generate_docs.py
"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
import os


def set_doc_styles(doc):
    """Configure les styles du document."""
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    for level in range(1, 5):
        heading_style = doc.styles[f"Heading {level}"]
        heading_style.font.color.rgb = RGBColor(0, 51, 102)


def add_cover_page(doc, title, subtitle):
    """Ajoute une page de couverture."""
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
    run = p.add_run("Projet HVAC Market Analysis\nAuvergne-Rhone-Alpes")
    run.font.size = Pt(14)

    doc.add_paragraph("")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Patrice DUCLOS\nData Analyst Senior\nFevrier 2026")
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(100, 100, 100)

    doc.add_page_break()


def add_table(doc, headers, rows):
    """Ajoute un tableau formate."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # En-tetes
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    # Donnees
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            table.rows[r + 1].cells[c].text = str(val)

    doc.add_paragraph("")
    return table


# ================================================================
# DOCUMENT 1 : Documentation Technique
# ================================================================

def generate_technical_doc():
    """Genere la documentation technique complete."""
    doc = Document()
    set_doc_styles(doc)
    add_cover_page(
        doc,
        "Documentation Technique",
        "Architecture, Processus et Methodologie"
    )

    # --- TABLE DES MATIERES ---
    doc.add_heading("Table des matieres", level=1)
    toc_items = [
        "1. Vue d'ensemble du projet",
        "2. Architecture technique",
        "3. Sources de donnees",
        "4. Pipeline de collecte (Phase 1)",
        "5. Traitement des donnees (Phase 2)",
        "6. Feature Engineering",
        "7. Schema de base de donnees",
        "8. Modelisation ML (Phase 4)",
        "9. Analyse de l'auto-correlation Ridge",
        "10. Evaluation des modeles",
        "11. Tests unitaires",
        "12. Configuration et parametrage",
    ]
    for item in toc_items:
        doc.add_paragraph(item, style="List Number")
    doc.add_page_break()

    # === 1. VUE D'ENSEMBLE ===
    doc.add_heading("1. Vue d'ensemble du projet", level=1)

    doc.add_heading("1.1 Objectif", level=2)
    doc.add_paragraph(
        "Ce projet construit un pipeline data complet pour analyser et predire "
        "les installations d'equipements HVAC (pompes a chaleur, climatisation) "
        "dans les 8 departements de la region Auvergne-Rhone-Alpes (AURA)."
    )
    doc.add_paragraph(
        "Il s'agit d'un projet portfolio Data Analyst demonstrant la maitrise "
        "de la chaine complete : collecte de donnees ouvertes, nettoyage, "
        "modelisation en etoile, feature engineering, modelisation ML et evaluation."
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
            ["Region", "Auvergne-Rhone-Alpes (code 84)"],
            ["Departements", "01, 07, 26, 38, 42, 69, 73, 74"],
            ["Periode", "Juillet 2021 - Fevrier 2026 (55 mois)"],
            ["Grain", "Mois x Departement (8 depts = 440 observations)"],
            ["Volume DPE", "~1.4 million de diagnostics"],
            ["PAC detectees", "~95 140 (6.9% des DPE)"],
            ["Features ML", "~90 features engineerees"],
        ]
    )

    doc.add_heading("1.4 Resultats cles", level=2)
    add_table(doc,
        ["Modele", "Val RMSE", "Val R2", "Test RMSE", "Test R2"],
        [
            ["Ridge (avec lags)", "1.09", "0.9975", "0.96", "0.9982"],
            ["Ridge (exogenes)", "-", "-", "-", "-"],
            ["LightGBM", "5.45", "0.937", "8.37", "0.865"],
            ["Prophet", "13.75", "0.599", "12.05", "0.719"],
        ]
    )
    doc.add_paragraph(
        "Note : Le R2 tres eleve de Ridge est principalement du a l'auto-correlation "
        "de la variable cible via les features lag. Le modele 'Ridge exogenes' "
        "permet d'evaluer la vraie contribution des features meteo et economiques "
        "(voir section 9)."
    )

    doc.add_page_break()

    # === 2. ARCHITECTURE TECHNIQUE ===
    doc.add_heading("2. Architecture technique", level=1)

    doc.add_heading("2.1 Structure du projet", level=2)
    doc.add_paragraph(
        "Le projet suit une architecture modulaire en couches :"
    )
    add_table(doc,
        ["Couche", "Repertoire", "Responsabilite"],
        [
            ["Configuration", "config/settings.py", "Parametres centralises (dataclasses frozen)"],
            ["Collecte", "src/collectors/", "5 collecteurs plugins (auto-enregistrement)"],
            ["Traitement", "src/processing/", "Nettoyage, fusion, feature engineering"],
            ["Base de donnees", "src/database/", "Schema en etoile, import CSV"],
            ["Analyse", "src/analysis/", "EDA automatisee, correlations"],
            ["Modelisation", "src/models/", "Ridge, LightGBM, Prophet, LSTM"],
            ["Orchestration", "src/pipeline.py", "CLI avec 10 etapes + mode 'all'"],
            ["Tests", "tests/", "57 tests unitaires (pytest)"],
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
        "Cela garantit que la configuration ne peut pas etre modifiee accidentellement "
        "pendant l'execution. Les valeurs sont chargees depuis le fichier .env "
        "via python-dotenv."
    )

    doc.add_heading("Pipeline idempotent", level=3)
    doc.add_paragraph(
        "Chaque etape du pipeline peut etre relancee sans risque. Les fichiers "
        "de sortie sont simplement ecrases. Les doublons sont geres en amont "
        "(deduplication dans le save et le clean)."
    )

    doc.add_heading("2.3 Technologies", level=2)
    add_table(doc,
        ["Categorie", "Technologies"],
        [
            ["Langage", "Python 3.10+"],
            ["Data", "pandas, numpy, SQLAlchemy"],
            ["ML", "scikit-learn (Ridge, StandardScaler, TimeSeriesSplit)"],
            ["Gradient Boosting", "LightGBM (regularise pour petit dataset)"],
            ["Series temporelles", "Prophet (Meta/Facebook)"],
            ["Deep Learning", "PyTorch (LSTM, exploratoire)"],
            ["Interpretabilite", "SHAP"],
            ["Visualisation", "matplotlib, seaborn, plotly"],
            ["BDD", "SQLite (defaut), SQL Server, PostgreSQL"],
            ["Tests", "pytest"],
            ["HTTP", "requests (retry automatique via urllib3)"],
            ["XML", "lxml (parsing SDMX pour INSEE)"],
        ]
    )

    doc.add_page_break()

    # === 3. SOURCES DE DONNEES ===
    doc.add_heading("3. Sources de donnees", level=1)

    doc.add_paragraph(
        "Toutes les sources sont Open Data — aucune cle API n'est requise."
    )

    doc.add_heading("3.1 DPE ADEME (variable cible)", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Source", "data.ademe.fr (API JSON paginee)"],
            ["Volume", "~1.4 million de DPE pour AURA"],
            ["Periode", "Juillet 2021+ (DPE v2)"],
            ["Grain", "1 ligne par diagnostic"],
            ["Champs cles", "numero_dpe, date_etablissement, etiquette_dpe, "
             "type_generateur_chauffage_principal, type_generateur_froid"],
            ["Detection PAC", "Regex sur chauffage/froid : PAC, pompe a chaleur, thermodynamique"],
            ["Collecteur", "DpeCollector (src/collectors/dpe.py)"],
        ]
    )
    doc.add_paragraph(
        "La collecte DPE prend environ 30-60 minutes en raison de la pagination "
        "API (1000 resultats par page, ~1400 pages)."
    )

    doc.add_heading("3.2 Open-Meteo (features meteo)", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Source", "archive-api.open-meteo.com"],
            ["Volume", "~20 000 lignes (8 villes x ~2500 jours)"],
            ["Variables", "temperature max/min/mean, precipitations, vent"],
            ["Derivees", "HDD (Heating Degree Days), CDD (Cooling Degree Days)"],
            ["Base HDD/CDD", "18 degres Celsius"],
            ["Collecteur", "WeatherCollector (src/collectors/weather.py)"],
        ]
    )

    doc.add_heading("3.3 INSEE BDM (features economiques)", level=2)
    add_table(doc,
        ["Serie", "idbank", "Description"],
        [
            ["confiance_menages", "001759970", "Indicateur synthetique de confiance des menages"],
            ["climat_affaires_industrie", "001565530", "Climat des affaires tous secteurs"],
            ["climat_affaires_batiment", "001586808", "Climat des affaires batiment"],
            ["opinion_achats_importants", "001759974", "Opportunite achats importants"],
            ["situation_financiere_future", "001759972", "Situation financiere future menages"],
            ["ipi_industrie_manuf", "010768261", "IPI industrie manufacturiere"],
        ]
    )
    doc.add_paragraph(
        "Format : XML SDMX 2.1 (StructureSpecific). Les idbanks ont ete verifies "
        "et corriges lors de l'audit de fevrier 2026."
    )

    doc.add_heading("3.4 Eurostat (IPI HVAC)", level=2)
    add_table(doc,
        ["Parametre", "Valeur"],
        [
            ["Dataset", "sts_inpr_m (Short-term statistics, Industrial Production)"],
            ["Codes NACE", "C28 (machines), C2825 (equipements clim)"],
            ["Filtre geo", "France (FR)"],
            ["Unite", "I21 (indice base 2021)"],
            ["Ajustement", "SCA (CVS-CJO)"],
            ["Collecteur", "EurostatCollector (src/collectors/eurostat_col.py)"],
        ]
    )

    doc.add_heading("3.5 SITADEL (permis de construire)", level=2)
    doc.add_paragraph(
        "Source en cours de migration vers l'API DiDo du SDES. "
        "Le collecteur SitadelCollector existe mais n'est pas pleinement fonctionnel."
    )

    doc.add_page_break()

    # === 4. PIPELINE DE COLLECTE ===
    doc.add_heading("4. Pipeline de collecte (Phase 1)", level=1)

    doc.add_heading("4.1 Architecture BaseCollector", level=2)
    doc.add_paragraph(
        "Tous les collecteurs heritent de BaseCollector (classe abstraite) qui fournit :"
    )
    bullets = [
        "Session HTTP avec retry automatique (codes 429, 500-504)",
        "Helpers fetch_json(), fetch_xml(), fetch_bytes()",
        "Cycle de vie orchestre : collect() -> validate() -> save()",
        "Logging structure avec horodatage",
        "Auto-enregistrement dans CollectorRegistry via __init_subclass__",
        "Pause de politesse entre appels API (rate_limit_delay)",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("4.2 Cycle de vie d'une collecte", level=2)
    doc.add_paragraph(
        "La methode run() orchestre le cycle complet :"
    )
    steps = [
        "Log de demarrage avec parametres (periode, departements)",
        "collect() — Recuperation des donnees brutes (specifique a chaque source)",
        "validate() — Verification qualite (colonnes, types, plages de valeurs)",
        "save() — Persistance CSV dans data/raw/ + copie dans data/processed/",
        "Log de fin avec bilan (lignes collectees, duree, statut)",
    ]
    for i, s in enumerate(steps, 1):
        doc.add_paragraph(f"{i}. {s}")

    doc.add_heading("4.3 Resilience", level=2)
    doc.add_paragraph(
        "Le systeme est concu pour etre tolerant aux pannes :"
    )
    bullets = [
        "Si une ville echoue (meteo), les autres sont quand meme collectees",
        "Si une serie echoue (INSEE), les autres sont fusionnees",
        "Le retry HTTP est automatique avec backoff exponentiel",
        "Le statut PARTIAL est retourne si certains elements ont echoue",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_page_break()

    # === 5. TRAITEMENT DES DONNEES ===
    doc.add_heading("5. Traitement des donnees (Phase 2)", level=1)

    doc.add_heading("5.1 Nettoyage (DataCleaner)", level=2)
    doc.add_paragraph(
        "Le module clean_data.py applique des regles de nettoyage specifiques "
        "a chaque source. Le nettoyage est idempotent."
    )

    doc.add_heading("Meteo", level=3)
    bullets = [
        "Conversion dates en datetime",
        "Suppression doublons (date x ville)",
        "Clipping : temperature [-30, 50], precipitations [0, 300], vent [0, 200]",
        "Recalcul HDD/CDD (base 18 degres)",
        "Ajout colonnes derivees : year, month, date_id",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("INSEE", level=3)
    bullets = [
        "Filtrage periodes mensuelles (exclusion trimestrielles)",
        "Interpolation lineaire des gaps courts (max 3 mois consecutifs)",
        "Conversion date_id (YYYYMM)",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("DPE", level=3)
    bullets = [
        "Traitement par chunks de 200 000 lignes (gestion memoire)",
        "Deduplication sur numero_dpe",
        "Validation dates (>= 2021-07-01, <= aujourd'hui)",
        "Validation etiquettes DPE (A-G uniquement)",
        "Clipping : surface [5, 1000], conso [0, 1000], cout [0, 50000]",
        "Detection PAC : regex sur type_generateur_chauffage et type_generateur_froid",
        "Detection climatisation : generateur froid renseigne",
        "Detection classe A-B : batiment performant",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("5.2 Fusion (DatasetMerger)", level=2)
    doc.add_paragraph(
        "Le DatasetMerger fusionne toutes les sources en un dataset unique "
        "au grain mois x departement :"
    )
    doc.add_paragraph(
        "DPE (agrege mois x dept) "
        "LEFT JOIN Meteo (agrege mois x dept) ON date_id + dept "
        "LEFT JOIN INSEE (mois) ON date_id "
        "LEFT JOIN Eurostat (mois x NACE) ON date_id"
    )
    doc.add_paragraph(
        "Les indicateurs economiques (INSEE, Eurostat) sont nationaux — "
        "ils sont broadcast sur tous les departements lors de la jointure. "
        "Le dataset final contient ~35 colonnes et ~440 lignes."
    )

    doc.add_page_break()

    # === 6. FEATURE ENGINEERING ===
    doc.add_heading("6. Feature Engineering", level=1)

    doc.add_paragraph(
        "Le FeatureEngineer transforme le dataset de ~35 colonnes en ~90 features. "
        "Toutes les operations temporelles (lags, rolling) sont faites PAR DEPARTEMENT "
        "(groupby dept) pour eviter les fuites inter-departements."
    )

    doc.add_heading("6.1 Lags temporels", level=2)
    add_table(doc,
        ["Feature", "Description", "Colonnes source"],
        [
            ["*_lag_1m", "Valeur du mois precedent", "nb_dpe_total, nb_installations_pac, "
             "nb_installations_clim, temp_mean, hdd_sum, cdd_sum, confiance_menages"],
            ["*_lag_3m", "Valeur de 3 mois avant", "Memes colonnes"],
            ["*_lag_6m", "Valeur de 6 mois avant", "Memes colonnes"],
        ]
    )
    doc.add_paragraph(
        "Les lags creent des NaN en debut de serie. Ces NaN sont geres par "
        "imputation median lors de l'entrainement."
    )

    doc.add_heading("6.2 Rolling windows", level=2)
    add_table(doc,
        ["Feature", "Description"],
        [
            ["*_rmean_3m", "Moyenne glissante sur 3 mois (min_periods=1)"],
            ["*_rmean_6m", "Moyenne glissante sur 6 mois"],
            ["*_rstd_3m", "Ecart-type glissant sur 3 mois (min_periods=2)"],
            ["*_rstd_6m", "Ecart-type glissant sur 6 mois"],
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
        ["Feature", "Formule", "Hypothese metier"],
        [
            ["interact_hdd_confiance", "HDD_norm x confiance/100",
             "Hiver froid + confiance = investissement chauffage"],
            ["interact_cdd_ipi", "CDD_norm x IPI_C28/100",
             "Ete chaud + activite industrielle = installations clim"],
            ["interact_confiance_bat", "confiance/100 x climat_bat/100",
             "Double proxy d'intention d'investissement"],
            ["jours_extremes", "nb_jours_canicule + nb_jours_gel",
             "Meteo extreme = declencheur d'achat"],
        ]
    )

    doc.add_heading("6.5 Features de tendance", level=2)
    add_table(doc,
        ["Feature", "Description"],
        [
            ["year_trend", "Annee normalisee (2021=0, 2022=1, ...)"],
            ["delta_temp_vs_mean", "Ecart de temperature vs moyenne historique dept x mois"],
        ]
    )

    doc.add_page_break()

    # === 7. SCHEMA BDD ===
    doc.add_heading("7. Schema de base de donnees", level=1)

    doc.add_paragraph(
        "Le projet utilise un schema en etoile (Star Schema) avec SQLAlchemy "
        "comme couche d'abstraction. Trois moteurs sont supportes :"
    )
    bullets = [
        "SQLite (defaut) — aucune installation, fichier local",
        "SQL Server — via pyodbc, pour environnement entreprise",
        "PostgreSQL — pour deploiement cloud/production",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("7.1 Tables dimensionnelles", level=2)
    add_table(doc,
        ["Table", "Grain", "Lignes", "Description"],
        [
            ["dim_time", "Mois", "84", "Dimension temporelle (YYYYMM, saisons HVAC)"],
            ["dim_geo", "Departement", "8", "8 departements AURA avec ville de reference"],
            ["dim_equipment_type", "Type", "10", "Catalogue des equipements HVAC"],
        ]
    )

    doc.add_heading("7.2 Tables de faits", level=2)
    add_table(doc,
        ["Table", "Grain", "Description"],
        [
            ["fact_hvac_installations", "Mois x Dept", "Variable cible + meteo + permis"],
            ["fact_economic_context", "Mois", "Indicateurs economiques nationaux"],
            ["raw_dpe", "DPE unitaire", "~1.4M diagnostics individuels"],
        ]
    )

    doc.add_page_break()

    # === 8. MODELISATION ML ===
    doc.add_heading("8. Modelisation ML (Phase 4)", level=1)

    doc.add_heading("8.1 Split temporel", level=2)
    doc.add_paragraph(
        "Le split respecte la chronologie (pas de fuite temporelle) :"
    )
    add_table(doc,
        ["Split", "Periode", "Duree", "Lignes (8 depts)"],
        [
            ["Train", "2021-07 a 2024-06", "36 mois", "~288"],
            ["Validation", "2024-07 a 2024-12", "6 mois", "~48"],
            ["Test", "2025-01 a 2025-12+", "12+ mois", "~96"],
        ]
    )

    doc.add_heading("8.2 Preprocessing", level=2)
    steps = [
        "Separation features (X) / cible (y)",
        "Suppression des identifiants (date_id, dept, city_ref, etc.)",
        "Suppression des lignes avec cible NaN",
        "Imputation median des NaN dans les features (SimpleImputer)",
        "Standardisation (StandardScaler) pour Ridge et LSTM",
    ]
    for i, s in enumerate(steps, 1):
        doc.add_paragraph(f"{i}. {s}")

    doc.add_heading("8.3 Modeles", level=2)

    doc.add_heading("Ridge Regression (Tier 1 — robuste)", level=3)
    doc.add_paragraph(
        "Regression lineaire regularisee L2. Alpha selectionne par cross-validation "
        "temporelle (TimeSeriesSplit, 3 folds). Tres stable sur petit dataset. "
        "Predictions negatives clippees a 0 (comptages)."
    )

    doc.add_heading("Ridge Exogenes (sans lags cible)", level=3)
    doc.add_paragraph(
        "Variante de Ridge qui exclut les features auto-regressives de la variable "
        "cible (lags, rolling, diff, pct_change de nb_installations_pac). "
        "Permet d'evaluer la vraie contribution predictive des features exogenes "
        "(meteo, economie) sans l'auto-correlation qui domine le R2 standard."
    )

    doc.add_heading("LightGBM (Tier 2 — gradient boosting)", level=3)
    doc.add_paragraph(
        "Gradient boosting regularise pour petit dataset : "
        "max_depth=4, num_leaves=15, min_child_samples=20, "
        "reg_alpha=0.1, reg_lambda=0.1, subsample=0.8. "
        "Early stopping sur la validation (20 rounds)."
    )

    doc.add_heading("Prophet (Tier 1 — series temporelles)", level=3)
    doc.add_paragraph(
        "Modele additif de Meta/Facebook entraine par departement. "
        "Capture tendance + saisonnalite annuelle. Regresseurs externes : "
        "temp_mean, hdd_sum, cdd_sum, confiance_menages, ipi_hvac_c28."
    )

    doc.add_heading("LSTM (exploratoire)", level=3)
    doc.add_paragraph(
        "Reseau de neurones recurrent (PyTorch). Exploratoire uniquement — "
        "le volume de donnees (~288 lignes train) est trop faible pour un DL robuste."
    )

    doc.add_page_break()

    # === 9. ANALYSE AUTO-CORRELATION ===
    doc.add_heading("9. Analyse de l'auto-correlation Ridge", level=1)

    doc.add_heading("9.1 Constat", level=2)
    doc.add_paragraph(
        "Le modele Ridge atteint un R2 de 0.998 sur le jeu de test, "
        "ce qui est exceptionnellement eleve. L'analyse des coefficients "
        "revele que les features dominantes sont les lags de la variable cible :"
    )
    add_table(doc,
        ["Feature", "Importance (coeff. absolu)"],
        [
            ["nb_installations_pac_lag_1m", "Tres elevee"],
            ["nb_installations_pac_rmean_3m", "Elevee"],
            ["nb_installations_pac_lag_3m", "Elevee"],
            ["nb_dpe_total_lag_1m", "Moyenne"],
        ]
    )

    doc.add_heading("9.2 Explication", level=2)
    doc.add_paragraph(
        "Ce n'est pas une fuite de donnees au sens strict (les lags utilisent "
        "des valeurs du passe, pas du futur), mais le modele fait essentiellement "
        "de l'auto-regression : il predit le nombre d'installations du mois M "
        "a partir du nombre d'installations du mois M-1. C'est une strategie "
        "valide mais qui ne demontre pas la valeur predictive des features "
        "exogenes (meteo, economie)."
    )

    doc.add_heading("9.3 Solution implementee", level=2)
    doc.add_paragraph(
        "Le ModelTrainer dispose d'un parametre exclude_target_lags=True qui "
        "exclut automatiquement toutes les features derivees de la variable cible "
        "(pattern : *_lag_*, *_rmean_*, *_rstd_*, *_diff_*, *_pct_*). "
        "Un modele 'ridge_exogenes' est entraine automatiquement dans train_all() "
        "pour comparer les performances avec et sans auto-correlation."
    )
    doc.add_paragraph(
        "Cela permet de repondre a la question : "
        "'Les features meteo et economiques apportent-elles une information "
        "predictive reelle, ou est-ce que seule l'inertie du marche explique "
        "les installations ?'"
    )

    doc.add_page_break()

    # === 10. EVALUATION ===
    doc.add_heading("10. Evaluation des modeles", level=1)

    doc.add_heading("10.1 Metriques", level=2)
    add_table(doc,
        ["Metrique", "Formule", "Interpretation"],
        [
            ["RMSE", "sqrt(mean((y-yhat)^2))", "Erreur moyenne en unites de la cible"],
            ["MAE", "mean(|y-yhat|)", "Erreur absolue moyenne"],
            ["MAPE", "mean(|y-yhat|/|y|) x 100", "Erreur relative en pourcentage"],
            ["R2", "1 - SS_res/SS_tot", "Part de variance expliquee (1 = parfait)"],
        ]
    )

    doc.add_heading("10.2 Visualisations generees", level=2)
    bullets = [
        "Predictions vs reel (validation + test) pour chaque modele",
        "Comparaison des metriques (graphique en barres groupees)",
        "Analyse des residus (distribution + scatter residus vs predictions)",
        "Feature importance (coefficients Ridge, gain LightGBM)",
        "Analyse SHAP (si le package shap est installe)",
    ]
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

    doc.add_heading("10.3 Cross-validation", level=2)
    doc.add_paragraph(
        "Une cross-validation temporelle (TimeSeriesSplit, 3 folds) est effectuee "
        "sur Ridge et LightGBM pour estimer la stabilite des performances. "
        "Les scores moyens et ecarts-types sont reportes dans training_results.csv."
    )

    doc.add_page_break()

    # === 11. TESTS UNITAIRES ===
    doc.add_heading("11. Tests unitaires", level=1)

    doc.add_paragraph(
        "Le projet dispose de 57 tests unitaires repartis en 3 categories :"
    )
    add_table(doc,
        ["Fichier", "Tests", "Couverture"],
        [
            ["tests/test_config.py", "11", "GeoConfig, TimeConfig, DatabaseConfig, "
             "ProjectConfig, ModelConfig"],
            ["tests/test_collectors/test_base.py", "12", "CollectorStatus, CollectorConfig, "
             "CollectorResult, CollectorRegistry"],
            ["tests/test_collectors/test_weather.py", "6", "Validation, collecte mockee, "
             "resilience partielle"],
            ["tests/test_collectors/test_insee.py", "5", "Validation, structure des series"],
            ["tests/test_processing/test_clean_data.py", "7", "Nettoyage meteo, INSEE, Eurostat"],
            ["tests/test_processing/test_feature_engineering.py", "9+7=16", "Lags, rolling, "
             "interactions, tendances, cas limites"],
        ]
    )

    doc.add_paragraph(
        "Les tests utilisent des fixtures partagees definies dans tests/conftest.py "
        "(configurations de test, DataFrames synthetiques). Les appels HTTP sont "
        "mockes pour eviter les dependances reseau."
    )

    doc.add_page_break()

    # === 12. CONFIGURATION ===
    doc.add_heading("12. Configuration et parametrage", level=1)

    doc.add_heading("12.1 Variables d'environnement", level=2)
    add_table(doc,
        ["Variable", "Defaut", "Description"],
        [
            ["DB_TYPE", "sqlite", "Type de BDD (sqlite, mssql, postgresql)"],
            ["DB_PATH", "data/hvac_market.db", "Chemin fichier SQLite"],
            ["DATA_START_DATE", "2019-01-01", "Debut de la collecte"],
            ["DATA_END_DATE", "2026-02-28", "Fin de la collecte"],
            ["TARGET_REGION", "84", "Code region INSEE"],
            ["TARGET_DEPARTMENTS", "01,07,26,38,42,69,73,74", "Departements cibles"],
            ["REQUEST_TIMEOUT", "30", "Timeout HTTP en secondes"],
            ["MAX_RETRIES", "3", "Nombre max de retries"],
            ["LOG_LEVEL", "INFO", "Niveau de logging"],
        ]
    )

    doc.add_heading("12.2 Configuration ML", level=2)
    add_table(doc,
        ["Parametre", "Defaut", "Description"],
        [
            ["max_lag_months", "6", "Lag maximum pour les features retardees"],
            ["rolling_windows", "[3, 6]", "Tailles des fenetres glissantes"],
            ["hdd_base_temp", "18.0", "Temperature de base HDD (degres C)"],
            ["cdd_base_temp", "18.0", "Temperature de base CDD (degres C)"],
            ["lightgbm max_depth", "4", "Profondeur max des arbres"],
            ["lightgbm num_leaves", "15", "Nombre max de feuilles"],
            ["lightgbm learning_rate", "0.05", "Taux d'apprentissage"],
        ]
    )

    # Sauvegarder
    path = "docs/documentation_technique.docx"
    doc.save(path)
    print(f"Documentation technique generee : {path}")
    return path


# ================================================================
# DOCUMENT 2 : Guide d'utilisation
# ================================================================

def generate_user_guide():
    """Genere le guide d'utilisation."""
    doc = Document()
    set_doc_styles(doc)
    add_cover_page(
        doc,
        "Guide d'Utilisation",
        "Installation, Commandes et Utilisation Quotidienne"
    )

    # --- TABLE DES MATIERES ---
    doc.add_heading("Table des matieres", level=1)
    toc_items = [
        "1. Prerequis",
        "2. Installation",
        "3. Configuration",
        "4. Commandes du pipeline",
        "5. Workflow quotidien",
        "6. Travailler sur plusieurs machines",
        "7. Tests",
        "8. Structure des donnees",
        "9. Depannage",
    ]
    for item in toc_items:
        doc.add_paragraph(item, style="List Number")
    doc.add_page_break()

    # === 1. PREREQUIS ===
    doc.add_heading("1. Prerequis", level=1)
    add_table(doc,
        ["Composant", "Version", "Notes"],
        [
            ["Python", "3.10+", "Requis"],
            ["Git", "2.x+", "Pour le versionnement"],
            ["pip", "Derniere", "Gestionnaire de packages"],
            ["SQLite", "Integre", "Aucune installation (inclus dans Python)"],
            ["PyTorch", "2.1+", "Optionnel (LSTM uniquement)"],
        ]
    )

    doc.add_page_break()

    # === 2. INSTALLATION ===
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
        "python -m venv venv\n"
        "venv\\Scripts\\activate",
        style="No Spacing"
    )
    doc.add_paragraph("Linux / Mac :", style="List Bullet")
    doc.add_paragraph(
        "python -m venv venv\n"
        "source venv/bin/activate",
        style="No Spacing"
    )

    doc.add_heading("2.3 Dependances", level=2)
    doc.add_paragraph(
        "# Core (requis)\n"
        "pip install -r requirements.txt\n\n"
        "# Deep Learning (optionnel)\n"
        "pip install -r requirements-dl.txt",
        style="No Spacing"
    )

    doc.add_heading("2.4 Configuration", level=2)
    doc.add_paragraph("Windows :", style="List Bullet")
    doc.add_paragraph("copy .env.example .env", style="No Spacing")
    doc.add_paragraph("Linux / Mac :", style="List Bullet")
    doc.add_paragraph("cp .env.example .env", style="No Spacing")
    doc.add_paragraph(
        "Le fichier .env contient les parametres du projet. "
        "Les valeurs par defaut fonctionnent pour un setup local standard."
    )

    doc.add_heading("2.5 Initialisation rapide", level=2)
    doc.add_paragraph(
        "python setup_project.py",
        style="No Spacing"
    )
    doc.add_paragraph(
        "Ce script cree les repertoires data/, initialise la BDD SQLite "
        "et verifie que les dependances sont installees."
    )

    doc.add_page_break()

    # === 3. CONFIGURATION ===
    doc.add_heading("3. Configuration", level=1)

    doc.add_heading("3.1 Fichier .env", level=2)
    doc.add_paragraph(
        "Le fichier .env est lu au demarrage du pipeline via python-dotenv. "
        "Il n'est pas versionne dans Git (secrets). "
        "Voir .env.example pour le template complet."
    )

    doc.add_heading("3.2 Base de donnees", level=2)
    doc.add_paragraph(
        "Par defaut, le projet utilise SQLite (aucune installation). "
        "Pour basculer vers SQL Server ou PostgreSQL, modifier DB_TYPE "
        "dans le fichier .env :"
    )
    doc.add_paragraph(
        "# SQLite (defaut)\n"
        "DB_TYPE=sqlite\n"
        "DB_PATH=data/hvac_market.db\n\n"
        "# SQL Server\n"
        "DB_TYPE=mssql\n"
        "DB_HOST=localhost\n"
        "DB_PORT=1433\n"
        "ALLOW_NON_LOCAL=true\n\n"
        "# PostgreSQL\n"
        "DB_TYPE=postgresql\n"
        "DB_HOST=localhost\n"
        "DB_PORT=5432\n"
        "ALLOW_NON_LOCAL=true",
        style="No Spacing"
    )

    doc.add_heading("3.3 Periode de collecte", level=2)
    doc.add_paragraph(
        "Les dates de debut/fin de collecte sont configurables :\n\n"
        "DATA_START_DATE=2019-01-01\n"
        "DATA_END_DATE=2026-02-28",
        style="No Spacing"
    )

    doc.add_page_break()

    # === 4. COMMANDES ===
    doc.add_heading("4. Commandes du pipeline", level=1)

    doc.add_paragraph(
        "Toutes les commandes s'executent depuis la racine du projet :"
    )

    doc.add_heading("4.1 Syntaxe", level=2)
    doc.add_paragraph(
        "python -m src.pipeline <commande> [options]",
        style="No Spacing"
    )

    doc.add_heading("4.2 Commandes disponibles", level=2)
    add_table(doc,
        ["Commande", "Description", "Duree estimee"],
        [
            ["list", "Lister les collecteurs disponibles", "< 1s"],
            ["init_db", "Initialiser la BDD (tables, index, donnees de reference)", "< 5s"],
            ["collect", "Collecter toutes les sources", "30-60 min (DPE)"],
            ["collect --sources weather,insee", "Collecter des sources specifiques", "< 5 min"],
            ["import_data", "Importer les CSV collectes dans la BDD", "5-10 min"],
            ["clean", "Nettoyer les donnees brutes", "1-3 min"],
            ["merge", "Fusionner en dataset ML", "< 30s"],
            ["features", "Feature engineering", "< 10s"],
            ["process", "clean + merge + features (raccourci)", "2-5 min"],
            ["eda", "Analyse exploratoire + correlations", "1-2 min"],
            ["train", "Entrainer les modeles ML", "2-5 min"],
            ["train --target nb_dpe_total", "Entrainer avec une autre cible", "2-5 min"],
            ["evaluate", "Evaluer et comparer les modeles", "3-5 min"],
            ["all", "Pipeline complet de bout en bout", "40-70 min"],
        ]
    )

    doc.add_heading("4.3 Options globales", level=2)
    add_table(doc,
        ["Option", "Valeurs", "Description"],
        [
            ["--log-level", "DEBUG, INFO, WARNING, ERROR", "Niveau de verbosity (defaut: INFO)"],
            ["--sources", "weather,insee,eurostat,dpe", "Sources a collecter (virgule)"],
            ["--target", "nb_installations_pac, nb_dpe_total, ...", "Variable cible ML"],
        ]
    )

    doc.add_heading("4.4 Exemples d'utilisation", level=2)

    doc.add_paragraph("Pipeline complet :", style="List Bullet")
    doc.add_paragraph("python -m src.pipeline all", style="No Spacing")

    doc.add_paragraph("")
    doc.add_paragraph("Collecte meteo + INSEE uniquement :", style="List Bullet")
    doc.add_paragraph("python -m src.pipeline collect --sources weather,insee", style="No Spacing")

    doc.add_paragraph("")
    doc.add_paragraph("Retraiter les donnees sans recollecte :", style="List Bullet")
    doc.add_paragraph("python -m src.pipeline process", style="No Spacing")

    doc.add_paragraph("")
    doc.add_paragraph("Entrainer un modele avec logs detailles :", style="List Bullet")
    doc.add_paragraph("python -m src.pipeline train --log-level DEBUG", style="No Spacing")

    doc.add_page_break()

    # === 5. WORKFLOW ===
    doc.add_heading("5. Workflow quotidien", level=1)

    doc.add_heading("5.1 Premiere utilisation", level=2)
    steps = [
        "Cloner le depot et installer l'environnement (Section 2)",
        "Copier .env.example en .env",
        "Lancer : python -m src.pipeline all",
        "Attendre 30-60 minutes (la collecte DPE est longue)",
        "Les resultats sont dans data/models/ et data/analysis/",
    ]
    for i, s in enumerate(steps, 1):
        doc.add_paragraph(f"{i}. {s}")

    doc.add_heading("5.2 Mise a jour des donnees", level=2)
    doc.add_paragraph(
        "Pour mettre a jour les donnees avec les derniers mois disponibles :"
    )
    steps = [
        "Modifier DATA_END_DATE dans .env si necessaire",
        "python -m src.pipeline collect    # Re-collecter",
        "python -m src.pipeline process    # Retraiter",
        "python -m src.pipeline train      # Re-entrainer",
    ]
    for i, s in enumerate(steps, 1):
        doc.add_paragraph(f"{i}. {s}")

    doc.add_heading("5.3 Changer de variable cible", level=2)
    doc.add_paragraph(
        "Le projet peut predire differentes variables :"
    )
    add_table(doc,
        ["Cible", "Description"],
        [
            ["nb_installations_pac", "Nombre de pompes a chaleur (defaut)"],
            ["nb_installations_clim", "Nombre de climatisations"],
            ["nb_dpe_total", "Volume total de DPE"],
            ["nb_dpe_classe_ab", "Nombre de batiments performants (A ou B)"],
        ]
    )
    doc.add_paragraph(
        "python -m src.pipeline train --target nb_dpe_total",
        style="No Spacing"
    )

    doc.add_page_break()

    # === 6. MULTI-MACHINES ===
    doc.add_heading("6. Travailler sur plusieurs machines", level=1)

    doc.add_heading("6.1 Ce qui est dans Git vs ce qui ne l'est pas", level=2)
    add_table(doc,
        ["Dans Git", "Pas dans Git (.gitignore)"],
        [
            ["Code source (src/, config/)", "Donnees brutes (data/raw/) ~600 Mo"],
            ["Configuration (.env.example)", "Donnees traitees (data/processed/)"],
            ["Schemas SQL", "Features ML (data/features/)"],
            ["Tests (tests/)", "Base SQLite (data/*.db) ~316 Mo"],
            ["Notebooks (structure)", "Fichier .env (secrets)"],
            ["Documentation (docs/)", "Modeles entraines (*.pkl, *.pt)"],
        ]
    )

    doc.add_heading("6.2 Option A — Telecharger les donnees (rapide)", level=2)
    doc.add_paragraph(
        "Les donnees pre-collectees (~1.5 Go) sont disponibles sur pCloud. "
        "Telecharger et extraire dans le repertoire data/ du projet."
    )

    doc.add_heading("6.3 Option B — Regenerer les donnees", level=2)
    doc.add_paragraph(
        "git clone https://github.com/PDUCLOS/Projet-HVAC.git\n"
        "cd Projet-HVAC\n"
        "python -m venv venv && source venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "cp .env.example .env\n"
        "python -m src.pipeline all",
        style="No Spacing"
    )

    doc.add_heading("6.4 Synchronisation", level=2)
    doc.add_paragraph("Machine A (apres modifications) :")
    doc.add_paragraph(
        "git add -A && git commit -m \"description\" && git push",
        style="No Spacing"
    )
    doc.add_paragraph("Machine B (recuperer) :")
    doc.add_paragraph(
        "git pull\n"
        "python -m src.pipeline process  # Si les donnees doivent etre retraitees",
        style="No Spacing"
    )

    doc.add_page_break()

    # === 7. TESTS ===
    doc.add_heading("7. Tests", level=1)

    doc.add_heading("7.1 Lancer tous les tests", level=2)
    doc.add_paragraph(
        "python -m pytest tests/ -v",
        style="No Spacing"
    )

    doc.add_heading("7.2 Lancer un fichier de test specifique", level=2)
    doc.add_paragraph(
        "python -m pytest tests/test_config.py -v\n"
        "python -m pytest tests/test_collectors/test_weather.py -v\n"
        "python -m pytest tests/test_processing/ -v",
        style="No Spacing"
    )

    doc.add_heading("7.3 Tests avec couverture", level=2)
    doc.add_paragraph(
        "pip install pytest-cov\n"
        "python -m pytest tests/ --cov=src --cov=config --cov-report=term-missing",
        style="No Spacing"
    )

    doc.add_page_break()

    # === 8. STRUCTURE DES DONNEES ===
    doc.add_heading("8. Structure des donnees", level=1)

    doc.add_heading("8.1 Repertoires", level=2)
    add_table(doc,
        ["Repertoire", "Contenu"],
        [
            ["data/raw/weather/", "weather_aura.csv — donnees meteo quotidiennes"],
            ["data/raw/insee/", "indicateurs_economiques.csv — series INSEE"],
            ["data/raw/eurostat/", "ipi_hvac_france.csv — indices de production"],
            ["data/raw/dpe/", "dpe_aura_all.csv — DPE bruts (~300 Mo)"],
            ["data/processed/", "Donnees nettoyees (meme structure que raw/)"],
            ["data/features/", "hvac_ml_dataset.csv et hvac_features_dataset.csv"],
            ["data/models/", "Modeles .pkl, metriques, rapports"],
            ["data/models/figures/", "Graphiques de visualisation"],
            ["data/analysis/", "Resultats EDA et correlations"],
            ["data/analysis/figures/", "Graphiques exploratoires"],
        ]
    )

    doc.add_heading("8.2 Fichiers de sortie cles", level=2)
    add_table(doc,
        ["Fichier", "Contenu", "Taille typique"],
        [
            ["hvac_ml_dataset.csv", "Dataset fusionne (35 colonnes)", "~50 Ko"],
            ["hvac_features_dataset.csv", "Dataset avec features (90 colonnes)", "~150 Ko"],
            ["training_results.csv", "Resume des metriques de tous les modeles", "< 5 Ko"],
            ["evaluation_report.txt", "Rapport textuel complet", "< 10 Ko"],
            ["ridge_model.pkl", "Modele Ridge entraine (pickle)", "< 1 Mo"],
            ["lightgbm_model.pkl", "Modele LightGBM entraine (pickle)", "< 5 Mo"],
            ["hvac_market.db", "Base SQLite complete", "~316 Mo"],
        ]
    )

    doc.add_page_break()

    # === 9. DEPANNAGE ===
    doc.add_heading("9. Depannage", level=1)

    doc.add_heading("9.1 Erreurs courantes", level=2)

    problems = [
        (
            "ModuleNotFoundError: No module named 'xxx'",
            "Installer le package manquant : pip install xxx\n"
            "Ou reinstaller toutes les dependances : pip install -r requirements.txt"
        ),
        (
            "La collecte DPE echoue ou est tres lente",
            "L'API ADEME peut etre temporairement surchargee. "
            "Relancer la commande — le collecteur est idempotent. "
            "Verifier la connexion internet."
        ),
        (
            "FileNotFoundError: Dataset features introuvable",
            "Les etapes doivent etre executees dans l'ordre : "
            "collect -> process -> train. "
            "Lancer d'abord : python -m src.pipeline process"
        ),
        (
            "ValueError: Base non locale desactivee",
            "Pour utiliser SQL Server ou PostgreSQL, ajouter "
            "ALLOW_NON_LOCAL=true dans le fichier .env"
        ),
        (
            "Les tests echouent",
            "Verifier que pytest est installe : pip install pytest\n"
            "Verifier les dependances : pip install -r requirements.txt"
        ),
        (
            "Ridge R2 tres eleve (>0.99)",
            "C'est normal avec les features lag — le modele fait de "
            "l'auto-regression. Voir le modele 'ridge_exogenes' pour "
            "l'evaluation sans auto-correlation."
        ),
    ]

    for problem, solution in problems:
        doc.add_heading(problem, level=3)
        doc.add_paragraph(solution)

    doc.add_heading("9.2 Logs", level=2)
    doc.add_paragraph(
        "Pour augmenter le niveau de detail des logs :"
    )
    doc.add_paragraph(
        "python -m src.pipeline <commande> --log-level DEBUG",
        style="No Spacing"
    )
    doc.add_paragraph(
        "Les niveaux disponibles sont : DEBUG, INFO, WARNING, ERROR. "
        "En mode DEBUG, tous les appels HTTP et les details de parsing "
        "sont affiches."
    )

    doc.add_heading("9.3 Contact", level=2)
    doc.add_paragraph(
        "Auteur : Patrice DUCLOS — Data Analyst Senior\n"
        "Depot GitHub : https://github.com/PDUCLOS/Projet-HVAC"
    )

    # Sauvegarder
    path = "docs/guide_utilisation.docx"
    doc.save(path)
    print(f"Guide d'utilisation genere : {path}")
    return path


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    generate_technical_doc()
    generate_user_guide()
    print("\nDocumentation generee avec succes dans docs/")
