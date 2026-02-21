# -*- coding: utf-8 -*-
"""HVAC dashboard home page."""

import streamlit as st
from pathlib import Path
import pandas as pd


def render():
    st.title("Analyse du Marche HVAC en France")
    st.markdown(
        """
        Ce dashboard presente l'analyse predictive du marche des equipements
        HVAC (chauffage, ventilation, climatisation) en France metropolitaine,
        a travers l'exploitation de donnees publiques Open Data.
        """
    )

    # --- Key metrics ---
    st.header("Vue d'ensemble")

    col1, col2, col3, col4 = st.columns(4)

    # Count available data
    raw_dir = Path("data/raw")
    features_path = Path("data/features/hvac_features_dataset.csv")

    n_weather = _count_rows(raw_dir / "weather" / "weather_france.csv")
    n_dpe = _count_rows(raw_dir / "dpe" / "dpe_france_all.csv")
    n_sitadel = _count_rows(raw_dir / "sitadel" / "permis_construire_france.csv")
    n_features = _count_rows(features_path)

    col1.metric("Donnees meteo", f"{n_weather:,}" if n_weather else "Non collecte")
    col2.metric("DPE ADEME", f"{n_dpe:,}" if n_dpe else "Non collecte")
    col3.metric("Permis SITADEL", f"{n_sitadel:,}" if n_sitadel else "Non collecte")
    col4.metric("Dataset ML", f"{n_features:,}" if n_features else "Non genere")

    # --- Pipeline architecture ---
    st.header("Architecture du pipeline")
    st.markdown(
        """
        ```
        1. COLLECTE         Open-Meteo, ADEME DPE, INSEE, Eurostat, SITADEL
              |
        2. NETTOYAGE        Deduplication, outliers, types, valeurs manquantes
              |
        3. FUSION           Jointure mois x departement (96 depts x N mois)
              |
        4. FEATURES         Lags, rolling, interactions, tendances
              |
        5. OUTLIERS         IQR + Z-score + Isolation Forest (consensus 2/3)
              |
        6. MODELISATION     LightGBM, XGBoost, Ridge, Prophet, LSTM
              |
        7. EVALUATION       MAE, RMSE, MAPE, R2, Feature Importance (SHAP)
              |
        8. PCLOUD           Upload automatique des resultats
        ```
        """
    )

    # --- Data sources ---
    st.header("Sources de donnees")
    sources = pd.DataFrame({
        "Source": ["Open-Meteo", "ADEME DPE", "INSEE BDM", "Eurostat", "SITADEL"],
        "Type": ["Meteo", "Energie", "Economie", "Industrie", "Construction"],
        "Granularite": [
            "Jour x Ville",
            "Unitaire (DPE)",
            "Mois (national)",
            "Mois (national)",
            "Mois x Departement",
        ],
        "API": [
            "open-meteo.com",
            "data.ademe.fr",
            "bdm.insee.fr (SDMX)",
            "ec.europa.eu/eurostat",
            "data.statistiques.developpement-durable.gouv.fr",
        ],
        "Authentification": ["Aucune"] * 5,
    })
    st.dataframe(sources, use_container_width=True, hide_index=True)

    # --- CLI commands ---
    st.header("Commandes CLI")
    st.code(
        """
# Mise a jour complete (collecte + pipeline + upload pCloud)
python -m src.pipeline update_all

# Collecte seule
python -m src.pipeline collect
python -m src.pipeline collect --sources weather,dpe

# Processing
python -m src.pipeline process       # clean + merge + features + outliers

# ML
python -m src.pipeline train
python -m src.pipeline evaluate

# pCloud
python -m src.pipeline sync_pcloud   # Telecharger depuis pCloud
python -m src.pipeline upload_pcloud  # Uploader vers pCloud

# Lancer le dashboard
streamlit run app/app.py
        """,
        language="bash",
    )


def _count_rows(filepath: Path) -> int:
    """Count the rows of a CSV without loading it all into memory."""
    if not filepath.exists():
        return 0
    try:
        # Count lines without reading the content
        with open(filepath) as f:
            return sum(1 for _ in f) - 1  # -1 for the header
    except Exception:
        return 0
