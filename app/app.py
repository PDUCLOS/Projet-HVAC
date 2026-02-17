# -*- coding: utf-8 -*-
"""
HVAC Market Analysis — Dashboard Streamlit.
============================================

Point d'entree de l'interface web.
Lancer avec : streamlit run app/app.py
"""

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# --- Configuration de la page ---
st.set_page_config(
    page_title="HVAC Market Analysis - France",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar : Navigation ---
st.sidebar.title("HVAC Market Analysis")
st.sidebar.caption("Analyse du marche HVAC en France")

page = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Exploration des donnees",
        "Carte de France",
        "Predictions ML",
        "Comparaison des modeles",
        "Pipeline & mise a jour",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Donnees** : 96 departements\n\n"
    "**Sources** : Open-Meteo, ADEME DPE, INSEE, Eurostat, SITADEL\n\n"
    "**ML** : LightGBM, XGBoost, Ridge, Prophet, LSTM"
)

# --- Dispatch des pages ---
if page == "Accueil":
    from pages import home
    home.render()
elif page == "Exploration des donnees":
    from pages import exploration
    exploration.render()
elif page == "Carte de France":
    from pages import carte
    carte.render()
elif page == "Predictions ML":
    from pages import predictions
    predictions.render()
elif page == "Comparaison des modeles":
    from pages import models
    models.render()
elif page == "Pipeline & mise a jour":
    from pages import pipeline_page
    pipeline_page.render()
