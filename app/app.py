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
PAGE_MODULES = {
    "Accueil": "home",
    "Exploration des donnees": "exploration",
    "Carte de France": "carte",
    "Predictions ML": "predictions",
    "Comparaison des modeles": "models",
    "Pipeline & mise a jour": "pipeline_page",
}

module_name = PAGE_MODULES.get(page, "home")
try:
    module = __import__(f"pages.{module_name}", fromlist=[module_name])
    module.render()
except Exception as e:
    st.error(f"Erreur lors du chargement de la page '{page}' : {e}")
    st.info("Verifiez que toutes les dependances sont installees.")
