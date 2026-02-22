# -*- coding: utf-8 -*-
"""
HVAC Market Analysis — Streamlit Dashboard.
=============================================

Entry point for the web interface.
Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

# Add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# --- Page configuration ---
st.set_page_config(
    page_title="HVAC Market Analysis - France",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar: Navigation ---
st.sidebar.title("HVAC Market Analysis")
st.sidebar.caption("HVAC market analysis in France")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Exploration",
        "France Map",
        "ML Predictions",
        "Model Comparison",
        "Pipeline & Update",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Data** : 96 departements\n\n"
    "**Sources** : Open-Meteo, ADEME DPE, INSEE, Eurostat, SITADEL\n\n"
    "**ML** : LightGBM, XGBoost, Ridge, Prophet, LSTM"
)

# --- Page dispatch ---
PAGE_MODULES = {
    "Home": "home",
    "Data Exploration": "exploration",
    "France Map": "carte",
    "ML Predictions": "predictions",
    "Model Comparison": "models",
    "Pipeline & Update": "pipeline_page",
}

module_name = PAGE_MODULES.get(page, "home")
try:
    module = __import__(f"pages.{module_name}", fromlist=[module_name])
    module.render()
except Exception as e:
    st.error(f"Error loading page '{page}' : {e}")
    st.info("Check that all dependencies are installed.")
