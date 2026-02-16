# -*- coding: utf-8 -*-
"""
Utilitaires pour le dashboard Streamlit.
Chargement des donnees, styles CSS, fonctions partagees.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Chemins
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"


# =============================================================================
# Chargement des donnees (cache Streamlit)
# =============================================================================

@st.cache_data(ttl=3600)
def load_ml_dataset() -> pd.DataFrame:
    """Charge le dataset ML-ready (34 colonnes, grain mois x dept)."""
    path = FEATURES_DIR / "hvac_ml_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["date_id"].astype(str).str[:4] + "-"
        + df["date_id"].astype(str).str[4:] + "-01"
    )
    df["dept"] = df["dept"].astype(str).str.zfill(2)
    return df


@st.cache_data(ttl=3600)
def load_features_dataset() -> pd.DataFrame:
    """Charge le dataset features (90 colonnes)."""
    path = FEATURES_DIR / "hvac_features_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(
        df["date_id"].astype(str).str[:4] + "-"
        + df["date_id"].astype(str).str[4:] + "-01"
    )
    df["dept"] = df["dept"].astype(str).str.zfill(2)
    return df


@st.cache_data(ttl=3600)
def load_training_results() -> pd.DataFrame:
    """Charge les resultats d'entrainement."""
    path = MODELS_DIR / "training_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model(name: str) -> Optional[Any]:
    """Charge un modele serialise depuis data/models/."""
    path = MODELS_DIR / f"{name}_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_scaler_and_imputer() -> Tuple[Optional[Any], Optional[Any]]:
    """Charge le scaler et l'imputer."""
    scaler_path = MODELS_DIR / "scaler.pkl"
    imputer_path = MODELS_DIR / "imputer.pkl"
    scaler = None
    imputer = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    if imputer_path.exists():
        with open(imputer_path, "rb") as f:
            imputer = pickle.load(f)
    return scaler, imputer


# =============================================================================
# Constantes
# =============================================================================

DEPARTMENTS = {
    "01": "Ain",
    "07": "Ardeche",
    "26": "Drome",
    "38": "Isere",
    "42": "Loire",
    "69": "Rhone",
    "73": "Savoie",
    "74": "Haute-Savoie",
}

TARGETS = {
    "nb_installations_pac": "Installations PAC",
    "nb_installations_clim": "Installations Climatisation",
    "nb_dpe_total": "Volume DPE Total",
    "nb_dpe_classe_ab": "DPE Classe A-B",
}

MOIS_FR = [
    "Jan", "Fev", "Mar", "Avr", "Mai", "Jun",
    "Jul", "Aou", "Sep", "Oct", "Nov", "Dec",
]


# =============================================================================
# Styles CSS
# =============================================================================

def inject_custom_css() -> None:
    """Injecte le CSS personnalise dans la page Streamlit."""
    st.markdown("""
    <style>
    /* ---- KPI Cards ---- */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-card h3 {
        font-size: 14px;
        font-weight: 400;
        margin: 0;
        opacity: 0.9;
    }
    .kpi-card h1 {
        font-size: 32px;
        font-weight: 700;
        margin: 8px 0 4px 0;
    }
    .kpi-card p {
        font-size: 12px;
        margin: 0;
        opacity: 0.8;
    }

    .kpi-blue { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .kpi-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .kpi-teal { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }

    /* ---- Section headers ---- */
    .section-header {
        border-left: 4px solid #667eea;
        padding-left: 12px;
        margin: 30px 0 15px 0;
    }

    /* ---- Model comparison table ---- */
    .best-model {
        background-color: #d4edda !important;
        font-weight: bold;
    }

    /* ---- Footer ---- */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 12px;
        border-top: 1px solid #eee;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(title: str, value: str, subtitle: str = "", css_class: str = "kpi-blue") -> str:
    """Genere le HTML d'une carte KPI."""
    return f"""
    <div class="kpi-card {css_class}">
        <h3>{title}</h3>
        <h1>{value}</h1>
        <p>{subtitle}</p>
    </div>
    """


def section_header(title: str) -> None:
    """Affiche un titre de section stylise."""
    st.markdown(f'<div class="section-header"><h2>{title}</h2></div>', unsafe_allow_html=True)
