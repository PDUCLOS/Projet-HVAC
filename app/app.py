# -*- coding: utf-8 -*-
"""
HVAC Market Analysis — Dashboard Streamlit.
=============================================

Point d'entree du dashboard interactif.

Lancement :
    streamlit run app/app.py

Structure :
    app/
    ├── app.py                  # Page principale (Accueil)
    └── pages/
        ├── 1_Donnees.py        # Exploration des donnees
        ├── 2_EDA.py            # Visualisations exploratoires
        ├── 3_Modelisation.py   # Resultats ML, comparaison
        └── 4_Predictions.py    # Predictions interactives
"""

import sys
from pathlib import Path

# Ajouter le projet au path pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.helpers import (
    DEPARTMENTS,
    MOIS_FR,
    TARGETS,
    inject_custom_css,
    kpi_card,
    load_ml_dataset,
    load_training_results,
    section_header,
)

# =============================================================================
# Configuration de la page
# =============================================================================
st.set_page_config(
    page_title="HVAC Market Analysis — AURA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/air-conditioner.png", width=60)
    st.title("HVAC Market")
    st.caption("Analyse du marche HVAC\nRegion Auvergne-Rhone-Alpes")
    st.divider()
    st.markdown("**Navigation**")
    st.page_link("app/app.py", label="Accueil", icon="🏠")
    st.page_link("app/pages/1_Donnees.py", label="Donnees", icon="📊")
    st.page_link("app/pages/2_EDA.py", label="Analyse Exploratoire", icon="🔍")
    st.page_link("app/pages/3_Modelisation.py", label="Modelisation ML", icon="🤖")
    st.page_link("app/pages/4_Predictions.py", label="Predictions", icon="📈")
    st.divider()
    st.markdown(
        "<div style='text-align:center; font-size:11px; color:#888;'>"
        "Patrice Duclos<br>Data Analyst Senior<br>"
        "<a href='https://github.com/PDUCLOS/Projet-HVAC'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Page Accueil
# =============================================================================

# Header
st.markdown(
    """
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="font-size:2.5rem; margin-bottom:5px;">
            Analyse du Marche HVAC
        </h1>
        <p style="font-size:1.2rem; color:#666;">
            Region Auvergne-Rhone-Alpes — Prediction des installations PAC & Climatisation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# Charger les donnees
df = load_ml_dataset()
df_results = load_training_results()

if df.empty:
    st.warning(
        "Pas de donnees disponibles. Executez le pipeline d'abord :\n\n"
        "```bash\npython -m src.pipeline process\npython -m src.pipeline train\n```"
    )
    st.stop()


# --- KPIs principaux ---
section_header("Chiffres cles")

col1, col2, col3, col4 = st.columns(4)

total_dpe = int(df["nb_dpe_total"].sum())
total_pac = int(df["nb_installations_pac"].sum())
total_clim = int(df["nb_installations_clim"].sum())
taux_pac = round(100 * total_pac / max(1, total_dpe), 1)

with col1:
    st.markdown(kpi_card("Volume DPE Total", f"{total_dpe:,}".replace(",", " "),
                         "Diagnostics sur la periode", "kpi-blue"), unsafe_allow_html=True)
with col2:
    st.markdown(kpi_card("Installations PAC", f"{total_pac:,}".replace(",", " "),
                         "Pompes a chaleur detectees", "kpi-green"), unsafe_allow_html=True)
with col3:
    st.markdown(kpi_card("Installations Clim", f"{total_clim:,}".replace(",", " "),
                         "Climatisations detectees", "kpi-orange"), unsafe_allow_html=True)
with col4:
    st.markdown(kpi_card("Taux PAC", f"{taux_pac}%",
                         "Part des DPE avec PAC", "kpi-teal"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Evolution temporelle ---
section_header("Evolution mensuelle")

# Agreger au niveau AURA
df_aura = df.groupby("date").agg({
    "nb_dpe_total": "sum",
    "nb_installations_pac": "sum",
    "nb_installations_clim": "sum",
    "temp_mean": "mean",
}).reset_index()

metric_choice = st.selectbox(
    "Indicateur :",
    ["nb_installations_pac", "nb_dpe_total", "nb_installations_clim"],
    format_func=lambda x: TARGETS.get(x, x),
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_aura["date"], y=df_aura[metric_choice],
    mode="lines+markers", name=TARGETS.get(metric_choice, metric_choice),
    line=dict(color="#667eea", width=2.5),
    marker=dict(size=5),
    fill="tozeroy", fillcolor="rgba(102,126,234,0.1)",
))

# Temperature en axe secondaire
fig.add_trace(go.Scatter(
    x=df_aura["date"], y=df_aura["temp_mean"],
    mode="lines", name="Temperature moy. (°C)",
    line=dict(color="#f5576c", width=1.5, dash="dot"),
    yaxis="y2", opacity=0.7,
))

fig.update_layout(
    yaxis=dict(title=TARGETS.get(metric_choice, metric_choice)),
    yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
    hovermode="x unified",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", y=-0.15),
)
st.plotly_chart(fig, use_container_width=True)


# --- Resultats ML (resume) ---
if not df_results.empty:
    section_header("Performances des modeles")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Barres RMSE val vs test
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Validation RMSE",
            x=df_results["model"],
            y=df_results["val_rmse"],
            marker_color="#667eea",
            text=df_results["val_rmse"].round(2),
            textposition="auto",
        ))
        fig_comp.add_trace(go.Bar(
            name="Test RMSE",
            x=df_results["model"],
            y=df_results["test_rmse"],
            marker_color="#f5576c",
            text=df_results["test_rmse"].round(2),
            textposition="auto",
        ))
        fig_comp.update_layout(
            barmode="group", height=350,
            yaxis_title="RMSE",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        # Meilleur modele
        best = df_results.loc[df_results["val_rmse"].idxmin()]
        st.markdown(
            f"""
            <div class="kpi-card kpi-green" style="margin-top:20px;">
                <h3>Meilleur modele</h3>
                <h1>{best['model'].upper()}</h1>
                <p>Val RMSE = {best['val_rmse']:.2f}<br>
                Test RMSE = {best['test_rmse']:.2f}<br>
                R² = {best.get('val_r2', 0):.3f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Carte departements ---
section_header("Repartition par departement")

dept_stats = df.groupby("dept").agg({
    "nb_dpe_total": "sum",
    "nb_installations_pac": "sum",
}).reset_index()
dept_stats["dept_name"] = dept_stats["dept"].map(DEPARTMENTS)
dept_stats["pct_pac"] = (100 * dept_stats["nb_installations_pac"] / dept_stats["nb_dpe_total"].clip(lower=1)).round(1)

fig_dept = px.bar(
    dept_stats.sort_values("nb_installations_pac", ascending=True),
    x="nb_installations_pac", y="dept_name",
    orientation="h",
    color="pct_pac",
    color_continuous_scale="Viridis",
    labels={
        "nb_installations_pac": "Installations PAC (total)",
        "dept_name": "Departement",
        "pct_pac": "Taux PAC (%)",
    },
    text="nb_installations_pac",
)
fig_dept.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_dept, use_container_width=True)


# --- Pipeline ---
section_header("Architecture du pipeline")

st.markdown(
    """
    ```
    Sources externes              Pipeline                    Resultats
    ──────────────              ────────                    ─────────
    Open-Meteo (meteo)    ──┐
    INSEE BDM (economie)  ──┤   Collecte → Nettoyage       data/raw/
    Eurostat (IPI HVAC)   ──┼──→ → Fusion → Features ──→   data/features/
    SITADEL (permis)      ──┤        448 × 90 colonnes      hvac_features_dataset.csv
    ADEME DPE (1.4M)      ──┘                                      │
                                                                   ▼
                                                        Modelisation ML
                                                        Ridge | LightGBM
                                                        Prophet | LSTM
                                                                   │
                                                                   ▼
                                                         Dashboard
                                                         (cette page)
    ```
    """,
)


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "Projet HVAC Market Analysis — Patrice Duclos — 2025"
    "</div>",
    unsafe_allow_html=True,
)
