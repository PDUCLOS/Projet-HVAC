# -*- coding: utf-8 -*-
"""
Page Donnees — Exploration du dataset et sources.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
    load_features_dataset,
    load_ml_dataset,
    section_header,
)

# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(
    page_title="Donnees — HVAC Market",
    page_icon="📊",
    layout="wide",
)
inject_custom_css()

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1>📊 Exploration des Donnees</h1>
        <p style="color:#666;">Sources, structure et qualite du dataset</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# =============================================================================
# Chargement
# =============================================================================
df_ml = load_ml_dataset()
df_features = load_features_dataset()

if df_ml.empty:
    st.warning("Aucune donnee disponible. Lancez le pipeline d'abord.")
    st.stop()


# =============================================================================
# 1. Sources de donnees
# =============================================================================
section_header("Sources de donnees")

sources = [
    {
        "Source": "ADEME — DPE",
        "Description": "Diagnostics de Performance Energetique (1.4M enregistrements individuels)",
        "Grain": "Individuel → agrege mois × departement",
        "Variables cles": "nb_dpe_total, nb_installations_pac, nb_installations_clim, nb_dpe_classe_ab",
    },
    {
        "Source": "Open-Meteo",
        "Description": "Donnees meteorologiques historiques",
        "Grain": "Journalier → agrege mensuel",
        "Variables cles": "temp_mean, temp_max, temp_min, hdd_sum, cdd_sum, precipitation_sum",
    },
    {
        "Source": "INSEE BDM",
        "Description": "Indicateurs economiques (confiance menages, climat affaires)",
        "Grain": "Mensuel national",
        "Variables cles": "confiance_menages, climat_affaires_indus, climat_affaires_bat",
    },
    {
        "Source": "Eurostat",
        "Description": "Indice de Production Industrielle HVAC (C28)",
        "Grain": "Mensuel europeen",
        "Variables cles": "ipi_manufacturing, ipi_hvac_c28",
    },
    {
        "Source": "SITADEL",
        "Description": "Permis de construire et mises en chantier",
        "Grain": "Mensuel × departement",
        "Variables cles": "permis_construire, mises_en_chantier (via features)",
    },
]

st.dataframe(
    pd.DataFrame(sources),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Source": st.column_config.TextColumn(width="medium"),
        "Description": st.column_config.TextColumn(width="large"),
    },
)


# =============================================================================
# 2. Apercu du dataset
# =============================================================================
section_header("Apercu du dataset ML")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        kpi_card("Observations", str(len(df_ml)), "Lignes dans le dataset ML", "kpi-blue"),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        kpi_card("Variables", str(len(df_ml.columns)), "Colonnes disponibles", "kpi-green"),
        unsafe_allow_html=True,
    )
with col3:
    n_dept = df_ml["dept"].nunique()
    st.markdown(
        kpi_card("Departements", str(n_dept), "Region AURA", "kpi-orange"),
        unsafe_allow_html=True,
    )
with col4:
    date_min = df_ml["date"].min().strftime("%Y-%m")
    date_max = df_ml["date"].max().strftime("%Y-%m")
    st.markdown(
        kpi_card("Periode", f"{date_min} → {date_max}", "Couverture temporelle", "kpi-teal"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs pour basculer entre les datasets
tab_ml, tab_features = st.tabs(["Dataset ML (compact)", "Dataset Features (complet)"])

with tab_ml:
    st.markdown(f"**{len(df_ml)} lignes × {len(df_ml.columns)} colonnes** — Grain : mois × departement")

    # Filtres
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        dept_filter = st.multiselect(
            "Departement(s) :",
            options=sorted(df_ml["dept"].unique()),
            format_func=lambda x: f"{x} — {DEPARTMENTS.get(x, x)}",
            key="ml_dept",
        )
    with col_f2:
        date_range = st.date_input(
            "Periode :",
            value=(df_ml["date"].min(), df_ml["date"].max()),
            key="ml_date",
        )

    df_show = df_ml.copy()
    if dept_filter:
        df_show = df_show[df_show["dept"].isin(dept_filter)]
    if len(date_range) == 2:
        df_show = df_show[
            (df_show["date"] >= pd.Timestamp(date_range[0]))
            & (df_show["date"] <= pd.Timestamp(date_range[1]))
        ]

    st.dataframe(df_show.drop(columns=["date"], errors="ignore"), use_container_width=True, height=400)

with tab_features:
    if df_features.empty:
        st.info("Le dataset features n'est pas disponible.")
    else:
        st.markdown(
            f"**{len(df_features)} lignes × {len(df_features.columns)} colonnes** — "
            "Dataset complet apres feature engineering"
        )
        st.dataframe(df_features.drop(columns=["date"], errors="ignore"), use_container_width=True, height=400)


# =============================================================================
# 3. Qualite des donnees
# =============================================================================
section_header("Qualite des donnees")

col_q1, col_q2 = st.columns(2)

with col_q1:
    st.markdown("**Valeurs manquantes (dataset ML)**")
    missing = df_ml.drop(columns=["date"], errors="ignore").isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("Aucune valeur manquante dans le dataset ML.")
    else:
        fig_miss = px.bar(
            x=missing.values,
            y=missing.index,
            orientation="h",
            labels={"x": "Nb valeurs manquantes", "y": "Variable"},
            color=missing.values,
            color_continuous_scale="Reds",
        )
        fig_miss.update_layout(
            height=max(200, len(missing) * 25),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_miss, use_container_width=True)

with col_q2:
    st.markdown("**Types de variables**")
    dtypes = df_ml.dtypes.value_counts().reset_index()
    dtypes.columns = ["Type", "Nombre"]
    dtypes["Type"] = dtypes["Type"].astype(str)
    fig_types = px.pie(
        dtypes,
        values="Nombre",
        names="Type",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_types.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_types, use_container_width=True)


# =============================================================================
# 4. Statistiques descriptives
# =============================================================================
section_header("Statistiques descriptives")

numeric_cols = df_ml.select_dtypes(include=[np.number]).columns.tolist()
cols_to_describe = st.multiselect(
    "Variables a decrire :",
    options=numeric_cols,
    default=[c for c in ["nb_dpe_total", "nb_installations_pac", "temp_mean", "confiance_menages"] if c in numeric_cols],
    key="describe_cols",
)

if cols_to_describe:
    desc = df_ml[cols_to_describe].describe().T
    desc = desc.round(2)
    st.dataframe(desc, use_container_width=True)


# =============================================================================
# 5. Couverture temporelle par departement
# =============================================================================
section_header("Couverture temporelle")

coverage = df_ml.groupby("dept").agg(
    debut=("date", "min"),
    fin=("date", "max"),
    nb_mois=("date", "count"),
).reset_index()
coverage["dept_name"] = coverage["dept"].map(DEPARTMENTS)
coverage["debut"] = coverage["debut"].dt.strftime("%Y-%m")
coverage["fin"] = coverage["fin"].dt.strftime("%Y-%m")

fig_cov = px.bar(
    coverage,
    x="nb_mois",
    y="dept_name",
    orientation="h",
    text="nb_mois",
    color="nb_mois",
    color_continuous_scale="Blues",
    labels={"nb_mois": "Nombre de mois", "dept_name": "Departement"},
)
fig_cov.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_cov, use_container_width=True)


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "Projet HVAC Market Analysis — Patrice Duclos — 2025"
    "</div>",
    unsafe_allow_html=True,
)
