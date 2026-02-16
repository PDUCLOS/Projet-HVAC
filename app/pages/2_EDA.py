# -*- coding: utf-8 -*-
"""
Page EDA — Visualisations exploratoires interactives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.utils.helpers import (
    DEPARTMENTS,
    MOIS_FR,
    TARGETS,
    inject_custom_css,
    load_ml_dataset,
    section_header,
)

# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(
    page_title="Analyse Exploratoire — HVAC Market",
    page_icon="🔍",
    layout="wide",
)
inject_custom_css()

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1>🔍 Analyse Exploratoire</h1>
        <p style="color:#666;">Tendances, saisonnalite, correlations</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# Chargement
df = load_ml_dataset()
if df.empty:
    st.warning("Aucune donnee disponible.")
    st.stop()

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["month_name"] = df["month"].map(dict(enumerate(MOIS_FR, 1)))


# =============================================================================
# 1. Evolution temporelle par departement
# =============================================================================
section_header("Evolution temporelle par departement")

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    target_choice = st.selectbox(
        "Variable cible :",
        list(TARGETS.keys()),
        format_func=lambda x: TARGETS[x],
        key="eda_target",
    )
with col_sel2:
    dept_sel = st.multiselect(
        "Departement(s) :",
        options=sorted(df["dept"].unique()),
        default=sorted(df["dept"].unique()),
        format_func=lambda x: f"{x} — {DEPARTMENTS.get(x, x)}",
        key="eda_dept",
    )

df_filt = df[df["dept"].isin(dept_sel)] if dept_sel else df

fig_ts = px.line(
    df_filt,
    x="date",
    y=target_choice,
    color="dept",
    labels={target_choice: TARGETS[target_choice], "date": "Date", "dept": "Departement"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_ts.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=-0.15),
    hovermode="x unified",
)
st.plotly_chart(fig_ts, use_container_width=True)


# =============================================================================
# 2. Saisonnalite
# =============================================================================
section_header("Profil saisonnier")

col_s1, col_s2 = st.columns(2)

with col_s1:
    # Box plot mensuel
    month_order = list(range(1, 13))
    df_filt_sorted = df_filt.copy()
    df_filt_sorted["month_name"] = pd.Categorical(
        df_filt_sorted["month_name"],
        categories=MOIS_FR,
        ordered=True,
    )
    fig_box = px.box(
        df_filt_sorted,
        x="month_name",
        y=target_choice,
        color="month_name",
        labels={target_choice: TARGETS[target_choice], "month_name": "Mois"},
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_box.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col_s2:
    # Heatmap mois x annee (moyenne AURA)
    pivot = df_filt.groupby(["year", "month"])[target_choice].mean().reset_index()
    pivot_table = pivot.pivot(index="year", columns="month", values=target_choice)
    pivot_table.columns = MOIS_FR[:pivot_table.shape[1]]

    fig_heat = px.imshow(
        pivot_table.values,
        x=pivot_table.columns.tolist(),
        y=[str(y) for y in pivot_table.index.tolist()],
        color_continuous_scale="YlOrRd",
        labels={"color": TARGETS[target_choice]},
        aspect="auto",
    )
    fig_heat.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Mois",
        yaxis_title="Annee",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# =============================================================================
# 3. Relations meteo / cible
# =============================================================================
section_header("Impact de la meteo")

meteo_vars = [c for c in ["temp_mean", "hdd_sum", "cdd_sum", "precipitation_sum", "nb_jours_canicule", "nb_jours_gel"] if c in df.columns]

col_m1, col_m2 = st.columns(2)
with col_m1:
    meteo_x = st.selectbox("Variable meteo (axe X) :", meteo_vars, key="meteo_x")
with col_m2:
    color_by = st.selectbox("Couleur :", ["dept", "year", "month"], key="color_by")

fig_scatter = px.scatter(
    df_filt,
    x=meteo_x,
    y=target_choice,
    color=color_by,
    trendline="ols",
    labels={
        target_choice: TARGETS[target_choice],
        meteo_x: meteo_x.replace("_", " ").title(),
    },
    opacity=0.7,
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_scatter.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig_scatter, use_container_width=True)


# =============================================================================
# 4. Correlations
# =============================================================================
section_header("Matrice de correlations")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclure certaines colonnes non pertinentes
exclude = {"date_id", "latitude", "longitude"}
corr_cols = [c for c in numeric_cols if c not in exclude]

# Selecteur pour filtrer
default_corr = [c for c in [
    "nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
    "temp_mean", "hdd_sum", "cdd_sum", "confiance_menages",
    "ipi_hvac_c28", "precipitation_sum",
] if c in corr_cols]

selected_corr = st.multiselect(
    "Variables pour la correlation :",
    options=corr_cols,
    default=default_corr,
    key="corr_vars",
)

if len(selected_corr) >= 2:
    corr_matrix = df[selected_corr].corr()
    fig_corr = px.imshow(
        corr_matrix.values,
        x=selected_corr,
        y=selected_corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_corr.update_layout(
        height=max(400, len(selected_corr) * 35),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Selectionnez au moins 2 variables pour afficher la matrice de correlation.")


# =============================================================================
# 5. Comparaison departementale
# =============================================================================
section_header("Comparaison departementale")

dept_agg = df.groupby("dept").agg({
    "nb_dpe_total": "sum",
    "nb_installations_pac": "sum",
    "nb_installations_clim": "sum",
    "temp_mean": "mean",
}).reset_index()
dept_agg["dept_name"] = dept_agg["dept"].map(DEPARTMENTS)
dept_agg["pct_pac"] = (100 * dept_agg["nb_installations_pac"] / dept_agg["nb_dpe_total"].clip(lower=1)).round(1)
dept_agg["pct_clim"] = (100 * dept_agg["nb_installations_clim"] / dept_agg["nb_dpe_total"].clip(lower=1)).round(1)

col_d1, col_d2 = st.columns(2)

with col_d1:
    fig_dept = px.bar(
        dept_agg.sort_values("nb_installations_pac", ascending=True),
        x="nb_installations_pac",
        y="dept_name",
        orientation="h",
        color="pct_pac",
        color_continuous_scale="Viridis",
        text="nb_installations_pac",
        labels={
            "nb_installations_pac": "Installations PAC",
            "dept_name": "Departement",
            "pct_pac": "Taux PAC (%)",
        },
    )
    fig_dept.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_dept, use_container_width=True)

with col_d2:
    fig_radar = go.Figure()
    for _, row in dept_agg.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["nb_dpe_total"] / dept_agg["nb_dpe_total"].max(),
               row["nb_installations_pac"] / dept_agg["nb_installations_pac"].max(),
               row["nb_installations_clim"] / dept_agg["nb_installations_clim"].max(),
               row["pct_pac"] / dept_agg["pct_pac"].max(),
               row["temp_mean"] / dept_agg["temp_mean"].max()],
            theta=["DPE Total", "PAC", "Clim", "Taux PAC", "Temp moy"],
            fill="toself",
            name=row["dept_name"],
            opacity=0.6,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
        height=400,
        margin=dict(l=40, r=40, t=30, b=20),
        legend=dict(font=dict(size=10)),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "Projet HVAC Market Analysis — Patrice Duclos — 2025"
    "</div>",
    unsafe_allow_html=True,
)
