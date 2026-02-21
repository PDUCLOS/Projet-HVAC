# -*- coding: utf-8 -*-
"""Interactive France map page."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


def render():
    st.title("Carte de France - HVAC")

    # Load department coordinates from settings
    try:
        from config.settings import FRANCE_DEPARTMENTS, REGION_NAMES
    except ImportError:
        st.error("Impossible de charger la configuration des departements.")
        return

    # --- Build the departments DataFrame ---
    dept_data = []
    for code, info in FRANCE_DEPARTMENTS.items():
        dept_data.append({
            "dept_code": code,
            "prefecture": info["name"],
            "lat": info["lat"],
            "lon": info["lon"],
            "region": REGION_NAMES.get(info.get("region", ""), info.get("region", "")),
        })
    df_depts = pd.DataFrame(dept_data)

    # --- Load data if available ---
    features_path = Path("data/features/hvac_features_dataset.csv")
    ml_path = Path("data/features/hvac_ml_dataset.csv")

    df_data = None
    if features_path.exists():
        df_data = pd.read_csv(features_path, low_memory=False)
    elif ml_path.exists():
        df_data = pd.read_csv(ml_path, low_memory=False)

    if df_data is not None and "dept" in df_data.columns:
        # Aggregate by department
        agg = df_data.groupby("dept").agg(
            nb_dpe_total=("nb_dpe_total", "sum") if "nb_dpe_total" in df_data.columns else ("dept", "count"),
        ).reset_index()
        agg.columns = ["dept_code", "nb_dpe_total"]
        agg["dept_code"] = agg["dept_code"].astype(str).str.zfill(2)

        # Enrich if other columns exist
        numeric_aggs = {}
        for col in ["nb_installations_pac", "nb_installations_clim", "nb_dpe_classe_ab"]:
            if col in df_data.columns:
                extra = df_data.groupby("dept")[col].sum().reset_index()
                extra.columns = ["dept_code", col]
                extra["dept_code"] = extra["dept_code"].astype(str).str.zfill(2)
                agg = agg.merge(extra, on="dept_code", how="left")

        df_depts = df_depts.merge(agg, on="dept_code", how="left")

        # Map with data
        metric_cols = [c for c in ["nb_dpe_total", "nb_installations_pac", "nb_installations_clim"] if c in df_depts.columns]
        if metric_cols:
            selected_metric = st.selectbox("Metrique a afficher", metric_cols)

            fig = px.scatter_mapbox(
                df_depts,
                lat="lat",
                lon="lon",
                color=selected_metric,
                size=selected_metric,
                hover_name="prefecture",
                hover_data=["dept_code", "region"] + metric_cols,
                color_continuous_scale="YlOrRd",
                size_max=30,
                zoom=5,
                center={"lat": 46.6, "lon": 2.3},
                mapbox_style="carto-positron",
                title=f"{selected_metric} par departement",
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Detailed table
            st.subheader("Donnees par departement")
            display_cols = ["dept_code", "prefecture", "region"] + metric_cols
            st.dataframe(
                df_depts[display_cols].sort_values(metric_cols[0], ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            _render_base_map(df_depts)
    else:
        st.info("Pas de donnees ML disponibles. Affichage des 96 departements.")
        _render_base_map(df_depts)

    # --- Stats by region ---
    st.subheader("Repartition par region")
    region_counts = df_depts.groupby("region")["dept_code"].count().reset_index()
    region_counts.columns = ["Region", "Departements"]
    fig = px.bar(
        region_counts.sort_values("Departements", ascending=True),
        x="Departements", y="Region",
        orientation="h",
        title="Nombre de departements par region",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_base_map(df_depts: pd.DataFrame):
    """Display the base map without ML data."""
    fig = px.scatter_mapbox(
        df_depts,
        lat="lat",
        lon="lon",
        hover_name="prefecture",
        hover_data=["dept_code", "region"],
        zoom=5,
        center={"lat": 46.6, "lon": 2.3},
        mapbox_style="carto-positron",
        title="96 departements metropolitains",
    )
    fig.update_traces(marker=dict(size=8, color="#1f77b4"))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
