# -*- coding: utf-8 -*-
"""Interactive France map page."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


def render():
    st.title("France Map — HVAC Market")

    # Load department coordinates from settings
    try:
        from config.settings import FRANCE_DEPARTMENTS, REGION_NAMES, DEPT_NAMES
    except ImportError:
        st.error("Unable to load department configuration.")
        return

    # --- Build the departments DataFrame ---
    dept_data = []
    for code, info in FRANCE_DEPARTMENTS.items():
        dept_code = info["dept"]
        dept_data.append({
            "dept_code": dept_code,
            "dept_name": DEPT_NAMES.get(dept_code, dept_code),
            "prefecture": code,
            "lat": info["lat"],
            "lon": info["lon"],
            "region": REGION_NAMES.get(info.get("region", ""), info.get("region", "")),
        })
    df_depts = pd.DataFrame(dept_data)

    # --- Load data if available ---
    from config.settings import config as cfg
    features_path = cfg.features_dataset_path
    ml_path = cfg.ml_dataset_path

    df_data = None
    if features_path.exists():
        df_data = pd.read_csv(features_path, low_memory=False)
    elif ml_path.exists():
        df_data = pd.read_csv(ml_path, low_memory=False)

    if df_data is not None and "dept" in df_data.columns:
        df_data["dept"] = df_data["dept"].astype(str).str.zfill(2)

        # Aggregate by department
        agg_cols = {}
        for col in ["nb_dpe_total", "nb_installations_pac", "nb_installations_clim",
                     "nb_dpe_classe_ab"]:
            if col in df_data.columns:
                agg_cols[col] = (col, "sum")

        if agg_cols:
            agg = df_data.groupby("dept").agg(**agg_cols).reset_index()
            agg = agg.rename(columns={"dept": "dept_code"})
            agg["dept_code"] = agg["dept_code"].astype(str).str.zfill(2)

            # Add averages for reference features if available
            for col in ["revenu_median", "prix_m2_median", "pct_maisons"]:
                if col in df_data.columns:
                    avg = df_data.groupby("dept")[col].mean().reset_index()
                    avg = avg.rename(columns={"dept": "dept_code"})
                    avg["dept_code"] = avg["dept_code"].astype(str).str.zfill(2)
                    agg = agg.merge(avg, on="dept_code", how="left")

            df_depts = df_depts.merge(agg, on="dept_code", how="left")

        # --- Metric selector ---
        metric_cols = [c for c in ["nb_dpe_total", "nb_installations_pac",
                                    "nb_installations_clim", "revenu_median",
                                    "prix_m2_median", "pct_maisons"]
                       if c in df_depts.columns and df_depts[c].notna().any()]

        if metric_cols:
            col_sel, col_info = st.columns([2, 3])
            with col_sel:
                selected_metric = st.selectbox("Metric to display", metric_cols)

            with col_info:
                valid = df_depts[selected_metric].dropna()
                if len(valid) > 0:
                    st.caption(
                        f"Min: {valid.min():,.0f} | "
                        f"Mean: {valid.mean():,.0f} | "
                        f"Max: {valid.max():,.0f} | "
                        f"Departments with data: {len(valid)}"
                    )

            # Fill NaN for map display
            df_map = df_depts.dropna(subset=[selected_metric])

            fig = px.scatter_mapbox(
                df_map,
                lat="lat",
                lon="lon",
                color=selected_metric,
                size=selected_metric,
                hover_name="dept_name",
                hover_data={
                    "dept_code": True,
                    "prefecture": True,
                    "region": True,
                    selected_metric: ":.0f",
                    "lat": False,
                    "lon": False,
                },
                color_continuous_scale="YlOrRd",
                size_max=30,
                zoom=5,
                center={"lat": 46.6, "lon": 2.3},
                mapbox_style="carto-positron",
                title=f"{selected_metric} by department",
            )
            fig.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # --- Department ranking table ---
            st.subheader("Department Ranking")
            display_cols = ["dept_code", "dept_name", "region"]
            display_cols += [c for c in metric_cols if c in df_depts.columns]
            df_display = df_depts[display_cols].dropna(subset=[selected_metric])
            df_display = df_display.sort_values(selected_metric, ascending=False)
            df_display.insert(0, "Rank", range(1, len(df_display) + 1))
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                height=400,
            )
        else:
            _render_base_map(df_depts)
    else:
        st.info("No ML data available. Displaying the 96 departments.")
        _render_base_map(df_depts)

    # --- Stats by region ---
    st.subheader("Distribution by Region")
    col_left, col_right = st.columns(2)

    with col_left:
        region_counts = df_depts.groupby("region")["dept_code"].count().reset_index()
        region_counts.columns = ["Region", "Departments"]
        fig = px.bar(
            region_counts.sort_values("Departments", ascending=True),
            x="Departments", y="Region",
            orientation="h",
            title="Departments per region",
        )
        fig.update_layout(height=400, margin=dict(l=0, r=20, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Regional aggregation if data available
    if df_data is not None and "nb_installations_pac" in df_data.columns:
        with col_right:
            df_data["dept"] = df_data["dept"].astype(str).str.zfill(2)
            dept_to_region = dict(zip(df_depts["dept_code"], df_depts["region"]))
            df_data["region"] = df_data["dept"].map(dept_to_region)
            region_pac = (
                df_data.groupby("region")["nb_installations_pac"]
                .sum()
                .sort_values(ascending=True)
                .reset_index()
            )
            region_pac.columns = ["Region", "PAC Installations"]
            fig2 = px.bar(
                region_pac,
                x="PAC Installations", y="Region",
                orientation="h",
                title="PAC installations per region",
                color="PAC Installations",
                color_continuous_scale="YlOrRd",
            )
            fig2.update_layout(height=400, margin=dict(l=0, r=20, t=40, b=0),
                               showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)


def _render_base_map(df_depts: pd.DataFrame):
    """Display the base map without ML data."""
    fig = px.scatter_mapbox(
        df_depts,
        lat="lat",
        lon="lon",
        hover_name="dept_name",
        hover_data={"dept_code": True, "prefecture": True, "region": True,
                    "lat": False, "lon": False},
        zoom=5,
        center={"lat": 46.6, "lon": 2.3},
        mapbox_style="carto-positron",
        title="96 metropolitan departments",
    )
    fig.update_traces(marker=dict(size=8, color="#1f77b4"))
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
