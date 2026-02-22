# -*- coding: utf-8 -*-
"""HVAC dashboard home page."""

import streamlit as st
from pathlib import Path
import pandas as pd


def render():
    st.title("HVAC Market Analysis in France")
    st.markdown(
        """
        Predictive analysis of the HVAC equipment market
        (heating, ventilation, air conditioning) in metropolitan France,
        through the exploitation of Open Data public datasets.
        """
    )

    # --- Key metrics ---
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    from config.settings import config as cfg
    raw_dir = cfg.raw_data_dir
    features_path = cfg.features_dataset_path
    ml_path = cfg.ml_dataset_path

    n_weather = _count_rows(raw_dir / "weather" / "weather_france.csv")
    n_dpe = _count_rows(raw_dir / "dpe" / "dpe_france_all.csv")
    n_sitadel = _count_rows(raw_dir / "sitadel" / "permis_construire_france.csv")
    n_features = _count_rows(features_path)

    col1.metric("Weather Data", f"{n_weather:,}" if n_weather else "N/A")
    col2.metric("DPE ADEME", f"{n_dpe:,}" if n_dpe else "N/A")
    col3.metric("SITADEL Permits", f"{n_sitadel:,}" if n_sitadel else "N/A")
    col4.metric("ML Dataset", f"{n_features:,}" if n_features else "N/A")

    # --- Quick data insights ---
    if ml_path.exists():
        _display_data_insights(ml_path)

    # --- Pipeline architecture ---
    st.header("Pipeline Architecture")
    st.markdown(
        """
        ```
        1. COLLECTION       Open-Meteo, ADEME DPE, INSEE, Eurostat, SITADEL
              |
        2. CLEANING         Deduplication, outliers, types, missing values
              |
        3. MERGE            Join month x department (96 depts x N months)
              |                + SITADEL permits + INSEE reference data
        4. FEATURES         Lags, rolling, interactions, trends
              |
        5. OUTLIERS         IQR + Z-score + Isolation Forest (consensus 2/3)
              |
        6. MODELING          LightGBM, Ridge, Prophet, LSTM
              |
        7. EVALUATION       MAE, RMSE, MAPE, R2, Feature Importance (SHAP)
              |
        8. PCLOUD           Automatic upload of results
        ```
        """
    )

    # --- Data sources ---
    st.header("Data Sources")
    sources = pd.DataFrame({
        "Source": ["Open-Meteo", "ADEME DPE", "INSEE BDM", "INSEE Filosofi",
                   "Eurostat", "SITADEL"],
        "Type": ["Weather", "Energy", "Economy", "Socioeconomic",
                 "Industry", "Construction"],
        "Granularity": [
            "Day x City",
            "Per unit (DPE)",
            "Month (national)",
            "Department (static)",
            "Month (national)",
            "Month x Department",
        ],
        "Features": [
            "temp, HDD, CDD, precipitation, wind",
            "DPE count, PAC, climatisation, class A/B",
            "confidence, business climate, IPI",
            "median income, housing stock, % houses",
            "IPI HVAC (C28, C2825)",
            "permits, housing units, surface",
        ],
    })
    st.dataframe(sources, use_container_width=True, hide_index=True)

    # --- Model performance summary ---
    _display_model_summary()

    # --- CLI commands ---
    with st.expander("CLI Commands"):
        st.code(
            """
# Full update (collection + pipeline + upload pCloud)
python -m src.pipeline update_all

# Collection only
python -m src.pipeline collect

# Processing
python -m src.pipeline process       # clean + merge + features + outliers

# ML
python -m src.pipeline train
python -m src.pipeline evaluate

# pCloud
python -m src.pipeline upload_pcloud

# Launch the dashboard
streamlit run app/app.py
            """,
            language="bash",
        )


def _display_data_insights(ml_path: Path):
    """Display key insights from the ML dataset."""
    try:
        df = pd.read_csv(ml_path, low_memory=False)
        df["dept"] = df["dept"].astype(str).str.zfill(2)
    except Exception:
        return

    st.header("Key Insights")

    col1, col2, col3 = st.columns(3)

    n_depts = df["dept"].nunique()
    date_min = str(df["date_id"].min())
    date_max = str(df["date_id"].max())
    period = f"{date_min[:4]}-{date_min[4:]} to {date_max[:4]}-{date_max[4:]}"

    col1.metric("Departments", n_depts)
    col2.metric("Period", period)
    col3.metric("Features", len(df.columns))

    # Top departments by PAC installations
    if "nb_installations_pac" in df.columns:
        top_pac = (
            df.groupby("dept")["nb_installations_pac"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Top 10 departments (PAC)")
            import plotly.express as px
            fig = px.bar(
                x=top_pac.values,
                y=top_pac.index,
                orientation="h",
                labels={"x": "Total PAC installations", "y": "Department"},
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=350,
                              margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Monthly trend (national)")
            if "nb_installations_pac" in df.columns:
                monthly = df.groupby("date_id")["nb_installations_pac"].sum().reset_index()
                monthly["date"] = pd.to_datetime(
                    monthly["date_id"].astype(str), format="%Y%m"
                )
                fig2 = px.line(
                    monthly, x="date", y="nb_installations_pac",
                    labels={"nb_installations_pac": "PAC installations", "date": ""},
                )
                fig2.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig2, use_container_width=True)


def _display_model_summary():
    """Display model performance if available."""
    import json
    from config.settings import config as cfg
    results_path = cfg.evaluation_results_path
    if not results_path.exists():
        return

    try:
        with open(results_path) as f:
            results = json.load(f)
    except Exception:
        return

    st.header("Model Performance")

    cols = st.columns(len(results))
    for i, (model_name, metrics) in enumerate(results.items()):
        if not isinstance(metrics, dict):
            continue
        with cols[i]:
            r2 = metrics.get("test_r2", metrics.get("r2", None))
            rmse = metrics.get("test_rmse", metrics.get("rmse", None))
            st.metric(
                model_name,
                f"R2={r2:.3f}" if r2 is not None else "N/A",
                f"RMSE={rmse:.2f}" if rmse is not None else None,
            )


def _count_rows(filepath: Path) -> int:
    """Count the rows of a CSV without loading it all into memory."""
    if not filepath.exists():
        return 0
    try:
        with open(filepath) as f:
            return sum(1 for _ in f) - 1
    except Exception:
        return 0
