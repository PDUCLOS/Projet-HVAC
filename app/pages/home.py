# -*- coding: utf-8 -*-
"""HVAC dashboard home page."""

import streamlit as st
from pathlib import Path
import pandas as pd


def render():
    st.title("HVAC Market Analysis in France")
    st.markdown(
        """
        This dashboard presents the predictive analysis of the HVAC equipment market
        (heating, ventilation, air conditioning) in metropolitan France,
        through the exploitation of Open Data public datasets.
        """
    )

    # --- Key metrics ---
    st.header("Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Count available data
    raw_dir = Path("data/raw")
    features_path = Path("data/features/hvac_features_dataset.csv")

    n_weather = _count_rows(raw_dir / "weather" / "weather_france.csv")
    n_dpe = _count_rows(raw_dir / "dpe" / "dpe_france_all.csv")
    n_sitadel = _count_rows(raw_dir / "sitadel" / "permis_construire_france.csv")
    n_features = _count_rows(features_path)

    col1.metric("Weather Data", f"{n_weather:,}" if n_weather else "Not collected")
    col2.metric("DPE ADEME", f"{n_dpe:,}" if n_dpe else "Not collected")
    col3.metric("Permis SITADEL", f"{n_sitadel:,}" if n_sitadel else "Not collected")
    col4.metric("Dataset ML", f"{n_features:,}" if n_features else "Not generated")

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
              |
        4. FEATURES         Lags, rolling, interactions, trends
              |
        5. OUTLIERS         IQR + Z-score + Isolation Forest (consensus 2/3)
              |
        6. MODELING          LightGBM, XGBoost, Ridge, Prophet, LSTM
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
        "Source": ["Open-Meteo", "ADEME DPE", "INSEE BDM", "Eurostat", "SITADEL"],
        "Type": ["Weather", "Energy", "Economy", "Industry", "Construction"],
        "Granularity": [
            "Day x City",
            "Per unit (DPE)",
            "Month (national)",
            "Month (national)",
            "Month x Department",
        ],
        "API": [
            "open-meteo.com",
            "data.ademe.fr",
            "bdm.insee.fr (SDMX)",
            "ec.europa.eu/eurostat",
            "data.statistiques.developpement-durable.gouv.fr",
        ],
        "Authentication": ["None"] * 5,
    })
    st.dataframe(sources, use_container_width=True, hide_index=True)

    # --- CLI commands ---
    st.header("CLI Commands")
    st.code(
        """
# Full update (collection + pipeline + upload pCloud)
python -m src.pipeline update_all

# Collection only
python -m src.pipeline collect
python -m src.pipeline collect --sources weather,dpe

# Processing
python -m src.pipeline process       # clean + merge + features + outliers

# ML
python -m src.pipeline train
python -m src.pipeline evaluate

# pCloud
python -m src.pipeline sync_pcloud   # Download from pCloud
python -m src.pipeline upload_pcloud  # Upload to pCloud

# Launch the dashboard
streamlit run app/app.py
        """,
        language="bash",
    )


def _count_rows(filepath: Path) -> int:
    """Count the rows of a CSV without loading it all into memory."""
    if not filepath.exists():
        return 0
    try:
        # Count lines without reading the content
        with open(filepath) as f:
            return sum(1 for _ in f) - 1  # -1 for the header
    except Exception:
        return 0
