# -*- coding: utf-8 -*-
"""Data exploration page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


def render():
    st.title("Exploration des donnees")

    # --- Dataset selection ---
    datasets = _list_available_datasets()
    if not datasets:
        st.warning("Aucun dataset disponible. Lancez d'abord la collecte.")
        st.code("python -m src.pipeline collect", language="bash")
        return

    selected = st.selectbox("Dataset a explorer", list(datasets.keys()))
    filepath = datasets[selected]

    # --- Loading ---
    with st.spinner(f"Chargement de {selected}..."):
        df = _load_dataset(str(filepath))

    if df is None or df.empty:
        st.error(f"Impossible de charger {filepath}")
        return

    # --- Overview ---
    st.subheader(f"Apercu : {selected}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", f"{len(df):,}")
    col2.metric("Colonnes", len(df.columns))
    col3.metric("Taille memoire", f"{df.memory_usage(deep=True).sum() / 1e6:.1f} Mo")

    st.dataframe(df.head(50), use_container_width=True)

    # --- Descriptive statistics ---
    st.subheader("Statistiques descriptives")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        st.dataframe(
            df[numeric_cols].describe().round(2),
            use_container_width=True,
        )

    # --- Missing values ---
    st.subheader("Valeurs manquantes")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("Aucune valeur manquante.")
    else:
        fig = px.bar(
            x=missing.index,
            y=missing.values,
            labels={"x": "Colonne", "y": "Nombre de NaN"},
            title="Valeurs manquantes par colonne",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Distribution ---
    if numeric_cols:
        st.subheader("Distribution")
        col_to_plot = st.selectbox("Colonne", numeric_cols)
        fig = px.histogram(
            df, x=col_to_plot, nbins=50,
            title=f"Distribution de {col_to_plot}",
            marginal="box",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Correlation ---
    if len(numeric_cols) >= 2:
        st.subheader("Matrice de correlation")
        max_cols = st.slider("Nombre de colonnes", 2, min(20, len(numeric_cols)), min(10, len(numeric_cols)))
        corr_cols = numeric_cols[:max_cols]
        corr = df[corr_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlations",
            aspect="auto",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Time series ---
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "period" in c.lower()]
    if date_cols and numeric_cols:
        st.subheader("Evolution temporelle")
        date_col = st.selectbox("Colonne de date", date_cols)
        value_col = st.selectbox("Valeur", numeric_cols, key="ts_value")

        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col], errors="coerce")
        df_ts = df_ts.dropna(subset=[date_col]).sort_values(date_col)

        fig = px.line(
            df_ts, x=date_col, y=value_col,
            title=f"{value_col} dans le temps",
        )
        st.plotly_chart(fig, use_container_width=True)


def _list_available_datasets() -> dict:
    """List available datasets."""
    datasets = {}
    paths = {
        "Dataset ML (features)": Path("data/features/hvac_features_dataset.csv"),
        "Dataset fusionne": Path("data/features/hvac_ml_dataset.csv"),
        "Meteo France (brut)": Path("data/raw/weather/weather_france.csv"),
        "Meteo France (clean)": Path("data/processed/weather/weather_france.csv"),
        "DPE ADEME (brut)": Path("data/raw/dpe/dpe_france_all.csv"),
        "DPE ADEME (clean)": Path("data/processed/dpe/dpe_france_clean.csv"),
        "INSEE economique": Path("data/raw/insee/indicateurs_economiques.csv"),
        "Eurostat IPI": Path("data/raw/eurostat/ipi_hvac_france.csv"),
        "SITADEL permis": Path("data/raw/sitadel/permis_construire_france.csv"),
    }
    for name, path in paths.items():
        if path.exists():
            datasets[name] = path
    return datasets


@st.cache_data(ttl=300)
def _load_dataset(filepath: str, max_rows: int = 100_000) -> pd.DataFrame:
    """Load a dataset with row limit and 5-minute cache."""
    try:
        return pd.read_csv(filepath, nrows=max_rows, low_memory=False,
                           encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(filepath, nrows=max_rows, low_memory=False,
                               encoding="latin-1")
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        return pd.DataFrame()
