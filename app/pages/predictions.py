# -*- coding: utf-8 -*-
"""ML predictions page."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import pickle
import json


def render():
    st.title("Predictions ML")

    # --- Load evaluation results ---
    eval_dir = Path("data/models")
    results_file = eval_dir / "evaluation_results.json"
    models_dir = eval_dir

    if not eval_dir.exists():
        st.warning("Aucun modele entraine. Lancez d'abord l'entrainement.")
        st.code("python -m src.pipeline train", language="bash")
        return

    # --- Load the dataset ---
    features_path = Path("data/features/hvac_features_dataset.csv")
    if not features_path.exists():
        st.warning("Dataset de features non disponible.")
        return

    df = pd.read_csv(features_path, low_memory=False)
    st.success(f"Dataset charge : {len(df):,} lignes x {len(df.columns)} colonnes")

    # --- Target variable selection ---
    target_cols = [c for c in df.columns if c.startswith("nb_") or c.startswith("pct_")]
    if not target_cols:
        st.error("Aucune variable cible trouvee dans le dataset.")
        return

    target = st.selectbox("Variable cible", target_cols, index=0)

    # --- Load results ---
    if results_file.exists():
        with open(results_file) as f:
            eval_results = json.load(f)
        _display_eval_results(eval_results)

    # --- Visualize predictions ---
    st.subheader("Predictions vs Realite")

    # Look for prediction files
    pred_files = list(eval_dir.glob("predictions_*.csv"))
    if pred_files:
        selected_pred = st.selectbox(
            "Fichier de predictions",
            [f.name for f in pred_files],
        )
        df_pred = pd.read_csv(eval_dir / selected_pred)

        if "y_true" in df_pred.columns and "y_pred" in df_pred.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_pred["y_true"],
                y=df_pred["y_pred"],
                mode="markers",
                name="Predictions",
                marker=dict(opacity=0.5),
            ))
            # Perfect line
            min_val = min(df_pred["y_true"].min(), df_pred["y_pred"].min())
            max_val = max(df_pred["y_true"].max(), df_pred["y_pred"].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Parfait",
                line=dict(dash="dash", color="red"),
            ))
            fig.update_layout(
                title="Predictions vs Valeurs reelles",
                xaxis_title="Valeurs reelles",
                yaxis_title="Predictions",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            df_pred["residual"] = df_pred["y_true"] - df_pred["y_pred"]
            fig_res = px.histogram(
                df_pred, x="residual", nbins=50,
                title="Distribution des residus",
                marginal="box",
            )
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Pas de fichier de predictions. Lancez l'evaluation.")

    # --- Feature Importance ---
    st.subheader("Feature Importance")
    fi_files = list(eval_dir.glob("feature_importance_*.csv"))
    if fi_files:
        selected_fi = st.selectbox(
            "Modele",
            [f.stem.replace("feature_importance_", "") for f in fi_files],
        )
        fi_path = eval_dir / f"feature_importance_{selected_fi}.csv"
        if fi_path.exists():
            df_fi = pd.read_csv(fi_path)
            if "feature" in df_fi.columns and "importance" in df_fi.columns:
                df_fi = df_fi.sort_values("importance", ascending=True).tail(20)
                fig = px.barh(
                    df_fi, x="importance", y="feature",
                    title=f"Top 20 Features ({selected_fi})",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Try to load importances from SHAP
        shap_dir = eval_dir / "shap"
        if shap_dir.exists():
            shap_files = list(shap_dir.glob("*.png"))
            if shap_files:
                for img_file in shap_files[:3]:
                    st.image(str(img_file), caption=img_file.stem)
        else:
            st.info("Pas de feature importance disponible.")


def _display_eval_results(results: dict):
    """Display evaluation metrics."""
    st.subheader("Metriques d'evaluation")

    if isinstance(results, dict):
        metrics_data = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                row = {"Modele": model_name}
                row.update(metrics)
                metrics_data.append(row)

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # Comparison chart
            metric_cols = [c for c in df_metrics.columns if c != "Modele"]
            if metric_cols:
                selected_metric = st.selectbox("Metrique a comparer", metric_cols)
                fig = px.bar(
                    df_metrics, x="Modele", y=selected_metric,
                    title=f"Comparaison : {selected_metric}",
                    color="Modele",
                )
                st.plotly_chart(fig, use_container_width=True)
