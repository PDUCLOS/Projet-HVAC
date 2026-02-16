# -*- coding: utf-8 -*-
"""
Page Modelisation — Resultats ML, comparaison, feature importance.
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
    TARGETS,
    inject_custom_css,
    kpi_card,
    load_features_dataset,
    load_ml_dataset,
    load_model,
    load_scaler_and_imputer,
    load_training_results,
    section_header,
    MODELS_DIR,
)

# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(
    page_title="Modelisation ML — HVAC Market",
    page_icon="🤖",
    layout="wide",
)
inject_custom_css()

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1>🤖 Modelisation Machine Learning</h1>
        <p style="color:#666;">Comparaison des modeles, metriques et interpretation</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# Chargement
df_results = load_training_results()
df = load_ml_dataset()
df_feat = load_features_dataset()

if df_results.empty:
    st.warning(
        "Aucun resultat de modelisation disponible. Lancez l'entrainement :\n\n"
        "```bash\npython -m src.pipeline train\n```"
    )
    st.stop()


# =============================================================================
# 1. Vue d'ensemble des modeles
# =============================================================================
section_header("Comparaison des modeles")

# KPIs du meilleur modele
best = df_results.loc[df_results["val_rmse"].idxmin()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        kpi_card("Meilleur modele", best["model"].upper(), "Validation RMSE la plus basse", "kpi-green"),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        kpi_card("Val RMSE", f"{best['val_rmse']:.2f}", "Erreur de validation", "kpi-blue"),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        kpi_card("Test RMSE", f"{best['test_rmse']:.2f}", "Erreur sur le jeu test", "kpi-orange"),
        unsafe_allow_html=True,
    )
with col4:
    r2_val = best.get("val_r2", 0)
    st.markdown(
        kpi_card("R² Validation", f"{r2_val:.3f}", "Coefficient de determination", "kpi-teal"),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tableau comparatif complet
st.markdown("**Tableau comparatif**")
display_cols = ["model", "val_rmse", "val_mae", "val_mape", "val_r2", "test_rmse", "test_mae", "test_mape", "test_r2"]
display_cols = [c for c in display_cols if c in df_results.columns]
df_display = df_results[display_cols].copy()
df_display.columns = [
    c.replace("val_", "Val ").replace("test_", "Test ").replace("_", " ").upper()
    if c != "model" else "Modele"
    for c in display_cols
]

# Arrondir les colonnes numeriques
for col in df_display.select_dtypes(include=[np.number]).columns:
    df_display[col] = df_display[col].round(4)

st.dataframe(df_display, use_container_width=True, hide_index=True)


# =============================================================================
# 2. Graphiques de comparaison
# =============================================================================
section_header("Visualisation des performances")

col_g1, col_g2 = st.columns(2)

with col_g1:
    # RMSE val vs test
    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        name="Validation RMSE",
        x=df_results["model"],
        y=df_results["val_rmse"],
        marker_color="#667eea",
        text=df_results["val_rmse"].round(2),
        textposition="auto",
    ))
    fig_rmse.add_trace(go.Bar(
        name="Test RMSE",
        x=df_results["model"],
        y=df_results["test_rmse"],
        marker_color="#f5576c",
        text=df_results["test_rmse"].round(2),
        textposition="auto",
    ))
    fig_rmse.update_layout(
        barmode="group",
        height=380,
        yaxis_title="RMSE",
        title="RMSE : Validation vs Test",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

with col_g2:
    # R² val vs test
    r2_cols = [c for c in ["val_r2", "test_r2"] if c in df_results.columns]
    if r2_cols:
        fig_r2 = go.Figure()
        if "val_r2" in df_results.columns:
            fig_r2.add_trace(go.Bar(
                name="Validation R²",
                x=df_results["model"],
                y=df_results["val_r2"],
                marker_color="#11998e",
                text=df_results["val_r2"].round(3),
                textposition="auto",
            ))
        if "test_r2" in df_results.columns:
            fig_r2.add_trace(go.Bar(
                name="Test R²",
                x=df_results["model"],
                y=df_results["test_r2"],
                marker_color="#38ef7d",
                text=df_results["test_r2"].round(3),
                textposition="auto",
            ))
        fig_r2.update_layout(
            barmode="group",
            height=380,
            yaxis_title="R²",
            title="R² : Validation vs Test",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_r2, use_container_width=True)

# Cross-validation results (si disponibles)
cv_cols = [c for c in ["cv_rmse_mean", "cv_rmse_std"] if c in df_results.columns]
if cv_cols and not df_results["cv_rmse_mean"].isna().all():
    section_header("Cross-Validation temporelle")

    df_cv = df_results.dropna(subset=["cv_rmse_mean"])
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=df_cv["model"],
        y=df_cv["cv_rmse_mean"],
        error_y=dict(type="data", array=df_cv["cv_rmse_std"].values, visible=True),
        marker_color="#667eea",
        text=df_cv["cv_rmse_mean"].round(2),
        textposition="auto",
        name="CV RMSE (mean ± std)",
    ))
    fig_cv.update_layout(
        height=350,
        yaxis_title="RMSE",
        title="RMSE moyenne par Cross-Validation temporelle (5 folds)",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig_cv, use_container_width=True)


# =============================================================================
# 3. Feature importance (LightGBM)
# =============================================================================
section_header("Importance des variables (LightGBM)")

lgbm_model = load_model("lightgbm")

if lgbm_model is not None:
    try:
        importance = lgbm_model.feature_importances_
        feature_names = lgbm_model.feature_name_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=True).tail(20)

        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Importance (gain)", "feature": "Variable"},
        )
        fig_fi.update_layout(
            height=max(400, len(fi_df) * 22),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.info(f"Impossible d'extraire les importances : {e}")
else:
    st.info("Modele LightGBM non disponible. Lancez l'entrainement pour obtenir les feature importances.")


# =============================================================================
# 4. Analyse des residus (Ridge)
# =============================================================================
section_header("Analyse des residus (Ridge)")

ridge_model = load_model("ridge")
scaler, imputer = load_scaler_and_imputer()

if ridge_model is not None and scaler is not None and not df_feat.empty:
    try:
        from src.models.train import ModelTrainer

        # Preparer les features comme a l'entrainement (dataset complet)
        exclude = ModelTrainer.EXCLUDE_COLS | ModelTrainer.TARGET_COLS
        target = "nb_installations_pac"
        feature_cols = [c for c in df_feat.columns if c not in exclude and c not in ["date"]]

        X = df_feat[feature_cols].copy()
        y = df_feat[target].copy()

        # Imputer et scaler
        if imputer is not None:
            X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
        else:
            X_imp = X.fillna(0)
        X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=X_imp.columns, index=X_imp.index)

        y_pred = ridge_model.predict(X_scaled)
        residuals = y.values - y_pred

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            # Predictions vs Actual
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y.values, y=y_pred,
                mode="markers",
                marker=dict(color="#667eea", opacity=0.6),
                name="Observations",
            ))
            # Ligne parfaite
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Prediction parfaite",
            ))
            fig_pred.update_layout(
                xaxis_title="Valeurs reelles",
                yaxis_title="Predictions",
                title="Predictions vs Reel (Ridge)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        with col_r2:
            # Distribution des residus
            fig_res = px.histogram(
                x=residuals,
                nbins=30,
                labels={"x": "Residu"},
                color_discrete_sequence=["#667eea"],
            )
            fig_res.update_layout(
                title="Distribution des residus (Ridge)",
                xaxis_title="Residu (reel - predit)",
                yaxis_title="Frequence",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_res, use_container_width=True)

    except Exception as e:
        st.info(f"Analyse des residus non disponible : {e}")
else:
    st.info("Le modele Ridge ou le scaler n'est pas disponible.")


# =============================================================================
# 5. Methodologie
# =============================================================================
section_header("Methodologie")

st.markdown(
    """
    | Aspect | Detail |
    |--------|--------|
    | **Split temporel** | Train: 2021-07 → 2024-06, Val: 2024-07 → 2024-12, Test: 2025-01 → 2025-12 |
    | **Grain** | Mois × Departement (8 departements AURA) |
    | **Cross-validation** | TimeSeriesSplit (5 folds) pour Ridge et LightGBM |
    | **Preprocessing** | Imputation mediane + StandardScaler |
    | **Ridge** | Regression lineaire regularisee L2, alpha optimise par CV |
    | **LightGBM** | Gradient Boosting regularise (max_depth=4, num_leaves=15) |
    | **Prophet** | Par departement, regresseurs externes (temp, HDD, CDD, confiance, IPI) |
    | **LSTM** | Exploratoire/pedagogique — lookback=3, 32 unites, dropout=0.3 |
    """
)


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "Projet HVAC Market Analysis — Patrice Duclos — 2025"
    "</div>",
    unsafe_allow_html=True,
)
