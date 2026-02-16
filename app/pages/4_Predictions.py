# -*- coding: utf-8 -*-
"""
Page Predictions — Predictions interactives avec les modeles entraines.
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
    load_model,
    load_scaler_and_imputer,
    section_header,
)

# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(
    page_title="Predictions — HVAC Market",
    page_icon="📈",
    layout="wide",
)
inject_custom_css()

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0;">
        <h1>📈 Predictions Interactives</h1>
        <p style="color:#666;">Simulez des scenarios et obtenez des estimations</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# Chargement
df_ml = load_ml_dataset()
df = load_features_dataset()
ridge_model = load_model("ridge")
lgbm_model = load_model("lightgbm")
scaler, imputer = load_scaler_and_imputer()

if df.empty:
    st.warning("Aucune donnee disponible (features dataset).")
    st.stop()

if df_ml.empty:
    st.warning("Aucune donnee disponible (ML dataset).")
    st.stop()

if ridge_model is None and lgbm_model is None:
    st.warning(
        "Aucun modele entraine disponible. Lancez l'entrainement :\n\n"
        "```bash\npython -m src.pipeline train\n```"
    )
    st.stop()


# =============================================================================
# Preparer les colonnes features
# =============================================================================
try:
    from src.models.train import ModelTrainer
    exclude = ModelTrainer.EXCLUDE_COLS | ModelTrainer.TARGET_COLS
except ImportError:
    exclude = {
        "date_id", "dept", "dept_name", "city_ref",
        "latitude", "longitude", "n_valid_features", "pct_valid_features",
        "nb_installations_pac", "nb_installations_clim",
        "nb_dpe_total", "nb_dpe_classe_ab",
    }

feature_cols = [c for c in df.columns if c not in exclude and c not in ["date"]]


# =============================================================================
# 1. Predictions sur donnees historiques
# =============================================================================
section_header("Predictions vs Reel (donnees historiques)")

col_m, col_d = st.columns(2)
with col_m:
    model_choice = st.selectbox(
        "Modele :",
        [m for m in ["ridge", "lightgbm"] if load_model(m) is not None],
        format_func=lambda x: x.upper(),
        key="pred_model",
    )
with col_d:
    dept_pred = st.selectbox(
        "Departement :",
        sorted(df["dept"].unique()),
        format_func=lambda x: f"{x} — {DEPARTMENTS.get(x, x)}",
        key="pred_dept",
    )

model = load_model(model_choice)
target = "nb_installations_pac"

if model is not None and scaler is not None:
    df_dept = df[df["dept"] == dept_pred].copy().sort_values("date")
    X = df_dept[feature_cols].copy()
    y = df_dept[target].copy()

    # Imputer + scaler
    if imputer is not None:
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    else:
        X_imp = X.fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=X_imp.columns, index=X_imp.index)

    y_pred = model.predict(X_scaled)
    df_dept["prediction"] = y_pred

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df_dept["date"], y=df_dept[target],
        mode="lines+markers",
        name="Reel",
        line=dict(color="#667eea", width=2.5),
        marker=dict(size=4),
    ))
    fig_pred.add_trace(go.Scatter(
        x=df_dept["date"], y=df_dept["prediction"],
        mode="lines+markers",
        name=f"Prediction ({model_choice.upper()})",
        line=dict(color="#f5576c", width=2, dash="dash"),
        marker=dict(size=4),
    ))

    # Zone test (2025+)
    fig_pred.add_vrect(
        x0="2025-01-01", x1=df_dept["date"].max(),
        fillcolor="rgba(245,87,108,0.08)",
        line_width=0,
        annotation_text="Test",
        annotation_position="top left",
    )
    fig_pred.add_vrect(
        x0="2024-07-01", x1="2024-12-31",
        fillcolor="rgba(102,126,234,0.08)",
        line_width=0,
        annotation_text="Validation",
        annotation_position="top left",
    )

    fig_pred.update_layout(
        height=420,
        yaxis_title=TARGETS[target],
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Metriques pour ce departement
    rmse = np.sqrt(np.mean((y.values - y_pred) ** 2))
    mae = np.mean(np.abs(y.values - y_pred))
    r2 = 1 - np.sum((y.values - y_pred) ** 2) / np.sum((y.values - y.mean()) ** 2)

    col_k1, col_k2, col_k3 = st.columns(3)
    with col_k1:
        st.markdown(kpi_card("RMSE", f"{rmse:.2f}", f"Dept {dept_pred}", "kpi-blue"), unsafe_allow_html=True)
    with col_k2:
        st.markdown(kpi_card("MAE", f"{mae:.2f}", f"Dept {dept_pred}", "kpi-orange"), unsafe_allow_html=True)
    with col_k3:
        st.markdown(kpi_card("R²", f"{r2:.3f}", f"Dept {dept_pred}", "kpi-green"), unsafe_allow_html=True)


# =============================================================================
# 2. Simulateur de scenario
# =============================================================================
section_header("Simulateur de scenario")

st.markdown(
    "Ajustez les parametres ci-dessous pour simuler un scenario et obtenir "
    "une estimation du nombre d'installations PAC."
)

# Utiliser les statistiques du dataset pour les bornes des sliders
col_sim1, col_sim2, col_sim3 = st.columns(3)

with col_sim1:
    st.markdown("**Parametres meteo**")
    sim_temp = st.slider(
        "Temperature moyenne (°C)",
        min_value=float(df["temp_mean"].min()) if "temp_mean" in df.columns else -5.0,
        max_value=float(df["temp_mean"].max()) if "temp_mean" in df.columns else 35.0,
        value=float(df["temp_mean"].median()) if "temp_mean" in df.columns else 12.0,
        step=0.5,
        key="sim_temp",
    )
    sim_hdd = st.slider(
        "Degres-jours chauffage (HDD)",
        min_value=0.0,
        max_value=float(df["hdd_sum"].max()) if "hdd_sum" in df.columns else 600.0,
        value=float(df["hdd_sum"].median()) if "hdd_sum" in df.columns else 150.0,
        step=10.0,
        key="sim_hdd",
    )
    sim_cdd = st.slider(
        "Degres-jours climatisation (CDD)",
        min_value=0.0,
        max_value=float(df["cdd_sum"].max()) if "cdd_sum" in df.columns else 300.0,
        value=float(df["cdd_sum"].median()) if "cdd_sum" in df.columns else 20.0,
        step=5.0,
        key="sim_cdd",
    )

with col_sim2:
    st.markdown("**Parametres economiques**")
    sim_confiance = st.slider(
        "Confiance menages",
        min_value=float(df["confiance_menages"].min()) if "confiance_menages" in df.columns else 70.0,
        max_value=float(df["confiance_menages"].max()) if "confiance_menages" in df.columns else 110.0,
        value=float(df["confiance_menages"].median()) if "confiance_menages" in df.columns else 90.0,
        step=1.0,
        key="sim_confiance",
    )
    sim_ipi = st.slider(
        "IPI HVAC (C28)",
        min_value=float(df["ipi_hvac_c28"].min()) if "ipi_hvac_c28" in df.columns else 60.0,
        max_value=float(df["ipi_hvac_c28"].max()) if "ipi_hvac_c28" in df.columns else 130.0,
        value=float(df["ipi_hvac_c28"].median()) if "ipi_hvac_c28" in df.columns else 95.0,
        step=1.0,
        key="sim_ipi",
    )

with col_sim3:
    st.markdown("**Parametres localisation / temps**")
    sim_dept = st.selectbox(
        "Departement :",
        sorted(df["dept"].unique()),
        format_func=lambda x: f"{x} — {DEPARTMENTS.get(x, x)}",
        key="sim_dept",
    )
    sim_month = st.selectbox(
        "Mois :",
        list(range(1, 13)),
        format_func=lambda x: MOIS_FR[x - 1],
        index=5,
        key="sim_month",
    )
    sim_model = st.selectbox(
        "Modele :",
        [m for m in ["ridge", "lightgbm"] if load_model(m) is not None],
        format_func=lambda x: x.upper(),
        key="sim_model",
    )

# Construire le vecteur de features a partir des medianes du dept selectionne
if st.button("Lancer la simulation", type="primary"):
    sim_model_obj = load_model(sim_model)
    if sim_model_obj is not None and scaler is not None:
        # Prendre la ligne mediane du departement comme base
        df_dept_base = df[df["dept"] == sim_dept]
        if not df_dept_base.empty:
            X_base = df_dept_base[feature_cols].median().to_frame().T

            # Injecter les valeurs du simulateur
            override_map = {
                "temp_mean": sim_temp,
                "hdd_sum": sim_hdd,
                "cdd_sum": sim_cdd,
                "confiance_menages": sim_confiance,
                "ipi_hvac_c28": sim_ipi,
            }
            # Colonnes temporelles
            if "month" in feature_cols:
                X_base["month"] = sim_month
            if "quarter" in feature_cols:
                X_base["quarter"] = (sim_month - 1) // 3 + 1
            if "sin_month" in feature_cols:
                X_base["sin_month"] = np.sin(2 * np.pi * sim_month / 12)
            if "cos_month" in feature_cols:
                X_base["cos_month"] = np.cos(2 * np.pi * sim_month / 12)

            for col, val in override_map.items():
                if col in feature_cols:
                    X_base[col] = val

            # Imputer + scaler
            if imputer is not None:
                X_sim = pd.DataFrame(
                    imputer.transform(X_base), columns=X_base.columns
                )
            else:
                X_sim = X_base.fillna(0)
            X_sim_scaled = pd.DataFrame(
                scaler.transform(X_sim), columns=X_sim.columns
            )

            prediction = sim_model_obj.predict(X_sim_scaled)[0]
            prediction = max(0, prediction)  # Pas de valeurs negatives

            st.markdown("<br>", unsafe_allow_html=True)
            col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
            with col_r2:
                st.markdown(
                    kpi_card(
                        f"Estimation ({sim_model.upper()})",
                        f"{prediction:.0f}",
                        f"Installations PAC — {DEPARTMENTS.get(sim_dept, sim_dept)} — {MOIS_FR[sim_month - 1]}",
                        "kpi-green",
                    ),
                    unsafe_allow_html=True,
                )

            # Comparaison avec les valeurs historiques du meme mois
            hist_month = df_dept_base[df_dept_base["date"].dt.month == sim_month][target]
            if not hist_month.empty:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Comparaison avec l'historique :**")
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.metric("Moyenne historique", f"{hist_month.mean():.0f}")
                with col_h2:
                    st.metric("Min historique", f"{hist_month.min():.0f}")
                with col_h3:
                    st.metric("Max historique", f"{hist_month.max():.0f}")
        else:
            st.error("Pas de donnees pour ce departement.")
    else:
        st.error("Modele ou scaler non disponible.")


# =============================================================================
# 3. Predictions multi-departements
# =============================================================================
section_header("Vue multi-departements")

st.markdown("Predictions pour tous les departements a la derniere date disponible.")

model_multi = st.selectbox(
    "Modele :",
    [m for m in ["ridge", "lightgbm"] if load_model(m) is not None],
    format_func=lambda x: x.upper(),
    key="multi_model",
)

model_obj = load_model(model_multi)
if model_obj is not None and scaler is not None:
    last_date = df["date"].max()
    df_last = df[df["date"] == last_date].copy()

    if not df_last.empty:
        X_last = df_last[feature_cols].copy()
        if imputer is not None:
            X_imp = pd.DataFrame(imputer.transform(X_last), columns=X_last.columns, index=X_last.index)
        else:
            X_imp = X_last.fillna(0)
        X_scaled = pd.DataFrame(scaler.transform(X_imp), columns=X_imp.columns, index=X_imp.index)

        df_last["prediction"] = model_obj.predict(X_scaled)
        df_last["dept_name"] = df_last["dept"].map(DEPARTMENTS)
        df_last["ecart"] = df_last["prediction"] - df_last[target]
        df_last["ecart_pct"] = (100 * df_last["ecart"] / df_last[target].clip(lower=1)).round(1)

        col_mv1, col_mv2 = st.columns(2)

        with col_mv1:
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Bar(
                name="Reel",
                x=df_last["dept_name"],
                y=df_last[target],
                marker_color="#667eea",
                text=df_last[target].round(0).astype(int),
                textposition="auto",
            ))
            fig_multi.add_trace(go.Bar(
                name=f"Prediction ({model_multi.upper()})",
                x=df_last["dept_name"],
                y=df_last["prediction"],
                marker_color="#f5576c",
                text=df_last["prediction"].round(0).astype(int),
                textposition="auto",
            ))
            fig_multi.update_layout(
                barmode="group",
                height=400,
                yaxis_title=TARGETS[target],
                title=f"Derniere date : {last_date.strftime('%Y-%m')}",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_multi, use_container_width=True)

        with col_mv2:
            fig_ecart = px.bar(
                df_last.sort_values("ecart"),
                x="ecart",
                y="dept_name",
                orientation="h",
                color="ecart",
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                text="ecart_pct",
                labels={"ecart": "Ecart (pred - reel)", "dept_name": "Departement"},
            )
            fig_ecart.update_traces(texttemplate="%{text}%", textposition="auto")
            fig_ecart.update_layout(
                height=400,
                title="Ecart prediction vs reel",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_ecart, use_container_width=True)


# --- Footer ---
st.markdown(
    '<div class="footer">'
    "Projet HVAC Market Analysis — Patrice Duclos — 2025"
    "</div>",
    unsafe_allow_html=True,
)
