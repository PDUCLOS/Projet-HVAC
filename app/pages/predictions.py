# -*- coding: utf-8 -*-
"""Interactive ML predictions page — department selection + PAC forecast."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime as dt
from pathlib import Path
import pickle


# ---------------------------------------------------------------------------
# Data loading (cached to avoid reloading on every interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load trained models, scaler, and imputer from disk."""
    models_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models"

    artifacts = {}
    for name, filename in [
        ("ridge", "ridge_model.pkl"),
        ("lightgbm", "lightgbm_model.pkl"),
        ("scaler", "scaler.pkl"),
        ("imputer", "imputer.pkl"),
    ]:
        path = models_dir / filename
        if path.exists():
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)  # noqa: S301 — trusted local files

    # Training results
    results_path = models_dir / "training_results.csv"
    if results_path.exists():
        artifacts["training_results"] = pd.read_csv(results_path)

    return artifacts


@st.cache_data
def load_features():
    """Load the features dataset."""
    features_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "features" / "hvac_features_dataset.csv"
    )
    if features_path.exists():
        return pd.read_csv(features_path, dtype={"dept": str})
    return None


def get_department_options(df):
    """Build department dropdown options: '69 — Rhone'."""
    from config.settings import DEPT_NAMES

    depts = sorted(df["dept"].unique())
    options = {}
    for d in depts:
        name = DEPT_NAMES.get(d, d)
        options[f"{d} — {name}"] = d
    return options



def predict_department(artifacts, df, dept, horizon, model_name):
    """Generate auto-regressive predictions for a department.

    Seasonality strategy:
    - Weather/climate features: injected from historical monthly averages so
      the model sees realistic temperature, HDD, CDD, etc. per season.
    - PAC lag + derived features: maintained via a rolling pac_buffer.
      Historical values anchor the start; future months use seasonal proxies
      (same calendar month mean × recent-trend scale), so rmean_3m, rstd_3m,
      diff_1m, pct_1m all vary with the seasonal cycle instead of freezing at
      the last observed value.  Using the proxy (not our own prediction) avoids
      error amplification while preserving seasonal patterns.
    """
    model = artifacts.get(model_name)
    ridge = artifacts.get("ridge")
    imputer = artifacts.get("imputer")
    scaler = artifacts.get("scaler")

    if model is None or imputer is None:
        return []

    feature_names = list(ridge.feature_names_in_)

    df_dept = df[df["dept"] == dept].sort_values("date_id").reset_index(drop=True)
    if df_dept.empty:
        return []

    # --- Seasonal weather profile (historical monthly means) ---
    SEASONAL_KEYWORDS = (
        "temp", "precip", "wind", "hdd", "cdd",
        "canicule", "gel", "pac_ineff", "delta_temp",
    )
    seasonal_feats = [
        f for f in feature_names
        if any(k in f.lower() for k in SEASONAL_KEYWORDS)
    ]
    monthly_weather = (
        df_dept.groupby("month")[seasonal_feats].mean() if seasonal_feats else None
    )

    # --- Seasonal PAC profile and trend scale ---
    monthly_pac = df_dept.groupby("month")["nb_installations_pac"].mean()
    hist_mean = df_dept["nb_installations_pac"].mean()
    recent_mean = df_dept["nb_installations_pac"].tail(12).mean()
    seasonal_scale = recent_mean / hist_mean if hist_mean > 0 else 1.0

    # --- Seasonal DPE A/B profile and trend scale ---
    dpe_ab_col = "nb_dpe_classe_ab"
    has_dpe_ab = dpe_ab_col in df_dept.columns and not df_dept[dpe_ab_col].isna().all()
    if has_dpe_ab:
        monthly_dpe_ab = df_dept.groupby("month")[dpe_ab_col].mean()
        dpe_ab_hist_mean = df_dept[dpe_ab_col].mean()
        dpe_ab_recent_mean = df_dept[dpe_ab_col].tail(12).mean()
        dpe_ab_scale = dpe_ab_recent_mean / dpe_ab_hist_mean if dpe_ab_hist_mean > 0 else 1.0
        dpe_ab_buffer = list(df_dept[dpe_ab_col].tail(24).values)
    else:
        dpe_ab_buffer = []

    last_row = df_dept.iloc[-1].copy()
    last_date_str = str(int(last_row["date_id"]))
    rmse = _get_rmse(artifacts, model_name)

    # pac_buffer: rolling window of PAC values used to compute ALL derived
    # features (lags, rmean, rstd, diff, pct).  Starts from actual historical
    # data; each step appends a seasonal proxy for that month so future
    # derived features vary with the seasonal cycle.
    pac_buffer = list(df_dept["nb_installations_pac"].tail(24).values)

    results = []
    current_row = last_row.copy()

    for step in range(1, horizon + 1):
        next_date = _next_month(last_date_str, step)
        target_month = int(next_date[4:])

        current_row = _advance_features(current_row, next_date)

        # Inject seasonal weather
        if monthly_weather is not None and target_month in monthly_weather.index:
            for feat in seasonal_feats:
                current_row[feat] = monthly_weather.loc[target_month, feat]

        # --- Update ALL PAC-derived features from pac_buffer ---
        # buffer[-1] = pac at (t + step - 1), i.e. the month just before
        # the one we are predicting, so features are consistent with training.
        n = len(pac_buffer)

        # Direct lag features
        for lag_col, lag_months in [
            ("nb_installations_pac_lag_1m", 1),
            ("nb_installations_pac_lag_3m", 3),
            ("nb_installations_pac_lag_6m", 6),
            ("nb_installations_pac_lag_12m", 12),
        ]:
            if lag_col in current_row.index and n >= lag_months:
                current_row[lag_col] = float(pac_buffer[-lag_months])

        # Rolling mean and std (windows 3 and 6)
        for window in (3, 6):
            slice_ = pac_buffer[-window:] if n >= window else pac_buffer[:]
            mean_col = f"nb_installations_pac_rmean_{window}m"
            std_col = f"nb_installations_pac_rstd_{window}m"
            if mean_col in current_row.index:
                current_row[mean_col] = float(np.mean(slice_))
            if std_col in current_row.index:
                current_row[std_col] = float(np.std(slice_)) if len(slice_) >= 2 else 0.0

        # Absolute diff and pct change vs previous month
        if n >= 2:
            diff_col = "nb_installations_pac_diff_1m"
            pct_col = "nb_installations_pac_pct_1m"
            delta = pac_buffer[-1] - pac_buffer[-2]
            if diff_col in current_row.index:
                current_row[diff_col] = float(delta)
            if pct_col in current_row.index:
                prev = pac_buffer[-2]
                pct = float(delta / prev * 100) if prev > 0 else 0.0
                current_row[pct_col] = float(np.clip(pct, -200, 500))

        # --- Update nb_dpe_classe_ab lag/rolling features from dpe_ab_buffer ---
        if dpe_ab_buffer:
            nb = len(dpe_ab_buffer)
            for lag_col, lag_months in [
                ("nb_dpe_classe_ab_lag_1m", 1),
                ("nb_dpe_classe_ab_lag_3m", 3),
                ("nb_dpe_classe_ab_lag_6m", 6),
            ]:
                if lag_col in current_row.index and nb >= lag_months:
                    current_row[lag_col] = float(dpe_ab_buffer[-lag_months])
            for window in (3, 6):
                ab_slice = dpe_ab_buffer[-window:] if nb >= window else dpe_ab_buffer[:]
                m_col = f"nb_dpe_classe_ab_rmean_{window}m"
                s_col = f"nb_dpe_classe_ab_rstd_{window}m"
                if m_col in current_row.index:
                    current_row[m_col] = float(np.mean(ab_slice))
                if s_col in current_row.index:
                    current_row[s_col] = float(np.std(ab_slice)) if len(ab_slice) >= 2 else 0.0

        # --- Predict ---
        X = current_row[feature_names].values.reshape(1, -1).astype(float)
        X = imputer.transform(X)

        if model_name == "lightgbm":
            pred = float(model.predict(X)[0])
        else:
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])

        pred = max(pred, 0.0)
        margin = 1.96 * rmse
        year = int(next_date[:4])
        month = int(next_date[4:])

        results.append({
            "date": dt(year=year, month=month, day=1),
            "predicted": round(pred, 1),
            "lower": round(max(pred - margin, 0), 1),
            "upper": round(pred + margin, 1),
        })

        # Extend buffers with seasonal proxies for the next step's features.
        pac_seasonal = float(monthly_pac.get(target_month, hist_mean)) * seasonal_scale
        pac_buffer.append(pac_seasonal)
        if len(pac_buffer) > 36:
            pac_buffer = pac_buffer[-36:]

        if dpe_ab_buffer:
            ab_seasonal = float(monthly_dpe_ab.get(target_month, dpe_ab_hist_mean)) * dpe_ab_scale
            dpe_ab_buffer.append(ab_seasonal)
            if len(dpe_ab_buffer) > 36:
                dpe_ab_buffer = dpe_ab_buffer[-36:]

    return results


def _get_rmse(artifacts, model_name):
    """Get test RMSE for a model."""
    tr = artifacts.get("training_results")
    if tr is not None:
        row = tr[tr["model"] == model_name]
        if not row.empty:
            val = row.iloc[0].get("test_rmse", np.nan)
            if not np.isnan(val):
                return float(val)
    return 10.0


def _get_r2(artifacts, model_name):
    """Get test R2 for a model."""
    tr = artifacts.get("training_results")
    if tr is not None:
        row = tr[tr["model"] == model_name]
        if not row.empty:
            val = row.iloc[0].get("test_r2", np.nan)
            if not np.isnan(val):
                return float(val)
    return 0.0


def _next_month(base_yyyymm, offset):
    """Calculate YYYYMM date after offset months."""
    year = int(base_yyyymm[:4])
    month = int(base_yyyymm[4:])
    total = (year * 12 + month - 1) + offset
    return f"{total // 12}{total % 12 + 1:02d}"


def _advance_features(row, new_date):
    """Update temporal columns for the new date."""
    row = row.copy()
    year = int(new_date[:4])
    month = int(new_date[4:])
    row["year"] = year
    row["month"] = month
    row["quarter"] = (month - 1) // 3 + 1
    row["is_heating"] = int(month in (1, 2, 3, 10, 11, 12))
    row["is_cooling"] = int(month in (6, 7, 8))
    row["month_sin"] = round(np.sin(2 * np.pi * month / 12), 3)
    row["month_cos"] = round(np.cos(2 * np.pi * month / 12), 3)
    row["year_trend"] = (year - 2020) + month / 12
    return row


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render():
    st.markdown(
        "<h1 style='text-align: center;'>Predictions PAC</h1>"
        "<p style='text-align: center; color: gray;'>"
        "Heat pump installation forecast by department</p>",
        unsafe_allow_html=True,
    )

    # Load data and models
    artifacts = load_models()
    df = load_features()

    if df is None:
        st.error("Features dataset not found.")
        st.code("python -m src.pipeline features", language="bash")
        return

    if "ridge" not in artifacts:
        st.error("No trained model found.")
        st.code("python -m src.pipeline train", language="bash")
        return

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Prediction settings")

    dept_options = get_department_options(df)
    default_idx = (
        list(dept_options.keys()).index("69 — Rhone")
        if "69 — Rhone" in dept_options else 0
    )
    dept_label = st.sidebar.selectbox("Department", list(dept_options.keys()), index=default_idx)
    dept_code = dept_options[dept_label]

    # Model selection
    available_models = []
    model_keys = []
    if "lightgbm" in artifacts:
        available_models.append("LightGBM")
        model_keys.append("lightgbm")
    available_models.append("Ridge")
    model_keys.append("ridge")

    model_choice = st.sidebar.radio("Model", available_models, index=0)
    model_name = model_keys[available_models.index(model_choice)]

    horizon = st.sidebar.slider("Forecast horizon (months)", 1, 24, 12)

    # --- Historical data for selected department ---
    df_dept = df[df["dept"] == dept_code].sort_values("date_id").copy()
    # Convert date_id (int or float like 202401.0) to proper Python datetimes
    # Using native datetime to avoid pandas Timestamp deprecation warnings with Plotly
    date_str = df_dept["date_id"].astype(int).astype(str)
    df_dept["date"] = [
        dt(int(s[:4]), int(s[4:]), 1) for s in date_str
    ]

    # =====================================================================
    # KPI cards row
    # =====================================================================
    r2 = _get_r2(artifacts, model_name)
    rmse = _get_rmse(artifacts, model_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Monthly mean", f"{df_dept['nb_installations_pac'].mean():.0f}")
    with col2:
        last_val = df_dept["nb_installations_pac"].iloc[-1]
        prev_val = df_dept["nb_installations_pac"].iloc[-2] if len(df_dept) > 1 else last_val
        delta = last_val - prev_val
        st.metric("Last month", f"{last_val:.0f}", delta=f"{delta:+.0f}")
    with col3:
        if len(df_dept) >= 12:
            recent = df_dept["nb_installations_pac"].tail(6).mean()
            previous = df_dept["nb_installations_pac"].iloc[-12:-6].mean()
            trend_pct = (recent - previous) / previous * 100 if previous > 0 else 0
            st.metric("6-month trend", f"{trend_pct:+.1f}%")
        else:
            st.metric("6-month trend", "N/A")
    with col4:
        st.metric("Model R2", f"{r2:.3f}")
    with col5:
        st.metric("Model RMSE", f"{rmse:.1f}")

    st.markdown("---")

    # =====================================================================
    # Generate predictions
    # =====================================================================
    with st.spinner(f"Generating {horizon}-month forecast..."):
        predictions = predict_department(artifacts, df, dept_code, horizon, model_name)

    if not predictions:
        st.warning("Unable to generate predictions for this department.")
        return

    df_pred = pd.DataFrame(predictions)

    # Convert dates to ISO strings to avoid pandas Timestamp arithmetic errors
    # (pandas auto-converts datetime to Timestamp in DataFrames; Plotly handles
    # ISO strings natively and builds proper date axes from them)
    hist_dates = [d.strftime("%Y-%m-%d") for d in df_dept["date"]]
    pred_dates = [d.strftime("%Y-%m-%d") for d in df_pred["date"]]

    # =====================================================================
    # Main chart: Historical + Forecast
    # =====================================================================
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=df_dept["nb_installations_pac"].tolist(),
        mode="lines+markers",
        name="Historical data",
        line=dict(color="#1f77b4", width=2.5),
        marker=dict(size=3),
        hovertemplate="%{x|%b %Y}: <b>%{y:.0f}</b> PAC<extra></extra>",
    ))

    # Confidence interval band
    upper_vals = df_pred["upper"].tolist()
    lower_vals = df_pred["lower"].tolist()[::-1]

    fig.add_trace(go.Scatter(
        x=pred_dates + pred_dates[::-1],
        y=upper_vals + lower_vals,
        fill="toself",
        fillcolor="rgba(255, 127, 14, 0.12)",
        line=dict(color="rgba(255, 127, 14, 0)"),
        name="95% confidence",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=df_pred["predicted"].tolist(),
        mode="lines+markers",
        name=f"Forecast ({model_choice})",
        line=dict(color="#ff7f0e", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="%{x|%b %Y}: <b>%{y:.0f}</b> PAC<extra></extra>",
    ))

    # Connecting line between last historical point and first prediction
    fig.add_trace(go.Scatter(
        x=[hist_dates[-1], pred_dates[0]],
        y=[df_dept["nb_installations_pac"].iloc[-1], df_pred["predicted"].iloc[0]],
        mode="lines",
        line=dict(color="#aaa", width=1.5, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Vertical separator line (annotation added separately to avoid Plotly
    # _mean() bug with date strings in add_vline's annotation_text)
    fig.add_vline(
        x=hist_dates[-1],
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )
    fig.add_annotation(
        x=hist_dates[-1], y=1.06, yref="paper",
        text="Forecast start", showarrow=False,
        font=dict(size=11, color="gray"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{dept_label}</b> — Heat pump installations",
            font=dict(size=18),
        ),
        xaxis_title="",
        yaxis_title="Installations PAC / month",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            font=dict(size=11),
        ),
        height=520,
        margin=dict(l=60, r=30, t=80, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # Forecast summary
    # =====================================================================
    st.subheader("Forecast summary")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        total_pred = df_pred["predicted"].sum()
        st.metric(f"Total PAC ({horizon}m)", f"{total_pred:,.0f}")
    with col_b:
        st.metric("Monthly average", f"{df_pred['predicted'].mean():.1f}")
    with col_c:
        peak_date = pred_dates[df_pred["predicted"].idxmax()]
        st.metric("Peak month", pd.Timestamp(peak_date).strftime("%b %Y"))
    with col_d:
        min_date = pred_dates[df_pred["predicted"].idxmin()]
        st.metric("Min month", pd.Timestamp(min_date).strftime("%b %Y"))

    # =====================================================================
    # Two columns: Prediction table + Seasonal pattern
    # =====================================================================
    tab1, tab2 = st.tabs(["Forecast table", "Seasonal pattern"])

    with tab1:
        df_display = df_pred.copy()
        df_display["date"] = [d[:7] for d in pred_dates]  # "YYYY-MM"
        df_display.columns = ["Month", "Predicted PAC", "Lower bound (95%)", "Upper bound (95%)"]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    with tab2:
        # Monthly seasonality for this department
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        seasonal = df_dept.groupby("month")["nb_installations_pac"].agg(["mean", "std"]).reset_index()
        seasonal["month_name"] = seasonal["month"].map(
            lambda m: month_labels[int(m) - 1]
        )

        fig_season = go.Figure()
        fig_season.add_trace(go.Bar(
            x=seasonal["month_name"],
            y=seasonal["mean"],
            error_y=dict(type="data", array=seasonal["std"].fillna(0), visible=True),
            marker_color=[
                "#4e79a7" if m in (1, 2, 3, 10, 11, 12) else "#e15759"
                for m in seasonal["month"]
            ],
            hovertemplate="%{x}: <b>%{y:.0f}</b> PAC (+/-%{error_y.array:.0f})<extra></extra>",
        ))
        fig_season.update_layout(
            title=f"<b>Seasonal pattern</b> — {dept_label}",
            xaxis_title="",
            yaxis_title="Mean PAC / month",
            height=380,
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )
        fig_season.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text="<span style='color:#4e79a7'>Heating season</span> | "
                 "<span style='color:#e15759'>Cooling season</span>",
            showarrow=False, font=dict(size=11),
        )
        st.plotly_chart(fig_season, use_container_width=True)

    # =====================================================================
    # Model comparison table
    # =====================================================================
    st.markdown("---")
    with st.expander("Model performance details"):
        tr = artifacts.get("training_results")
        if tr is not None:
            display_cols = [c for c in ["model", "val_rmse", "test_rmse", "val_r2", "test_r2",
                                         "val_mae", "test_mae"] if c in tr.columns]
            df_tr = tr[display_cols].copy()
            col_map = {
                "model": "Model", "val_rmse": "Val RMSE", "test_rmse": "Test RMSE",
                "val_r2": "Val R2", "test_r2": "Test R2",
                "val_mae": "Val MAE", "test_mae": "Test MAE",
            }
            df_tr = df_tr.rename(columns=col_map)
            st.dataframe(df_tr, use_container_width=True, hide_index=True)

            # Mini bar chart
            if "Test R2" in df_tr.columns:
                fig_comp = px.bar(
                    df_tr, x="Model", y="Test R2",
                    color="Model",
                    color_discrete_sequence=["#4e79a7", "#59a14f", "#e15759", "#f28e2b"],
                    title="Test R2 by model",
                    height=300,
                )
                fig_comp.update_layout(
                    showlegend=False,
                    margin=dict(l=40, r=20, t=40, b=30),
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                )
                st.plotly_chart(fig_comp, use_container_width=True)
