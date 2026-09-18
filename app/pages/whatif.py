# -*- coding: utf-8 -*-
"""What-If Simulator page — scenario analysis for HVAC installations."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime as dt
from pathlib import Path
import pickle


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_ridge_model():
    """Load the Ridge model, scaler, and imputer from disk."""
    models_dir = Path(__file__).resolve().parent.parent.parent / "data" / "models"
    artifacts = {}

    for name, filename in [
        ("ridge", "ridge_model.pkl"),
        ("scaler", "scaler.pkl"),
        ("imputer", "imputer.pkl"),
    ]:
        path = models_dir / filename
        if path.exists():
            with open(path, "rb") as f:
                artifacts[name] = pickle.load(f)  # noqa: S301 — trusted local files

    # Training results for RMSE
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
    """Build department dropdown options: '69 -- Rhone'."""
    from config.settings import DEPT_NAMES

    depts = sorted(df["dept"].unique())
    options = {}
    for d in depts:
        name = DEPT_NAMES.get(d, d)
        options[f"{d} -- {name}"] = d
    return options


def _get_rmse(artifacts, model_name="ridge"):
    """Get test RMSE for a model from training results."""
    tr = artifacts.get("training_results")
    if tr is not None:
        row = tr[tr["model"] == model_name]
        if not row.empty:
            val = row.iloc[0].get("test_rmse", np.nan)
            if not np.isnan(val):
                return float(val)
    return 10.0


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


def predict_scenario(artifacts, df, dept, horizon, adjustments=None):
    """Generate baseline and adjusted predictions for a department.

    Args:
        artifacts: dict with ridge model, scaler, imputer, training_results.
        df: features DataFrame.
        dept: department code.
        horizon: number of months to predict.
        adjustments: dict of {feature_name: multiplier} to apply.

    Returns:
        Tuple of (baseline_results, adjusted_results) where each is a list
        of dicts with 'date' and 'predicted' keys.
    """
    ridge = artifacts.get("ridge")
    imputer = artifacts.get("imputer")
    scaler = artifacts.get("scaler")

    if ridge is None or imputer is None or scaler is None:
        return [], []

    feature_names = list(ridge.feature_names_in_)
    df_dept = df[df["dept"] == dept].sort_values("date_id").reset_index(drop=True)
    if df_dept.empty:
        return [], []

    last_row = df_dept.iloc[-1].copy()
    last_date_str = str(int(last_row["date_id"]))
    rmse = _get_rmse(artifacts)

    baseline_results = []
    adjusted_results = []

    for scenario_adjustments in [None, adjustments]:
        results = []
        current_row = last_row.copy()

        for step in range(1, horizon + 1):
            next_date = _next_month(last_date_str, step)
            current_row = _advance_features(current_row, next_date)

            # Apply adjustments (multipliers) for the adjusted scenario
            if scenario_adjustments:
                for feat, multiplier in scenario_adjustments.items():
                    if feat in current_row.index:
                        current_row[feat] = float(current_row[feat]) * multiplier

            X = current_row[feature_names].values.reshape(1, -1).astype(float)
            X = imputer.transform(X)
            X_scaled = scaler.transform(X)
            pred = float(ridge.predict(X_scaled)[0])
            pred = max(pred, 0.0)

            year = int(next_date[:4])
            month = int(next_date[4:])
            results.append({
                "date": dt(year=year, month=month, day=1),
                "predicted": round(pred, 1),
            })

            # Re-inject prediction into lag for next step
            if "nb_installations_pac_lag_1m" in current_row.index:
                current_row["nb_installations_pac_lag_1m"] = pred

        if scenario_adjustments is None:
            baseline_results = results
        else:
            adjusted_results = results

    # If no adjustments provided, adjusted == baseline
    if not adjustments:
        adjusted_results = baseline_results.copy()

    return baseline_results, adjusted_results


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render():
    """Render the What-If Simulator page."""
    st.markdown(
        "<h1 style='text-align: center;'>What-If Simulator</h1>"
        "<p style='text-align: center; color: gray;'>"
        "Explore how changes in temperature, economic conditions, "
        "and construction activity affect heat pump installations</p>",
        unsafe_allow_html=True,
    )

    # Load data and models
    artifacts = load_ridge_model()
    df = load_features()

    if df is None:
        st.error(
            "Features dataset not found. "
            "Run the pipeline to generate it first."
        )
        st.code("python -m src.pipeline features", language="bash")
        return

    if "ridge" not in artifacts:
        st.error(
            "No trained Ridge model found. "
            "Run the training pipeline first."
        )
        st.code("python -m src.pipeline train", language="bash")
        return

    # --- Sidebar controls ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Scenario Parameters")

    # Department selector
    dept_options = get_department_options(df)
    default_idx = (
        list(dept_options.keys()).index("69 -- Rhone")
        if "69 -- Rhone" in dept_options else 0
    )
    dept_label = st.sidebar.selectbox(
        "Department",
        list(dept_options.keys()),
        index=default_idx,
        key="whatif_dept",
    )
    dept_code = dept_options[dept_label]

    # Forecast horizon
    horizon = st.sidebar.slider(
        "Forecast horizon (months)", 1, 24, 12, key="whatif_horizon"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Adjustment Sliders")

    # Temperature change slider (-5 to +5 degrees -> multiplier)
    temp_change = st.sidebar.slider(
        "Temperature change (C)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
        key="whatif_temp",
        help="Shift the average temperature. Positive = warmer.",
    )

    # Economic confidence change (-20% to +20%)
    econ_change = st.sidebar.slider(
        "Economic confidence change (%)",
        min_value=-20,
        max_value=20,
        value=0,
        step=1,
        key="whatif_econ",
        help="Change in consumer/business confidence index.",
    )

    # Construction permits change (-50% to +50%)
    construction_change = st.sidebar.slider(
        "Construction permits change (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        key="whatif_construction",
        help="Change in new construction activity (SITADEL permits).",
    )

    # --- Build adjustments dict ---
    adjustments = {}

    # Temperature: apply as an additive offset
    # We need to figure out which temp-related features are in the model
    feature_names = list(artifacts["ridge"].feature_names_in_)
    df_dept = df[df["dept"] == dept_code].sort_values("date_id")

    if not df_dept.empty and temp_change != 0.0:
        last_row = df_dept.iloc[-1]
        # Temperature features: adjust as additive offset converted to multiplier
        temp_features = [f for f in feature_names if "temp_mean" in f.lower()]
        for feat in temp_features:
            current_val = float(last_row.get(feat, 0))
            if current_val != 0:
                adjustments[feat] = (current_val + temp_change) / current_val

        # HDD/CDD: modify inversely/proportionally with temperature
        hdd_features = [f for f in feature_names if "hdd" in f.lower()]
        for feat in hdd_features:
            # Warmer = less HDD
            current_val = float(last_row.get(feat, 0))
            if current_val > 0:
                # Approximate: each +1C reduces HDD by ~30 per month
                new_val = max(current_val - temp_change * 30, 0)
                adjustments[feat] = new_val / current_val if current_val > 0 else 1.0

        cdd_features = [f for f in feature_names if "cdd" in f.lower()]
        for feat in cdd_features:
            # Warmer = more CDD
            current_val = float(last_row.get(feat, 0))
            if current_val > 0:
                new_val = max(current_val + temp_change * 30, 0)
                adjustments[feat] = new_val / current_val

    if econ_change != 0:
        econ_multiplier = 1.0 + econ_change / 100.0
        econ_features = [
            f for f in feature_names
            if any(k in f.lower() for k in ("confiance", "climat_affaires", "ipi"))
        ]
        for feat in econ_features:
            adjustments[feat] = econ_multiplier

    if construction_change != 0:
        construction_multiplier = 1.0 + construction_change / 100.0
        construction_features = [
            f for f in feature_names
            if any(k in f.lower() for k in ("sitadel", "logements", "permis"))
        ]
        for feat in construction_features:
            adjustments[feat] = construction_multiplier

    # --- Run Simulation button ---
    has_adjustments = temp_change != 0.0 or econ_change != 0 or construction_change != 0

    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running scenario simulation..."):
            baseline, adjusted = predict_scenario(
                artifacts, df, dept_code, horizon,
                adjustments if has_adjustments else None,
            )

        if not baseline:
            st.warning(
                "Unable to generate predictions for this department. "
                "The dataset may not contain enough data."
            )
            return

        df_baseline = pd.DataFrame(baseline)
        df_adjusted = pd.DataFrame(adjusted)

        # Convert dates to strings for Plotly
        baseline_dates = [d.strftime("%Y-%m-%d") for d in df_baseline["date"]]
        adjusted_dates = [d.strftime("%Y-%m-%d") for d in df_adjusted["date"]]

        # --- Impact summary ---
        baseline_total = df_baseline["predicted"].sum()
        adjusted_total = df_adjusted["predicted"].sum()
        if baseline_total > 0:
            impact_pct = (adjusted_total - baseline_total) / baseline_total * 100
        else:
            impact_pct = 0.0

        st.markdown("---")
        st.subheader("Impact Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                f"Baseline total ({horizon}m)",
                f"{baseline_total:,.0f}",
            )
        with col2:
            st.metric(
                f"Adjusted total ({horizon}m)",
                f"{adjusted_total:,.0f}",
                delta=f"{adjusted_total - baseline_total:+,.0f}",
            )
        with col3:
            st.metric(
                "Impact",
                f"{impact_pct:+.1f}%",
            )
        with col4:
            st.metric(
                "Adjustments applied",
                f"{len(adjustments)} features",
            )

        st.markdown("---")

        # --- Chart: Baseline vs Adjusted ---
        fig = go.Figure()

        # Baseline
        fig.add_trace(go.Scatter(
            x=baseline_dates,
            y=df_baseline["predicted"].tolist(),
            mode="lines+markers",
            name="Baseline",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=5),
            hovertemplate="%{x|%b %Y}: <b>%{y:.0f}</b> PAC<extra>Baseline</extra>",
        ))

        # Adjusted (only show if different)
        if has_adjustments:
            fig.add_trace(go.Scatter(
                x=adjusted_dates,
                y=df_adjusted["predicted"].tolist(),
                mode="lines+markers",
                name="Adjusted scenario",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
                hovertemplate="%{x|%b %Y}: <b>%{y:.0f}</b> PAC<extra>Adjusted</extra>",
            ))

            # Fill between baseline and adjusted
            fig.add_trace(go.Scatter(
                x=baseline_dates + adjusted_dates[::-1],
                y=(df_baseline["predicted"].tolist()
                   + df_adjusted["predicted"].tolist()[::-1]),
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{dept_label}</b> -- "
                    f"What-If Scenario ({horizon} months)"
                ),
                font=dict(size=18),
            ),
            xaxis_title="",
            yaxis_title="Predicted PAC installations / month",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="center", x=0.5,
                font=dict(size=11),
            ),
            height=520,
            margin=dict(l=60, r=30, t=80, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Details table ---
        with st.expander("Detailed predictions"):
            df_compare = pd.DataFrame({
                "Month": [d[:7] for d in baseline_dates],
                "Baseline": df_baseline["predicted"].tolist(),
                "Adjusted": df_adjusted["predicted"].tolist(),
                "Difference": (
                    df_adjusted["predicted"] - df_baseline["predicted"]
                ).round(1).tolist(),
                "Change (%)": [
                    round((a - b) / b * 100, 1) if b > 0 else 0.0
                    for a, b in zip(
                        df_adjusted["predicted"], df_baseline["predicted"]
                    )
                ],
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # --- Scenario description ---
        with st.expander("Scenario details"):
            if temp_change != 0:
                st.write(f"- Temperature change: **{temp_change:+.1f} C**")
            if econ_change != 0:
                st.write(
                    f"- Economic confidence change: **{econ_change:+d}%**"
                )
            if construction_change != 0:
                st.write(
                    f"- Construction permits change: **{construction_change:+d}%**"
                )
            if adjustments:
                st.write(f"- Total features adjusted: **{len(adjustments)}**")
                st.write("- Affected features:")
                for feat, mult in sorted(adjustments.items()):
                    st.write(f"  - `{feat}`: x{mult:.3f}")
            else:
                st.info("No adjustments applied. Move the sliders and click 'Run Simulation'.")

    else:
        st.info(
            "Adjust the scenario parameters in the sidebar, "
            "then click **Run Simulation** to see the results."
        )
