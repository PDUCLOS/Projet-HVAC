# -*- coding: utf-8 -*-
"""ML model comparison page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json


def render():
    st.title("Model comparison")

    from config.settings import config as cfg
    eval_dir = cfg.models_dir

    # --- Evaluation report ---
    report_path = eval_dir / "evaluation_report.txt"
    results_path = cfg.evaluation_results_path

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        # Build the comparison table
        rows = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict):
                row = {"Model": model_name}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        row[k] = round(v, 4) if isinstance(v, float) else v
                rows.append(row)

        if rows:
            df_compare = pd.DataFrame(rows)
            st.subheader("Comparison table")
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

            # --- Radar chart ---
            metric_cols = [c for c in df_compare.columns if c != "Model"]
            if len(metric_cols) >= 3:
                st.subheader("Radar chart")
                fig = go.Figure()
                for _, row in df_compare.iterrows():
                    values = [row[c] for c in metric_cols]
                    # Normalize for the radar
                    max_vals = df_compare[metric_cols].max()
                    normalized = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
                    fig.add_trace(go.Scatterpolar(
                        r=normalized + [normalized[0]],
                        theta=metric_cols + [metric_cols[0]],
                        name=row["Model"],
                        fill="toself",
                        opacity=0.6,
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
                    title="Normalized model comparison",
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- Ranking ---
            st.subheader("Ranking")
            ranking_metric = st.selectbox("Sort by", metric_cols)
            # For MAE/RMSE: smaller = better; for R2: larger = better
            ascending = "r2" not in ranking_metric.lower()
            df_ranked = df_compare.sort_values(ranking_metric, ascending=ascending)
            df_ranked.insert(0, "Rang", range(1, len(df_ranked) + 1))

            st.dataframe(df_ranked, use_container_width=True, hide_index=True)

            best_model = df_ranked.iloc[0]["Model"]
            best_value = df_ranked.iloc[0][ranking_metric]
            st.success(f"Best model: **{best_model}** ({ranking_metric} = {best_value})")

    elif report_path.exists():
        st.subheader("Evaluation report")
        st.text(report_path.read_text())
    else:
        st.warning("No evaluation results available.")
        st.code("python -m src.pipeline train && python -m src.pipeline evaluate", language="bash")
        return

    # --- Available models ---
    st.subheader("Saved models")
    model_files = list(eval_dir.glob("*.pkl")) + list(eval_dir.glob("*.joblib"))
    if model_files:
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            st.text(f"  {f.name:<40} {size_mb:.1f} Mo")
    else:
        st.info("No saved models.")

    # --- Outlier Detection Report ---
    outlier_report = cfg.outlier_report_path
    if outlier_report.exists():
        st.subheader("Outlier detection report")
        with open(outlier_report) as f:
            report = json.load(f)

        if "summary" in report:
            summary = report["summary"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Columns analyzed", summary.get("n_columns_analyzed", "N/A"))
            col2.metric("Outliers detected", summary.get("total_outliers_flagged", "N/A"))
            col3.metric("Outlier rate", f"{summary.get('outlier_rate_pct', 0):.1f}%")

        if "columns" in report:
            for col_name, col_info in report["columns"].items():
                with st.expander(f"Column: {col_name}"):
                    st.json(col_info)
