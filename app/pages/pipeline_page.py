# -*- coding: utf-8 -*-
"""Pipeline management and update page."""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys


def render():
    st.title("Pipeline & Update")

    # --- Current state ---
    st.header("Data status")
    _display_data_status()

    # --- Launch pipeline stages ---
    st.header("Run a pipeline stage")

    stages = {
        "update_all": "Full update (collect + process + train + upload)",
        "collect": "Data collection (APIs)",
        "process": "Processing (clean + merge + features + outliers)",
        "train": "ML training",
        "evaluate": "Model evaluation",
        "upload_pcloud": "Upload to pCloud",
        "sync_pcloud": "Sync from pCloud",
        "init_db": "Database initialization",
        "eda": "Exploratory analysis",
    }

    selected_stage = st.selectbox("Stage", list(stages.keys()), format_func=lambda x: f"{x} — {stages[x]}")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Run", type="primary"):
            _run_pipeline_stage(selected_stage)
    with col2:
        st.caption(f"Command: `python -m src.pipeline {selected_stage}`")

    # --- pCloud synchronization state ---
    st.header("pCloud synchronization")
    from config.settings import config as cfg
    sync_state_file = cfg.pcloud_sync_state_path
    if sync_state_file.exists():
        with open(sync_state_file) as f:
            sync_state = json.load(f)

        sync_data = []
        for filename, info in sync_state.items():
            sync_data.append({
                "File": filename,
                "Last sync": info.get("last_sync", "N/A"),
                "Hash": info.get("hash", "N/A"),
                "Size": info.get("size", 0),
            })

        if sync_data:
            st.dataframe(pd.DataFrame(sync_data), use_container_width=True, hide_index=True)
        else:
            st.info("No synchronized files.")
    else:
        st.info("Never synchronized with pCloud.")

    # --- Configuration ---

    st.header("Current configuration")
    try:
        from config.settings import config
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Geography**")
            st.text(f"Region: {config.geo.region_code}")
            st.text(f"Departments: {len(config.geo.departments)}")
            st.text(f"Cities: {len(config.geo.cities)}")

        with col2:
            st.markdown("**Temporal**")
            st.text(f"Start: {config.time.start_date}")
            st.text(f"End: {config.time.end_date}")
            st.text(f"Train/Test split: {config.time.train_end}")
    except Exception as e:
        st.error(f"Config error: {e}")

    # --- Useful commands ---
    st.header("Useful commands")
    st.code(
        """
# Full update
python -m src.pipeline update_all

# Specific collection
python -m src.pipeline collect --sources weather,dpe

# With detailed logging
python -m src.pipeline process --log-level DEBUG

# Launch the dashboard
streamlit run app/app.py
        """,
        language="bash",
    )


def _display_data_status():
    """Display the data status."""
    from config.settings import config as cfg
    raw_dir = cfg.raw_data_dir
    processed_dir = cfg.processed_data_dir

    status = []

    files_to_check = {
        "Weather (raw)": raw_dir / "weather" / "weather_france.csv",
        "Weather (clean)": processed_dir / "weather" / "weather_france.csv",
        "DPE (raw)": raw_dir / "dpe" / "dpe_france_all.csv",
        "DPE (clean)": processed_dir / "dpe" / "dpe_france_clean.csv",
        "INSEE": raw_dir / "insee" / "indicateurs_economiques.csv",
        "Eurostat": raw_dir / "eurostat" / "ipi_hvac_france.csv",
        "SITADEL": raw_dir / "sitadel" / "permis_construire_france.csv",
        "ML Dataset": cfg.ml_dataset_path,
        "Features Dataset": cfg.features_dataset_path,
        "SQLite DB": cfg.db_path,
    }

    for name, path in files_to_check.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            status.append({
                "Data": name,
                "Status": "OK",
                "Size": f"{size_mb:.1f} MB",
                "Last modified": mod_time,
            })
        else:
            status.append({
                "Data": name,
                "Status": "Missing",
                "Size": "-",
                "Last modified": "-",
            })

    df_status = pd.DataFrame(status)
    st.dataframe(df_status, use_container_width=True, hide_index=True)

    # Count
    ok_count = sum(1 for s in status if s["Status"] == "OK")
    total = len(status)
    if ok_count == total:
        st.success(f"All data is available ({ok_count}/{total})")
    elif ok_count > 0:
        st.warning(f"{ok_count}/{total} sources available")
    else:
        st.error("No data available. Run the collection.")


def _run_pipeline_stage(stage: str):
    """Run a pipeline stage via subprocess."""
    with st.spinner(f"Running '{stage}'..."):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.pipeline", stage],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
            )

            if result.returncode == 0:
                st.success(f"Stage '{stage}' completed successfully.")
                if result.stdout:
                    with st.expander("Logs"):
                        st.text(result.stdout[-5000:])
            else:
                st.error(f"Stage '{stage}' failed (code {result.returncode}).")
                if result.stderr:
                    with st.expander("Errors"):
                        st.text(result.stderr[-3000:])

        except subprocess.TimeoutExpired:
            st.error("Timeout: the stage took more than 10 minutes.")
        except Exception as e:
            st.error(f"Error: {e}")
