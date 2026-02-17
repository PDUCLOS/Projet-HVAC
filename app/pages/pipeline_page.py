# -*- coding: utf-8 -*-
"""Page de gestion du pipeline et mise a jour."""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys


def render():
    st.title("Pipeline & Mise a jour")

    # --- Etat actuel ---
    st.header("Etat des donnees")
    _display_data_status()

    # --- Lancement des etapes ---
    st.header("Lancer une etape du pipeline")

    stages = {
        "update_all": "Mise a jour complete (collect + process + train + upload)",
        "collect": "Collecte des donnees (APIs)",
        "process": "Processing (clean + merge + features + outliers)",
        "train": "Entrainement ML",
        "evaluate": "Evaluation des modeles",
        "upload_pcloud": "Upload vers pCloud",
        "sync_pcloud": "Synchronisation depuis pCloud",
        "init_db": "Initialisation de la base de donnees",
        "eda": "Analyse exploratoire",
    }

    selected_stage = st.selectbox("Etape", list(stages.keys()), format_func=lambda x: f"{x} — {stages[x]}")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Lancer", type="primary"):
            _run_pipeline_stage(selected_stage)
    with col2:
        st.caption(f"Commande : `python -m src.pipeline {selected_stage}`")

    # --- Etat de synchronisation pCloud ---
    st.header("Synchronisation pCloud")
    sync_state_file = Path("data/.pcloud_sync_state.json")
    if sync_state_file.exists():
        with open(sync_state_file) as f:
            sync_state = json.load(f)

        sync_data = []
        for filename, info in sync_state.items():
            sync_data.append({
                "Fichier": filename,
                "Derniere sync": info.get("last_sync", "N/A"),
                "Hash": info.get("hash", "N/A"),
                "Taille": info.get("size", 0),
            })

        if sync_data:
            st.dataframe(pd.DataFrame(sync_data), use_container_width=True, hide_index=True)
        else:
            st.info("Aucun fichier synchronise.")
    else:
        st.info("Jamais synchronise avec pCloud.")

    # --- Configuration ---
    st.header("Configuration actuelle")
    try:
        from config.settings import config
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Geographie**")
            st.text(f"Region : {config.geo.region_code}")
            st.text(f"Departements : {len(config.geo.departments)}")
            st.text(f"Villes : {len(config.geo.cities)}")

        with col2:
            st.markdown("**Temporel**")
            st.text(f"Debut : {config.time.start_date}")
            st.text(f"Fin : {config.time.end_date}")
            st.text(f"Train/Test split : {config.time.train_end}")
    except Exception as e:
        st.error(f"Erreur config : {e}")

    # --- Logs ---
    st.header("Commandes utiles")
    st.code(
        """
# Mise a jour complete
python -m src.pipeline update_all

# Collecte specifique
python -m src.pipeline collect --sources weather,dpe

# Avec logging detaille
python -m src.pipeline process --log-level DEBUG

# Lancer le dashboard
streamlit run app/app.py
        """,
        language="bash",
    )


def _display_data_status():
    """Affiche l'etat des donnees."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    features_dir = Path("data/features")
    models_dir = Path("data/models")

    status = []

    files_to_check = {
        "Meteo (brut)": raw_dir / "weather" / "weather_france.csv",
        "Meteo (clean)": processed_dir / "weather" / "weather_france.csv",
        "DPE (brut)": raw_dir / "dpe" / "dpe_france_all.csv",
        "DPE (clean)": processed_dir / "dpe" / "dpe_france_clean.csv",
        "INSEE": raw_dir / "insee" / "indicateurs_economiques.csv",
        "Eurostat": raw_dir / "eurostat" / "ipi_hvac_france.csv",
        "SITADEL": raw_dir / "sitadel" / "permis_construire_france.csv",
        "Dataset ML": features_dir / "hvac_ml_dataset.csv",
        "Dataset Features": features_dir / "hvac_features_dataset.csv",
        "BDD SQLite": Path("data/hvac_market.db"),
    }

    for name, path in files_to_check.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            status.append({
                "Donnee": name,
                "Statut": "OK",
                "Taille": f"{size_mb:.1f} Mo",
                "Derniere modif": mod_time,
            })
        else:
            status.append({
                "Donnee": name,
                "Statut": "Manquant",
                "Taille": "-",
                "Derniere modif": "-",
            })

    df_status = pd.DataFrame(status)
    st.dataframe(df_status, use_container_width=True, hide_index=True)

    # Compter
    ok_count = sum(1 for s in status if s["Statut"] == "OK")
    total = len(status)
    if ok_count == total:
        st.success(f"Toutes les donnees sont disponibles ({ok_count}/{total})")
    elif ok_count > 0:
        st.warning(f"{ok_count}/{total} sources disponibles")
    else:
        st.error("Aucune donnee disponible. Lancez la collecte.")


def _run_pipeline_stage(stage: str):
    """Lance une etape du pipeline via subprocess."""
    with st.spinner(f"Execution de '{stage}'..."):
        try:
            result = subprocess.run(
                [sys.executable, "-m", "src.pipeline", stage],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
            )

            if result.returncode == 0:
                st.success(f"Etape '{stage}' terminee avec succes.")
                if result.stdout:
                    with st.expander("Logs"):
                        st.text(result.stdout[-5000:])
            else:
                st.error(f"Etape '{stage}' echouee (code {result.returncode}).")
                if result.stderr:
                    with st.expander("Erreurs"):
                        st.text(result.stderr[-3000:])

        except subprocess.TimeoutExpired:
            st.error("Timeout : l'etape a pris plus de 10 minutes.")
        except Exception as e:
            st.error(f"Erreur : {e}")
