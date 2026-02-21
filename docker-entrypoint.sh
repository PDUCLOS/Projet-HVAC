#!/bin/bash
# =============================================================================
# HVAC Market Analysis — Point d'entree Docker
# =============================================================================
# Modes disponibles :
#   all        — API + Dashboard (defaut)
#   api        — API FastAPI uniquement
#   dashboard  — Dashboard Streamlit uniquement
#   pipeline   — Executer le pipeline puis quitter
#   demo       — Generer donnees demo + pipeline + lancer all
# =============================================================================

set -e

MODE=${1:-all}

echo "============================================="
echo "  HVAC Market Analysis"
echo "  Mode : $MODE"
echo "============================================="

case "$MODE" in
    api)
        echo "  Lancement de l'API FastAPI sur :8000..."
        exec uvicorn api.main:app --host 0.0.0.0 --port 8000
        ;;

    dashboard)
        echo "  Lancement du Dashboard Streamlit sur :8501..."
        exec streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;

    all)
        echo "  Lancement API (8000) + Dashboard (8501)..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        exec streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;

    pipeline)
        echo "  Execution du pipeline complet..."
        python -m src.pipeline process
        python -m src.pipeline train
        python -m src.pipeline evaluate
        echo "  Pipeline termine."
        ;;

    demo)
        echo "  Generation des donnees de demo..."
        python scripts/generate_demo_data.py
        echo "  Execution du pipeline..."
        python -m src.pipeline process
        python -m src.pipeline train
        python -m src.pipeline evaluate
        echo "  Lancement API (8000) + Dashboard (8501)..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        exec streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;

    *)
        echo "  Mode inconnu : $MODE"
        echo "  Modes disponibles : all, api, dashboard, pipeline, demo"
        exit 1
        ;;
esac
