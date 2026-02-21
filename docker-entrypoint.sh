#!/bin/bash
# =============================================================================
# HVAC Market Analysis — Docker Entrypoint
# =============================================================================
# Available modes :
#   all        — API + Dashboard (default)
#   api        — FastAPI API only
#   dashboard  — Streamlit Dashboard only
#   pipeline   — Run the pipeline then exit
#   demo       — Generate demo data + pipeline + launch all
# =============================================================================

set -e

MODE=${1:-all}

echo "============================================="
echo "  HVAC Market Analysis"
echo "  Mode : $MODE"
echo "============================================="

case "$MODE" in
    api)
        echo "  Launching FastAPI API on :8000..."
        exec uvicorn api.main:app --host 0.0.0.0 --port 8000
        ;;

    dashboard)
        echo "  Launching Streamlit Dashboard on :8501..."
        exec streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;

    all)
        echo "  Launching API (8000) + Dashboard (8501)..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        exec streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;

    pipeline)
        echo "  Running the full pipeline..."
        python -m src.pipeline process
        python -m src.pipeline train
        python -m src.pipeline evaluate
        echo "  Pipeline completed."
        ;;

    demo)
        echo "  Generating demo data..."
        python scripts/generate_demo_data.py
        echo "  Running the pipeline..."
        python -m src.pipeline process
        python -m src.pipeline train
        python -m src.pipeline evaluate
        echo "  Launching API (8000) + Dashboard (8501)..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        exec streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;

    *)
        echo "  Unknown mode : $MODE"
        echo "  Available modes : all, api, dashboard, pipeline, demo"
        exit 1
        ;;
esac
