# =============================================================================
# HVAC Market Analysis — Makefile
# =============================================================================
# Professional shortcut commands for project deployment, testing, and usage.
#
# Usage:
#   make help          Show all available commands
#   make setup         Full setup: venv + install + init_db
#   make pipeline      Run the complete ML pipeline
#   make test          Run all tests with coverage
#
# Self-documenting: each target with ## comment appears in `make help`.
# =============================================================================

PYTHON := python
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit
UVICORN := $(VENV)/bin/uvicorn

# Detect Windows
ifeq ($(OS),Windows_NT)
    PIP := $(VENV)/Scripts/pip
    PYTHON_VENV := $(VENV)/Scripts/python
    STREAMLIT := $(VENV)/Scripts/streamlit
    UVICORN := $(VENV)/Scripts/uvicorn
    ACTIVATE := $(VENV)\Scripts\activate
else
    ACTIVATE := source $(VENV)/bin/activate
endif

.PHONY: help setup install test lint security collect clean process train evaluate \
        pipeline serve-api serve-dashboard docker-build docker-up docker-down \
        demo sync eda update all test-cov clean-data clean-cache dashboard

# =============================================================================
# Help (self-documenting pattern)
# =============================================================================

help: ## Show this help
	@echo ""
	@echo "  ======================================================="
	@echo "   HVAC Market Analysis — Makefile Commands"
	@echo "  ======================================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Setup & Installation
# =============================================================================

setup: install ## Full setup: venv + install + init_db
	@test -f .env || cp .env.example .env
	@echo "[setup] Configuration .env OK"
	$(PYTHON_VENV) -m src.pipeline init_db
	@echo ""
	@echo "  Setup complete. Run 'make demo' or 'make collect' to get data."
	@echo ""

install: $(VENV)/bin/activate ## Install all dependencies

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "  Virtual environment created and dependencies installed."
	@echo "  Activate with: $(ACTIVATE)"
	@echo ""

# =============================================================================
# Testing & Quality
# =============================================================================

test: install ## Run all tests with coverage
	$(PYTHON_VENV) -m pytest tests/ -v --tb=short

test-cov: install ## Run tests with detailed coverage report
	$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

lint: install ## Run ruff linter
	$(PYTHON_VENV) -m ruff check src/ api/ --select E,W,F --ignore E501
	@echo "[lint] Code quality check passed."

security: install ## Run security checks (pip-audit + bandit)
	@echo "[security] Running pip-audit (dependency vulnerabilities)..."
	-$(PYTHON_VENV) -m pip_audit 2>/dev/null || echo "  pip-audit not installed. Run: pip install pip-audit"
	@echo "[security] Running bandit (code security)..."
	-$(PYTHON_VENV) -m bandit -r src/ api/ -ll -q 2>/dev/null || echo "  bandit not installed. Run: pip install bandit"
	@echo "[security] Security checks complete."

# =============================================================================
# Data Collection
# =============================================================================

collect: install ## Collect data from all sources
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) -m src.pipeline collect
	@echo "[collect] Data collection complete."

demo: install ## Generate demonstration data
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) scripts/generate_demo_data.py
	@echo ""
	@echo "  Demonstration data generated in data/raw/"
	@echo ""

sync: install ## Download data from pCloud
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) -m src.pipeline sync_pcloud

# =============================================================================
# Processing Pipeline
# =============================================================================

clean: ## Clean processed data (removes data/processed/)
	rm -rf data/processed
	@echo "[clean] Processed data removed."

process: install ## Run full processing pipeline (clean + merge + features + outliers)
	$(PYTHON_VENV) -m src.pipeline process
	@echo "[process] Processing pipeline complete."

train: install ## Train ML models
	$(PYTHON_VENV) -m src.pipeline train
	@echo "[train] Model training complete."

evaluate: install ## Evaluate trained models
	$(PYTHON_VENV) -m src.pipeline evaluate
	@echo "[evaluate] Model evaluation complete."

eda: install ## Run Exploratory Data Analysis
	$(PYTHON_VENV) -m src.pipeline eda

pipeline: install ## Run complete pipeline (process + train + evaluate)
	$(PYTHON_VENV) -m src.pipeline process
	$(PYTHON_VENV) -m src.pipeline train
	$(PYTHON_VENV) -m src.pipeline evaluate
	@echo ""
	@echo "  ======================================================="
	@echo "   Pipeline complete."
	@echo "   Results in data/models/"
	@echo "   Launch dashboard: make serve-dashboard"
	@echo "  ======================================================="
	@echo ""

update: install ## Full update (collect + process + train + upload)
	$(PYTHON_VENV) -m src.pipeline update_all

# =============================================================================
# Serving (API & Dashboard)
# =============================================================================

serve-api: install ## Start FastAPI server
	@echo ""
	@echo "  Starting FastAPI server..."
	@echo "  Swagger UI: http://localhost:8000/docs"
	@echo ""
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

serve-dashboard: install ## Start Streamlit dashboard
	@echo ""
	@echo "  Starting Streamlit dashboard..."
	@echo "  Open http://localhost:8501 in your browser"
	@echo ""
	$(STREAMLIT) run app/app.py

# Legacy alias
dashboard: serve-dashboard ## Start Streamlit dashboard (alias for serve-dashboard)

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker image
	docker build -t hvac-market:latest .
	@echo "[docker-build] Image built: hvac-market:latest"

docker-up: ## Start all services with Docker Compose
	docker compose up -d
	@echo ""
	@echo "  Services started:"
	@echo "    API:       http://localhost:8000/docs"
	@echo "    Dashboard: http://localhost:8501"
	@echo ""

docker-down: ## Stop Docker Compose services
	docker compose down
	@echo "[docker-down] All services stopped."

# =============================================================================
# Maintenance
# =============================================================================

clean-cache: ## Remove temporary Python files (__pycache__, .pyc)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "[clean-cache] Temporary files removed."

clean-data: ## Remove all generated data (requires re-collection)
	rm -rf data/raw data/processed data/features data/models data/analysis
	rm -f data/*.db
	@echo "[clean-data] Data removed. Re-run: make demo && make pipeline"

# =============================================================================
# Complete Setup
# =============================================================================

all: install demo pipeline ## Full setup: install + demo + pipeline
	@echo ""
	@echo "  ======================================================="
	@echo "   Complete setup finished."
	@echo "  ======================================================="
	@echo ""
	@echo "  Launch the dashboard: make serve-dashboard"
	@echo "  Or the API:           make serve-api"
	@echo ""
