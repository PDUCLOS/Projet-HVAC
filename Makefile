# =============================================================================
# HVAC Market Analysis — Makefile
# =============================================================================
# Shortcut commands for project deployment and usage.
#
# Usage :
#   make install       Create venv + install dependencies
#   make demo          Generate demonstration data
#   make pipeline      Run the full pipeline (process + train + evaluate)
#   make dashboard     Launch the Streamlit dashboard
#   make test          Run unit tests
#   make all           install + demo + pipeline (complete setup)
# =============================================================================

PYTHON := python
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit

# Detect Windows
ifeq ($(OS),Windows_NT)
    PIP := $(VENV)/Scripts/pip
    PYTHON_VENV := $(VENV)/Scripts/python
    STREAMLIT := $(VENV)/Scripts/streamlit
    ACTIVATE := $(VENV)\Scripts\activate
else
    ACTIVATE := source $(VENV)/bin/activate
endif

.PHONY: help install demo pipeline dashboard test clean collect update all setup

# --- Help ---
help:
	@echo ""
	@echo "  HVAC Market Analysis — Available commands"
	@echo "  ============================================="
	@echo ""
	@echo "  Setup :"
	@echo "    make install       Create venv + install dependencies"
	@echo "    make setup         install + configure .env"
	@echo "    make all           Complete setup (install + demo + pipeline)"
	@echo ""
	@echo "  Data :"
	@echo "    make demo          Generate demonstration data"
	@echo "    make collect       Collect data from APIs"
	@echo "    make sync          Download data from pCloud"
	@echo ""
	@echo "  Pipeline :"
	@echo "    make pipeline      Run clean + merge + features + outliers + train + evaluate"
	@echo "    make process       Run clean + merge + features + outliers"
	@echo "    make train         Train ML models"
	@echo "    make evaluate      Evaluate and compare models"
	@echo "    make eda           Exploratory Data Analysis (EDA)"
	@echo "    make update        Full update (collect + process + train + upload)"
	@echo ""
	@echo "  Interface :"
	@echo "    make dashboard     Launch the Streamlit dashboard"
	@echo ""
	@echo "  Tests :"
	@echo "    make test          Run the 119 unit tests"
	@echo "    make test-cov      Tests with code coverage"
	@echo ""
	@echo "  Maintenance :"
	@echo "    make clean         Remove temporary Python files"
	@echo "    make clean-data    Remove all generated data"
	@echo ""

# --- Installation ---
install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "  Virtual environment created and dependencies installed."
	@echo "  Activate with : $(ACTIVATE)"
	@echo ""

setup: install
	@test -f .env || cp .env.example .env
	@echo "  Configuration .env OK"
	$(PYTHON_VENV) setup_project.py

# --- Data ---
demo: install
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) scripts/generate_demo_data.py
	@echo ""
	@echo "  Demonstration data generated in data/raw/"

collect: install
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) -m src.pipeline collect

sync: install
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) -m src.pipeline sync_pcloud

# --- Pipeline ---
process: install
	$(PYTHON_VENV) -m src.pipeline process

train: install
	$(PYTHON_VENV) -m src.pipeline train

evaluate: install
	$(PYTHON_VENV) -m src.pipeline evaluate

eda: install
	$(PYTHON_VENV) -m src.pipeline eda

pipeline: install
	$(PYTHON_VENV) -m src.pipeline process
	$(PYTHON_VENV) -m src.pipeline train
	$(PYTHON_VENV) -m src.pipeline evaluate
	@echo ""
	@echo "  Full pipeline completed."
	@echo "  Results in data/models/"
	@echo "  Launch the dashboard : make dashboard"

update: install
	$(PYTHON_VENV) -m src.pipeline update_all

# --- Interface ---
dashboard: install
	@echo ""
	@echo "  Launching the Streamlit dashboard..."
	@echo "  Open http://localhost:8501 in your browser"
	@echo ""
	$(STREAMLIT) run app/app.py

# --- Tests ---
test: install
	$(PYTHON_VENV) -m pytest tests/ -v

test-cov: install
	$(PYTHON_VENV) -m pytest tests/ -v --cov=src --cov-report=term-missing

# --- Maintenance ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "  Temporary files removed."

clean-data:
	rm -rf data/raw data/processed data/features data/models data/analysis
	rm -f data/*.db
	@echo "  Data removed. Re-run : make demo && make pipeline"

# --- Complete setup ---
all: install demo pipeline
	@echo ""
	@echo "  ============================================="
	@echo "  Complete setup finished !"
	@echo "  ============================================="
	@echo ""
	@echo "  Launch the dashboard : make dashboard"
	@echo "  Or : $(STREAMLIT) run app/app.py"
	@echo ""
