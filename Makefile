# =============================================================================
# HVAC Market Analysis — Makefile
# =============================================================================
# Commandes raccourcies pour le deploiement et l'utilisation du projet.
#
# Usage :
#   make install       Creer venv + installer dependances
#   make demo          Generer les donnees de demonstration
#   make pipeline      Executer le pipeline complet (process + train + evaluate)
#   make dashboard     Lancer le dashboard Streamlit
#   make test          Lancer les tests unitaires
#   make all           install + demo + pipeline (setup complet)
# =============================================================================

PYTHON := python
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit

# Detecter Windows
ifeq ($(OS),Windows_NT)
    PIP := $(VENV)/Scripts/pip
    PYTHON_VENV := $(VENV)/Scripts/python
    STREAMLIT := $(VENV)/Scripts/streamlit
    ACTIVATE := $(VENV)\Scripts\activate
else
    ACTIVATE := source $(VENV)/bin/activate
endif

.PHONY: help install demo pipeline dashboard test clean collect update all setup

# --- Aide ---
help:
	@echo ""
	@echo "  HVAC Market Analysis — Commandes disponibles"
	@echo "  ============================================="
	@echo ""
	@echo "  Setup :"
	@echo "    make install       Creer venv + installer dependances"
	@echo "    make setup         install + config .env"
	@echo "    make all           Setup complet (install + demo + pipeline)"
	@echo ""
	@echo "  Donnees :"
	@echo "    make demo          Generer les donnees de demonstration"
	@echo "    make collect       Collecter les donnees depuis les APIs"
	@echo "    make sync          Telecharger les donnees depuis pCloud"
	@echo ""
	@echo "  Pipeline :"
	@echo "    make pipeline      Executer clean + merge + features + outliers + train + evaluate"
	@echo "    make process       Executer clean + merge + features + outliers"
	@echo "    make train         Entrainer les modeles ML"
	@echo "    make evaluate      Evaluer et comparer les modeles"
	@echo "    make eda           Analyse exploratoire (EDA)"
	@echo "    make update        Mise a jour complete (collect + process + train + upload)"
	@echo ""
	@echo "  Interface :"
	@echo "    make dashboard     Lancer le dashboard Streamlit"
	@echo ""
	@echo "  Tests :"
	@echo "    make test          Lancer les 119 tests unitaires"
	@echo "    make test-cov      Tests avec couverture de code"
	@echo ""
	@echo "  Maintenance :"
	@echo "    make clean         Supprimer les fichiers temporaires Python"
	@echo "    make clean-data    Supprimer toutes les donnees generees"
	@echo ""

# --- Installation ---
install: $(VENV)/bin/activate

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "  Environnement virtuel cree et dependances installees."
	@echo "  Activer avec : $(ACTIVATE)"
	@echo ""

setup: install
	@test -f .env || cp .env.example .env
	@echo "  Configuration .env OK"
	$(PYTHON_VENV) setup_project.py

# --- Donnees ---
demo: install
	@test -f .env || cp .env.example .env
	$(PYTHON_VENV) scripts/generate_demo_data.py
	@echo ""
	@echo "  Donnees de demonstration generees dans data/raw/"

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
	@echo "  Pipeline complet termine."
	@echo "  Resultats dans data/models/"
	@echo "  Lancer le dashboard : make dashboard"

update: install
	$(PYTHON_VENV) -m src.pipeline update_all

# --- Interface ---
dashboard: install
	@echo ""
	@echo "  Lancement du dashboard Streamlit..."
	@echo "  Ouvrir http://localhost:8501 dans le navigateur"
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
	@echo "  Fichiers temporaires supprimes."

clean-data:
	rm -rf data/raw data/processed data/features data/models data/analysis
	rm -f data/*.db
	@echo "  Donnees supprimees. Relancer : make demo && make pipeline"

# --- Setup complet ---
all: install demo pipeline
	@echo ""
	@echo "  ============================================="
	@echo "  Setup complet termine !"
	@echo "  ============================================="
	@echo ""
	@echo "  Lancer le dashboard : make dashboard"
	@echo "  Ou : $(STREAMLIT) run app/app.py"
	@echo ""
