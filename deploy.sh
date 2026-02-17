#!/usr/bin/env bash
# =============================================================================
# HVAC Market Analysis — Script de deploiement local
# =============================================================================
# Deploie le projet complet en une seule commande :
#   1. Environnement virtuel Python
#   2. Installation des dependances
#   3. Configuration .env
#   4. Generation des donnees de demonstration
#   5. Pipeline complet (clean → merge → features → outliers → train → evaluate)
#   6. Lancement du dashboard Streamlit
#
# Usage :
#   chmod +x deploy.sh
#   ./deploy.sh                 # Deploiement complet + lancement dashboard
#   ./deploy.sh --no-dashboard  # Deploiement sans lancer le dashboard
#   ./deploy.sh --api           # Utiliser les APIs reelles au lieu du demo
# =============================================================================

set -e  # Arreter en cas d'erreur

# --- Couleurs ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# --- Options ---
LAUNCH_DASHBOARD=true
USE_API=false

for arg in "$@"; do
    case $arg in
        --no-dashboard) LAUNCH_DASHBOARD=false ;;
        --api) USE_API=true ;;
        --help|-h)
            echo "Usage: ./deploy.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-dashboard  Ne pas lancer le dashboard a la fin"
            echo "  --api           Utiliser les APIs reelles (necessite internet)"
            echo "  --help, -h      Afficher cette aide"
            exit 0
            ;;
    esac
done

# --- Fonctions ---
print_header() {
    echo ""
    echo -e "${BLUE}=============================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=============================================${NC}"
}

print_step() {
    echo -e "${GREEN}  [OK]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}  [!]${NC} $1"
}

print_error() {
    echo -e "${RED}  [ERREUR]${NC} $1"
}

# --- Verifications ---
print_header "HVAC Market Analysis — Deploiement local"

echo ""
echo "  Verification des prerequis..."

# Python 3.10+
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python non trouve. Installez Python 3.10+"
    exit 1
fi

# Determiner la commande Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python $PYTHON_VERSION detecte. Python 3.10+ requis."
    exit 1
fi
print_step "Python $PYTHON_VERSION"

# --- Etape 1 : Environnement virtuel ---
print_header "Etape 1/6 — Environnement virtuel"

if [ -d "venv" ]; then
    print_warn "Environnement venv/ existe deja, reutilisation."
else
    $PYTHON -m venv venv
    print_step "Environnement virtuel cree (venv/)"
fi

# Activer le venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    print_error "Impossible d'activer l'environnement virtuel"
    exit 1
fi
print_step "Environnement virtuel active"

# --- Etape 2 : Dependances ---
print_header "Etape 2/6 — Installation des dependances"

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
print_step "Dependances installees ($(pip list 2>/dev/null | wc -l) packages)"

# --- Etape 3 : Configuration ---
print_header "Etape 3/6 — Configuration"

if [ -f ".env" ]; then
    print_warn ".env existe deja, conservation du fichier existant."
else
    cp .env.example .env
    print_step ".env cree depuis .env.example"
fi

# Creer les repertoires de donnees
mkdir -p data/raw/weather data/raw/insee data/raw/eurostat data/raw/sitadel data/raw/dpe
mkdir -p data/processed/weather data/processed/dpe data/processed/insee data/processed/eurostat
mkdir -p data/features data/models/figures data/analysis/figures
print_step "Repertoires de donnees crees"

# --- Etape 4 : Donnees ---
print_header "Etape 4/6 — Generation des donnees"

if [ "$USE_API" = true ]; then
    echo "  Mode API : collecte depuis les sources reelles..."
    python -m src.pipeline collect
    print_step "Donnees collectees depuis les APIs"
else
    # Verifier si les donnees existent deja
    if [ -f "data/raw/weather/weather_france.csv" ] && [ -f "data/raw/dpe/dpe_france_all.csv" ]; then
        print_warn "Donnees deja presentes dans data/raw/, pas de regeneration."
    else
        echo "  Mode demo : generation de donnees synthetiques..."
        python scripts/generate_demo_data.py
        print_step "Donnees de demonstration generees (96 departements x 2019-2025)"
    fi
fi

# --- Etape 5 : Pipeline ---
print_header "Etape 5/6 — Execution du pipeline"

echo "  Nettoyage + fusion + features + outliers..."
python -m src.pipeline process
print_step "Traitement des donnees termine"

echo "  Entrainement des modeles ML..."
python -m src.pipeline train
print_step "Modeles entraines"

echo "  Evaluation et comparaison..."
python -m src.pipeline evaluate
print_step "Evaluation terminee"

# --- Etape 6 : Tests ---
print_header "Etape 6/6 — Verification"

echo "  Lancement des tests unitaires..."
if python -m pytest tests/ -q --tb=no 2>/dev/null; then
    TEST_COUNT=$(python -m pytest tests/ --co -q 2>/dev/null | tail -1)
    print_step "Tests passes : $TEST_COUNT"
else
    print_warn "Certains tests ont echoue (non bloquant)"
fi

# --- Resume ---
print_header "Deploiement termine !"

echo ""
echo "  Fichiers generes :"
echo "    - data/raw/          Donnees brutes (5 sources)"
echo "    - data/processed/    Donnees nettoyees"
echo "    - data/features/     Dataset ML avec features"
echo "    - data/models/       Modeles entraines + evaluation"
echo ""
echo "  Commandes utiles :"
echo "    make dashboard       Lancer le dashboard Streamlit"
echo "    make test            Lancer les tests"
echo "    make pipeline        Re-executer le pipeline"
echo "    make update          Mise a jour complete (API + train)"
echo ""

# --- Lancement du dashboard ---
if [ "$LAUNCH_DASHBOARD" = true ]; then
    echo -e "${GREEN}  Lancement du dashboard Streamlit...${NC}"
    echo -e "${GREEN}  Ouvrir http://localhost:8501 dans le navigateur${NC}"
    echo ""
    streamlit run app/app.py
else
    echo "  Pour lancer le dashboard :"
    echo "    source venv/bin/activate"
    echo "    streamlit run app/app.py"
    echo ""
fi
