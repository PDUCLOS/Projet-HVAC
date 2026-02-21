#!/usr/bin/env bash
# =============================================================================
# HVAC Market Analysis — Local Deployment Script
# =============================================================================
# Deploys the complete project in a single command :
#   1. Python virtual environment
#   2. Dependency installation
#   3. .env configuration
#   4. Demo data generation
#   5. Full pipeline (clean -> merge -> features -> outliers -> train -> evaluate)
#   6. Launch the Streamlit dashboard
#
# Usage :
#   chmod +x deploy.sh
#   ./deploy.sh                 # Full deployment + launch dashboard
#   ./deploy.sh --no-dashboard  # Deployment without launching the dashboard
#   ./deploy.sh --api           # Use real APIs instead of demo data
# =============================================================================

set -e  # Stop on error

# --- Colors ---
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
            echo "  --no-dashboard  Do not launch the dashboard at the end"
            echo "  --api           Use real APIs (requires internet access)"
            echo "  --help, -h      Show this help"
            exit 0
            ;;
    esac
done

# --- Functions ---
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
    echo -e "${RED}  [ERROR]${NC} $1"
}

# --- Checks ---
print_header "HVAC Market Analysis — Local Deployment"

echo ""
echo "  Checking prerequisites..."

# Python 3.10+
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.10+"
    exit 1
fi

# Determine the Python command
if command -v python3 &> /dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python $PYTHON_VERSION detected. Python 3.10+ required."
    exit 1
fi
print_step "Python $PYTHON_VERSION"

# --- Step 1 : Virtual environment ---
print_header "Step 1/6 — Virtual environment"

if [ -d "venv" ]; then
    print_warn "venv/ environment already exists, reusing it."
else
    $PYTHON -m venv venv
    print_step "Virtual environment created (venv/)"
fi

# Activate the venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    print_error "Unable to activate the virtual environment"
    exit 1
fi
print_step "Virtual environment activated"

# --- Step 2 : Dependencies ---
print_header "Step 2/6 — Installing dependencies"

pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
print_step "Dependencies installed ($(pip list 2>/dev/null | wc -l) packages)"

# --- Step 3 : Configuration ---
print_header "Step 3/6 — Configuration"

if [ -f ".env" ]; then
    print_warn ".env already exists, keeping the existing file."
else
    cp .env.example .env
    print_step ".env created from .env.example"
fi

# Create data directories
mkdir -p data/raw/weather data/raw/insee data/raw/eurostat data/raw/sitadel data/raw/dpe
mkdir -p data/processed/weather data/processed/dpe data/processed/insee data/processed/eurostat
mkdir -p data/features data/models/figures data/analysis/figures
print_step "Data directories created"

# --- Step 4 : Data ---
print_header "Step 4/6 — Data generation"

if [ "$USE_API" = true ]; then
    echo "  API mode : collecting from real sources..."
    python -m src.pipeline collect
    print_step "Data collected from APIs"
else
    # Check if data already exists
    if [ -f "data/raw/weather/weather_france.csv" ] && [ -f "data/raw/dpe/dpe_france_all.csv" ]; then
        print_warn "Data already present in data/raw/, skipping regeneration."
    else
        echo "  Demo mode : generating synthetic data..."
        python scripts/generate_demo_data.py
        print_step "Demo data generated (96 departments x 2019-2025)"
    fi
fi

# --- Step 5 : Pipeline ---
print_header "Step 5/6 — Running the pipeline"

echo "  Cleaning + merging + features + outliers..."
python -m src.pipeline process
print_step "Data processing completed"

echo "  Training ML models..."
python -m src.pipeline train
print_step "Models trained"

echo "  Evaluating and comparing..."
python -m src.pipeline evaluate
print_step "Evaluation completed"

# --- Step 6 : Tests ---
print_header "Step 6/6 — Verification"

echo "  Running unit tests..."
if python -m pytest tests/ -q --tb=no 2>/dev/null; then
    TEST_COUNT=$(python -m pytest tests/ --co -q 2>/dev/null | tail -1)
    print_step "Tests passed : $TEST_COUNT"
else
    print_warn "Some tests failed (non-blocking)"
fi

# --- Summary ---
print_header "Deployment completed !"

echo ""
echo "  Generated files :"
echo "    - data/raw/          Raw data (5 sources)"
echo "    - data/processed/    Cleaned data"
echo "    - data/features/     ML dataset with features"
echo "    - data/models/       Trained models + evaluation"
echo ""
echo "  Useful commands :"
echo "    make dashboard       Launch the Streamlit dashboard"
echo "    make test            Run the tests"
echo "    make pipeline        Re-run the pipeline"
echo "    make update          Full update (API + train)"
echo ""

# --- Launch the dashboard ---
if [ "$LAUNCH_DASHBOARD" = true ]; then
    echo -e "${GREEN}  Launching the Streamlit dashboard...${NC}"
    echo -e "${GREEN}  Open http://localhost:8501 in your browser${NC}"
    echo ""
    streamlit run app/app.py
else
    echo "  To launch the dashboard :"
    echo "    source venv/bin/activate"
    echo "    streamlit run app/app.py"
    echo ""
fi
