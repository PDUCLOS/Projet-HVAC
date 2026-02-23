# Commands Cheat Sheet — HVAC Market Analysis

## Project Commands

### Full Pipeline
```bash
# Run everything (collection + processing + training)
python -m src.pipeline all

# Interactive CLI menu
python -m src.pipeline menu
python -m src.cli_menu
```

### Individual Steps
```bash
# 1. Data Collection (APIs)
python -m src.pipeline collect

# 2. Data Processing (clean, merge, features)
python -m src.pipeline process

# 3. ML Training
python -m src.pipeline train
```

### Tests
```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_cli_menu.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Jupyter
```bash
# Notebook server
jupyter notebook --notebook-dir=notebooks/

# JupyterLab
jupyter lab --notebook-dir=notebooks/
```

### API (FastAPI)
```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000

# Access docs: http://localhost:8000/docs
```

---

## Git Commands

### Daily Workflow
```bash
# Check current status
git status

# View changes
git diff                    # unstaged changes
git diff --staged           # staged changes

# Stage and commit
git add <file>              # stage specific file
git add -A                  # stage all changes
git commit -m "message"     # commit

# Push to GitHub
git push origin main
```

### Branches
```bash
# Create and switch to a new branch
git checkout -b feature/my-feature

# Switch branches
git checkout main

# Merge a branch into main
git checkout main
git merge feature/my-feature

# Delete a branch (local)
git branch -d feature/my-feature
```

### History & Logs
```bash
# View recent commits
git log --oneline -10

# View commit details
git log --stat -5

# View a specific file's history
git log --oneline -- src/cli_menu.py
```

### Sync with Remote
```bash
# Pull latest changes
git pull origin main

# Fetch without merging
git fetch origin

# Check remote URL
git remote -v
```

### Undo / Fix
```bash
# Unstage a file
git restore --staged <file>

# Discard changes in a file (CAUTION)
git restore <file>

# Amend last commit message
git commit --amend -m "new message"

# View diff between branches
git diff main..feature/my-branch
```

### GitHub CLI (gh)
```bash
# Create a pull request
gh pr create --title "Title" --body "Description"

# View PRs
gh pr list
gh pr view 123

# Create an issue
gh issue create --title "Title" --body "Description"
```

---

## Virtual Environment
```bash
# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dl.txt   # PyTorch for LSTM
```

---

## CLI Menu Structure

| Menu | Key | Description |
|------|-----|-------------|
| **1 - Collection** | `1` | API data collection + manual pCloud copy instructions |
| **2 - Processing** | `2` | Clean, merge, feature engineering, outliers |
| **3 - ML Training** | `3` | Ridge, LightGBM, Prophet + Jupyter notebooks |
| **4 - DL & RL** | `4` | LSTM standalone, RL demo, results exploration |
| **Exit** | `0` | Quit |

---

## Key Directories

```
data/
  raw/          # Raw collected data (CSV)
  processed/    # Cleaned/merged data
  features/     # Feature-engineered dataset
  models/       # Trained model artifacts (.pkl, .pt)
  analysis/     # EDA figures
notebooks/      # Jupyter notebooks (EDA, ML, DL)
reports/        # Evaluation reports, figures
src/
  collectors/   # Data collection modules
  processing/   # ETL pipeline
  models/       # ML/DL model code
  api/          # FastAPI endpoints
config/         # Project settings
tests/          # Test suite
```
