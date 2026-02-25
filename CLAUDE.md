# CLAUDE.md — Project Rules for AI Assistant

## Code Standards

- **All code comments MUST be written in English** — no exceptions
- **Docstrings in English** for all functions, classes, and modules
- Keep code clean, readable, and well-structured — this project is a portfolio showcase

## Testing Requirements

- **Security tests**: Run security checks (input validation, injection prevention, dependency vulnerabilities) on every change
- **Regression tests**: Every improvement or evolution MUST include regression tests to ensure existing functionality is not broken
- **Never skip tests** before committing — run `pytest tests/ -v` to validate all changes
- Add new tests for every new feature or bug fix

## Documentation

- **Update documentation** (`docs/`, `README.md`, `PROJECT_BRIEF.md`) whenever code changes affect architecture, features, or usage
- Keep documentation in sync with the codebase at all times
- Document new features, API changes, and configuration updates
- **Always update ALL related doc files** when a change impacts multiple documents:
  - `docs/DATA_PIPELINE.md` — pipeline stages, feature tables, data flow diagrams
  - `PROJECT_BRIEF.md` — feature counts, formulas, domain logic, data sources
  - `README.md` — test count badge, data sources table, architecture tree
  - `CLAUDE.md` — project rules (if new conventions are added)
- **Update the test count badge** in `README.md` after every test suite run

## Project Purpose

- This project is a **portfolio showcase** for the Data Science Lead certification (Jedha Bootcamp, Bac+5 RNCP Level 7)
- Every change should reflect **professional quality** — clean code, proper testing, clear documentation
- The project must demonstrate end-to-end data science skills: collection, ETL, ML, deep learning, API, dashboard, DevOps

## Local Execution

- The project MUST always remain **runnable locally** (SQLite, local venv, no mandatory cloud dependency)
- Docker support is optional but local Python execution is the primary mode
- Preserve portability: `python -m src.pipeline all` must work on any machine after setup

## ML / Deep Learning Optimization (Pending Full Data)

- When full data is available (~5-8M DPE from 96 departments), optimize:
  - Feature importance analysis and feature selection
  - Hyperparameter tuning (Ridge, LightGBM)
  - Deep learning models (LSTM/Transformer) with sufficient data volume
  - Cross-validation strategy adapted to temporal data (TimeSeriesSplit)
- Always measure and document model impact of changes (metrics before/after)

## TODO: Data Column Review (Before Optimization)

- **Before optimizing ML/DL**, conduct a full review of selected columns with the user:
  - **DPE columns**: Document each selected column, its content, and its relevance to the model
  - **Weather columns**: Same review — explain what each variable brings to prediction
  - **Evaluate with more data**: With full 96-department data, assess whether additional columns
    (currently excluded) could improve model performance
  - **Decision**: For each column, justify keep/drop/add based on feature importance and domain logic
- **Candidate new variables to evaluate with full data**:
  - **Revenu médian par département** (INSEE Filosofi) — directly linked to aid eligibility
    (MaPrimeRénov', CEE), strong hypothesis for HVAC installation driver
  - **Prix du m² par département** — proxy for real estate market activity and housing type
  - **Parc de logements par département** (INSEE Recensement) — total housing stock
    with breakdown by type (maison / appartement), enables normalization of HVAC counts
    into installation rates and captures structural differences (houses → more heat pumps,
    apartments → more individual AC, constrained by co-ownership rules)
  - Test all candidates, let SHAP / feature importance decide which adds predictive value
- This review is a **collaborative discussion** — do not skip it, wait for user input
- **Document the reasoning in a Jupyter notebook** — the column review, keep/drop decisions,
  and feature importance analysis must be written up in a notebook (not just in code comments)
  to serve as a clear, readable deliverable for the portfolio

## Data Quality — NaN Detection & Traceability

### Mandatory NaN Controls

Every notebook and pipeline stage that loads data MUST include a **systematic NaN audit**:

1. **Per-column NaN count and percentage** — identify which columns are affected
2. **Per-source NaN tracing** — determine the origin of each NaN:
   - `weather_*` columns → check `data/raw/weather/weather_france.csv` (all 96 depts collected?)
   - `lag_*` / `rolling_*` columns → structural NaN from feature engineering (expected at series start)
   - `insee_*` / `eurostat_*` columns → national indicators not yet published for recent months
   - `sitadel_*` columns → SITADEL construction permits data coverage
   - `dpe_*` columns → check DPE collection completeness by department
3. **Department coverage check** — verify that all 96 metropolitan departments are present
4. **Temporal coverage check** — verify no months are missing in the expected date range
5. **Threshold alerts**:
   - Any column with **>5% NaN** must be flagged with a WARNING
   - Any column with **>50% NaN** must trigger an investigation of the data source
   - **0 NaN** after imputation must be confirmed before training

### Root Cause Documentation

When NaN are found, **always document the root cause and provenance**:
- Which raw data source is incomplete?
- Is it a collection failure (API rate limit, network error)?
- Is it a structural gap (feature engineering lag at series start)?
- Is it a temporal coverage issue (data not yet available)?

### Where to Apply

- **Notebook 01** (EDA): Full NaN heatmap + per-source breakdown + department coverage
- **Notebook 02** (ML): Pre-training NaN audit + post-imputation verification
- **Notebook 03** (LSTM): Pre-training NaN audit + post-imputation verification
- **Notebook 04** (Results): Verify dataset used for evaluation is NaN-free
- **Pipeline** (`merge`, `features`, `train`): Log NaN stats at each stage

## Git Workflow

- Commit messages in English, clear and descriptive
- Never push secrets, credentials, or large data files
- Keep `.gitignore` up to date
