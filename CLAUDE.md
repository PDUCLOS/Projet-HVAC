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
  - Test both, let SHAP / feature importance decide which adds predictive value
- This review is a **collaborative discussion** — do not skip it, wait for user input
- **Document the reasoning in a Jupyter notebook** — the column review, keep/drop decisions,
  and feature importance analysis must be written up in a notebook (not just in code comments)
  to serve as a clear, readable deliverable for the portfolio

## Git Workflow

- Commit messages in English, clear and descriptive
- Never push secrets, credentials, or large data files
- Keep `.gitignore` up to date
