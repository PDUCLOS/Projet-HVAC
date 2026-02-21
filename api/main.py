"""
Prediction API for the HVAC market in France.

Exposes trained Machine Learning models (Ridge, LightGBM) via a
FastAPI REST API. Allows generating predictions of heat pump
installation counts by department and month.

Launch: uvicorn api.main:app --reload
Documentation: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    API_VERSION,
    DEPARTEMENTS,
    FEATURES_CSV,
    RAW_DIR,
    TARGET_COL,
    state,
)
from api.models import (
    CustomPredictRequest,
    CustomPredictResponse,
    DataSummaryResponse,
    DepartmentInfo,
    DepartmentsResponse,
    HealthResponse,
    ModelMetric,
    ModelMetricsResponse,
    PredictionPoint,
    PredictionResponse,
    SourceSummary,
)


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load ML artifacts once at startup."""
    state.load()
    yield


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HVAC Market Prediction API",
    description=(
        "API de prediction du marche HVAC (pompes a chaleur) en France.\n\n"
        "Fournit des previsions mensuelles par departement, les metriques "
        "des modeles entraines et un resume des donnees disponibles."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS — restrict origins in production via CORS_ORIGINS env var
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Systeme"])
async def health() -> HealthResponse:
    """Check the API status and return version information."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        modele_principal="ridge",
        nb_features=len(state.feature_names),
        derniere_date_entrainement=state.model_date,
        uptime_secondes=round(time.time() - state.start_time, 1),
    )


# ---------------------------------------------------------------------------
# GET /predictions
# ---------------------------------------------------------------------------

@app.get("/predictions", response_model=PredictionResponse, tags=["Predictions"])
async def get_predictions(
    departement: str = Query(
        ...,
        min_length=1,
        max_length=3,
        description="Code departement (ex: 69, 2A)",
        examples=["69"],
    ),
    horizon: int = Query(
        default=6,
        ge=1,
        le=24,
        description="Nombre de mois a predire (1-24)",
    ),
) -> PredictionResponse:
    """
    Generate predictions of heat pump installation counts for a department.

    Uses the trained Ridge model, scaler, and imputer loaded at
    startup. Predictions are generated iteratively from the
    last known observation for the department.
    """
    dept = departement.upper().zfill(2)
    _validate_department_in_data(dept)

    predictions = state.predict(dept, horizon)
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnee trouvee pour le departement {dept}",
        )

    return PredictionResponse(
        departement=dept,
        horizon_mois=horizon,
        modele_utilise="ridge",
        variable_cible=TARGET_COL,
        predictions=[PredictionPoint(**p) for p in predictions],
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=CustomPredictResponse, tags=["Predictions"])
async def post_predict(body: CustomPredictRequest) -> CustomPredictResponse:
    """
    Custom prediction with free parameters.

    Accepts a department code, a horizon, and a dictionary of additional
    features to inject into the model input vector.
    """
    dept = body.departement
    _validate_department_in_data(dept)

    predictions = state.predict(
        dept,
        body.horizon,
        extra_features=body.features if body.features else None,
    )
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune donnee trouvee pour le departement {dept}",
        )

    # Model test R2
    r2 = 0.0
    if state.training_results is not None:
        ridge_row = state.training_results[
            state.training_results["model"] == "ridge"
        ]
        if not ridge_row.empty:
            r2 = float(ridge_row.iloc[0].get("test_r2", 0.0))

    return CustomPredictResponse(
        departement=dept,
        modele_utilise="ridge",
        horizon_mois=body.horizon,
        confiance_modele_r2=round(r2, 4),
        predictions=[PredictionPoint(**p) for p in predictions],
    )


# ---------------------------------------------------------------------------
# GET /data/summary
# ---------------------------------------------------------------------------

@app.get("/data/summary", response_model=DataSummaryResponse, tags=["Donnees"])
async def data_summary() -> DataSummaryResponse:
    """
    Summary of available data.

    Number of departments, date range, row counts per source,
    and last modification dates of raw files.
    """
    df = state.features_df
    nb_depts = df["dept"].nunique() if df is not None else 0

    # Date range
    plage: dict[str, str] = {"debut": "", "fin": ""}
    if df is not None and "date_id" in df.columns:
        dates = df["date_id"].dropna().astype(str)
        plage = {"debut": dates.min(), "fin": dates.max()}

    nb_lignes = len(df) if df is not None else 0

    # Raw sources
    sources: list[SourceSummary] = []
    if RAW_DIR.exists():
        for sub in sorted(RAW_DIR.iterdir()):
            if not sub.is_dir():
                continue
            for csv_file in sorted(sub.glob("*.csv")):
                stat = csv_file.stat()
                # Count lines (minus the header)
                with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
                    n_lines = sum(1 for _ in f) - 1
                sources.append(
                    SourceSummary(
                        nom=f"{sub.name}/{csv_file.name}",
                        nb_lignes=max(n_lines, 0),
                        derniere_modification=datetime.fromtimestamp(
                            stat.st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                    )
                )

    return DataSummaryResponse(
        nb_departements=nb_depts,
        plage_dates=plage,
        nb_lignes_features=nb_lignes,
        sources_brutes=sources,
    )


# ---------------------------------------------------------------------------
# GET /model/metrics
# ---------------------------------------------------------------------------

@app.get("/model/metrics", response_model=ModelMetricsResponse, tags=["Modeles"])
async def model_metrics() -> ModelMetricsResponse:
    """
    Evaluation metrics for all trained models.

    Reads results from training_results.csv and returns RMSE, MAE,
    R2, and MAPE for the validation and test sets.
    """
    tr = state.training_results
    if tr is None or tr.empty:
        raise HTTPException(
            status_code=404,
            detail="Aucun resultat d'entrainement disponible",
        )

    modeles: list[ModelMetric] = []
    for _, row in tr.iterrows():
        modeles.append(
            ModelMetric(
                modele=str(row.get("model", "")),
                cible=str(row.get("target", TARGET_COL)),
                val_rmse=_safe_float(row.get("val_rmse")),
                val_mae=_safe_float(row.get("val_mae")),
                val_mape=_safe_float(row.get("val_mape")),
                val_r2=_safe_float(row.get("val_r2")),
                test_rmse=_safe_float(row.get("test_rmse")),
                test_mae=_safe_float(row.get("test_mae")),
                test_mape=_safe_float(row.get("test_mape")),
                test_r2=_safe_float(row.get("test_r2")),
                cv_rmse_mean=_safe_float(row.get("cv_rmse_mean")),
                cv_r2_mean=_safe_float(row.get("cv_r2_mean")),
            )
        )

    # Best model = lowest test RMSE (excluding NaN)
    valid = [m for m in modeles if m.test_rmse is not None]
    best = min(valid, key=lambda m: m.test_rmse).modele if valid else "inconnu"

    return ModelMetricsResponse(
        meilleur_modele=best,
        nb_modeles=len(modeles),
        modeles=modeles,
    )


# ---------------------------------------------------------------------------
# GET /departments
# ---------------------------------------------------------------------------

@app.get("/departments", response_model=DepartmentsResponse, tags=["Reference"])
async def list_departments() -> DepartmentsResponse:
    """
    List of 96 metropolitan departments with codes and names.

    Includes Corsica (2A, 2B). Departments present in the dataset
    are marked in the full list.
    """
    dept_list = [
        DepartmentInfo(code=code, nom=nom)
        for code, nom in sorted(DEPARTEMENTS.items())
    ]
    return DepartmentsResponse(
        nb_departements=len(dept_list),
        departements=dept_list,
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _validate_department_in_data(dept: str) -> None:
    """Verify that the department exists in the features dataset."""
    if state.features_df is None:
        raise HTTPException(
            status_code=503,
            detail="Donnees non chargees — l'API est en cours de demarrage",
        )
    available = set(state.features_df["dept"].unique())
    if dept not in available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Departement '{dept}' absent du dataset. "
                f"Departements disponibles : {sorted(available)}"
            ),
        )


def _safe_float(val: object) -> float | None:
    """Convert a value to float, return None if NaN or missing."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None
