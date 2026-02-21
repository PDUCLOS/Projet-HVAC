"""
API de prediction pour le marche HVAC en France.

Expose les modeles de Machine Learning entraines (Ridge, LightGBM) via une
API REST FastAPI. Permet de generer des predictions de nombre d'installations
de pompes a chaleur par departement et par mois.

Lancement : uvicorn api.main:app --reload
Documentation : http://localhost:8000/docs
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
# Lifespan : chargement des modeles au demarrage
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Charge les artefacts ML une seule fois au demarrage."""
    state.load()
    yield


# ---------------------------------------------------------------------------
# Application FastAPI
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Systeme"])
async def health() -> HealthResponse:
    """Verifie l'etat de l'API et renvoie les informations de version."""
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
    Genere des predictions de nombre d'installations PAC pour un departement.

    Utilise le modele Ridge entraine, le scaler et l'imputer charges au
    demarrage. Les predictions sont generees iterativement a partir de la
    derniere observation connue du departement.
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
    Prediction personnalisee avec parametres libres.

    Accepte un code departement, un horizon et un dictionnaire de features
    supplementaires a injecter dans le vecteur d'entree du modele.
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

    # R2 test du modele
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
    Resume des donnees disponibles.

    Nombre de departements, plage de dates, nombre de lignes par source
    et dates de derniere modification des fichiers bruts.
    """
    df = state.features_df
    nb_depts = df["dept"].nunique() if df is not None else 0

    # Plage de dates
    plage: dict[str, str] = {"debut": "", "fin": ""}
    if df is not None and "date_id" in df.columns:
        dates = df["date_id"].dropna().astype(str)
        plage = {"debut": dates.min(), "fin": dates.max()}

    nb_lignes = len(df) if df is not None else 0

    # Sources brutes
    sources: list[SourceSummary] = []
    if RAW_DIR.exists():
        for sub in sorted(RAW_DIR.iterdir()):
            if not sub.is_dir():
                continue
            for csv_file in sorted(sub.glob("*.csv")):
                stat = csv_file.stat()
                # Compter les lignes (moins l'en-tete)
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
    Metriques d'evaluation de tous les modeles entraines.

    Lit les resultats depuis training_results.csv et retourne RMSE, MAE,
    R2 et MAPE pour les jeux de validation et de test.
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

    # Meilleur modele = RMSE test le plus bas (hors NaN)
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
    Liste des 96 departements metropolitains avec codes et noms.

    Inclut la Corse (2A, 2B). Les departements presents dans le dataset
    sont marques dans la liste complete.
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
# Utilitaires internes
# ---------------------------------------------------------------------------

def _validate_department_in_data(dept: str) -> None:
    """Verifie que le departement existe dans le dataset de features."""
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
    """Convertit une valeur en float, retourne None si NaN ou absent."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None
