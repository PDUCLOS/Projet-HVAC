"""
Pydantic models for the HVAC prediction API.

Defines request and response schemas for all endpoints,
with strict validation via Pydantic v2.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Generic responses
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response for the /health endpoint."""

    status: str = Field(..., examples=["ok"])
    version: str = Field(..., examples=["1.0.0"])
    modele_principal: str = Field(..., examples=["ridge"])
    nb_features: int = Field(..., ge=0, examples=[81])
    derniere_date_entrainement: str | None = Field(
        None, examples=["2025-02-17"]
    )
    uptime_secondes: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionPoint(BaseModel):
    """A single monthly prediction point."""

    date: str = Field(..., examples=["2026-01"])
    valeur_predite: float = Field(..., examples=[25.4])
    intervalle_bas: float = Field(..., examples=[20.1])
    intervalle_haut: float = Field(..., examples=[30.7])


class PredictionResponse(BaseModel):
    """Response for the GET /predictions endpoint."""

    departement: str = Field(..., examples=["69"])
    horizon_mois: int = Field(..., ge=1, le=24, examples=[6])
    modele_utilise: str = Field(..., examples=["ridge"])
    variable_cible: str = Field(
        default="nb_installations_pac",
        examples=["nb_installations_pac"],
    )
    predictions: list[PredictionPoint]


class CustomPredictRequest(BaseModel):
    """Request body for the POST /predict endpoint."""

    departement: str = Field(
        ...,
        min_length=1,
        max_length=3,
        examples=["69"],
        description="Department code (01-95, 2A, 2B)",
    )
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of additional features to inject",
    )
    horizon: int = Field(
        default=1,
        ge=1,
        le=24,
        description="Number of months to predict",
    )

    @model_validator(mode="after")
    def _normaliser_dept(self) -> "CustomPredictRequest":
        """Normalize the department code to uppercase."""
        self.departement = self.departement.upper().zfill(2)
        return self


class CustomPredictResponse(BaseModel):
    """Response for the POST /predict endpoint."""

    departement: str
    modele_utilise: str
    horizon_mois: int
    confiance_modele_r2: float
    predictions: list[PredictionPoint]


# ---------------------------------------------------------------------------
# Data and metrics
# ---------------------------------------------------------------------------

class SourceSummary(BaseModel):
    """Summary of a raw data source."""

    nom: str
    nb_lignes: int
    derniere_modification: str


class DataSummaryResponse(BaseModel):
    """Response for the GET /data/summary endpoint."""

    nb_departements: int
    plage_dates: dict[str, str] = Field(
        ..., examples=[{"debut": "202107", "fin": "202512"}]
    )
    nb_lignes_features: int
    sources_brutes: list[SourceSummary]


class ModelMetric(BaseModel):
    """Metrics for a trained model."""

    modele: str
    cible: str
    val_rmse: float | None = None
    val_mae: float | None = None
    val_mape: float | None = None
    val_r2: float | None = None
    test_rmse: float | None = None
    test_mae: float | None = None
    test_mape: float | None = None
    test_r2: float | None = None
    cv_rmse_mean: float | None = None
    cv_r2_mean: float | None = None


class ModelMetricsResponse(BaseModel):
    """Response for the GET /model/metrics endpoint."""

    meilleur_modele: str
    nb_modeles: int
    modeles: list[ModelMetric]


# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------

class DepartmentInfo(BaseModel):
    """Information about a department."""

    code: str = Field(..., examples=["69"])
    nom: str = Field(..., examples=["Rhone"])


class DepartmentsResponse(BaseModel):
    """Response for the GET /departments endpoint."""

    nb_departements: int
    departements: list[DepartmentInfo]
