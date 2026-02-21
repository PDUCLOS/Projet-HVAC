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
    primary_model: str = Field(..., examples=["ridge"])
    nb_features: int = Field(..., ge=0, examples=[81])
    last_training_date: str | None = Field(
        None, examples=["2025-02-17"]
    )
    uptime_seconds: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionPoint(BaseModel):
    """A single monthly prediction point."""

    date: str = Field(..., examples=["2026-01"])
    predicted_value: float = Field(..., examples=[25.4])
    lower_bound: float = Field(..., examples=[20.1])
    upper_bound: float = Field(..., examples=[30.7])


class PredictionResponse(BaseModel):
    """Response for the GET /predictions endpoint."""

    departement: str = Field(..., examples=["69"])
    horizon_months: int = Field(..., ge=1, le=24, examples=[6])
    model_used: str = Field(..., examples=["ridge"])
    target_variable: str = Field(
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
    def _normalize_dept(self) -> "CustomPredictRequest":
        """Normalize the department code to uppercase."""
        self.departement = self.departement.upper().zfill(2)
        return self


class CustomPredictResponse(BaseModel):
    """Response for the POST /predict endpoint."""

    departement: str
    model_used: str
    horizon_months: int
    model_confidence_r2: float
    predictions: list[PredictionPoint]


# ---------------------------------------------------------------------------
# Data and metrics
# ---------------------------------------------------------------------------

class SourceSummary(BaseModel):
    """Summary of a raw data source."""

    name: str
    row_count: int
    last_modified: str


class DataSummaryResponse(BaseModel):
    """Response for the GET /data/summary endpoint."""

    department_count: int
    date_range: dict[str, str] = Field(
        ..., examples=[{"start": "202107", "end": "202512"}]
    )
    feature_row_count: int
    raw_sources: list[SourceSummary]


class ModelMetric(BaseModel):
    """Metrics for a trained model."""

    model: str
    target: str
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

    best_model: str
    model_count: int
    models: list[ModelMetric]


# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------

class DepartmentInfo(BaseModel):
    """Information about a department."""

    code: str = Field(..., examples=["69"])
    name: str = Field(..., examples=["Rhone"])


class DepartmentsResponse(BaseModel):
    """Response for the GET /departments endpoint."""

    department_count: int
    departments: list[DepartmentInfo]
