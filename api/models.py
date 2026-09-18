"""
Pydantic models for the HVAC prediction API.

Defines request and response schemas for all endpoints,
with strict validation via Pydantic v2.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

# Valid French metropolitan department codes (01-19, 21-95, 2A, 2B)
_VALID_DEPT_PATTERN = re.compile(r"^(0[1-9]|[1-8]\d|9[0-5]|2[AB])$")

# Allowed metric names for the comparison endpoint
ALLOWED_METRICS: set[str] = {
    "nb_installations_pac",
    "nb_installations_clim",
    "nb_dpe_total",
    "nb_dpe_classe_ab",
    "temp_mean",
    "hdd_sum",
    "cdd_sum",
}


def _validate_dept_code(code: str) -> str:
    """Validate and normalize a French department code.

    Accepts codes like '1', '01', '69', '2a', '2A'.
    Returns the normalized uppercase, zero-padded code.

    Raises ValueError if the code does not match a valid
    metropolitan French department (01-95, 2A, 2B).
    """
    normalized = code.strip().upper().zfill(2)
    if not _VALID_DEPT_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid department code '{code}'. "
            "Must be a valid French metropolitan department (01-95, 2A, 2B)."
        )
    return normalized


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

    @field_validator("departement")
    @classmethod
    def _validate_dept(cls, v: str) -> str:
        """Validate and normalize the department code."""
        return _validate_dept_code(v)


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


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------

class TrendPoint(BaseModel):
    """A single point in a department trend series."""

    date: str = Field(..., examples=["2024-01"])
    actual: float | None = Field(None, examples=[42.0])
    predicted: float | None = Field(None, examples=[40.5])


class TrendResponse(BaseModel):
    """Response for the GET /trends/{dept} endpoint."""

    departement: str = Field(..., examples=["69"])
    dept_name: str = Field(..., examples=["Rhone"])
    points: list[TrendPoint]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class ComparisonValue(BaseModel):
    """A single date/value pair in a department comparison."""

    date: str = Field(..., examples=["2024-01"])
    value: float = Field(..., examples=[42.0])


class ComparisonDepartment(BaseModel):
    """Data series for one department in a comparison."""

    dept: str = Field(..., examples=["69"])
    dept_name: str = Field(..., examples=["Rhone"])
    values: list[ComparisonValue]


class ComparisonResponse(BaseModel):
    """Response for the GET /comparison endpoint."""

    metric: str = Field(..., examples=["nb_installations_pac"])
    departments: list[ComparisonDepartment]


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class FeatureImportance(BaseModel):
    """A single feature with its importance score."""

    feature: str = Field(..., examples=["temp_mean"])
    importance: float = Field(..., examples=[0.142])


class FeatureImportanceResponse(BaseModel):
    """Response for the GET /features/importance endpoint."""

    model: str = Field(default="lightgbm", examples=["lightgbm"])
    feature_count: int = Field(..., ge=0, examples=[20])
    features: list[FeatureImportance]


# ---------------------------------------------------------------------------
# Scenario (what-if)
# ---------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    """Request body for the POST /scenario endpoint."""

    dept: str = Field(
        ...,
        min_length=1,
        max_length=3,
        examples=["69"],
        description="Department code (01-95, 2A, 2B)",
    )
    horizon_months: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Number of months to forecast (1-24)",
    )
    adjustments: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Feature multipliers to apply. "
            "Example: {'temp_mean': 1.1} increases temperature by 10%."
        ),
    )

    @field_validator("dept")
    @classmethod
    def _validate_dept(cls, v: str) -> str:
        """Validate and normalize the department code."""
        return _validate_dept_code(v)

    @field_validator("horizon_months")
    @classmethod
    def _validate_horizon(cls, v: int) -> int:
        """Ensure horizon is within valid range."""
        if not 1 <= v <= 24:
            raise ValueError(
                f"horizon_months must be between 1 and 24, got {v}."
            )
        return v


class ScenarioPoint(BaseModel):
    """A single point in a scenario prediction."""

    date: str = Field(..., examples=["2026-01"])
    value: float = Field(..., examples=[25.4])


class ScenarioResponse(BaseModel):
    """Response for the POST /scenario endpoint."""

    departement: str = Field(..., examples=["69"])
    baseline: list[ScenarioPoint]
    adjusted: list[ScenarioPoint]
    impact_pct: float = Field(
        ...,
        description="Percentage change from baseline to adjusted total",
        examples=[12.5],
    )
