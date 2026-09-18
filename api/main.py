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
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import (
    API_VERSION,
    DEPARTEMENTS,
    RAW_DIR,
    TARGET_COL,
    state,
)
from api.models import (
    ALLOWED_METRICS,
    ComparisonDepartment,
    ComparisonResponse,
    ComparisonValue,
    CustomPredictRequest,
    CustomPredictResponse,
    DataSummaryResponse,
    DepartmentInfo,
    DepartmentsResponse,
    FeatureImportance,
    FeatureImportanceResponse,
    HealthResponse,
    ModelMetric,
    ModelMetricsResponse,
    PredictionPoint,
    PredictionResponse,
    ScenarioPoint,
    ScenarioRequest,
    ScenarioResponse,
    SourceSummary,
    TrendPoint,
    TrendResponse,
)

# ---------------------------------------------------------------------------
# Rate limiting (optional — graceful fallback if slowapi is not installed)
# ---------------------------------------------------------------------------

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
    _RATE_LIMITING_ENABLED = True
except ImportError:
    limiter = None  # type: ignore[assignment]
    _RATE_LIMITING_ENABLED = False


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
        "Prediction API for the HVAC market (heat pumps) in France.\n\n"
        "Provides monthly forecasts by department, trained model metrics, "
        "and a summary of available data."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

# Attach rate limiter (if available)
if _RATE_LIMITING_ENABLED:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Check the API status and return version information."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        primary_model="ridge",
        nb_features=len(state.feature_names),
        last_training_date=state.model_date,
        uptime_seconds=round(time.time() - state.start_time, 1),
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
        description="Department code (e.g., 69, 2A)",
        examples=["69"],
    ),
    horizon: int = Query(
        default=6,
        ge=1,
        le=24,
        description="Number of months to predict (1-24)",
    ),
) -> PredictionResponse:
    """
    Generate predictions of heat pump installation counts for a department.

    Uses the trained Ridge model, scaler, and imputer loaded at
    startup. Predictions are generated iteratively from the
    last known observation for the department.
    """
    dept = departement.upper().zfill(2)
    if dept not in DEPARTEMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department code '{dept}'. Must be a valid French metropolitan department (01-95, 2A, 2B).",
        )
    _validate_department_in_data(dept)

    predictions = state.predict(dept, horizon)
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for department {dept}",
        )

    return PredictionResponse(
        departement=dept,
        horizon_months=horizon,
        model_used="ridge",
        target_variable=TARGET_COL,
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
            detail=f"No data found for department {dept}",
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
        model_used="ridge",
        horizon_months=body.horizon,
        model_confidence_r2=round(r2, 4),
        predictions=[PredictionPoint(**p) for p in predictions],
    )


# ---------------------------------------------------------------------------
# GET /data/summary
# ---------------------------------------------------------------------------

@app.get("/data/summary", response_model=DataSummaryResponse, tags=["Data"])
async def data_summary() -> DataSummaryResponse:
    """
    Summary of available data.

    Number of departments, date range, row counts per source,
    and last modification dates of raw files.
    """
    df = state.features_df
    nb_depts = df["dept"].nunique() if df is not None else 0

    # Date range
    plage: dict[str, str] = {"start": "", "end": ""}
    if df is not None and "date_id" in df.columns:
        dates = df["date_id"].dropna().astype(str)
        plage = {"start": dates.min(), "end": dates.max()}

    row_count = len(df) if df is not None else 0

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
                        name=f"{sub.name}/{csv_file.name}",
                        row_count=max(n_lines, 0),
                        last_modified=datetime.fromtimestamp(
                            stat.st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                    )
                )

    return DataSummaryResponse(
        department_count=nb_depts,
        date_range=plage,
        feature_row_count=row_count,
        raw_sources=sources,
    )


# ---------------------------------------------------------------------------
# GET /model/metrics
# ---------------------------------------------------------------------------

@app.get("/model/metrics", response_model=ModelMetricsResponse, tags=["Models"])
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
            detail="No training results available",
        )

    models: list[ModelMetric] = []
    for _, row in tr.iterrows():
        models.append(
            ModelMetric(
                model=str(row.get("model", "")),
                target=str(row.get("target", TARGET_COL)),
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
    valid = [m for m in models if m.test_rmse is not None]
    best = min(valid, key=lambda m: m.test_rmse).model if valid else "unknown"

    return ModelMetricsResponse(
        best_model=best,
        model_count=len(models),
        models=models,
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
        DepartmentInfo(code=code, name=name)
        for code, name in sorted(DEPARTEMENTS.items())
    ]
    return DepartmentsResponse(
        department_count=len(dept_list),
        departments=dept_list,
    )


# ---------------------------------------------------------------------------
# GET /trends/{dept}
# ---------------------------------------------------------------------------

@app.get("/trends/{dept}", response_model=TrendResponse, tags=["Analysis"])
async def get_trends(dept: str) -> TrendResponse:
    """
    Monthly trend data for a specific department.

    Returns historical actual values and in-sample model predictions
    for each month in the dataset.
    """
    dept = dept.upper().zfill(2)
    if dept not in DEPARTEMENTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid department code '{dept}'. "
                "Must be a valid French metropolitan department (01-95, 2A, 2B)."
            ),
        )
    _validate_department_in_data(dept)

    df = state.features_df
    df_dept = df[df["dept"] == dept].sort_values("date_id").copy()
    if df_dept.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for department {dept}",
        )

    # Generate in-sample predictions using the Ridge model
    points: list[TrendPoint] = []
    for _, row in df_dept.iterrows():
        date_str = str(int(row["date_id"]))
        date_label = f"{date_str[:4]}-{date_str[4:]}"
        actual = _safe_float(row.get(TARGET_COL))

        # Compute in-sample prediction
        predicted = None
        try:
            feat_vals = row[state.feature_names].values.reshape(1, -1).astype(float)
            feat_vals = state.imputer.transform(feat_vals)
            feat_scaled = state.scaler.transform(feat_vals)
            pred = float(state.ridge_model.predict(feat_scaled)[0])
            predicted = round(max(pred, 0.0), 2)
        except Exception:
            predicted = None

        points.append(TrendPoint(
            date=date_label,
            actual=actual,
            predicted=predicted,
        ))

    return TrendResponse(
        departement=dept,
        dept_name=DEPARTEMENTS.get(dept, dept),
        points=points,
    )


# ---------------------------------------------------------------------------
# GET /comparison
# ---------------------------------------------------------------------------

@app.get("/comparison", response_model=ComparisonResponse, tags=["Analysis"])
async def get_comparison(
    depts: str = Query(
        ...,
        description="Comma-separated department codes (max 5)",
        examples=["69,38,75"],
    ),
    metric: str = Query(
        default="nb_installations_pac",
        description="Metric column to compare",
    ),
) -> ComparisonResponse:
    """
    Compare a metric across multiple departments.

    Returns time-series data for up to 5 departments for a given
    metric column from the features dataset.
    """
    # Parse and validate department codes
    dept_list = [d.strip().upper().zfill(2) for d in depts.split(",") if d.strip()]
    if not dept_list:
        raise HTTPException(status_code=400, detail="No department codes provided.")
    if len(dept_list) > 5:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum 5 departments allowed, got {len(dept_list)}.",
        )
    for d in dept_list:
        if d not in DEPARTEMENTS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid department code '{d}'. "
                    "Must be a valid French metropolitan department (01-95, 2A, 2B)."
                ),
            )

    # Validate metric
    if metric not in ALLOWED_METRICS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid metric '{metric}'. "
                f"Allowed values: {sorted(ALLOWED_METRICS)}"
            ),
        )

    df = state.features_df
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded.")

    if metric not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Metric '{metric}' not found in dataset.",
        )

    departments: list[ComparisonDepartment] = []
    for d in dept_list:
        df_dept = df[df["dept"] == d].sort_values("date_id")
        if df_dept.empty:
            continue
        values: list[ComparisonValue] = []
        for _, row in df_dept.iterrows():
            date_str = str(int(row["date_id"]))
            date_label = f"{date_str[:4]}-{date_str[4:]}"
            val = _safe_float(row.get(metric))
            if val is not None:
                values.append(ComparisonValue(date=date_label, value=val))
        departments.append(ComparisonDepartment(
            dept=d,
            dept_name=DEPARTEMENTS.get(d, d),
            values=values,
        ))

    return ComparisonResponse(metric=metric, departments=departments)


# ---------------------------------------------------------------------------
# GET /features/importance
# ---------------------------------------------------------------------------

@app.get(
    "/features/importance",
    response_model=FeatureImportanceResponse,
    tags=["Models"],
)
async def feature_importance() -> FeatureImportanceResponse:
    """
    Feature importance from the trained LightGBM model.

    Returns the top 20 features ranked by importance. Falls back
    to the Ridge model coefficients if LightGBM is unavailable.
    """
    # Try LightGBM first (has .feature_importances_)
    if state.lgb_model is not None:
        try:
            importances = state.lgb_model.feature_importances_
            names = list(state.lgb_model.feature_name_)
            model_name = "lightgbm"
        except AttributeError:
            importances = None
            names = None
            model_name = "lightgbm"
    else:
        importances = None
        names = None
        model_name = "lightgbm"

    # Fallback to Ridge coefficients
    if importances is None and state.ridge_model is not None:
        try:
            importances = np.abs(state.ridge_model.coef_)
            names = list(state.ridge_model.feature_names_in_)
            model_name = "ridge"
        except AttributeError:
            pass

    if importances is None or names is None:
        raise HTTPException(
            status_code=404,
            detail="No model with feature importance data available.",
        )

    # Sort by importance descending and take top 20
    pairs = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
    top_20 = pairs[:20]

    features = [
        FeatureImportance(feature=name, importance=round(float(imp), 6))
        for name, imp in top_20
    ]

    return FeatureImportanceResponse(
        model=model_name,
        feature_count=len(features),
        features=features,
    )


# ---------------------------------------------------------------------------
# POST /scenario
# ---------------------------------------------------------------------------

@app.post("/scenario", response_model=ScenarioResponse, tags=["Predictions"])
async def run_scenario(body: ScenarioRequest) -> ScenarioResponse:
    """
    What-if scenario analysis.

    Generates a baseline prediction and an adjusted prediction
    with feature multipliers applied, then computes the percentage
    impact on total installations.
    """
    dept = body.dept
    _validate_department_in_data(dept)

    # Generate baseline predictions
    baseline_raw = state.predict(dept, body.horizon_months)
    if not baseline_raw:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for department {dept}",
        )

    baseline = [
        ScenarioPoint(date=p["date"], value=p["predicted_value"])
        for p in baseline_raw
    ]

    # Generate adjusted predictions (apply multipliers)
    if body.adjustments:
        extra: dict[str, float] = {}
        df = state.features_df
        df_dept = df[df["dept"] == dept].sort_values("date_id")
        if not df_dept.empty:
            last_row = df_dept.iloc[-1]
            for feat, multiplier in body.adjustments.items():
                if feat in last_row.index:
                    extra[feat] = float(last_row[feat]) * multiplier

        adjusted_raw = state.predict(
            dept, body.horizon_months, extra_features=extra if extra else None
        )
    else:
        adjusted_raw = baseline_raw

    adjusted = [
        ScenarioPoint(date=p["date"], value=p["predicted_value"])
        for p in adjusted_raw
    ]

    # Compute impact percentage
    baseline_total = sum(p.value for p in baseline)
    adjusted_total = sum(p.value for p in adjusted)
    if baseline_total > 0:
        impact_pct = round(
            (adjusted_total - baseline_total) / baseline_total * 100, 2
        )
    else:
        impact_pct = 0.0

    return ScenarioResponse(
        departement=dept,
        baseline=baseline,
        adjusted=adjusted,
        impact_pct=impact_pct,
    )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _validate_department_in_data(dept: str) -> None:
    """Verify that the department exists in the features dataset."""
    if state.features_df is None:
        raise HTTPException(
            status_code=503,
            detail="Data not loaded — API is starting up",
        )
    available = set(state.features_df["dept"].unique())
    if dept not in available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Department '{dept}' not found in dataset. "
                f"Available departments: {sorted(available)}"
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
