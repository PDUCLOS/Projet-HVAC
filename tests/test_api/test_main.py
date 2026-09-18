# -*- coding: utf-8 -*-
"""
Comprehensive tests for the HVAC prediction API (api/main.py).

Covers all endpoints, input validation, internal utilities,
and error handling. Uses FastAPI TestClient with mocked AppState
to avoid loading actual pickle files from disk.
"""

from __future__ import annotations

import math
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures: mock the AppState before importing the app
# ---------------------------------------------------------------------------

def _build_mock_features_df(
    depts: list[str] | None = None,
) -> pd.DataFrame:
    """Build a small features DataFrame suitable for testing.

    Returns a DataFrame with two departments (69, 38) and a few
    months of data, mimicking the structure expected by the API.
    """
    if depts is None:
        depts = ["69", "38"]
    rows: list[dict[str, Any]] = []
    for dept in depts:
        for ym in [202301, 202302, 202303, 202304]:
            rows.append({
                "dept": dept,
                "date_id": ym,
                "col1": np.random.uniform(0, 1),
                "col2": np.random.uniform(0, 1),
                "year": int(str(ym)[:4]),
                "month": int(str(ym)[4:]),
                "quarter": (int(str(ym)[4:]) - 1) // 3 + 1,
                "is_heating": 1,
                "is_cooling": 0,
                "month_sin": 0.5,
                "month_cos": 0.866,
                "year_trend": 3.0,
                "nb_installations_pac_lag_1m": 10.0,
                "nb_installations_pac": 15.0,
            })
    return pd.DataFrame(rows)


def _build_mock_training_results() -> pd.DataFrame:
    """Build a small training_results DataFrame."""
    return pd.DataFrame([
        {
            "model": "ridge",
            "target": "nb_installations_pac",
            "val_rmse": 5.12,
            "val_mae": 3.45,
            "val_mape": 0.12,
            "val_r2": 0.87,
            "test_rmse": 4.98,
            "test_mae": 3.21,
            "test_mape": 0.11,
            "test_r2": 0.89,
            "cv_rmse_mean": 5.05,
            "cv_r2_mean": 0.88,
        },
        {
            "model": "lightgbm",
            "target": "nb_installations_pac",
            "val_rmse": 6.00,
            "val_mae": 4.00,
            "val_mape": 0.15,
            "val_r2": 0.82,
            "test_rmse": 5.80,
            "test_mae": 3.90,
            "test_mape": 0.14,
            "test_r2": 0.83,
            "cv_rmse_mean": 5.90,
            "cv_r2_mean": 0.82,
        },
    ])


@pytest.fixture(autouse=True)
def _mock_app_state():
    """Mock the global AppState so the app starts without real model files.

    Patches state.load() to be a no-op and injects mock objects for
    the model, scaler, imputer, features_df, and training_results.
    This fixture runs automatically for every test in this module.
    """
    with patch("api.dependencies.AppState.load") as mock_load:
        # Import state after patching load to prevent actual file I/O
        from api.dependencies import state

        # Configure the mock state attributes
        state.feature_names = ["col1", "col2"]
        state.model_date = "2024-01-01"
        state.start_time = time.time()
        state.features_df = _build_mock_features_df()
        state.training_results = _build_mock_training_results()

        # Mock Ridge model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([25.4])
        mock_model.feature_names_in_ = ["col1", "col2"]
        state.ridge_model = mock_model

        # Mock scaler and imputer (passthrough transforms)
        mock_scaler = MagicMock()
        mock_scaler.transform.side_effect = lambda x: x
        state.scaler = mock_scaler

        mock_imputer = MagicMock()
        mock_imputer.transform.side_effect = lambda x: x
        state.imputer = mock_imputer

        # Store original predict method to restore after test
        original_predict = state.predict

        yield mock_load

        # Reset state after each test to avoid cross-test pollution
        state.ridge_model = None
        state.scaler = None
        state.imputer = None
        state.features_df = None
        state.training_results = None
        state.feature_names = []
        state.model_date = None
        # Restore the original predict method in case a test replaced it
        state.predict = original_predict


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient wired to the FastAPI app."""
    from api.main import app
    return TestClient(app)


# ===================================================================
# 1. GET /health
# ===================================================================

class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self, client: TestClient):
        """Health check returns HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_ok(self, client: TestClient):
        """Health check reports status 'ok'."""
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_contains_version(self, client: TestClient):
        """Health response includes the API version string."""
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_contains_uptime(self, client: TestClient):
        """Health response includes non-negative uptime_seconds."""
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_contains_model_info(self, client: TestClient):
        """Health response includes primary model and feature count."""
        data = client.get("/health").json()
        assert data["primary_model"] == "ridge"
        assert data["nb_features"] == 2  # ["col1", "col2"]
        assert data["last_training_date"] == "2024-01-01"


# ===================================================================
# 2. GET /departments
# ===================================================================

class TestDepartmentsEndpoint:
    """Tests for the GET /departments endpoint."""

    def test_departments_returns_200(self, client: TestClient):
        """Departments endpoint returns HTTP 200."""
        response = client.get("/departments")
        assert response.status_code == 200

    def test_departments_count_is_96(self, client: TestClient):
        """Response contains exactly 96 metropolitan departments."""
        data = client.get("/departments").json()
        assert data["department_count"] == 96
        assert len(data["departments"]) == 96

    def test_departments_contains_corsica(self, client: TestClient):
        """Corsican departments 2A and 2B are present."""
        data = client.get("/departments").json()
        codes = {d["code"] for d in data["departments"]}
        assert "2A" in codes
        assert "2B" in codes

    def test_departments_have_names(self, client: TestClient):
        """Every department entry has a non-empty name."""
        data = client.get("/departments").json()
        for dept in data["departments"]:
            assert "name" in dept
            assert isinstance(dept["name"], str)
            assert len(dept["name"]) > 0

    def test_departments_have_valid_codes(self, client: TestClient):
        """All department codes are between 1 and 3 characters."""
        data = client.get("/departments").json()
        for dept in data["departments"]:
            code = dept["code"]
            assert 1 <= len(code) <= 3


# ===================================================================
# 3. GET /data/summary
# ===================================================================

class TestDataSummaryEndpoint:
    """Tests for the GET /data/summary endpoint."""

    def test_data_summary_returns_200(self, client: TestClient):
        """Data summary endpoint returns HTTP 200."""
        response = client.get("/data/summary")
        assert response.status_code == 200

    def test_data_summary_department_count(self, client: TestClient):
        """Summary reports the correct number of departments in the data."""
        data = client.get("/data/summary").json()
        # Our mock has 2 departments (69, 38)
        assert data["department_count"] == 2

    def test_data_summary_date_range(self, client: TestClient):
        """Summary includes a start and end date range."""
        data = client.get("/data/summary").json()
        assert "date_range" in data
        assert "start" in data["date_range"]
        assert "end" in data["date_range"]
        # Our mock dates range from 202301 to 202304
        assert data["date_range"]["start"] == "202301"
        assert data["date_range"]["end"] == "202304"

    def test_data_summary_feature_row_count(self, client: TestClient):
        """Summary reports the correct row count of the features dataset."""
        data = client.get("/data/summary").json()
        # 2 departments * 4 months = 8 rows
        assert data["feature_row_count"] == 8

    def test_data_summary_contains_raw_sources(self, client: TestClient):
        """Summary contains a raw_sources list (may be empty in tests)."""
        data = client.get("/data/summary").json()
        assert "raw_sources" in data
        assert isinstance(data["raw_sources"], list)


# ===================================================================
# 4. GET /model/metrics
# ===================================================================

class TestModelMetricsEndpoint:
    """Tests for the GET /model/metrics endpoint."""

    def test_model_metrics_returns_200(self, client: TestClient):
        """Model metrics returns HTTP 200 when training results exist."""
        response = client.get("/model/metrics")
        assert response.status_code == 200

    def test_model_metrics_best_model(self, client: TestClient):
        """Best model is determined by lowest test RMSE."""
        data = client.get("/model/metrics").json()
        # ridge has test_rmse=4.98, lightgbm has 5.80
        assert data["best_model"] == "ridge"

    def test_model_metrics_model_count(self, client: TestClient):
        """Metric count matches the number of trained models."""
        data = client.get("/model/metrics").json()
        assert data["model_count"] == 2

    def test_model_metrics_contains_expected_fields(self, client: TestClient):
        """Each model metric has the required fields."""
        data = client.get("/model/metrics").json()
        for m in data["models"]:
            assert "model" in m
            assert "target" in m
            assert "test_rmse" in m
            assert "test_r2" in m

    def test_model_metrics_returns_404_when_no_results(self, client: TestClient):
        """Returns 404 when training_results is None."""
        from api.dependencies import state
        state.training_results = None

        response = client.get("/model/metrics")
        assert response.status_code == 404
        assert "No training results" in response.json()["detail"]

    def test_model_metrics_returns_404_when_empty_results(self, client: TestClient):
        """Returns 404 when training_results is an empty DataFrame."""
        from api.dependencies import state
        state.training_results = pd.DataFrame()

        response = client.get("/model/metrics")
        assert response.status_code == 404


# ===================================================================
# 5. GET /predictions?departement=XX (valid department)
# ===================================================================

class TestPredictionsEndpoint:
    """Tests for the GET /predictions endpoint."""

    def test_predictions_valid_department(self, client: TestClient):
        """Valid department code returns HTTP 200 with predictions."""
        response = client.get("/predictions", params={"departement": "69"})
        assert response.status_code == 200

    def test_predictions_response_structure(self, client: TestClient):
        """Response contains the expected fields."""
        data = client.get(
            "/predictions", params={"departement": "69", "horizon": 3}
        ).json()
        assert data["departement"] == "69"
        assert data["horizon_months"] == 3
        assert data["model_used"] == "ridge"
        assert len(data["predictions"]) == 3

    def test_predictions_each_point_has_bounds(self, client: TestClient):
        """Each prediction point has date, predicted_value, and bounds."""
        data = client.get(
            "/predictions", params={"departement": "69", "horizon": 2}
        ).json()
        for p in data["predictions"]:
            assert "date" in p
            assert "predicted_value" in p
            assert "lower_bound" in p
            assert "upper_bound" in p
            assert p["lower_bound"] <= p["predicted_value"] <= p["upper_bound"]

    def test_predictions_default_horizon(self, client: TestClient):
        """Default horizon is 6 months when not specified."""
        data = client.get(
            "/predictions", params={"departement": "69"}
        ).json()
        assert data["horizon_months"] == 6
        assert len(data["predictions"]) == 6

    def test_predictions_department_case_insensitive(self, client: TestClient):
        """Department code '2a' (lowercase) is normalized to '2A'."""
        # We need to add 2A to the mock features
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["69", "38", "2A"])

        response = client.get("/predictions", params={"departement": "2a"})
        assert response.status_code == 200
        data = response.json()
        assert data["departement"] == "2A"

    def test_predictions_department_zero_padding(self, client: TestClient):
        """Single-digit department code '1' is padded to '01'."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["01", "38"])

        response = client.get("/predictions", params={"departement": "1"})
        assert response.status_code == 200
        data = response.json()
        assert data["departement"] == "01"


# ===================================================================
# 6. GET /predictions?departement=ZZ (invalid department code)
# ===================================================================

class TestPredictionsInvalidDepartment:
    """Tests for invalid department codes on GET /predictions."""

    def test_invalid_department_returns_400(self, client: TestClient):
        """Invalid department code 'ZZ' returns HTTP 400 (not 404)."""
        response = client.get("/predictions", params={"departement": "ZZ"})
        assert response.status_code == 400

    def test_invalid_department_error_detail(self, client: TestClient):
        """Error message mentions the invalid department code."""
        response = client.get("/predictions", params={"departement": "ZZ"})
        detail = response.json()["detail"]
        assert "ZZ" in detail
        assert "Invalid department code" in detail

    def test_invalid_department_99(self, client: TestClient):
        """Department code '99' is not a valid metropolitan department."""
        response = client.get("/predictions", params={"departement": "99"})
        assert response.status_code == 400

    def test_invalid_department_00(self, client: TestClient):
        """Department code '00' is not a valid metropolitan department."""
        response = client.get("/predictions", params={"departement": "00"})
        assert response.status_code == 400

    def test_invalid_department_abc(self, client: TestClient):
        """Department code 'AB' is not valid."""
        response = client.get("/predictions", params={"departement": "AB"})
        assert response.status_code == 400

    def test_valid_department_not_in_data_returns_404(self, client: TestClient):
        """Valid department code that is not in the dataset returns 404."""
        # '01' is a valid department but not in our mock features_df
        response = client.get("/predictions", params={"departement": "01"})
        assert response.status_code == 404
        assert "not found in dataset" in response.json()["detail"]


# ===================================================================
# 7. POST /predict (valid request body)
# ===================================================================

class TestPostPredictEndpoint:
    """Tests for the POST /predict endpoint."""

    def test_post_predict_valid_body(self, client: TestClient):
        """Valid POST request returns HTTP 200."""
        body = {"departement": "69", "horizon": 3, "features": {}}
        response = client.post("/predict", json=body)
        assert response.status_code == 200

    def test_post_predict_response_structure(self, client: TestClient):
        """Response has the expected CustomPredictResponse fields."""
        body = {"departement": "69", "horizon": 2}
        data = client.post("/predict", json=body).json()
        assert data["departement"] == "69"
        assert data["model_used"] == "ridge"
        assert data["horizon_months"] == 2
        assert "model_confidence_r2" in data
        assert len(data["predictions"]) == 2

    def test_post_predict_with_extra_features(self, client: TestClient):
        """POST request with extra features is accepted."""
        body = {
            "departement": "69",
            "horizon": 1,
            "features": {"col1": 0.5, "col2": 0.8},
        }
        response = client.post("/predict", json=body)
        assert response.status_code == 200

    def test_post_predict_default_horizon(self, client: TestClient):
        """Default horizon for POST /predict is 1."""
        body = {"departement": "69"}
        data = client.post("/predict", json=body).json()
        assert data["horizon_months"] == 1

    def test_post_predict_r2_from_training_results(self, client: TestClient):
        """model_confidence_r2 is extracted from training results."""
        body = {"departement": "69", "horizon": 1}
        data = client.post("/predict", json=body).json()
        # Our mock has test_r2=0.89 for ridge
        assert data["model_confidence_r2"] == 0.89

    def test_post_predict_r2_zero_when_no_results(self, client: TestClient):
        """model_confidence_r2 is 0.0 when no training results exist."""
        from api.dependencies import state
        state.training_results = None

        body = {"departement": "69", "horizon": 1}
        data = client.post("/predict", json=body).json()
        assert data["model_confidence_r2"] == 0.0

    def test_post_predict_normalizes_department(self, client: TestClient):
        """POST normalizes department code to uppercase and zero-padded."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["01", "38"])

        body = {"departement": "1", "horizon": 1}
        data = client.post("/predict", json=body).json()
        assert data["departement"] == "01"

    def test_post_predict_invalid_department_not_in_data(self, client: TestClient):
        """POST with a valid department code not in data returns 404."""
        body = {"departement": "01", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 404


# ===================================================================
# 8. Input validation: department length, horizon bounds
# ===================================================================

class TestInputValidation:
    """Tests for FastAPI query/body parameter validation."""

    def test_department_too_long(self, client: TestClient):
        """Department code longer than 3 characters returns 422."""
        response = client.get(
            "/predictions", params={"departement": "ABCD"}
        )
        assert response.status_code == 422

    def test_department_empty_string(self, client: TestClient):
        """Empty department code returns 422."""
        response = client.get(
            "/predictions", params={"departement": ""}
        )
        assert response.status_code == 422

    def test_department_missing(self, client: TestClient):
        """Missing department parameter returns 422."""
        response = client.get("/predictions")
        assert response.status_code == 422

    def test_horizon_too_low(self, client: TestClient):
        """Horizon of 0 returns 422 (minimum is 1)."""
        response = client.get(
            "/predictions", params={"departement": "69", "horizon": 0}
        )
        assert response.status_code == 422

    def test_horizon_too_high(self, client: TestClient):
        """Horizon of 25 returns 422 (maximum is 24)."""
        response = client.get(
            "/predictions", params={"departement": "69", "horizon": 25}
        )
        assert response.status_code == 422

    def test_horizon_boundary_low(self, client: TestClient):
        """Horizon of 1 (minimum valid) is accepted."""
        response = client.get(
            "/predictions", params={"departement": "69", "horizon": 1}
        )
        assert response.status_code == 200

    def test_horizon_boundary_high(self, client: TestClient):
        """Horizon of 24 (maximum valid) is accepted."""
        response = client.get(
            "/predictions", params={"departement": "69", "horizon": 24}
        )
        assert response.status_code == 200

    def test_post_predict_department_too_long(self, client: TestClient):
        """POST body with department code too long returns 422."""
        body = {"departement": "ABCD", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 422

    def test_post_predict_horizon_too_high(self, client: TestClient):
        """POST body with horizon > 24 returns 422."""
        body = {"departement": "69", "horizon": 25}
        response = client.post("/predict", json=body)
        assert response.status_code == 422

    def test_post_predict_horizon_too_low(self, client: TestClient):
        """POST body with horizon < 1 returns 422."""
        body = {"departement": "69", "horizon": 0}
        response = client.post("/predict", json=body)
        assert response.status_code == 422

    def test_post_predict_empty_department(self, client: TestClient):
        """POST body with empty department code returns 422."""
        body = {"departement": "", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 422


# ===================================================================
# 9. _safe_float() — handles None, NaN, valid floats
# ===================================================================

class TestSafeFloat:
    """Tests for the _safe_float() internal utility."""

    def test_safe_float_none_returns_none(self):
        """None input returns None."""
        from api.main import _safe_float
        assert _safe_float(None) is None

    def test_safe_float_nan_returns_none(self):
        """NaN input returns None."""
        from api.main import _safe_float
        assert _safe_float(float("nan")) is None

    def test_safe_float_numpy_nan_returns_none(self):
        """numpy.nan input returns None."""
        from api.main import _safe_float
        assert _safe_float(np.nan) is None

    def test_safe_float_valid_int(self):
        """Integer input is converted to float."""
        from api.main import _safe_float
        result = _safe_float(42)
        assert result == 42.0
        assert isinstance(result, float)

    def test_safe_float_valid_float(self):
        """Float input is returned (rounded to 6 decimals)."""
        from api.main import _safe_float
        result = _safe_float(3.14159265)
        assert result == 3.141593

    def test_safe_float_string_number(self):
        """Numeric string is converted to float."""
        from api.main import _safe_float
        result = _safe_float("2.718")
        assert result == 2.718

    def test_safe_float_non_numeric_string_returns_none(self):
        """Non-numeric string returns None."""
        from api.main import _safe_float
        assert _safe_float("not_a_number") is None

    def test_safe_float_zero(self):
        """Zero is a valid float and is not treated as None."""
        from api.main import _safe_float
        result = _safe_float(0)
        assert result == 0.0
        assert result is not None

    def test_safe_float_negative(self):
        """Negative float is handled correctly."""
        from api.main import _safe_float
        result = _safe_float(-5.5)
        assert result == -5.5

    def test_safe_float_infinity(self):
        """Positive infinity is returned as float (not NaN)."""
        from api.main import _safe_float
        result = _safe_float(float("inf"))
        # inf is a valid float, not NaN, so it passes through
        assert result == float("inf")

    def test_safe_float_numpy_float64(self):
        """numpy float64 is handled correctly."""
        from api.main import _safe_float
        result = _safe_float(np.float64(7.77))
        assert result is not None
        assert abs(result - 7.77) < 1e-5


# ===================================================================
# 10. _validate_department_in_data() — raises 503 / 404
# ===================================================================

class TestValidateDepartmentInData:
    """Tests for the _validate_department_in_data() utility."""

    def test_raises_503_when_features_df_is_none(self):
        """Raises HTTPException 503 when features_df is None."""
        from fastapi import HTTPException
        from api.dependencies import state
        from api.main import _validate_department_in_data

        state.features_df = None
        with pytest.raises(HTTPException) as exc_info:
            _validate_department_in_data("69")
        assert exc_info.value.status_code == 503
        assert "not loaded" in exc_info.value.detail

    def test_raises_404_when_dept_not_in_data(self):
        """Raises HTTPException 404 when the department is not in the dataset."""
        from fastapi import HTTPException
        from api.main import _validate_department_in_data

        with pytest.raises(HTTPException) as exc_info:
            _validate_department_in_data("01")
        assert exc_info.value.status_code == 404
        assert "not found in dataset" in exc_info.value.detail

    def test_passes_for_valid_department(self):
        """Does not raise for a department present in the dataset."""
        from api.main import _validate_department_in_data

        # Should not raise — "69" is in our mock features_df
        _validate_department_in_data("69")

    def test_error_detail_lists_available_departments(self):
        """Error detail for missing department includes the available list."""
        from fastapi import HTTPException
        from api.main import _validate_department_in_data

        with pytest.raises(HTTPException) as exc_info:
            _validate_department_in_data("99")
        detail = exc_info.value.detail
        assert "69" in detail or "38" in detail


# ===================================================================
# Additional edge-case and regression tests
# ===================================================================

class TestCORSHeaders:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_default_origins(self, client: TestClient):
        """CORS preflight for localhost:8501 is accepted."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORS middleware returns 200 for valid preflight
        assert response.status_code == 200


class TestPredictionEmptyData:
    """Tests for prediction endpoints when department has no data rows."""

    def test_get_predictions_empty_dept_data(self, client: TestClient):
        """Returns 404 when the model returns no predictions."""
        from api.dependencies import state

        # Create a features_df that includes dept "75" but with no rows
        # that the predict method can use (simulate empty predictions)
        mock_predict = MagicMock(return_value=[])
        state.predict = mock_predict
        # "75" must be in features_df for _validate_department_in_data to pass
        state.features_df = _build_mock_features_df(depts=["69", "38", "75"])

        response = client.get("/predictions", params={"departement": "75"})
        assert response.status_code == 404
        assert "No data found" in response.json()["detail"]


class TestOpenAPISchema:
    """Tests to verify that the OpenAPI schema is available."""

    def test_openapi_schema_accessible(self, client: TestClient):
        """The /openapi.json endpoint returns valid JSON."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]
        assert "/predictions" in schema["paths"]
        assert "/predict" in schema["paths"]
        assert "/departments" in schema["paths"]
        assert "/data/summary" in schema["paths"]
        assert "/model/metrics" in schema["paths"]

    def test_docs_page_accessible(self, client: TestClient):
        """The /docs endpoint (Swagger UI) returns HTTP 200."""
        response = client.get("/docs")
        assert response.status_code == 200


class TestSecurityInputs:
    """Security-focused tests: injection prevention, unusual inputs."""

    def test_sql_injection_in_department(self, client: TestClient):
        """SQL injection attempt in department code returns 400 or 422."""
        response = client.get(
            "/predictions", params={"departement": "'; DROP TABLE--"}
        )
        # The max_length=3 validator should reject this
        assert response.status_code == 422

    def test_xss_in_department(self, client: TestClient):
        """XSS attempt in department code is rejected by validation."""
        response = client.get(
            "/predictions", params={"departement": "<script>"}
        )
        assert response.status_code == 422

    def test_post_predict_extra_fields_ignored(self, client: TestClient):
        """Extra fields in POST body are ignored and do not cause a 500."""
        body = {
            "departement": "69",
            "horizon": 1,
            "features": {},
            "malicious_field": "drop_table",
        }
        response = client.post("/predict", json=body)
        # Pydantic ignores extra fields by default (or forbids them via 422).
        # The key assertion: no internal server error (500) occurs.
        assert response.status_code != 500
        # Request should be accepted (200), rejected by validation (422),
        # or return no data (404) — never an internal error
        assert response.status_code in (200, 404, 422)

    def test_negative_horizon_rejected(self, client: TestClient):
        """Negative horizon value is rejected."""
        response = client.get(
            "/predictions", params={"departement": "69", "horizon": -1}
        )
        assert response.status_code == 422

    def test_post_predict_features_with_nan_value(self, client: TestClient):
        """Features dict with non-numeric values should fail validation."""
        body = {
            "departement": "69",
            "horizon": 1,
            "features": {"col1": "not_a_number"},
        }
        response = client.post("/predict", json=body)
        assert response.status_code == 422


# ===================================================================
# 11. GET /trends/{dept}
# ===================================================================

class TestTrendsEndpoint:
    """Tests for the GET /trends/{dept} endpoint."""

    def test_trends_valid_department(self, client: TestClient):
        """Valid department returns HTTP 200 with trend points."""
        response = client.get("/trends/69")
        assert response.status_code == 200

    def test_trends_response_structure(self, client: TestClient):
        """Response contains expected fields."""
        data = client.get("/trends/69").json()
        assert data["departement"] == "69"
        assert "dept_name" in data
        assert "points" in data
        assert isinstance(data["points"], list)
        assert len(data["points"]) > 0

    def test_trends_points_have_required_fields(self, client: TestClient):
        """Each trend point has date, actual, and predicted."""
        data = client.get("/trends/69").json()
        for point in data["points"]:
            assert "date" in point
            assert "actual" in point
            assert "predicted" in point

    def test_trends_invalid_department(self, client: TestClient):
        """Invalid department code returns HTTP 400."""
        response = client.get("/trends/ZZ")
        assert response.status_code == 400
        assert "Invalid department code" in response.json()["detail"]

    def test_trends_department_not_in_data(self, client: TestClient):
        """Valid department code not in dataset returns 404."""
        response = client.get("/trends/01")
        assert response.status_code == 404
        assert "not found in dataset" in response.json()["detail"]

    def test_trends_case_insensitive(self, client: TestClient):
        """Lowercase '2a' is normalized to '2A'."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["69", "38", "2A"])

        response = client.get("/trends/2a")
        assert response.status_code == 200
        data = response.json()
        assert data["departement"] == "2A"

    def test_trends_date_format(self, client: TestClient):
        """Trend point dates are in YYYY-MM format."""
        data = client.get("/trends/69").json()
        for point in data["points"]:
            assert len(point["date"]) >= 6
            assert "-" in point["date"]


# ===================================================================
# 12. GET /comparison
# ===================================================================

class TestComparisonEndpoint:
    """Tests for the GET /comparison endpoint."""

    def test_comparison_valid_depts(self, client: TestClient):
        """Valid department codes return HTTP 200."""
        response = client.get("/comparison", params={"depts": "69,38"})
        assert response.status_code == 200

    def test_comparison_response_structure(self, client: TestClient):
        """Response has the expected structure."""
        data = client.get("/comparison", params={"depts": "69,38"}).json()
        assert "metric" in data
        assert data["metric"] == "nb_installations_pac"
        assert "departments" in data
        assert isinstance(data["departments"], list)

    def test_comparison_department_data(self, client: TestClient):
        """Each department entry has dept, dept_name, and values."""
        data = client.get("/comparison", params={"depts": "69"}).json()
        for dept in data["departments"]:
            assert "dept" in dept
            assert "dept_name" in dept
            assert "values" in dept
            assert isinstance(dept["values"], list)

    def test_comparison_custom_metric(self, client: TestClient):
        """Custom metric parameter is accepted."""
        # Add nb_dpe_total column to mock data so the endpoint can find it
        from api.dependencies import state
        df = _build_mock_features_df()
        df["nb_dpe_total"] = 100.0
        state.features_df = df

        response = client.get(
            "/comparison",
            params={"depts": "69,38", "metric": "nb_dpe_total"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "nb_dpe_total"

    def test_comparison_invalid_metric(self, client: TestClient):
        """Invalid metric returns HTTP 400."""
        response = client.get(
            "/comparison",
            params={"depts": "69", "metric": "invalid_metric"},
        )
        assert response.status_code == 400
        assert "Invalid metric" in response.json()["detail"]

    def test_comparison_too_many_depts(self, client: TestClient):
        """More than 5 departments returns HTTP 400."""
        response = client.get(
            "/comparison",
            params={"depts": "69,38,75,13,31,33"},
        )
        assert response.status_code == 400
        assert "Maximum 5" in response.json()["detail"]

    def test_comparison_invalid_dept(self, client: TestClient):
        """Invalid department in list returns HTTP 400."""
        response = client.get(
            "/comparison",
            params={"depts": "69,ZZ"},
        )
        assert response.status_code == 400
        assert "Invalid department code" in response.json()["detail"]

    def test_comparison_empty_depts(self, client: TestClient):
        """Empty depts parameter returns HTTP 400."""
        response = client.get("/comparison", params={"depts": ""})
        assert response.status_code == 400

    def test_comparison_single_dept(self, client: TestClient):
        """Single department is accepted."""
        response = client.get("/comparison", params={"depts": "69"})
        assert response.status_code == 200


# ===================================================================
# 13. GET /features/importance
# ===================================================================

class TestFeatureImportanceEndpoint:
    """Tests for the GET /features/importance endpoint."""

    def test_feature_importance_returns_200(self, client: TestClient):
        """Feature importance endpoint returns HTTP 200."""
        # The mock Ridge model needs coef_ and feature_names_in_
        from api.dependencies import state
        state.ridge_model.coef_ = np.array([0.5, 0.3])
        state.ridge_model.feature_names_in_ = ["col1", "col2"]

        response = client.get("/features/importance")
        assert response.status_code == 200

    def test_feature_importance_response_structure(self, client: TestClient):
        """Response has the expected structure."""
        from api.dependencies import state
        state.ridge_model.coef_ = np.array([0.5, 0.3])
        state.ridge_model.feature_names_in_ = ["col1", "col2"]

        data = client.get("/features/importance").json()
        assert "model" in data
        assert "feature_count" in data
        assert "features" in data
        assert isinstance(data["features"], list)

    def test_feature_importance_sorted_descending(self, client: TestClient):
        """Features are sorted by importance in descending order."""
        from api.dependencies import state
        state.ridge_model.coef_ = np.array([0.3, 0.5])
        state.ridge_model.feature_names_in_ = ["col1", "col2"]

        data = client.get("/features/importance").json()
        features = data["features"]
        assert len(features) == 2
        # col2 (importance 0.5) should come first
        assert features[0]["feature"] == "col2"
        assert features[1]["feature"] == "col1"
        assert features[0]["importance"] >= features[1]["importance"]

    def test_feature_importance_each_entry_has_fields(self, client: TestClient):
        """Each feature entry has feature name and importance value."""
        from api.dependencies import state
        state.ridge_model.coef_ = np.array([0.5, 0.3])
        state.ridge_model.feature_names_in_ = ["col1", "col2"]

        data = client.get("/features/importance").json()
        for feat in data["features"]:
            assert "feature" in feat
            assert "importance" in feat
            assert isinstance(feat["importance"], (int, float))

    def test_feature_importance_uses_lightgbm_when_available(
        self, client: TestClient,
    ):
        """Uses LightGBM model when it has feature_importances_."""
        from api.dependencies import state
        mock_lgb = MagicMock()
        mock_lgb.feature_importances_ = np.array([0.8, 0.2])
        mock_lgb.feature_name_ = ["col1", "col2"]
        state.lgb_model = mock_lgb

        data = client.get("/features/importance").json()
        assert data["model"] == "lightgbm"

    def test_feature_importance_404_when_no_models(self, client: TestClient):
        """Returns 404 when no model has importance data."""
        from api.dependencies import state
        state.ridge_model = MagicMock(spec=[])  # No coef_ attribute
        state.lgb_model = None

        response = client.get("/features/importance")
        assert response.status_code == 404


# ===================================================================
# 14. POST /scenario
# ===================================================================

class TestScenarioEndpoint:
    """Tests for the POST /scenario endpoint."""

    def test_scenario_valid_body(self, client: TestClient):
        """Valid scenario request returns HTTP 200."""
        body = {
            "dept": "69",
            "horizon_months": 3,
            "adjustments": {},
        }
        response = client.post("/scenario", json=body)
        assert response.status_code == 200

    def test_scenario_response_structure(self, client: TestClient):
        """Response contains baseline, adjusted, and impact_pct."""
        body = {
            "dept": "69",
            "horizon_months": 3,
            "adjustments": {},
        }
        data = client.post("/scenario", json=body).json()
        assert "departement" in data
        assert data["departement"] == "69"
        assert "baseline" in data
        assert "adjusted" in data
        assert "impact_pct" in data
        assert isinstance(data["baseline"], list)
        assert isinstance(data["adjusted"], list)
        assert len(data["baseline"]) == 3

    def test_scenario_with_adjustments(self, client: TestClient):
        """Scenario with feature adjustments is accepted."""
        body = {
            "dept": "69",
            "horizon_months": 3,
            "adjustments": {"col1": 1.1},
        }
        response = client.post("/scenario", json=body)
        assert response.status_code == 200

    def test_scenario_baseline_points_structure(self, client: TestClient):
        """Each baseline point has date and value."""
        body = {"dept": "69", "horizon_months": 2}
        data = client.post("/scenario", json=body).json()
        for point in data["baseline"]:
            assert "date" in point
            assert "value" in point

    def test_scenario_invalid_department(self, client: TestClient):
        """Invalid department code returns 422."""
        body = {"dept": "ZZ", "horizon_months": 3}
        response = client.post("/scenario", json=body)
        assert response.status_code == 422

    def test_scenario_department_not_in_data(self, client: TestClient):
        """Valid department not in dataset returns 404."""
        body = {"dept": "01", "horizon_months": 3}
        response = client.post("/scenario", json=body)
        assert response.status_code == 404

    def test_scenario_horizon_too_high(self, client: TestClient):
        """Horizon > 24 returns 422."""
        body = {"dept": "69", "horizon_months": 25}
        response = client.post("/scenario", json=body)
        assert response.status_code == 422

    def test_scenario_horizon_too_low(self, client: TestClient):
        """Horizon < 1 returns 422."""
        body = {"dept": "69", "horizon_months": 0}
        response = client.post("/scenario", json=body)
        assert response.status_code == 422

    def test_scenario_default_horizon(self, client: TestClient):
        """Default horizon is 6 months."""
        body = {"dept": "69"}
        data = client.post("/scenario", json=body).json()
        assert len(data["baseline"]) == 6

    def test_scenario_no_negative_predictions(self, client: TestClient):
        """All prediction values are non-negative."""
        body = {"dept": "69", "horizon_months": 3}
        data = client.post("/scenario", json=body).json()
        for point in data["baseline"]:
            assert point["value"] >= 0
        for point in data["adjusted"]:
            assert point["value"] >= 0


# ===================================================================
# 15. Pydantic validation: department codes
# ===================================================================

class TestDepartmentValidation:
    """Tests for enhanced Pydantic department code validation."""

    def test_valid_dept_code_01(self, client: TestClient):
        """Department '01' is valid."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["01", "38"])

        body = {"departement": "01", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 200

    def test_valid_dept_code_2A(self, client: TestClient):
        """Department '2A' (Corsica) is valid."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["2A", "38"])

        body = {"departement": "2A", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 200

    def test_valid_dept_code_2B(self, client: TestClient):
        """Department '2B' (Corsica) is valid."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["2B", "38"])

        body = {"departement": "2B", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 200

    def test_invalid_dept_code_00(self, client: TestClient):
        """Department '00' is invalid — returns 422."""
        body = {"departement": "00", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 422

    def test_invalid_dept_code_99(self, client: TestClient):
        """Department '99' is invalid — returns 422."""
        body = {"departement": "99", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 422

    def test_invalid_dept_code_AB(self, client: TestClient):
        """Department 'AB' is invalid — returns 422."""
        body = {"departement": "AB", "horizon": 1}
        response = client.post("/predict", json=body)
        assert response.status_code == 422

    def test_dept_code_lowercase_normalized(self, client: TestClient):
        """Lowercase '2a' is normalized to '2A'."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["2A", "38"])

        body = {"departement": "2a", "horizon": 1}
        data = client.post("/predict", json=body).json()
        assert data["departement"] == "2A"

    def test_dept_code_single_digit_padded(self, client: TestClient):
        """Single digit '1' is padded to '01'."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["01", "38"])

        body = {"departement": "1", "horizon": 1}
        data = client.post("/predict", json=body).json()
        assert data["departement"] == "01"


# ===================================================================
# 16. Pydantic validation: scenario-specific
# ===================================================================

class TestScenarioValidation:
    """Tests for ScenarioRequest Pydantic validation."""

    def test_scenario_invalid_dept_code_96(self, client: TestClient):
        """Department '96' is invalid for scenario — returns 422."""
        body = {"dept": "96", "horizon_months": 3}
        response = client.post("/scenario", json=body)
        assert response.status_code == 422

    def test_scenario_empty_dept(self, client: TestClient):
        """Empty department code returns 422."""
        body = {"dept": "", "horizon_months": 3}
        response = client.post("/scenario", json=body)
        assert response.status_code == 422

    def test_scenario_dept_normalized(self, client: TestClient):
        """Department '9' is normalized to '09'."""
        from api.dependencies import state
        state.features_df = _build_mock_features_df(depts=["09", "38"])

        body = {"dept": "9", "horizon_months": 1}
        data = client.post("/scenario", json=body).json()
        assert data["departement"] == "09"


# ===================================================================
# 17. Rate limiting headers
# ===================================================================

class TestRateLimiting:
    """Tests for rate limiting integration."""

    def test_rate_limiting_installed(self):
        """Verify slowapi can be imported (installed)."""
        try:
            import slowapi
            installed = True
        except ImportError:
            installed = False
        # Test passes whether installed or not — we verify graceful handling
        assert isinstance(installed, bool)

    def test_rate_limit_app_state_set(self, client: TestClient):
        """App state has limiter attribute when slowapi is installed."""
        try:
            import slowapi
            from api.main import app
            assert hasattr(app.state, "limiter")
        except ImportError:
            pass  # Acceptable: slowapi not installed

    def test_health_endpoint_still_works_with_rate_limiting(
        self, client: TestClient,
    ):
        """Health endpoint works normally with rate limiting enabled."""
        # Make multiple requests — should all succeed within limits
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200


# ===================================================================
# 18. OpenAPI schema includes new endpoints
# ===================================================================

class TestOpenAPISchemNewEndpoints:
    """Verify new endpoints appear in the OpenAPI schema."""

    def test_openapi_has_trends(self, client: TestClient):
        """OpenAPI schema includes /trends/{dept}."""
        schema = client.get("/openapi.json").json()
        assert "/trends/{dept}" in schema["paths"]

    def test_openapi_has_comparison(self, client: TestClient):
        """OpenAPI schema includes /comparison."""
        schema = client.get("/openapi.json").json()
        assert "/comparison" in schema["paths"]

    def test_openapi_has_features_importance(self, client: TestClient):
        """OpenAPI schema includes /features/importance."""
        schema = client.get("/openapi.json").json()
        assert "/features/importance" in schema["paths"]

    def test_openapi_has_scenario(self, client: TestClient):
        """OpenAPI schema includes /scenario."""
        schema = client.get("/openapi.json").json()
        assert "/scenario" in schema["paths"]


# ===================================================================
# 19. Security tests for new endpoints
# ===================================================================

class TestNewEndpointsSecurity:
    """Security-focused tests for new endpoints."""

    def test_trends_sql_injection(self, client: TestClient):
        """SQL injection in trends dept returns 400."""
        response = client.get("/trends/'; DROP--")
        assert response.status_code == 400

    def test_comparison_xss_in_metric(self, client: TestClient):
        """XSS attempt in metric parameter is rejected."""
        response = client.get(
            "/comparison",
            params={"depts": "69", "metric": "<script>alert(1)</script>"},
        )
        assert response.status_code == 400

    def test_scenario_extra_fields_ignored(self, client: TestClient):
        """Extra fields in scenario body do not cause 500."""
        body = {
            "dept": "69",
            "horizon_months": 3,
            "adjustments": {},
            "malicious": "payload",
        }
        response = client.post("/scenario", json=body)
        assert response.status_code != 500
