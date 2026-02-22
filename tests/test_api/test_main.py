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
