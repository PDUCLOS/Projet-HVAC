# -*- coding: utf-8 -*-
"""
Security tests — Input validation across the HVAC project.

Tests SQL injection, XSS, path traversal, oversized inputs, empty inputs,
special characters, and API parameter boundary conditions.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from api.models import CustomPredictRequest


# ---------------------------------------------------------------------------
# SQL injection attempts
# ---------------------------------------------------------------------------

class TestSQLInjection:
    """Verify that SQL injection payloads are rejected or neutralized."""

    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE dpe;--",
        "1 OR 1=1",
        "69; DELETE FROM users;--",
        "' UNION SELECT * FROM information_schema.tables--",
        "69' AND '1'='1",
        "1; EXEC xp_cmdshell('dir');--",
        "' OR ''='",
        "69\"; DROP TABLE predictions;--",
    ]

    def test_department_code_rejects_sql_injection(self):
        """CustomPredictRequest rejects SQL injection in department code."""
        for payload in self.SQL_INJECTION_PAYLOADS:
            # Pydantic enforces max_length=3 on departement
            if len(payload) > 3:
                with pytest.raises(ValidationError):
                    CustomPredictRequest(departement=payload, horizon=6)

    def test_short_sql_injection_rejected(self):
        """Short injection fragments like '1=1' are rejected by validation."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="1=1", horizon=1)

    @patch("api.dependencies.state")
    def test_api_predictions_rejects_sql_department(self, mock_state):
        """GET /predictions rejects SQL injection department codes."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_state.feature_names = ["col1", "col2"]
        mock_state.model_date = "2024-01-01"
        mock_state.start_time = 0
        client = TestClient(app, raise_server_exceptions=False)

        for payload in self.SQL_INJECTION_PAYLOADS:
            resp = client.get("/predictions", params={
                "departement": payload, "horizon": 6,
            })
            # Must be 400 (invalid dept) or 422 (validation error), never 200
            assert resp.status_code in (400, 422), (
                f"Payload {payload!r} returned {resp.status_code}"
            )

    @patch("api.dependencies.state")
    def test_api_predict_post_rejects_sql_body(self, mock_state):
        """POST /predict rejects SQL injection in request body."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_state.feature_names = ["col1", "col2"]
        mock_state.model_date = "2024-01-01"
        mock_state.start_time = 0
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/predict", json={
            "departement": "'; DROP TABLE--",
            "horizon": 6,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# XSS attempts
# ---------------------------------------------------------------------------

class TestXSSPrevention:
    """Verify that XSS payloads in inputs do not pass validation."""

    XSS_PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert('xss')",
        "<svg/onload=alert('xss')>",
        "'\"><script>alert(document.cookie)</script>",
        "<body onload=alert('xss')>",
    ]

    def test_department_rejects_xss(self):
        """CustomPredictRequest rejects XSS in department field."""
        for payload in self.XSS_PAYLOADS:
            with pytest.raises(ValidationError):
                CustomPredictRequest(departement=payload, horizon=6)

    @patch("api.dependencies.state")
    def test_api_rejects_xss_in_query_param(self, mock_state):
        """GET /predictions rejects XSS payloads in department param."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_state.feature_names = ["col1"]
        mock_state.model_date = "2024-01-01"
        mock_state.start_time = 0
        client = TestClient(app, raise_server_exceptions=False)

        for payload in self.XSS_PAYLOADS:
            resp = client.get("/predictions", params={
                "departement": payload, "horizon": 1,
            })
            assert resp.status_code in (400, 422), (
                f"XSS payload {payload!r} returned {resp.status_code}"
            )

    def test_features_dict_rejects_non_float_values(self):
        """Feature values must be floats, not script strings."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(
                departement="69",
                horizon=1,
                features={"temp_mean": "<script>alert(1)</script>"},
            )


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Verify that path traversal attempts are blocked."""

    PATH_TRAVERSAL_PAYLOADS = [
        "../../etc/passwd",
        "../../../etc/shadow",
        "..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ]

    def test_department_rejects_path_traversal(self):
        """CustomPredictRequest rejects path traversal in department."""
        for payload in self.PATH_TRAVERSAL_PAYLOADS:
            with pytest.raises(ValidationError):
                CustomPredictRequest(departement=payload, horizon=1)

    @patch("api.dependencies.state")
    def test_api_rejects_path_traversal_department(self, mock_state):
        """GET /predictions rejects path traversal in department."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_state.feature_names = ["col1"]
        mock_state.model_date = "2024-01-01"
        mock_state.start_time = 0
        client = TestClient(app, raise_server_exceptions=False)

        for payload in self.PATH_TRAVERSAL_PAYLOADS:
            resp = client.get("/predictions", params={
                "departement": payload, "horizon": 1,
            })
            assert resp.status_code in (400, 422), (
                f"Path traversal {payload!r} returned {resp.status_code}"
            )


# ---------------------------------------------------------------------------
# Oversized and empty inputs
# ---------------------------------------------------------------------------

class TestInputBoundaries:
    """Boundary testing for API parameters."""

    def test_empty_department_rejected(self):
        """Empty string department is rejected by Pydantic (min_length=1)."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="", horizon=1)

    def test_oversized_department_rejected(self):
        """Department code exceeding max_length=3 is rejected."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="12345", horizon=1)

    def test_very_long_department_rejected(self):
        """Very long department string is rejected."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="A" * 1000, horizon=1)

    def test_horizon_zero_rejected(self):
        """Horizon of 0 is below ge=1 constraint."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="69", horizon=0)

    def test_negative_horizon_rejected(self):
        """Negative horizon is rejected."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="69", horizon=-1)

    def test_huge_horizon_rejected(self):
        """Horizon exceeding le=24 is rejected."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="69", horizon=100)

    def test_horizon_boundary_min(self):
        """Horizon=1 (minimum valid value) is accepted."""
        req = CustomPredictRequest(departement="69", horizon=1)
        assert req.horizon == 1

    def test_horizon_boundary_max(self):
        """Horizon=24 (maximum valid value) is accepted."""
        req = CustomPredictRequest(departement="69", horizon=24)
        assert req.horizon == 24

    def test_horizon_just_over_max_rejected(self):
        """Horizon=25 (just over max) is rejected."""
        with pytest.raises(ValidationError):
            CustomPredictRequest(departement="69", horizon=25)

    @patch("api.dependencies.state")
    def test_api_query_negative_horizon(self, mock_state):
        """GET /predictions rejects negative horizon."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_state.feature_names = ["col1"]
        mock_state.model_date = "2024-01-01"
        mock_state.start_time = 0
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/predictions", params={
            "departement": "69", "horizon": -5,
        })
        assert resp.status_code == 422

    @patch("api.dependencies.state")
    def test_api_query_huge_horizon(self, mock_state):
        """GET /predictions rejects horizon > 24."""
        from fastapi.testclient import TestClient
        from api.main import app

        mock_state.feature_names = ["col1"]
        mock_state.model_date = "2024-01-01"
        mock_state.start_time = 0
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/predictions", params={
            "departement": "69", "horizon": 999,
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Special characters
# ---------------------------------------------------------------------------

class TestSpecialCharacters:
    """Verify that special characters are handled safely."""

    SPECIAL_CHARS = [
        "\x00",           # null byte
        "\n\r",           # newlines
        "\t",             # tab
        "69\x00",         # null byte injection
        "🔥",            # emoji
        "六九",           # CJK characters
        "69 ",            # trailing space
        " 69",            # leading space
    ]

    def test_special_chars_in_department(self):
        """Special characters are either rejected or safely normalized."""
        for char in self.SPECIAL_CHARS:
            try:
                req = CustomPredictRequest(departement=char, horizon=1)
                # If accepted, the value must be safely normalized
                assert isinstance(req.departement, str)
                assert len(req.departement) <= 3
            except ValidationError:
                # Rejection is the expected safe behavior
                pass

    def test_null_bytes_in_features_key(self):
        """Null bytes in feature dict keys do not crash validation."""
        # Pydantic should accept the dict structure but keys are just strings
        try:
            req = CustomPredictRequest(
                departement="69",
                horizon=1,
                features={"\x00": 1.0},
            )
            assert isinstance(req.features, dict)
        except ValidationError:
            pass  # Also acceptable

    def test_nan_in_features_value(self):
        """NaN as a feature value is handled (float('nan') is valid float)."""
        req = CustomPredictRequest(
            departement="69",
            horizon=1,
            features={"temp_mean": float("nan")},
        )
        assert "temp_mean" in req.features

    def test_infinity_in_features_value(self):
        """Infinity as a feature value is handled."""
        req = CustomPredictRequest(
            departement="69",
            horizon=1,
            features={"temp_mean": float("inf")},
        )
        assert req.features["temp_mean"] == float("inf")


# ---------------------------------------------------------------------------
# Collector input validation
# ---------------------------------------------------------------------------

class TestCollectorInputValidation:
    """Verify that collectors validate their inputs properly."""

    def test_dpe_validate_missing_critical_columns(self, collector_config):
        """DPE validate logs warning for missing critical columns."""
        from src.collectors.dpe import DpeCollector
        collector = DpeCollector(collector_config)
        df = pd.DataFrame({"random_col": [1, 2, 3]})
        # Should not raise but should log a warning
        result = collector.validate(df)
        assert len(result) == 3

    def test_dpe_validate_empty_dataframe(self, collector_config):
        """DPE validate handles empty DataFrame gracefully."""
        from src.collectors.dpe import DpeCollector
        collector = DpeCollector(collector_config)
        df = pd.DataFrame()
        result = collector.validate(df)
        assert result.empty

    def test_eurostat_validate_missing_columns_raises(self, collector_config):
        """Eurostat validate raises ValueError for missing required columns."""
        from src.collectors.eurostat_col import EurostatCollector
        collector = EurostatCollector(collector_config)
        df = pd.DataFrame({"wrong": [1, 2]})
        with pytest.raises(ValueError, match="Missing columns"):
            collector.validate(df)

    def test_eurostat_validate_suspicious_values(self, collector_config):
        """Eurostat validate warns on out-of-range IPI values."""
        from src.collectors.eurostat_col import EurostatCollector
        collector = EurostatCollector(collector_config)
        df = pd.DataFrame({
            "period": ["2023-01", "2023-02"],
            "nace_r2": ["C28", "C28"],
            "ipi_value": [-10.0, 350.0],  # Suspicious range
        })
        # Should not raise but should log a warning
        result = collector.validate(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Pydantic model robustness
# ---------------------------------------------------------------------------

class TestPydanticModelRobustness:
    """Regression tests for Pydantic model validation edge cases."""

    def test_valid_department_codes(self):
        """Standard department codes are accepted."""
        for dept in ["01", "69", "2A", "2B", "95", "75"]:
            req = CustomPredictRequest(departement=dept, horizon=1)
            assert req.departement == dept.upper().zfill(2)

    def test_department_normalized_uppercase(self):
        """Lowercase department codes are normalized to uppercase."""
        req = CustomPredictRequest(departement="2a", horizon=1)
        assert req.departement == "2A"

    def test_department_zero_padded(self):
        """Single-digit department codes are zero-padded."""
        req = CustomPredictRequest(departement="1", horizon=1)
        assert req.departement == "01"

    def test_features_default_empty_dict(self):
        """Features default to empty dict when not provided."""
        req = CustomPredictRequest(departement="69", horizon=1)
        assert req.features == {}

    def test_features_with_valid_floats(self):
        """Valid float features are accepted."""
        req = CustomPredictRequest(
            departement="69",
            horizon=6,
            features={"temp_mean": 15.0, "hdd_sum": 200.0},
        )
        assert req.features["temp_mean"] == 15.0

    def test_very_large_features_dict(self):
        """Large number of features in dict is accepted."""
        features = {f"feature_{i}": float(i) for i in range(200)}
        req = CustomPredictRequest(
            departement="69", horizon=1, features=features,
        )
        assert len(req.features) == 200
