# -*- coding: utf-8 -*-
"""Tests pour le module collectors.base (BaseCollector, Registry, Config)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collectors.base import (
    BaseCollector,
    CollectorConfig,
    CollectorRegistry,
    CollectorResult,
    CollectorStatus,
)


class TestCollectorStatus:
    """Tests pour l'énumération CollectorStatus."""

    def test_status_values(self):
        assert CollectorStatus.SUCCESS.value == "success"
        assert CollectorStatus.PARTIAL.value == "partial"
        assert CollectorStatus.FAILED.value == "failed"
        assert CollectorStatus.SKIPPED.value == "skipped"


class TestCollectorConfig:
    """Tests pour CollectorConfig."""

    def test_default_values(self):
        config = CollectorConfig()
        assert config.start_date == "2019-01-01"
        assert config.end_date == "2026-02-28"
        assert config.region_code == "FR"
        assert len(config.departments) == 96
        assert config.request_timeout == 30
        assert config.max_retries == 3

    def test_custom_values(self, collector_config):
        assert collector_config.start_date == "2023-01-01"
        assert collector_config.end_date == "2023-12-31"
        assert len(collector_config.departments) == 2
        assert collector_config.rate_limit_delay == 0.0

    def test_from_env(self):
        with patch.dict("os.environ", {
            "DATA_START_DATE": "2020-01-01",
            "DATA_END_DATE": "2024-12-31",
            "TARGET_REGION": "11",
        }):
            config = CollectorConfig.from_env()
            assert config.start_date == "2020-01-01"
            assert config.end_date == "2024-12-31"
            assert config.region_code == "11"

    def test_frozen(self, collector_config):
        with pytest.raises(AttributeError):
            collector_config.start_date = "2020-01-01"


class TestCollectorResult:
    """Tests pour CollectorResult."""

    def test_basic_result(self):
        result = CollectorResult(
            name="test",
            status=CollectorStatus.SUCCESS,
            rows_collected=100,
        )
        assert result.name == "test"
        assert result.rows_collected == 100
        assert result.duration_seconds is None

    def test_duration(self):
        from datetime import datetime, timedelta

        start = datetime(2024, 1, 1, 12, 0, 0)
        end = start + timedelta(seconds=5.5)
        result = CollectorResult(
            name="test",
            status=CollectorStatus.SUCCESS,
            started_at=start,
            finished_at=end,
        )
        assert result.duration_seconds == pytest.approx(5.5)

    def test_str_representation(self):
        result = CollectorResult(
            name="weather",
            status=CollectorStatus.SUCCESS,
            rows_collected=20456,
        )
        text = str(result)
        assert "SUCCESS" in text
        assert "weather" in text
        assert "20456" in text


class TestCollectorRegistry:
    """Tests pour le système de registry (plugins)."""

    def test_available_returns_sorted_list(self):
        names = CollectorRegistry.available()
        assert isinstance(names, list)
        assert names == sorted(names)

    def test_known_collectors_registered(self):
        # Les imports dans src/collectors/__init__.py enregistrent les collecteurs
        import src.collectors  # noqa: F401

        available = CollectorRegistry.available()
        assert "weather" in available
        assert "insee" in available
        assert "eurostat" in available
        assert "dpe" in available

    def test_get_existing_collector(self):
        import src.collectors  # noqa: F401

        cls = CollectorRegistry.get("weather")
        assert cls is not None
        assert hasattr(cls, "source_name")
        assert cls.source_name == "weather"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Collecteur inconnu"):
            CollectorRegistry.get("nonexistent_source")
