# -*- coding: utf-8 -*-
"""
Tests for the DPE collector (DpeCollector).

Tests class attributes, validate(), collect() with mocked API responses,
pagination logic, and error handling.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.collectors.base import CollectorConfig
from src.collectors.dpe import DpeCollector, DPE_API_BASE, DPE_SELECT_FIELDS, PAGE_SIZE


def _make_config(departments=None, raw_data_dir=None):
    """Create a CollectorConfig with custom departments/raw_data_dir."""
    return CollectorConfig(
        raw_data_dir=raw_data_dir or Path("/tmp/hvac_test/raw"),
        processed_data_dir=Path("/tmp/hvac_test/processed"),
        start_date="2023-01-01",
        end_date="2023-12-31",
        departments=departments or ["69", "38"],
        region_code="84",
        request_timeout=5,
        max_retries=1,
        retry_backoff_factor=0.1,
        rate_limit_delay=0.0,
    )


class TestDpeCollectorAttributes:
    """Tests for DpeCollector class attributes."""

    def test_source_name(self):
        """source_name is 'dpe'."""
        assert DpeCollector.source_name == "dpe"

    def test_output_filename(self):
        """output_filename is 'dpe_france_all.csv'."""
        assert DpeCollector.output_filename == "dpe_france_all.csv"

    def test_output_subdir(self):
        """output_subdir is 'dpe'."""
        assert DpeCollector.output_subdir == "dpe"

    def test_select_fields_not_empty(self):
        """DPE_SELECT_FIELDS has the expected key fields."""
        assert len(DPE_SELECT_FIELDS) > 0
        assert "numero_dpe" in DPE_SELECT_FIELDS
        assert "date_etablissement_dpe" in DPE_SELECT_FIELDS
        assert "etiquette_dpe" in DPE_SELECT_FIELDS
        assert "code_departement_ban" in DPE_SELECT_FIELDS

    def test_page_size(self):
        """PAGE_SIZE is 10000 (ADEME API maximum)."""
        assert PAGE_SIZE == 10000


class TestDpeValidate:
    """Tests for DpeCollector.validate()."""

    def test_validate_valid_data(self, collector_config):
        """validate() succeeds with valid DPE data."""
        collector = DpeCollector(collector_config)
        df = pd.DataFrame({
            "numero_dpe": ["DPE001", "DPE002"],
            "date_etablissement_dpe": ["2023-01-15", "2023-06-20"],
            "etiquette_dpe": ["A", "C"],
            "etiquette_ges": ["B", "D"],
            "code_departement_ban": ["69", "38"],
            "surface_habitable_logement": [80.0, 120.0],
        })
        result = collector.validate(df)
        assert len(result) == 2
        assert pd.api.types.is_datetime64_any_dtype(result["date_etablissement_dpe"])

    def test_validate_missing_critical_columns_warns(self, collector_config, caplog):
        """validate() warns when critical columns are missing."""
        import logging
        collector = DpeCollector(collector_config)
        df = pd.DataFrame({
            "numero_dpe": ["DPE001"],
            "surface_habitable_logement": [80.0],
        })
        with caplog.at_level(logging.WARNING):
            result = collector.validate(df)
        assert len(result) == 1
        # Should have logged a warning about missing columns
        assert any("missing" in r.message.lower() or "Expected" in r.message
                    for r in caplog.records)

    def test_validate_coerces_invalid_dates(self, collector_config):
        """validate() coerces invalid dates to NaT."""
        collector = DpeCollector(collector_config)
        df = pd.DataFrame({
            "date_etablissement_dpe": ["2023-01-15", "not-a-date", None],
            "etiquette_dpe": ["A", "B", "C"],
        })
        result = collector.validate(df)
        assert result["date_etablissement_dpe"].isna().sum() == 2

    def test_validate_empty_dataframe(self, collector_config):
        """validate() handles empty DataFrame."""
        collector = DpeCollector(collector_config)
        df = pd.DataFrame()
        result = collector.validate(df)
        assert result.empty

    def test_validate_logs_dpe_distribution(self, collector_config, caplog):
        """validate() logs DPE label distribution."""
        import logging
        collector = DpeCollector(collector_config)
        df = pd.DataFrame({
            "date_etablissement_dpe": ["2023-01-15"] * 5,
            "etiquette_dpe": ["A", "A", "B", "C", "D"],
            "code_departement_ban": ["69"] * 5,
        })
        with caplog.at_level(logging.INFO):
            collector.validate(df)
        assert any("distribution" in r.message.lower() or "DPE" in r.message
                    for r in caplog.records)


class TestDpeCollect:
    """Tests for DpeCollector.collect() with mocked API responses."""

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_single_department_single_page(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() retrieves data for a single page (no next cursor)."""
        mock_fetch.return_value = {
            "results": [
                {"numero_dpe": "DPE001", "code_departement_ban": "69"},
                {"numero_dpe": "DPE002", "code_departement_ban": "69"},
            ],
            "next": None,
        }
        cfg = _make_config(departments=["69"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        df = collector.collect()

        assert not df.empty
        assert len(df) == 2
        assert mock_fetch.call_count == 1

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_pagination_with_cursor(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() follows cursor-based pagination across pages."""
        # Page 1: has next cursor
        page1 = {
            "results": [{"numero_dpe": f"DPE{i:03d}"} for i in range(3)],
            "next": "https://data.ademe.fr/data-fair/api/v1/datasets/dpe03existant/lines?after=cursor123&size=10000",
        }
        # Page 2: no more data
        page2 = {
            "results": [{"numero_dpe": f"DPE{i:03d}"} for i in range(3, 5)],
            "next": None,
        }
        mock_fetch.side_effect = [page1, page2]

        cfg = _make_config(departments=["69"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        df = collector.collect()

        assert len(df) == 5
        assert mock_fetch.call_count == 2

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_stops_on_empty_results(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() stops when results list is empty."""
        mock_fetch.return_value = {
            "results": [],
            "next": None,
        }
        cfg = _make_config(departments=["69"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        df = collector.collect()

        assert df.empty

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_multiple_departments(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() iterates over all configured departments."""
        mock_fetch.return_value = {
            "results": [{"numero_dpe": "DPE001"}],
            "next": None,
        }
        cfg = _make_config(departments=["69", "38"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        df = collector.collect()

        assert len(df) == 2  # 1 row per department
        # fetch_json called once per department
        assert mock_fetch.call_count == 2

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_saves_intermediate_csv(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() saves intermediate CSV per department."""
        mock_fetch.return_value = {
            "results": [{"numero_dpe": "DPE001"}],
            "next": None,
        }
        cfg = _make_config(departments=["69"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        collector.collect()

        dept_csv = tmp_path / "raw" / "dpe" / "dpe_69.csv"
        assert dept_csv.exists()


class TestDpeErrorHandling:
    """Tests for error handling in DpeCollector."""

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_continues_on_department_failure(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() continues to next department when one fails."""
        mock_fetch.side_effect = [
            Exception("API timeout"),  # dept 69 fails
            {  # dept 38 succeeds
                "results": [{"numero_dpe": "DPE001"}],
                "next": None,
            },
        ]
        cfg = _make_config(departments=["69", "38"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        df = collector.collect()

        assert not df.empty
        assert len(df) == 1  # Only dept 38 data

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_returns_empty_on_all_failures(
        self, mock_fetch, mock_pause
    ):
        """collect() returns empty DataFrame when all departments fail."""
        mock_fetch.side_effect = Exception("Network error")
        cfg = _make_config(departments=["69", "38"])
        collector = DpeCollector(cfg)
        df = collector.collect()

        assert df.empty

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_department_handles_mid_pagination_error(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """collect() saves partial data when pagination fails mid-way."""
        mock_fetch.side_effect = [
            {  # Page 1: OK
                "results": [{"numero_dpe": "DPE001"}],
                "next": "https://example.com?after=xyz",
            },
            Exception("Connection reset"),  # Page 2: fails
        ]
        cfg = _make_config(departments=["69"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)
        df = collector.collect()

        # Should have the data from page 1
        assert len(df) == 1

    @patch.object(DpeCollector, "rate_limit_pause")
    @patch.object(DpeCollector, "fetch_json")
    def test_collect_department_respects_max_pages(
        self, mock_fetch, mock_pause, tmp_path
    ):
        """_collect_department stops after max_pages."""
        # Return data with cursor forever
        mock_fetch.return_value = {
            "results": [{"numero_dpe": "DPE001"}],
            "next": "https://example.com?after=cursor123",
        }
        cfg = _make_config(departments=["69"], raw_data_dir=tmp_path / "raw")
        collector = DpeCollector(cfg)

        # Call with max_pages=3 to limit
        df = collector._collect_department("69", max_pages=3)
        assert mock_fetch.call_count == 3
        assert len(df) == 3
