# -*- coding: utf-8 -*-
"""
Tests for the Eurostat collector (EurostatCollector).

Tests class attributes, validate(), collect() with mocked eurostat package
responses, and error handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.collectors.eurostat_col import (
    EurostatCollector,
    GEO_FILTER,
    NACE_CODES,
    SEASONAL_ADJ_FILTER,
    UNIT_FILTER,
)


# ---------------------------------------------------------------------------
# Helper to build a mock Eurostat raw DataFrame
# ---------------------------------------------------------------------------

def _build_mock_eurostat_df(
    geo_col: str = "geo\\TIME_PERIOD",
    include_france: bool = True,
    include_hvac: bool = True,
) -> pd.DataFrame:
    """Build a DataFrame mimicking the eurostat package output.

    The eurostat package returns a wide DataFrame with:
    - geo column (e.g., 'geo\\TIME_PERIOD')
    - nace_r2, s_adj, unit columns
    - One column per time period (e.g., '2023-01', '2023-02', ...)
    """
    rows = []
    geos = ["FR", "DE", "IT"] if include_france else ["DE", "IT"]
    naces = ["C28", "C2825", "C29"] if include_hvac else ["C29", "C30"]

    for geo in geos:
        for nace in naces:
            row = {
                geo_col: geo,
                "nace_r2": nace,
                "s_adj": SEASONAL_ADJ_FILTER,
                "unit": UNIT_FILTER,
            }
            # Add 12 months of data
            for m in range(1, 13):
                period = f"2023-{m:02d}"
                row[period] = np.random.uniform(80, 120)
            rows.append(row)

    return pd.DataFrame(rows)


class TestEurostatAttributes:
    """Tests for EurostatCollector class attributes."""

    def test_source_name(self):
        """source_name is 'eurostat'."""
        assert EurostatCollector.source_name == "eurostat"

    def test_output_filename(self):
        """output_filename is 'ipi_hvac_france.csv'."""
        assert EurostatCollector.output_filename == "ipi_hvac_france.csv"

    def test_output_subdir(self):
        """output_subdir is 'eurostat'."""
        assert EurostatCollector.output_subdir == "eurostat"

    def test_nace_codes(self):
        """NACE codes include both C28 and C2825."""
        assert "C28" in NACE_CODES
        assert "C2825" in NACE_CODES

    def test_geo_filter_is_france(self):
        """Geographic filter is set to France."""
        assert GEO_FILTER == "FR"


class TestEurostatValidate:
    """Tests for EurostatCollector.validate()."""

    def test_validate_valid_data(self, collector_config, sample_eurostat_df):
        """validate() succeeds with valid Eurostat data."""
        collector = EurostatCollector(collector_config)
        result = collector.validate(sample_eurostat_df)
        assert len(result) == len(sample_eurostat_df)

    def test_validate_missing_columns_raises(self, collector_config):
        """validate() raises ValueError when required columns are missing."""
        collector = EurostatCollector(collector_config)
        df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns"):
            collector.validate(df)

    def test_validate_missing_period_raises(self, collector_config):
        """validate() raises when 'period' column is missing."""
        collector = EurostatCollector(collector_config)
        df = pd.DataFrame({
            "nace_r2": ["C28"],
            "ipi_value": [100.0],
        })
        with pytest.raises(ValueError, match="Missing columns"):
            collector.validate(df)

    def test_validate_suspicious_values_warns(self, collector_config, caplog):
        """validate() warns on suspicious IPI values."""
        import logging
        collector = EurostatCollector(collector_config)
        df = pd.DataFrame({
            "period": ["2023-01", "2023-02"],
            "nace_r2": ["C28", "C28"],
            "ipi_value": [-5.0, 350.0],
        })
        with caplog.at_level(logging.WARNING):
            result = collector.validate(df)
        assert len(result) == 2
        assert any("Suspicious" in r.message or "suspicious" in r.message.lower()
                    for r in caplog.records)

    def test_validate_normal_range_no_warning(self, collector_config, caplog):
        """validate() does not warn when values are in normal range."""
        import logging
        collector = EurostatCollector(collector_config)
        df = pd.DataFrame({
            "period": ["2023-01", "2023-02"],
            "nace_r2": ["C28", "C2825"],
            "ipi_value": [95.0, 105.0],
        })
        with caplog.at_level(logging.WARNING):
            result = collector.validate(df)
        assert not any("Suspicious" in r.message or "suspicious" in r.message.lower()
                        for r in caplog.records)


class TestEurostatCollect:
    """Tests for EurostatCollector.collect() with mocked eurostat package."""

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_success(self, mock_fetch, collector_config):
        """collect() returns filtered data for France HVAC."""
        mock_df = _build_mock_eurostat_df()

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            result = collector.collect()

        assert not result.empty
        assert "period" in result.columns
        assert "nace_r2" in result.columns
        assert "ipi_value" in result.columns
        # Only France HVAC codes
        assert set(result["nace_r2"].unique()).issubset(set(NACE_CODES))

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_filters_by_date_range(self, mock_fetch, collector_config):
        """collect() filters data to configured date range."""
        mock_df = _build_mock_eurostat_df()

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            result = collector.collect()

        if not result.empty:
            # All periods should be within config range
            start = collector_config.start_date[:7]
            end = collector_config.end_date[:7]
            assert (result["period"] >= start).all()
            assert (result["period"] <= end).all()

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_handles_alternative_geo_column(self, mock_fetch, collector_config):
        """collect() works with alternative geo column names."""
        mock_df = _build_mock_eurostat_df(geo_col="geo\\time")

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            result = collector.collect()

        assert not result.empty

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_handles_geo_column_name(self, mock_fetch, collector_config):
        """collect() works with 'geo' as column name."""
        mock_df = _build_mock_eurostat_df(geo_col="geo")

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            result = collector.collect()

        assert not result.empty


class TestEurostatErrorHandling:
    """Tests for error handling in EurostatCollector."""

    def test_collect_raises_on_missing_eurostat_package(self, collector_config):
        """collect() raises ImportError when eurostat package is missing."""
        collector = EurostatCollector(collector_config)

        with patch.dict("sys.modules", {"eurostat": None}):
            with pytest.raises(ImportError, match="eurostat"):
                collector.collect()

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_raises_on_download_failure(self, mock_fetch, collector_config):
        """collect() raises RuntimeError when Eurostat download fails."""
        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.side_effect = Exception("Connection timeout")

            collector = EurostatCollector(collector_config)
            with pytest.raises(RuntimeError, match="Eurostat download failed"):
                collector.collect()

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_raises_on_missing_geo_column(self, mock_fetch, collector_config):
        """collect() raises ValueError when geo column is not found."""
        mock_df = pd.DataFrame({
            "unknown_col": ["FR"],
            "nace_r2": ["C28"],
            "s_adj": ["SCA"],
            "unit": ["I21"],
            "2023-01": [100.0],
        })

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            with pytest.raises(ValueError, match="Geographic column not found"):
                collector.collect()

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_returns_empty_when_no_matching_data(self, mock_fetch, collector_config):
        """collect() returns empty DataFrame when filters match nothing."""
        mock_df = _build_mock_eurostat_df(include_france=False)

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            result = collector.collect()

        assert result.empty

    @patch("src.collectors.eurostat_col.EurostatCollector.fetch_json")
    def test_collect_raises_on_no_time_columns(self, mock_fetch, collector_config):
        """collect() raises ValueError when no time columns are found."""
        mock_df = pd.DataFrame({
            "geo\\TIME_PERIOD": ["FR"],
            "nace_r2": ["C28"],
            "s_adj": [SEASONAL_ADJ_FILTER],
            "unit": [UNIT_FILTER],
            "some_non_date_col": [100.0],
        })

        with patch.dict("sys.modules", {"eurostat": MagicMock()}) as _:
            import sys
            mock_estat = sys.modules["eurostat"]
            mock_estat.get_data_df.return_value = mock_df

            collector = EurostatCollector(collector_config)
            with pytest.raises(ValueError, match="No time columns found"):
                collector.collect()
