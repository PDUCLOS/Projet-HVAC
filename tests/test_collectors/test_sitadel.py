# -*- coding: utf-8 -*-
"""Tests for the SITADEL collector (SitadelCollector)."""

from __future__ import annotations

import io
from unittest.mock import patch

import pandas as pd
import pytest

from src.collectors.sitadel import SitadelCollector, _DIDO_COLUMN_MAP


class TestSitadelCollector:
    """Tests for SitadelCollector."""

    def test_source_name(self):
        assert SitadelCollector.source_name == "sitadel"
        assert SitadelCollector.output_filename == "permis_construire_france.csv"

    def test_validate_valid_data(self, collector_config):
        collector = SitadelCollector(collector_config)
        df = pd.DataFrame({
            "DEP": ["69", "38", "69"],
            "DATE_PRISE_EN_COMPTE": ["2023-01-15", "2023-02-20", "2023-03-10"],
            "NB_LGT_TOT_CREES": [5, 10, 3],
            "CAT_DEM": ["P", "P", "S"],
        })
        result = collector.validate(df)
        assert len(result) == 3

    def test_validate_missing_dep_column(self, collector_config):
        collector = SitadelCollector(collector_config)
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Column 'DEP' missing"):
            collector.validate(df)


class TestColumnRenaming:
    """Tests for DiDo API column renaming (2026 migration)."""

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_new_column_names_dep_code(self, mock_fetch, collector_config):
        """DEP_CODE is renamed to DEP for backwards compatibility."""
        csv_data = (
            "REG_CODE,DEP_CODE,DATE_PRISE_EN_COMPTE,NB_LGT_TOT_CREES\n"
            "84,69,2023-01-15,5\n"
            "84,38,2023-02-20,10\n"
            "84,01,2023-03-10,3\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)
        df = collector.collect()

        # DEP column should exist after renaming
        assert "DEP" in df.columns
        assert "REG" in df.columns
        # Should only keep departments in collector_config (69, 38)
        assert set(df["DEP"].unique()) <= {"69", "38"}

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_old_column_names_still_work(self, mock_fetch, collector_config):
        """Legacy DEP/REG column names still work without renaming."""
        csv_data = (
            "REG,DEP,DATE_PRISE_EN_COMPTE,NB_LGT_TOT_CREES\n"
            "84,69,2023-01-15,5\n"
            "84,38,2023-02-20,10\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)
        df = collector.collect()

        assert "DEP" in df.columns
        assert "REG" in df.columns
        assert len(df) == 2

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_no_rename_if_both_exist(self, mock_fetch, collector_config):
        """No renaming if both DEP and DEP_CODE exist simultaneously."""
        csv_data = (
            "REG,DEP,DEP_CODE,DATE_PRISE_EN_COMPTE,NB_LGT_TOT_CREES\n"
            "84,69,69,2023-01-15,5\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)
        df = collector.collect()

        # DEP should still be the original, DEP_CODE should remain
        assert "DEP" in df.columns
        assert "DEP_CODE" in df.columns
        assert len(df) == 1

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_numeric_conversion(self, mock_fetch, collector_config):
        """NB_LGT_TOT_CREES is converted to numeric."""
        csv_data = (
            "DEP_CODE,REG_CODE,DATE_PRISE_EN_COMPTE,NB_LGT_TOT_CREES\n"
            "69,84,2023-01-15,5\n"
            "69,84,2023-02-20,invalid\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)
        df = collector.collect()

        assert pd.api.types.is_numeric_dtype(df["NB_LGT_TOT_CREES"])
        # 'invalid' should become NaN
        assert df["NB_LGT_TOT_CREES"].isna().sum() == 1

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_dep_zero_padded(self, mock_fetch, collector_config):
        """Department codes are zero-padded to 2 characters."""
        csv_data = (
            "DEP_CODE,REG_CODE,NB_LGT_TOT_CREES\n"
            "1,84,5\n"  # Should become "01"
            "69,84,10\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")

        # Override config to include dept "01"
        from src.collectors.base import CollectorConfig
        from pathlib import Path
        cfg = CollectorConfig(
            raw_data_dir=Path("/tmp/hvac_test/raw"),
            processed_data_dir=Path("/tmp/hvac_test/processed"),
            departments=["01", "69"],
            rate_limit_delay=0.0,
        )
        collector = SitadelCollector(cfg)
        df = collector.collect()

        assert "01" in df["DEP"].values

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_filter_keeps_only_target_departments(self, mock_fetch, collector_config):
        """Only target departments are kept after filtering."""
        csv_data = (
            "DEP_CODE,REG_CODE,NB_LGT_TOT_CREES\n"
            "69,84,5\n"
            "75,11,10\n"  # Paris — not in collector_config (69, 38)
            "38,84,3\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)
        df = collector.collect()

        assert set(df["DEP"].unique()) == {"69", "38"}
        assert "75" not in df["DEP"].values

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_collect_no_dep_no_reg_raises(self, mock_fetch, collector_config):
        """Raises ValueError when neither DEP nor REG columns exist."""
        csv_data = "X,Y,Z\n1,2,3\n"
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)

        with pytest.raises(ValueError, match="Neither 'DEP' nor 'REG'"):
            collector.collect()


class TestSecurityValidation:
    """Security tests for SITADEL collector."""

    @patch.object(SitadelCollector, "fetch_bytes")
    def test_special_chars_in_dep(self, mock_fetch, collector_config):
        """Special characters in DEP column do not cause errors."""
        csv_data = (
            "DEP_CODE,REG_CODE,NB_LGT_TOT_CREES\n"
            "69,84,5\n"
            "../etc,84,10\n"
            "38,84,3\n"
        )
        mock_fetch.return_value = csv_data.encode("utf-8")
        collector = SitadelCollector(collector_config)
        df = collector.collect()

        # Malicious dept code should be filtered out
        assert "../etc" not in df["DEP"].values
        assert len(df) == 2
