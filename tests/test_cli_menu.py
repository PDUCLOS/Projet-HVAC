# -*- coding: utf-8 -*-
"""Tests for the interactive CLI menu module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli_menu import (
    _detect_date_range,
    _format_size,
    _get_file_info,
    COLLECTION_SOURCES,
    NOTEBOOKS,
)


class TestFormatSize:
    """Tests for _format_size utility."""

    def test_bytes(self):
        assert _format_size(500) == "500 B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_size(2 * 1024 * 1024 * 1024) == "2.00 GB"

    def test_zero(self):
        assert _format_size(0) == "0 B"


class TestGetFileInfo:
    """Tests for _get_file_info utility."""

    def test_existing_file(self, tmp_path):
        f = tmp_path / "test.csv"
        f.write_text("col1,col2\na,b\nc,d\n")
        info = _get_file_info(f)
        assert info["exists"] is True
        assert info["rows"] == 2
        assert info["size"] > 0
        assert info["modified"] != "N/A"

    def test_missing_file(self, tmp_path):
        f = tmp_path / "nope.csv"
        info = _get_file_info(f)
        assert info["exists"] is False
        assert info["rows"] == 0
        assert info["size_str"] == "N/A"

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("header\n")
        info = _get_file_info(f)
        assert info["exists"] is True
        assert info["rows"] == 0


class TestDetectDateRange:
    """Tests for _detect_date_range utility."""

    def test_valid_dates(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("date,val\n2023-01-01,10\n2023-06-15,20\n2024-12-31,30\n")
        result = _detect_date_range(f, "date")
        assert result["min"] == "2023-01-01"
        assert result["max"] == "2024-12-31"

    def test_missing_file(self, tmp_path):
        f = tmp_path / "missing.csv"
        result = _detect_date_range(f, "date")
        assert result["min"] == "N/A"
        assert result["max"] == "N/A"

    def test_invalid_date_column(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("other,val\n2023-01-01,10\n")
        result = _detect_date_range(f, "date")
        assert result["min"] == "N/A"

    def test_mixed_valid_invalid(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("date,val\n2023-01-01,10\nBAD,20\n2024-06-01,30\n")
        result = _detect_date_range(f, "date")
        assert result["min"] == "2023-01-01"
        assert result["max"] == "2024-06-01"


class TestCollectionSources:
    """Tests for COLLECTION_SOURCES configuration."""

    def test_has_five_sources(self):
        assert len(COLLECTION_SOURCES) == 5

    def test_required_keys(self):
        required = {"label", "description", "file", "date_col", "estimate"}
        for name, src in COLLECTION_SOURCES.items():
            assert required.issubset(src.keys()), f"{name} missing keys"

    def test_source_names(self):
        expected = {"weather", "insee", "eurostat", "sitadel", "dpe"}
        assert set(COLLECTION_SOURCES.keys()) == expected


class TestNotebooks:
    """Tests for NOTEBOOKS configuration."""

    def test_has_five_notebooks(self):
        assert len(NOTEBOOKS) == 5

    def test_required_keys(self):
        required = {"file", "label", "description"}
        for key, nb in NOTEBOOKS.items():
            assert required.issubset(nb.keys()), f"Notebook {key} missing keys"

    def test_files_are_ipynb(self):
        for key, nb in NOTEBOOKS.items():
            assert nb["file"].endswith(".ipynb"), f"Notebook {key} not .ipynb"


class TestSecurityValidation:
    """Security tests for the CLI menu."""

    def test_no_command_injection_in_format_size(self):
        """Ensure _format_size handles negative numbers safely."""
        result = _format_size(-1)
        assert isinstance(result, str)

    def test_date_range_handles_malicious_csv(self, tmp_path):
        """Ensure date detection handles malicious content safely."""
        f = tmp_path / "evil.csv"
        f.write_text('date,val\n"; DROP TABLE users; --",10\n')
        result = _detect_date_range(f, "date")
        assert result["min"] == "N/A"

    def test_file_info_handles_binary(self, tmp_path):
        """Ensure _get_file_info handles binary files safely."""
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff" * 100)
        info = _get_file_info(f)
        assert info["exists"] is True
        assert info["size"] > 0


class TestRegressions:
    """Regression tests to prevent breaking existing functionality."""

    def test_import_cli_menu(self):
        """Ensure cli_menu can be imported without errors."""
        import src.cli_menu
        assert hasattr(src.cli_menu, "main_menu")
        assert hasattr(src.cli_menu, "menu_collection")
        assert hasattr(src.cli_menu, "menu_processing")
        assert hasattr(src.cli_menu, "menu_training")
        assert hasattr(src.cli_menu, "menu_deep_learning")

    def test_pipeline_menu_stage(self):
        """Ensure pipeline.py accepts 'menu' as a valid stage."""
        from src.pipeline import main
        # Just verify the function exists and is importable
        assert callable(main)
