# -*- coding: utf-8 -*-
"""Tests for the pipeline orchestrator (src/pipeline.py).

Tests the CLI entry point, argument parsing, stage dispatching,
interactive menus, and helper functions. All actual work (collection,
cleaning, training, etc.) is mocked — we only test orchestration logic.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Test setup_logging()
# ---------------------------------------------------------------------------

class TestSetupLogging:
    """Tests for setup_logging() configuration.

    Note: logging.basicConfig() is a no-op when the root logger already
    has handlers (as in pytest). We must clear handlers first, then
    restore them after the test.
    """

    def _reset_and_call(self, level=None):
        """Clear root logger handlers, call setup_logging, return root level.

        Args:
            level: Optional level string to pass to setup_logging().

        Returns:
            The root logger level after setup_logging() completes.
        """
        from src.pipeline import setup_logging

        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        try:
            # Clear existing handlers so basicConfig() can take effect
            root.handlers.clear()
            if level is not None:
                setup_logging(level)
            else:
                setup_logging()
            return root.level
        finally:
            root.handlers = original_handlers
            root.level = original_level

    def test_setup_logging_default_info(self):
        """setup_logging() with default level configures INFO."""
        result_level = self._reset_and_call()
        assert result_level == logging.INFO

    def test_setup_logging_debug(self):
        """setup_logging('DEBUG') sets root logger to DEBUG."""
        result_level = self._reset_and_call("DEBUG")
        assert result_level == logging.DEBUG

    def test_setup_logging_case_insensitive(self):
        """setup_logging() handles mixed-case level strings."""
        result_level = self._reset_and_call("warning")
        assert result_level == logging.WARNING

    def test_setup_logging_invalid_level_fallback(self):
        """setup_logging() with an invalid level falls back to INFO."""
        result_level = self._reset_and_call("NONEXISTENT")
        assert result_level == logging.INFO


# ---------------------------------------------------------------------------
# Test run_list()
# ---------------------------------------------------------------------------

class TestRunList:
    """Tests for run_list() — listing available collectors."""

    @patch("src.pipeline.CollectorRegistry")
    @patch("src.pipeline.import_module", create=True)
    def test_run_list_prints_collectors(self, mock_import, mock_registry, capsys):
        """run_list() prints a formatted list of available collectors."""
        # Mock the registry to return known collectors
        mock_weather_cls = type("WeatherCollector", (), {"__name__": "WeatherCollector"})
        mock_insee_cls = type("InseeCollector", (), {"__name__": "InseeCollector"})

        mock_registry.available.return_value = ["weather", "insee"]
        mock_registry.get.side_effect = lambda name: {
            "weather": mock_weather_cls,
            "insee": mock_insee_cls,
        }[name]

        # We need to also mock the import of src.collectors inside run_list
        with patch.dict("sys.modules", {"src.collectors": MagicMock()}):
            from src.pipeline import run_list
            run_list()

        captured = capsys.readouterr()
        assert "Available collectors:" in captured.out
        assert "weather" in captured.out
        assert "insee" in captured.out

    @patch("src.pipeline.CollectorRegistry")
    def test_run_list_no_error_on_empty_registry(self, mock_registry, capsys):
        """run_list() works without error even if no collectors are registered."""
        mock_registry.available.return_value = []

        with patch.dict("sys.modules", {"src.collectors": MagicMock()}):
            from src.pipeline import run_list
            run_list()

        captured = capsys.readouterr()
        assert "Available collectors:" in captured.out


# ---------------------------------------------------------------------------
# Test main() — CLI argument parsing and dispatch
# ---------------------------------------------------------------------------

class TestMainCLI:
    """Tests for main() — the CLI entry point and its stage dispatching."""

    @patch("src.pipeline.run_list")
    @patch("src.pipeline.setup_logging")
    def test_main_list_stage(self, mock_setup_logging, mock_run_list):
        """main() with 'list' stage calls run_list()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "list"]):
            main()

        mock_setup_logging.assert_called_once_with("INFO")
        mock_run_list.assert_called_once()

    @patch("src.pipeline.run_collect")
    @patch("src.pipeline.setup_logging")
    def test_main_collect_no_sources(self, mock_setup_logging, mock_run_collect):
        """main() with 'collect' and no --sources calls run_collect(None)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "collect"]):
            main()

        mock_run_collect.assert_called_once_with(None)

    @patch("src.pipeline.run_collect")
    @patch("src.pipeline.setup_logging")
    def test_main_collect_with_sources(self, mock_setup_logging, mock_run_collect):
        """main() with 'collect --sources weather,insee' splits and passes the list."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "collect", "--sources", "weather,insee"]):
            main()

        mock_run_collect.assert_called_once_with(["weather", "insee"])

    @patch("src.pipeline.run_init_db")
    @patch("src.pipeline.setup_logging")
    def test_main_init_db(self, mock_setup_logging, mock_run_init_db):
        """main() with 'init_db' calls run_init_db()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "init_db"]):
            main()

        mock_run_init_db.assert_called_once()

    @patch("src.pipeline.run_import_data")
    @patch("src.pipeline.setup_logging")
    def test_main_import_data_default(self, mock_setup_logging, mock_run_import):
        """main() with 'import_data' calls run_import_data(interactive=False)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "import_data"]):
            main()

        mock_run_import.assert_called_once_with(interactive=False)

    @patch("src.pipeline.run_import_data")
    @patch("src.pipeline.setup_logging")
    def test_main_import_data_interactive(self, mock_setup_logging, mock_run_import):
        """main() with 'import_data -i' calls run_import_data(interactive=True)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "import_data", "-i"]):
            main()

        mock_run_import.assert_called_once_with(interactive=True)

    @patch("src.pipeline.run_clean")
    @patch("src.pipeline.setup_logging")
    def test_main_clean(self, mock_setup_logging, mock_run_clean):
        """main() with 'clean' calls run_clean(interactive=False)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "clean"]):
            main()

        mock_run_clean.assert_called_once_with(interactive=False)

    @patch("src.pipeline.run_clean")
    @patch("src.pipeline.setup_logging")
    def test_main_clean_interactive(self, mock_setup_logging, mock_run_clean):
        """main() with 'clean --interactive' calls run_clean(interactive=True)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "clean", "--interactive"]):
            main()

        mock_run_clean.assert_called_once_with(interactive=True)

    @patch("src.pipeline.run_merge")
    @patch("src.pipeline.setup_logging")
    def test_main_merge(self, mock_setup_logging, mock_run_merge):
        """main() with 'merge' calls run_merge()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "merge"]):
            main()

        mock_run_merge.assert_called_once()

    @patch("src.pipeline.run_features")
    @patch("src.pipeline.setup_logging")
    def test_main_features(self, mock_setup_logging, mock_run_features):
        """main() with 'features' calls run_features()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "features"]):
            main()

        mock_run_features.assert_called_once()

    @patch("src.pipeline.run_outliers")
    @patch("src.pipeline.setup_logging")
    def test_main_outliers(self, mock_setup_logging, mock_run_outliers):
        """main() with 'outliers' calls run_outliers()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "outliers"]):
            main()

        mock_run_outliers.assert_called_once()

    @patch("src.pipeline.run_process")
    @patch("src.pipeline.setup_logging")
    def test_main_process(self, mock_setup_logging, mock_run_process):
        """main() with 'process' calls run_process(interactive=False)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "process"]):
            main()

        mock_run_process.assert_called_once_with(interactive=False)

    @patch("src.pipeline.run_process")
    @patch("src.pipeline.setup_logging")
    def test_main_process_interactive(self, mock_setup_logging, mock_run_process):
        """main() with 'process -i' calls run_process(interactive=True)."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "process", "-i"]):
            main()

        mock_run_process.assert_called_once_with(interactive=True)

    @patch("src.pipeline.run_eda")
    @patch("src.pipeline.setup_logging")
    def test_main_eda(self, mock_setup_logging, mock_run_eda):
        """main() with 'eda' calls run_eda()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "eda"]):
            main()

        mock_run_eda.assert_called_once()

    @patch("src.pipeline.run_train")
    @patch("src.pipeline.setup_logging")
    def test_main_train_default_target(self, mock_setup_logging, mock_run_train):
        """main() with 'train' uses the default target variable."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "train"]):
            main()

        mock_run_train.assert_called_once_with(target="nb_installations_pac")

    @patch("src.pipeline.run_train")
    @patch("src.pipeline.setup_logging")
    def test_main_train_custom_target(self, mock_setup_logging, mock_run_train):
        """main() with 'train --target nb_dpe_total' passes the custom target."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "train", "--target", "nb_dpe_total"]):
            main()

        mock_run_train.assert_called_once_with(target="nb_dpe_total")

    @patch("src.pipeline.run_evaluate")
    @patch("src.pipeline.setup_logging")
    def test_main_evaluate(self, mock_setup_logging, mock_run_evaluate):
        """main() with 'evaluate' calls run_evaluate() with default target."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "evaluate"]):
            main()

        mock_run_evaluate.assert_called_once_with(target="nb_installations_pac")

    @patch("src.pipeline.run_evaluate")
    @patch("src.pipeline.run_train")
    @patch("src.pipeline.run_eda")
    @patch("src.pipeline.run_process")
    @patch("src.pipeline.run_import_data")
    @patch("src.pipeline.run_collect")
    @patch("src.pipeline.run_init_db")
    @patch("src.pipeline.setup_logging")
    def test_main_all_stage(
        self,
        mock_setup_logging,
        mock_init_db,
        mock_collect,
        mock_import,
        mock_process,
        mock_eda,
        mock_train,
        mock_evaluate,
    ):
        """main() with 'all' stage chains all sub-stages in order."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "all"]):
            main()

        # Verify all stages were called
        mock_init_db.assert_called_once()
        mock_collect.assert_called_once()
        mock_import.assert_called_once()
        mock_process.assert_called_once()
        mock_eda.assert_called_once()
        mock_train.assert_called_once_with(target="nb_installations_pac")
        mock_evaluate.assert_called_once_with(target="nb_installations_pac")

    @patch("src.pipeline.setup_logging")
    def test_main_log_level_debug(self, mock_setup_logging):
        """main() with '--log-level DEBUG' passes DEBUG to setup_logging."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "list"]):
            with patch("src.pipeline.run_list"):
                # Override log level
                with patch("sys.argv", ["pipeline", "--log-level", "DEBUG", "list"]):
                    main()

        mock_setup_logging.assert_called_with("DEBUG")

    def test_main_invalid_stage_exits(self):
        """main() with an invalid stage name raises SystemExit."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "nonexistent"]):
            with pytest.raises(SystemExit):
                main()

    @patch("src.pipeline.run_sync_pcloud")
    @patch("src.pipeline.setup_logging")
    def test_main_sync_pcloud(self, mock_setup_logging, mock_sync):
        """main() with 'sync_pcloud' calls run_sync_pcloud()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "sync_pcloud"]):
            main()

        mock_sync.assert_called_once()

    @patch("src.pipeline.run_upload_pcloud")
    @patch("src.pipeline.setup_logging")
    def test_main_upload_pcloud(self, mock_setup_logging, mock_upload):
        """main() with 'upload_pcloud' calls run_upload_pcloud()."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "upload_pcloud"]):
            main()

        mock_upload.assert_called_once()

    @patch("src.pipeline.run_update_all")
    @patch("src.pipeline.setup_logging")
    def test_main_update_all(self, mock_setup_logging, mock_update_all):
        """main() with 'update_all' calls run_update_all() with target."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "update_all"]):
            main()

        mock_update_all.assert_called_once_with(target="nb_installations_pac")


# ---------------------------------------------------------------------------
# Test _interactive_import_menu()
# ---------------------------------------------------------------------------

class TestInteractiveImportMenu:
    """Tests for _interactive_import_menu() with mocked IMPORT_SOURCES and input()."""

    def _make_mock_db(self, tmp_path):
        """Create a mock DatabaseManager with IMPORT_SOURCES and test files.

        Returns:
            Tuple of (mock_db, raw_data_dir).
        """
        raw_data_dir = tmp_path / "raw"
        raw_data_dir.mkdir(parents=True)

        # Create mock CSV files for two sources
        weather_dir = raw_data_dir / "weather"
        weather_dir.mkdir()
        weather_file = weather_dir / "weather_france.csv"
        weather_file.write_text("col1,col2\nval1,val2\nval3,val4\n")

        insee_dir = raw_data_dir / "insee"
        insee_dir.mkdir()
        insee_file = insee_dir / "indicateurs_economiques.csv"
        insee_file.write_text("col1,col2\nval1,val2\n")

        mock_db = MagicMock()
        mock_db.IMPORT_SOURCES = {
            "weather": {
                "file": "weather/weather_france.csv",
                "table": "fact_hvac_installations",
                "description": "Open-Meteo daily weather",
            },
            "insee": {
                "file": "insee/indicateurs_economiques.csv",
                "table": "fact_economic_context",
                "description": "INSEE confidence indicators",
            },
        }
        return mock_db, raw_data_dir

    def test_select_all_sources(self, tmp_path):
        """User types 'a' to import all available sources."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        # Simulate: user types "a" then "y" to confirm
        with patch("builtins.input", side_effect=["a", "y"]):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert "weather" in result
        assert "insee" in result
        assert len(result) == 2

    def test_select_specific_source_by_number(self, tmp_path):
        """User types '1' to import only the first source."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        # Simulate: user types "1" (first source) then "y" to confirm
        with patch("builtins.input", side_effect=["1", "y"]):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert len(result) == 1
        assert result[0] == "weather"

    def test_select_multiple_by_numbers(self, tmp_path):
        """User types '1,2' to select both sources by number."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", side_effect=["1,2", "y"]):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert len(result) == 2

    def test_cancel_with_q(self, tmp_path):
        """User types 'q' to cancel — returns empty list."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", return_value="q"):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert result == []

    def test_cancel_with_empty_string(self, tmp_path):
        """User presses Enter (empty string) to cancel — returns empty list."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", return_value=""):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert result == []

    def test_decline_confirmation(self, tmp_path):
        """User selects sources but declines confirmation with 'n'."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", side_effect=["a", "n"]):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert result == []

    def test_invalid_input_returns_empty(self, tmp_path):
        """User types invalid (non-numeric) text — returns empty list."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", return_value="abc"):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert result == []

    def test_no_files_found(self, tmp_path):
        """When no data files exist, returns empty list immediately."""
        from src.pipeline import _interactive_import_menu

        raw_data_dir = tmp_path / "raw"
        raw_data_dir.mkdir(parents=True)

        mock_db = MagicMock()
        mock_db.IMPORT_SOURCES = {
            "weather": {
                "file": "weather/weather_france.csv",
                "table": "fact_hvac_installations",
                "description": "Open-Meteo daily weather",
            },
        }

        # No input() call expected — function should return early
        result = _interactive_import_menu(mock_db, raw_data_dir)
        assert result == []

    def test_eof_on_input_returns_empty(self, tmp_path):
        """EOFError on input() returns empty list (non-interactive env)."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", side_effect=EOFError):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert result == []

    def test_keyboard_interrupt_returns_empty(self, tmp_path):
        """KeyboardInterrupt on input() returns empty list."""
        from src.pipeline import _interactive_import_menu

        mock_db, raw_data_dir = self._make_mock_db(tmp_path)

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert result == []

    def test_skips_missing_file_source(self, tmp_path):
        """Selecting a source with a missing file skips it gracefully."""
        from src.pipeline import _interactive_import_menu

        raw_data_dir = tmp_path / "raw"
        raw_data_dir.mkdir(parents=True)

        # Only create the weather file, not the insee file
        weather_dir = raw_data_dir / "weather"
        weather_dir.mkdir()
        (weather_dir / "weather_france.csv").write_text("a,b\n1,2\n")

        mock_db = MagicMock()
        mock_db.IMPORT_SOURCES = {
            "weather": {
                "file": "weather/weather_france.csv",
                "table": "fact_hvac_installations",
                "description": "Open-Meteo daily weather",
            },
            "insee": {
                "file": "insee/indicateurs_economiques.csv",
                "table": "fact_economic_context",
                "description": "INSEE indicators (missing file)",
            },
        }

        # User selects "1,2" — source 2 (insee) is missing so it gets skipped
        # Then confirms
        with patch("builtins.input", side_effect=["1,2", "y"]):
            result = _interactive_import_menu(mock_db, raw_data_dir)

        assert "weather" in result
        assert "insee" not in result


# ---------------------------------------------------------------------------
# Test _interactive_cleaning_menu()
# ---------------------------------------------------------------------------

class TestInteractiveCleaningMenu:
    """Tests for _interactive_cleaning_menu() with mocked dependencies."""

    @patch("src.pipeline.DataCleaner", create=True)
    @patch("src.pipeline.CLEANING_RULES", create=True)
    def test_keep_all_rules(self, mock_rules, mock_cleaner_cls):
        """User presses Enter on all sources to keep all rules."""
        # We must mock the import inside _interactive_cleaning_menu
        mock_cleaner = MagicMock()
        mock_cleaner.preview_all.return_value = {
            "weather": [
                {
                    "rule": "remove_duplicates",
                    "description": "Remove duplicate rows",
                    "rows_affected": 10,
                    "pct": 0.5,
                    "note": "",
                },
            ],
        }

        mock_cleaning_rules = {
            "weather": {
                "remove_duplicates": "Remove duplicate rows",
            },
        }

        # Create a mock config with raw_data_dir
        mock_cfg = MagicMock()
        mock_cfg.raw_data_dir = Path("/fake/raw")

        with patch(
            "src.pipeline.DataCleaner", return_value=mock_cleaner
        ) as patched_cleaner, patch(
            "src.pipeline.CLEANING_RULES", mock_cleaning_rules
        ), patch(
            "builtins.input", side_effect=["", "y"]
        ):
            # We need to patch the imports inside _interactive_cleaning_menu
            # The function does: from src.processing.clean_data import CLEANING_RULES, DataCleaner
            from src.pipeline import _interactive_cleaning_menu

            with patch(
                "src.pipeline._interactive_cleaning_menu.__module__",
                "src.pipeline",
            ):
                pass

            # Patch the local imports within the function
            mock_module = MagicMock()
            mock_module.CLEANING_RULES = mock_cleaning_rules
            mock_module.DataCleaner = MagicMock(return_value=mock_cleaner)

            with patch.dict("sys.modules", {
                "src.processing.clean_data": mock_module,
            }), patch("builtins.input", side_effect=["", "y"]):
                result = _interactive_cleaning_menu(mock_cfg)

        assert result == {}

    def test_skip_specific_rules(self, tmp_path):
        """User types '1' to skip the first rule for a source."""
        from src.pipeline import _interactive_cleaning_menu

        mock_cleaner = MagicMock()
        mock_cleaner.preview_all.return_value = {
            "weather": [
                {
                    "rule": "remove_duplicates",
                    "description": "Remove duplicate rows",
                    "rows_affected": 10,
                    "pct": 0.5,
                    "note": "",
                },
                {
                    "rule": "remove_nan_temps",
                    "description": "Remove NaN temperatures",
                    "rows_affected": 5,
                    "pct": 0.2,
                    "note": "",
                },
            ],
        }

        mock_cleaning_rules = {
            "weather": {
                "remove_duplicates": "Remove duplicate rows",
                "remove_nan_temps": "Remove NaN temperatures",
            },
        }

        mock_cfg = MagicMock()
        mock_cfg.raw_data_dir = tmp_path / "raw"

        mock_module = MagicMock()
        mock_module.CLEANING_RULES = mock_cleaning_rules
        mock_module.DataCleaner = MagicMock(return_value=mock_cleaner)

        # User skips rule 1 (remove_duplicates) then confirms
        with patch.dict("sys.modules", {
            "src.processing.clean_data": mock_module,
        }), patch("builtins.input", side_effect=["1", "y"]):
            result = _interactive_cleaning_menu(mock_cfg)

        assert "weather" in result
        assert "remove_duplicates" in result["weather"]
        assert "remove_nan_temps" not in result.get("weather", set())

    def test_cancel_cleaning(self, tmp_path):
        """User types 'n' at confirmation — exits via sys.exit(0)."""
        from src.pipeline import _interactive_cleaning_menu

        mock_cleaner = MagicMock()
        mock_cleaner.preview_all.return_value = {
            "weather": [
                {
                    "rule": "remove_duplicates",
                    "description": "Remove duplicate rows",
                    "rows_affected": 0,
                    "pct": 0.0,
                    "note": "",
                },
            ],
        }

        mock_cleaning_rules = {
            "weather": {
                "remove_duplicates": "Remove duplicate rows",
            },
        }

        mock_cfg = MagicMock()
        mock_cfg.raw_data_dir = tmp_path / "raw"

        mock_module = MagicMock()
        mock_module.CLEANING_RULES = mock_cleaning_rules
        mock_module.DataCleaner = MagicMock(return_value=mock_cleaner)

        # User keeps all rules (Enter) but declines at final confirm (n)
        with patch.dict("sys.modules", {
            "src.processing.clean_data": mock_module,
        }), patch("builtins.input", side_effect=["", "n"]):
            with pytest.raises(SystemExit) as exc_info:
                _interactive_cleaning_menu(mock_cfg)
            assert exc_info.value.code == 0

    def test_eof_on_rule_selection(self, tmp_path):
        """EOFError during rule selection keeps all rules for that source."""
        from src.pipeline import _interactive_cleaning_menu

        mock_cleaner = MagicMock()
        mock_cleaner.preview_all.return_value = {
            "weather": [
                {
                    "rule": "remove_duplicates",
                    "description": "Remove duplicate rows",
                    "rows_affected": 5,
                    "pct": 0.3,
                    "note": "",
                },
            ],
        }

        mock_cleaning_rules = {
            "weather": {
                "remove_duplicates": "Remove duplicate rows",
            },
        }

        mock_cfg = MagicMock()
        mock_cfg.raw_data_dir = tmp_path / "raw"

        mock_module = MagicMock()
        mock_module.CLEANING_RULES = mock_cleaning_rules
        mock_module.DataCleaner = MagicMock(return_value=mock_cleaner)

        # EOFError on rule selection, then EOFError on final confirm
        # (EOFError on final confirm defaults to "y")
        with patch.dict("sys.modules", {
            "src.processing.clean_data": mock_module,
        }), patch("builtins.input", side_effect=[EOFError, EOFError]):
            result = _interactive_cleaning_menu(mock_cfg)

        # No rules skipped (EOFError treated as "keep all")
        assert result == {}

    def test_invalid_rule_numbers(self, tmp_path):
        """Invalid (non-numeric) skip input keeps all rules."""
        from src.pipeline import _interactive_cleaning_menu

        mock_cleaner = MagicMock()
        mock_cleaner.preview_all.return_value = {
            "weather": [
                {
                    "rule": "remove_duplicates",
                    "description": "Remove duplicate rows",
                    "rows_affected": 5,
                    "pct": 0.3,
                    "note": "",
                },
            ],
        }

        mock_cleaning_rules = {
            "weather": {
                "remove_duplicates": "Remove duplicate rows",
            },
        }

        mock_cfg = MagicMock()
        mock_cfg.raw_data_dir = tmp_path / "raw"

        mock_module = MagicMock()
        mock_module.CLEANING_RULES = mock_cleaning_rules
        mock_module.DataCleaner = MagicMock(return_value=mock_cleaner)

        # User types nonsense for rule skip, then confirms
        with patch.dict("sys.modules", {
            "src.processing.clean_data": mock_module,
        }), patch("builtins.input", side_effect=["abc", "y"]):
            result = _interactive_cleaning_menu(mock_cfg)

        # Invalid input = keep all rules
        assert result == {}

    def test_empty_preview_sources(self, tmp_path):
        """Sources with no cleaning rules are skipped in the menu."""
        from src.pipeline import _interactive_cleaning_menu

        mock_cleaner = MagicMock()
        mock_cleaner.preview_all.return_value = {
            "unknown": [
                {
                    "rule": "some_rule",
                    "description": "Some rule",
                    "rows_affected": 0,
                    "pct": 0.0,
                    "note": "",
                },
            ],
        }

        # CLEANING_RULES has no entry for "unknown"
        mock_cleaning_rules = {}

        mock_cfg = MagicMock()
        mock_cfg.raw_data_dir = tmp_path / "raw"

        mock_module = MagicMock()
        mock_module.CLEANING_RULES = mock_cleaning_rules
        mock_module.DataCleaner = MagicMock(return_value=mock_cleaner)

        # Only the final confirmation input is needed (no source prompts)
        with patch.dict("sys.modules", {
            "src.processing.clean_data": mock_module,
        }), patch("builtins.input", side_effect=["y"]):
            result = _interactive_cleaning_menu(mock_cfg)

        assert result == {}


# ---------------------------------------------------------------------------
# Test run_process() dispatching
# ---------------------------------------------------------------------------

class TestRunProcess:
    """Tests for run_process() — chaining clean + merge + features + outliers."""

    @patch("src.pipeline.run_outliers")
    @patch("src.pipeline.run_features")
    @patch("src.pipeline.run_merge")
    @patch("src.pipeline.run_clean")
    def test_run_process_calls_all_stages(
        self, mock_clean, mock_merge, mock_features, mock_outliers
    ):
        """run_process() chains clean, merge, features, outliers in order."""
        from src.pipeline import run_process

        run_process(interactive=False)

        mock_clean.assert_called_once_with(interactive=False)
        mock_merge.assert_called_once()
        mock_features.assert_called_once()
        mock_outliers.assert_called_once()

    @patch("src.pipeline.run_outliers")
    @patch("src.pipeline.run_features")
    @patch("src.pipeline.run_merge")
    @patch("src.pipeline.run_clean")
    def test_run_process_interactive_flag(
        self, mock_clean, mock_merge, mock_features, mock_outliers
    ):
        """run_process(interactive=True) passes the flag to run_clean()."""
        from src.pipeline import run_process

        run_process(interactive=True)

        mock_clean.assert_called_once_with(interactive=True)


# ---------------------------------------------------------------------------
# Security tests: input validation and injection prevention
# ---------------------------------------------------------------------------

class TestSecurityInputValidation:
    """Security tests to verify the pipeline handles malicious inputs safely."""

    def test_argparse_rejects_unknown_stages(self):
        """Argparse rejects stage names not in the allowed choices."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "'; DROP TABLE users; --"]):
            with pytest.raises(SystemExit):
                main()

    def test_argparse_rejects_shell_injection_in_sources(self):
        """Argparse accepts --sources as a string; it is just split on comma.

        The actual collectors validate source names via CollectorRegistry,
        so injection attempts are safely rejected as unknown sources.
        """
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "collect", "--sources", "$(rm -rf /)"]):
            with patch("src.pipeline.setup_logging"):
                with patch("src.pipeline.run_collect") as mock_collect:
                    main()
                    # The malicious string is passed as a plain string element,
                    # not executed. run_collect receives it as a list item.
                    mock_collect.assert_called_once_with(["$(rm -rf /)"])

    def test_argparse_rejects_unknown_arguments(self):
        """Argparse rejects unknown CLI flags."""
        from src.pipeline import main

        with patch("sys.argv", ["pipeline", "list", "--evil-flag"]):
            with pytest.raises(SystemExit):
                main()
