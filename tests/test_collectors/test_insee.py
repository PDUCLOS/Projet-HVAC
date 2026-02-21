# -*- coding: utf-8 -*-
"""Tests for the INSEE collector (InseeCollector)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.collectors.insee import InseeCollector, INSEE_SERIES


class TestInseeCollector:
    """Tests for InseeCollector."""

    def test_source_name(self):
        assert InseeCollector.source_name == "insee"
        assert InseeCollector.output_filename == "indicateurs_economiques.csv"

    def test_series_config_complete(self):
        """Verify that all series have an idbank and a description."""
        assert len(INSEE_SERIES) >= 4
        for name, info in INSEE_SERIES.items():
            assert "idbank" in info, f"Série '{name}' sans idbank"
            assert "desc" in info, f"Série '{name}' sans description"
            assert len(info["idbank"]) == 9, (
                f"Série '{name}' : idbank '{info['idbank']}' invalide (9 chiffres attendus)"
            )

    def test_validate_valid_data(self, collector_config, sample_insee_df):
        collector = InseeCollector(collector_config)
        result = collector.validate(sample_insee_df)
        assert len(result) == 12
        assert "period" in result.columns

    def test_validate_too_few_series(self, collector_config):
        collector = InseeCollector(collector_config)
        df = pd.DataFrame({
            "period": ["2023-01", "2023-02"],
            "confiance_menages": [95.0, 96.0],
            "climat_affaires_industrie": [100.0, 101.0],
        })
        with pytest.raises(ValueError, match="Trop peu de séries"):
            collector.validate(df)

    def test_validate_missing_period(self, collector_config):
        collector = InseeCollector(collector_config)
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="period"):
            collector.validate(df)
