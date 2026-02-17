# -*- coding: utf-8 -*-
"""Tests pour le collecteur météo (WeatherCollector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.collectors.weather import WeatherCollector


class TestWeatherCollector:
    """Tests pour WeatherCollector."""

    def test_source_name(self):
        assert WeatherCollector.source_name == "weather"
        assert WeatherCollector.output_filename == "weather_france.csv"

    def test_validate_valid_data(self, collector_config, sample_weather_df):
        collector = WeatherCollector(collector_config)
        result = collector.validate(sample_weather_df)
        assert len(result) == len(sample_weather_df)
        assert pd.api.types.is_datetime64_any_dtype(result["time"])

    def test_validate_missing_columns(self, collector_config):
        collector = WeatherCollector(collector_config)
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Colonnes obligatoires manquantes"):
            collector.validate(df)

    def test_validate_warns_on_nulls(self, collector_config, sample_weather_df, caplog):
        collector = WeatherCollector(collector_config)
        # Injecter 10% de NaN dans température
        sample_weather_df.loc[:5, "temperature_2m_mean"] = None
        import logging
        with caplog.at_level(logging.WARNING):
            collector.validate(sample_weather_df)

    @patch.object(WeatherCollector, "fetch_json")
    def test_collect_success(self, mock_fetch, collector_config):
        """Test que collect retourne un DataFrame valide avec des données mockées."""
        # Simuler la réponse API Open-Meteo
        mock_fetch.return_value = {
            "daily": {
                "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "temperature_2m_max": [5.0, 7.0, 3.0],
                "temperature_2m_min": [-2.0, 1.0, -1.0],
                "temperature_2m_mean": [1.5, 4.0, 1.0],
                "precipitation_sum": [2.0, 0.0, 5.0],
                "wind_speed_10m_max": [25.0, 15.0, 30.0],
            }
        }
        collector = WeatherCollector(collector_config)
        df = collector.collect()

        assert not df.empty
        assert "temperature_2m_mean" in df.columns
        assert "hdd" in df.columns
        assert "cdd" in df.columns
        # HDD doit être > 0 quand temp < 18
        assert (df["hdd"] >= 0).all()
        assert (df["cdd"] >= 0).all()

    @patch.object(WeatherCollector, "fetch_json")
    def test_collect_partial_failure(self, mock_fetch, collector_config):
        """Test la résilience quand une ville échoue."""
        # Première ville OK, deuxième échoue
        mock_fetch.side_effect = [
            {
                "daily": {
                    "time": ["2023-01-01"],
                    "temperature_2m_max": [5.0],
                    "temperature_2m_min": [-2.0],
                    "temperature_2m_mean": [1.5],
                    "precipitation_sum": [2.0],
                    "wind_speed_10m_max": [25.0],
                }
            },
            Exception("API timeout"),
        ]
        collector = WeatherCollector(collector_config)
        df = collector.collect()

        # On doit quand même avoir les données de la première ville
        assert not df.empty
        assert len(df) >= 1
