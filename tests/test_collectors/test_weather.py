# -*- coding: utf-8 -*-
"""Tests for the weather collector (WeatherCollector)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.collectors.weather import WeatherCollector


class TestWeatherCollector:
    """Tests for WeatherCollector."""

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
        with pytest.raises(ValueError, match="Required columns missing"):
            collector.validate(df)

    def test_validate_warns_on_nulls(self, collector_config, sample_weather_df, caplog):
        collector = WeatherCollector(collector_config)
        # Inject 10% NaN into temperature
        sample_weather_df.loc[:5, "temperature_2m_mean"] = None
        import logging
        with caplog.at_level(logging.WARNING):
            collector.validate(sample_weather_df)

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "_fetch_elevations", return_value={"69": 175.0, "38": 212.0})
    @patch.object(WeatherCollector, "fetch_json")
    def test_collect_success(self, mock_fetch, mock_elev, mock_sleep, collector_config):
        """Test that collect returns a valid DataFrame with mocked data."""
        # Simulate the Open-Meteo API response
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
        assert "pac_inefficient" in df.columns
        assert "elevation" in df.columns
        # HDD must be > 0 when temp < 18
        assert (df["hdd"] >= 0).all()
        assert (df["cdd"] >= 0).all()
        # PAC inefficiency: none of these temps are below -7
        assert (df["pac_inefficient"] == 0).all()

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "_fetch_elevations", return_value={"69": 175.0, "38": 212.0})
    @patch.object(WeatherCollector, "fetch_json")
    def test_collect_partial_failure(self, mock_fetch, mock_elev, mock_sleep, collector_config):
        """Test resilience when a city fails."""
        good_response = {
            "daily": {
                "time": ["2023-01-01"],
                "temperature_2m_max": [5.0],
                "temperature_2m_min": [-2.0],
                "temperature_2m_mean": [1.5],
                "precipitation_sum": [2.0],
                "wind_speed_10m_max": [25.0],
            }
        }
        # Pass 1: city1 OK, city2 fails
        # Retry pass: city2 fails again
        mock_fetch.side_effect = [
            good_response,
            Exception("API timeout"),
            Exception("API timeout"),
        ]
        collector = WeatherCollector(collector_config)
        df = collector.collect()

        # We should still have the data from the first city
        assert not df.empty
        assert len(df) >= 1
        assert "pac_inefficient" in df.columns
        assert "elevation" in df.columns


class TestFetchWithRetry:
    """Tests for _fetch_with_retry (429 backoff logic)."""

    @patch.object(WeatherCollector, "fetch_json")
    def test_success_on_first_attempt(self, mock_fetch, collector_config):
        """Returns data immediately when no error occurs."""
        mock_fetch.return_value = {"daily": {"time": ["2023-01-01"]}}
        collector = WeatherCollector(collector_config)
        result = collector._fetch_with_retry("http://test", {}, "TestCity")
        assert result == {"daily": {"time": ["2023-01-01"]}}
        assert mock_fetch.call_count == 1

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "fetch_json")
    def test_retries_on_429_then_succeeds(self, mock_fetch, mock_sleep, collector_config):
        """Retries on 429 and eventually succeeds."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        http_429 = requests.exceptions.HTTPError(response=resp_429)

        mock_fetch.side_effect = [
            http_429,
            http_429,
            {"daily": {"time": ["2023-01-01"]}},
        ]
        collector = WeatherCollector(collector_config)
        result = collector._fetch_with_retry("http://test", {}, "TestCity")

        assert result is not None
        assert result["daily"]["time"] == ["2023-01-01"]
        assert mock_fetch.call_count == 3
        # Should have slept twice (5s, 10s)
        assert mock_sleep.call_count == 2

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "fetch_json")
    def test_returns_none_after_max_retries(self, mock_fetch, mock_sleep, collector_config):
        """Returns None when all 429 retries are exhausted."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        http_429 = requests.exceptions.HTTPError(response=resp_429)

        mock_fetch.side_effect = http_429
        collector = WeatherCollector(collector_config)
        result = collector._fetch_with_retry("http://test", {}, "TestCity")

        assert result is None
        assert mock_fetch.call_count == WeatherCollector._MAX_429_RETRIES

    @patch.object(WeatherCollector, "fetch_json")
    def test_non_429_error_raised_immediately(self, mock_fetch, collector_config):
        """Non-429 HTTP errors are raised without retry."""
        resp_500 = MagicMock()
        resp_500.status_code = 500
        http_500 = requests.exceptions.HTTPError(response=resp_500)

        mock_fetch.side_effect = http_500
        collector = WeatherCollector(collector_config)

        with pytest.raises(requests.exceptions.HTTPError):
            collector._fetch_with_retry("http://test", {}, "TestCity")
        assert mock_fetch.call_count == 1

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "fetch_json")
    def test_connection_error_with_429_retried(self, mock_fetch, mock_sleep, collector_config):
        """ConnectionError containing '429' is caught and retried."""
        conn_err = requests.exceptions.ConnectionError(
            "Max retries exceeded: too many 429 error responses"
        )
        mock_fetch.side_effect = [
            conn_err,
            {"daily": {"time": ["2023-01-01"]}},
        ]
        collector = WeatherCollector(collector_config)
        result = collector._fetch_with_retry("http://test", {}, "TestCity")

        assert result is not None
        assert mock_fetch.call_count == 2

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "fetch_json")
    def test_exponential_backoff_timing(self, mock_fetch, mock_sleep, collector_config):
        """Backoff follows exponential pattern: 5, 10, 20, 40, 80."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        http_429 = requests.exceptions.HTTPError(response=resp_429)

        mock_fetch.side_effect = http_429
        collector = WeatherCollector(collector_config)
        collector._fetch_with_retry("http://test", {}, "TestCity")

        expected_waits = [5.0, 10.0, 20.0, 40.0, 80.0]
        actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
        assert actual_waits == expected_waits


class TestRetryPass:
    """Tests for the retry pass logic in collect()."""

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "_fetch_elevations", return_value={"69": 175.0, "38": 212.0})
    @patch.object(WeatherCollector, "fetch_json")
    def test_failed_cities_retried_on_second_pass(
        self, mock_fetch, mock_elev, mock_sleep, collector_config
    ):
        """Cities that fail on pass 1 are retried on pass 2."""
        good_response = {
            "daily": {
                "time": ["2023-01-01"],
                "temperature_2m_max": [5.0],
                "temperature_2m_min": [-2.0],
                "temperature_2m_mean": [1.5],
                "precipitation_sum": [2.0],
                "wind_speed_10m_max": [25.0],
            }
        }
        conn_err = requests.exceptions.ConnectionError(
            "too many 429 error responses"
        )

        # Pass 1: city1 OK, city2 fails all 5 retries
        # Pass 2: city2 succeeds on retry
        side_effects = [good_response]  # city1 pass 1
        side_effects += [conn_err] * 5  # city2 pass 1 (5 retries, all fail)
        side_effects += [good_response]  # city2 pass 2
        mock_fetch.side_effect = side_effects

        collector = WeatherCollector(collector_config)
        df = collector.collect()

        # Both cities should be in the result
        assert not df.empty
        assert len(df) == 2  # 1 row per city

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "_fetch_elevations", return_value={"69": 175.0, "38": 212.0})
    @patch.object(WeatherCollector, "fetch_json")
    def test_retry_pass_waits_before_starting(
        self, mock_fetch, mock_elev, mock_sleep, collector_config
    ):
        """The retry pass waits _RETRY_PASS_PAUSE before starting."""
        good_response = {
            "daily": {
                "time": ["2023-01-01"],
                "temperature_2m_max": [5.0],
                "temperature_2m_min": [-2.0],
                "temperature_2m_mean": [1.5],
                "precipitation_sum": [2.0],
                "wind_speed_10m_max": [25.0],
            }
        }
        conn_err = requests.exceptions.ConnectionError(
            "too many 429 error responses"
        )

        # city1 OK, city2 fails pass 1, succeeds pass 2
        side_effects = [good_response]
        side_effects += [conn_err] * 5
        side_effects += [good_response]
        mock_fetch.side_effect = side_effects

        collector = WeatherCollector(collector_config)
        collector.collect()

        # Find the 60s cooldown pause in the sleep calls
        sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
        assert WeatherCollector._RETRY_PASS_PAUSE in sleep_values

    @patch("src.collectors.weather.time.sleep")
    @patch.object(WeatherCollector, "_fetch_elevations", return_value={"69": 175.0})
    @patch.object(WeatherCollector, "fetch_json")
    def test_no_retry_pass_when_all_succeed(
        self, mock_fetch, mock_elev, mock_sleep, collector_config
    ):
        """No retry pass triggered when all cities succeed on pass 1."""
        good_response = {
            "daily": {
                "time": ["2023-01-01"],
                "temperature_2m_max": [5.0],
                "temperature_2m_min": [-2.0],
                "temperature_2m_mean": [1.5],
                "precipitation_sum": [2.0],
                "wind_speed_10m_max": [25.0],
            }
        }
        mock_fetch.return_value = good_response

        collector = WeatherCollector(collector_config)
        collector.collect()

        # The 60s retry pass pause should NOT appear
        sleep_values = [call.args[0] for call in mock_sleep.call_args_list]
        assert WeatherCollector._RETRY_PASS_PAUSE not in sleep_values
