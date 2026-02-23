# -*- coding: utf-8 -*-
"""
Weather collector — Open-Meteo Archive API.
============================================

Retrieves historical daily weather data for reference cities
in Auvergne-Rhone-Alpes via the Open-Meteo API.

Source: https://open-meteo.com/en/docs/historical-weather-api
Authentication: None (free API, fair use ~10,000 calls/day)

Collected data (daily):
    - temperature_2m_max/min/mean (degC)
    - precipitation_sum (mm)
    - wind_speed_10m_max (km/h)

Computed data:
    - HDD (Heating Degree Days) = max(0, 18 - temp_mean)
      -> Indicator of heating demand
    - CDD (Cooling Degree Days) = max(0, temp_mean - 18)
      -> Indicator of cooling demand
    - pac_inefficient (bool): 1 if temp_min < -7degC
      -> Below -7degC, air-source heat pump COP drops critically (<2.0)
      -> Key domain knowledge: PAC viability depends on outdoor temperature

NOTE: The archive API does not directly provide HDD/CDD;
      they are computed from mean temperature (base 18degC).

Extensibility:
    To add a city, simply add it to the `config.geo.cities`
    dictionary in config/settings.py.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any, ClassVar, Dict, List, Optional

import pandas as pd
import requests

from src.collectors.base import BaseCollector

# Open-Meteo Archive API base URL
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo Elevation API (Copernicus DEM GLO-90, 90m resolution)
OPENMETEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

# Daily weather variables to collect
DAILY_VARIABLES = [
    "temperature_2m_max",     # Daily maximum temperature (degC)
    "temperature_2m_min",     # Daily minimum temperature (degC)
    "temperature_2m_mean",    # Daily mean temperature (degC)
    "precipitation_sum",      # Cumulative precipitation (mm)
    "wind_speed_10m_max",     # Maximum wind speed at 10m (km/h)
]


class WeatherCollector(BaseCollector):
    """Collector for historical weather data via Open-Meteo.

    Collects daily data for each reference city configured in
    `config.geo.cities`. One city = one department.

    The collector is fault-tolerant: if one city fails, the others
    are still collected (partial collection).

    Auto-registered as 'weather' in the CollectorRegistry.
    """

    source_name: ClassVar[str] = "weather"
    output_subdir: ClassVar[str] = "weather"
    output_filename: ClassVar[str] = "weather_france.csv"

    # Rate-limiting constants for Open-Meteo free tier
    _MAX_429_RETRIES: ClassVar[int] = 5           # Max retries on 429 per city
    _INITIAL_BACKOFF: ClassVar[float] = 5.0        # Initial wait on 429 (seconds)
    _BATCH_SIZE: ClassVar[int] = 10                # Cities per batch before extra pause
    _BATCH_PAUSE: ClassVar[float] = 30.0           # Extra pause between batches (seconds)
    _RETRY_PASS_PAUSE: ClassVar[float] = 60.0      # Cooldown before retrying failed cities
    _MAX_RETRY_PASSES: ClassVar[int] = 5           # Max number of retry passes

    def _fetch_with_retry(
        self, url: str, params: Dict[str, Any], city: str
    ) -> Optional[Any]:
        """Fetch JSON with 429-specific exponential backoff.

        The urllib3 Retry handles transient errors, but for sustained
        429 rate limiting across 96 cities, we need a higher-level
        retry loop with longer waits.

        Args:
            url: API endpoint URL.
            params: Query parameters.
            city: City name (for logging).

        Returns:
            Parsed JSON data, or None if all retries exhausted.

        Raises:
            Exception: Re-raises non-429 errors immediately.
        """
        for attempt in range(1, self._MAX_429_RETRIES + 1):
            try:
                return self.fetch_json(url, params=params)
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    wait = self._INITIAL_BACKOFF * (2 ** (attempt - 1))
                    self.logger.warning(
                        "  429 Too Many Requests for %s (attempt %d/%d) "
                        "— waiting %.0fs before retry...",
                        city, attempt, self._MAX_429_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                # Non-429 HTTP errors: raise immediately
                raise
            except requests.exceptions.ConnectionError as exc:
                # urllib3 Retry wraps 429 as ConnectionError after max retries
                error_str = str(exc)
                if "429" in error_str or "too many" in error_str.lower():
                    wait = self._INITIAL_BACKOFF * (2 ** (attempt - 1))
                    self.logger.warning(
                        "  429 rate limit for %s (attempt %d/%d) "
                        "— waiting %.0fs before retry...",
                        city, attempt, self._MAX_429_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                raise

        self.logger.error(
            "  All %d retries exhausted for %s — skipping city",
            self._MAX_429_RETRIES, city,
        )
        return None

    def _collect_single_city(
        self,
        city: str,
        coords: Dict[str, Any],
        effective_end: str,
        idx: int,
        total: int,
    ) -> Optional[pd.DataFrame]:
        """Collect weather data for a single city.

        Args:
            city: City name.
            coords: Dictionary with lat, lon, dept, region.
            effective_end: End date (clamped to yesterday).
            idx: Current index for progress logging.
            total: Total number of cities for progress logging.

        Returns:
            DataFrame with weather data for the city, or None on failure.
        """
        self.logger.info(
            "Weather collection [%d/%d]: %s (lat=%.2f, lon=%.2f, dept=%s)",
            idx, total, city, coords["lat"], coords["lon"], coords["dept"],
        )

        params = {
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start_date": self.config.start_date,
            "end_date": effective_end,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": "Europe/Paris",
        }

        try:
            data = self._fetch_with_retry(
                OPENMETEO_BASE_URL, params=params, city=city,
            )

            if data is None:
                return None

            if "daily" not in data:
                raise ValueError(
                    f"Unexpected API response for {city}: "
                    f"'daily' key missing. Keys received: {list(data.keys())}"
                )

            df = pd.DataFrame(data["daily"])
            df["city"] = city
            df["dept"] = coords["dept"]

            self.logger.info(
                "  ✓ %s: %d days collected (%s → %s)",
                city, len(df),
                df["time"].iloc[0] if len(df) > 0 else "?",
                df["time"].iloc[-1] if len(df) > 0 else "?",
            )
            return df

        except Exception as exc:
            self.logger.error("  ✗ Failed for %s: %s", city, exc)
            return None

    def collect(self) -> pd.DataFrame:
        """Collect weather data for all reference cities.

        For each city, makes a single API call covering the entire
        configured period (2019-2025). The API returns daily data
        in a single JSON block.

        After collection, computes HDD and CDD (base 18degC):
        - HDD (Heating Degree Days) = heating demand
        - CDD (Cooling Degree Days) = cooling demand

        Returns:
            DataFrame with columns: time, city, dept,
            temperature_2m_max/min/mean, precipitation_sum,
            wind_speed_10m_max, hdd, cdd.
        """
        # Load city configuration from settings
        from config.settings import config as project_config
        cities = project_config.geo.cities

        all_frames: List[pd.DataFrame] = []
        errors: List[str] = []

        # Clamp end_date to yesterday: the Open-Meteo Archive API
        # only serves historical data (up to the previous day).
        # Requesting future dates returns HTTP 400 Bad Request.
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        effective_end = min(self.config.end_date, yesterday)
        if effective_end != self.config.end_date:
            self.logger.info(
                "  end_date clamped: %s → %s (Archive API only has historical data)",
                self.config.end_date, effective_end,
            )

        city_items = list(cities.items())
        total_cities = len(city_items)
        pending_cities: List[tuple] = list(city_items)

        # Multi-pass collection: pass 1 collects all, subsequent passes
        # retry only the cities that failed (up to _MAX_RETRY_PASSES total).
        for pass_num in range(1, self._MAX_RETRY_PASSES + 1):
            if not pending_cities:
                break

            if pass_num == 1:
                self.logger.info(
                    "═══ Pass %d/%d: collecting %d cities ═══",
                    pass_num, self._MAX_RETRY_PASSES, len(pending_cities),
                )
            else:
                self.logger.info(
                    "═══ Pass %d/%d: retrying %d failed cities "
                    "(cooldown %.0fs) ═══",
                    pass_num, self._MAX_RETRY_PASSES,
                    len(pending_cities), self._RETRY_PASS_PAUSE,
                )
                time.sleep(self._RETRY_PASS_PAUSE)

            still_failed: List[tuple] = []

            for idx, (city, coords) in enumerate(pending_cities, 1):
                df = self._collect_single_city(
                    city, coords, effective_end, idx, len(pending_cities),
                )
                if df is not None:
                    all_frames.append(df)
                else:
                    still_failed.append((city, coords))

                # Courtesy pause between calls (10s)
                self.rate_limit_pause()

                # Extra pause between batches to stay well under rate limits
                if idx % self._BATCH_SIZE == 0 and idx < len(pending_cities):
                    self.logger.info(
                        "  Batch pause (%d/%d cities done) — waiting %.0fs...",
                        idx, len(pending_cities), self._BATCH_PAUSE,
                    )
                    time.sleep(self._BATCH_PAUSE)

            # Summary for this pass
            succeeded = len(pending_cities) - len(still_failed)
            self.logger.info(
                "═══ Pass %d result: %d/%d succeeded, %d failed ═══",
                pass_num, succeeded, len(pending_cities), len(still_failed),
            )

            pending_cities = still_failed

            # All cities collected — no more passes needed
            if not pending_cities:
                self.logger.info("All %d cities collected!", total_cities)
                break

        # Record any cities still failed after all passes
        for city, coords in pending_cities:
            errors.append(
                f"Failed for {city} (dept {coords['dept']}): "
                f"exhausted all {self._MAX_RETRY_PASSES} passes"
            )

        # Check that at least one city was collected
        if not all_frames:
            self.logger.error(
                "No weather data collected. Errors: %s", errors
            )
            return pd.DataFrame()

        # Concatenate all cities
        result = pd.concat(all_frames, ignore_index=True)

        # Compute HDD and CDD (base 18degC)
        # HDD = heating demand: the colder it is, the higher the HDD
        # CDD = cooling demand: the hotter it is, the higher the CDD
        if "temperature_2m_mean" in result.columns:
            result["hdd"] = (18.0 - result["temperature_2m_mean"]).clip(lower=0)
            result["cdd"] = (result["temperature_2m_mean"] - 18.0).clip(lower=0)
            self.logger.info(
                "HDD/CDD computed (base 18°C) — "
                "mean HDD=%.1f, mean CDD=%.1f",
                result["hdd"].mean(), result["cdd"].mean(),
            )

        # PAC inefficiency flag: days where T_min < -7°C
        # Below -7°C, air-source heat pump COP drops below ~2.0,
        # making it economically unviable vs gas/oil in most cases.
        pac_threshold = project_config.thresholds.pac_inefficiency_temp
        if "temperature_2m_min" in result.columns:
            result["pac_inefficient"] = (
                result["temperature_2m_min"] < pac_threshold
            ).astype(int)
            n_inefficient = result["pac_inefficient"].sum()
            pct = 100 * n_inefficient / len(result) if len(result) > 0 else 0
            self.logger.info(
                "PAC inefficiency flag: %d days (%.1f%%) with T_min < %.0f°C",
                n_inefficient, pct, pac_threshold,
            )

        # Fetch elevation for each department and attach to data
        elevations = self._fetch_elevations(cities)
        result["elevation"] = result["dept"].map(elevations).fillna(0).astype(float)
        self.logger.info(
            "Elevation attached — range: %.0fm (min) to %.0fm (max)",
            result["elevation"].min(), result["elevation"].max(),
        )

        # Log summary if partial collection
        if errors:
            self.logger.warning(
                "⚠ Partial collection: %d/%d cities succeeded. "
                "Cities with errors: %s",
                len(all_frames), len(cities), errors,
            )

        return result

    def _fetch_elevations(self, cities: Dict[str, Dict]) -> Dict[str, float]:
        """Fetch elevation for each reference city via Open-Meteo API.

        Falls back to static PREFECTURE_ELEVATIONS from config if the API
        is unreachable (offline mode, proxy, etc.).

        Args:
            cities: Dictionary {city_name: {lat, lon, dept, region}}.

        Returns:
            Dictionary {dept_code: elevation_meters}.
        """
        from config.settings import PREFECTURE_ELEVATIONS

        items = list(cities.items())
        lats = ",".join(str(c["lat"]) for _, c in items)
        lons = ",".join(str(c["lon"]) for _, c in items)

        try:
            data = self.fetch_json(
                OPENMETEO_ELEVATION_URL,
                params={"latitude": lats, "longitude": lons},
            )
            elevations = data.get("elevation", [])
            if len(elevations) == len(items):
                result = {
                    items[i][1]["dept"]: elevations[i]
                    for i in range(len(items))
                }
                self.logger.info(
                    "Elevation fetched via API for %d cities", len(result),
                )
                return result
        except Exception as exc:
            self.logger.warning(
                "Elevation API unavailable (%s), using static reference", exc,
            )

        # Fallback: use static reference data
        result = {}
        for _, coords in items:
            dept = coords["dept"]
            result[dept] = float(PREFECTURE_ELEVATIONS.get(dept, 0))
        self.logger.info(
            "Elevation loaded from static reference for %d departments",
            len(result),
        )
        return result

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the structure and quality of weather data.

        Checks:
        1. Presence of required columns
        2. Type conversion (dates, numerics)
        3. Detection of abnormal null values (threshold 5%)
        4. Statistical summary for visual diagnostics

        Args:
            df: Raw DataFrame from collect().

        Returns:
            Validated DataFrame with correct types.

        Raises:
            ValueError: If critical columns are missing.
        """
        # 1. Check required columns
        required_cols = {"time", "city", "dept", "temperature_2m_mean"}
        expected_cols = {"pac_inefficient", "elevation"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Required columns missing from weather data: {missing}"
            )
        missing_optional = expected_cols - set(df.columns)
        if missing_optional:
            self.logger.info(
                "Optional columns not yet present (will be added on next collection): %s",
                missing_optional,
            )

        # 2. Type conversion
        df["time"] = pd.to_datetime(df["time"])

        # 3. Null value detection
        null_pct = df.isnull().mean()
        for col in null_pct[null_pct > 0.05].index:
            self.logger.warning(
                "⚠ Column '%s': %.1f%% null values",
                col, null_pct[col] * 100,
            )

        # 4. Log validation summary
        self.logger.info(
            "Validation OK: %d rows | %d cities | %s → %s | "
            "mean T°=%.1f°C [%.1f, %.1f]",
            len(df),
            df["city"].nunique(),
            df["time"].min().strftime("%Y-%m-%d"),
            df["time"].max().strftime("%Y-%m-%d"),
            df["temperature_2m_mean"].mean(),
            df["temperature_2m_mean"].min(),
            df["temperature_2m_mean"].max(),
        )

        return df
