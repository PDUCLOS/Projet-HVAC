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

from typing import Any, ClassVar, Dict, List

import pandas as pd

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

        for city, coords in cities.items():
            self.logger.info(
                "Weather collection: %s (lat=%.2f, lon=%.2f, dept=%s)",
                city, coords["lat"], coords["lon"], coords["dept"],
            )

            # API request parameters
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": "Europe/Paris",
            }

            try:
                # API call — retry is handled automatically by the session
                data = self.fetch_json(OPENMETEO_BASE_URL, params=params)

                # Check that the 'daily' key exists in the response
                if "daily" not in data:
                    raise ValueError(
                        f"Unexpected API response for {city}: "
                        f"'daily' key missing. Keys received: {list(data.keys())}"
                    )

                # Convert to DataFrame and enrich with metadata
                df = pd.DataFrame(data["daily"])
                df["city"] = city
                df["dept"] = coords["dept"]
                all_frames.append(df)

                self.logger.info(
                    "  ✓ %s: %d days collected (%s → %s)",
                    city, len(df),
                    df["time"].iloc[0] if len(df) > 0 else "?",
                    df["time"].iloc[-1] if len(df) > 0 else "?",
                )

            except Exception as exc:
                # Record the error but continue with other cities
                error_msg = f"Failed for {city}: {exc}"
                errors.append(error_msg)
                self.logger.error("  ✗ %s", error_msg)
                continue

            # Courtesy pause between calls
            self.rate_limit_pause()

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
