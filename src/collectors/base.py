# -*- coding: utf-8 -*-
"""
BaseCollector & CollectorRegistry — Foundation of the data collection system.
=============================================================================

This module defines the extensible architecture of the data collection system.
Two main components:

1. **BaseCollector** (abstract class):
   Provides the common skeleton for all collectors: HTTP management with
   automatic retry, structured logging, data validation, and
   standardized saving. Each data source only needs to implement
   `collect()` and `validate()`.

2. **CollectorRegistry** (plugin system):
   Automatically registers each concrete subclass of BaseCollector
   via the `__init_subclass__` hook. Allows discovering, listing, and
   running collectors without manual configuration.

Architecture:
    BaseCollector (ABC)
    ├── collect()     → Retrieves raw data (to be implemented)
    ├── validate()    → Checks data quality (to be implemented)
    ├── save()        → Persists to disk (CSV by default, overridable)
    ├── run()         → Orchestrates the full collect→validate→save cycle
    ├── fetch_json()  → HTTP GET with retry + JSON parsing
    ├── fetch_xml()   → HTTP GET with retry + XML parsing
    └── fetch_bytes() → HTTP GET with retry + binary content (ZIP, etc.)

Extensibility:
    To add a new data source:
    1. Create a file in src/collectors/ (e.g., src/collectors/my_source.py)
    2. Define a class inheriting from BaseCollector with source_name
    3. Implement collect() and validate()
    4. That's it! The source is auto-registered and available in the registry.

    Minimal example:
        class MySourceCollector(BaseCollector):
            source_name = "my_source"

            def collect(self) -> pd.DataFrame:
                data = self.fetch_json("https://api.example.com/data")
                return pd.DataFrame(data)

            def validate(self, df: pd.DataFrame) -> pd.DataFrame:
                assert "id" in df.columns, "Column 'id' missing"
                return df

Usage:
    >>> from src.collectors.base import CollectorRegistry, CollectorConfig
    >>> config = CollectorConfig.from_env()
    >>> # Run a single collector
    >>> result = CollectorRegistry.run("weather", config)
    >>> print(f"{result.name}: {result.rows_collected} rows")
    >>> # Run all collectors
    >>> results = CollectorRegistry.run_all(config)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type

import pandas as pd
import requests
from lxml import etree
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# Collection status enumeration
# =============================================================================

class CollectorStatus(Enum):
    """Possible states of a collector execution.

    Values:
        SUCCESS: Complete collection, all data retrieved.
        PARTIAL: Partial collection (some items failed).
        FAILED: Total collection failure.
        SKIPPED: Collector skipped (data already up to date, etc.).
    """
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# Collector configuration
# =============================================================================

@dataclass(frozen=True)
class CollectorConfig:
    """Shared configuration for all collectors.

    Can be built manually (tests) or from environment variables
    via `from_env()`.

    Attributes:
        raw_data_dir: Root directory for raw data.
        processed_data_dir: Root directory for structured data.
        start_date: Collection start date (ISO YYYY-MM-DD).
        end_date: Collection end date.
        departments: List of target department codes.
        region_code: INSEE region code ("FR" = metropolitan France).
        request_timeout: HTTP timeout in seconds.
        max_retries: Maximum number of retries on transient errors.
        retry_backoff_factor: Exponential factor between retries.
        rate_limit_delay: Minimum pause between API calls (seconds).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    start_date: str = "2019-01-01"
    end_date: str = "2026-02-28"
    departments: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "14", "15", "16", "17", "18", "19",
        "21", "22", "23", "24", "25", "26", "27", "28", "29",
        "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
        "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
        "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
        "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
        "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
        "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
        "90", "91", "92", "93", "94", "95", "2A", "2B",
    ])
    region_code: str = "FR"
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    rate_limit_delay: float = 1.5
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> CollectorConfig:
        """Build configuration from environment variables.

        Loads the .env file via python-dotenv then maps variables
        to attributes. Default values are used if absent.

        Returns:
            CollectorConfig initialized from the environment.
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()

        return cls(
            raw_data_dir=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
            processed_data_dir=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
            start_date=os.getenv("DATA_START_DATE", "2019-01-01"),
            end_date=os.getenv("DATA_END_DATE", "2026-02-28"),
            departments=os.getenv(
                "TARGET_DEPARTMENTS", ""
            ).split(",") if os.getenv("TARGET_DEPARTMENTS") else [
                "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                "11", "12", "13", "14", "15", "16", "17", "18", "19",
                "21", "22", "23", "24", "25", "26", "27", "28", "29",
                "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
                "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
                "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
                "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
                "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
                "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
                "90", "91", "92", "93", "94", "95", "2A", "2B",
            ],
            region_code=os.getenv("TARGET_REGION", "FR"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff_factor=float(os.getenv("RETRY_BACKOFF", "1.0")),
            rate_limit_delay=float(os.getenv("RATE_LIMIT_DELAY", "1.5")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


# =============================================================================
# Collection result
# =============================================================================

@dataclass
class CollectorResult:
    """Result of a collector execution.

    Provides a structured report of what happened: number of rows
    collected, save path, duration, errors encountered.

    Attributes:
        name: Name of the collected source.
        status: Final status (SUCCESS, PARTIAL, FAILED, SKIPPED).
        rows_collected: Number of rows in the final DataFrame.
        output_path: Path of the saved file (if applicable).
        started_at: Collection start timestamp.
        finished_at: Collection end timestamp.
        errors: List of error messages encountered.
    """
    name: str
    status: CollectorStatus
    rows_collected: int = 0
    output_path: Optional[Path] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Total collection duration in seconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def __str__(self) -> str:
        """Human-readable summary of the result."""
        duration = f"{self.duration_seconds:.1f}s" if self.duration_seconds else "N/A"
        return (
            f"[{self.status.value.upper():>7}] {self.name:<15} "
            f"| {self.rows_collected:>6} rows | {duration}"
        )


# =============================================================================
# Abstract class BaseCollector
# =============================================================================

class BaseCollector(ABC):
    """Abstract base class for all data collectors.

    Provides:
    - HTTP session with automatic retry (429, 500, 502, 503, 504)
    - Helpers `fetch_json()`, `fetch_xml()`, `fetch_bytes()`
    - Structured logging with timestamps
    - Lifecycle orchestrated by `run()`: collect -> validate -> save
    - Auto-registration in the CollectorRegistry

    Subclasses MUST implement:
        - `source_name` (ClassVar[str]): unique identifier for the source
        - `collect()`: data retrieval logic -> DataFrame
        - `validate(df)`: quality checks -> cleaned DataFrame

    Subclasses MAY override:
        - `output_subdir`: output subdirectory (default = source_name)
        - `output_filename`: file name (default = "{source_name}.csv")
        - `save(df, path)`: custom persistence logic (Parquet, etc.)
    """

    # --- Class contract (to be defined by subclasses) ---------------------

    source_name: ClassVar[str]                       # Unique identifier
    output_subdir: ClassVar[Optional[str]] = None    # Output subdirectory
    output_filename: ClassVar[Optional[str]] = None  # Output file name

    # --- Auto-registration hook -------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically registers each concrete subclass.

        This Python hook is called every time a class inherits from
        BaseCollector. If the class is concrete (no unimplemented abstract
        methods), it is added to the CollectorRegistry.
        """
        super().__init_subclass__(**kwargs)
        # Only register concrete classes (not intermediate ABCs)
        if hasattr(cls, "source_name") and not getattr(cls, "__abstractmethods__", None):
            CollectorRegistry._register(cls)

    # --- Initialization ---------------------------------------------------

    def __init__(self, config: CollectorConfig) -> None:
        """Initialize the collector with its configuration.

        Args:
            config: Shared configuration (paths, dates, network, etc.).
        """
        self.config = config
        self.logger = logging.getLogger(f"collectors.{self.source_name}")
        self._setup_logging()
        self._session: Optional[requests.Session] = None

    # --- Main entry point -------------------------------------------------

    def run(self) -> CollectorResult:
        """Execute the full collection cycle.

        Orchestrates the steps:
        1. Log startup with parameters
        2. `collect()` — retrieve raw data
        3. `validate()` — check data quality
        4. `save()` — persist to disk
        5. Log completion with summary

        On exception, the status becomes FAILED and the error is logged.
        The collection never abruptly terminates (always returns a result).

        Returns:
            CollectorResult with the complete collection summary.
        """
        result = CollectorResult(
            name=self.source_name,
            status=CollectorStatus.FAILED,
            started_at=datetime.now(),
        )

        self.logger.info(
            "="*60 + "\n"
            "  COLLECTION : %s\n"
            "  Period     : %s → %s\n"
            "  Departments : %s\n"
            + "="*60,
            self.source_name.upper(),
            self.config.start_date,
            self.config.end_date,
            ", ".join(self.config.departments),
        )

        try:
            # Step 1: Collect raw data
            self.logger.info("Step 1/3 — Collecting data...")
            df = self.collect()

            if df.empty:
                self.logger.warning("⚠ Collection returned an empty DataFrame.")
                result.status = CollectorStatus.PARTIAL
                result.errors.append("Empty DataFrame returned by collect().")
            else:
                # Step 2: Validation and light cleaning
                self.logger.info("Step 2/3 — Validating data...")
                df = self.validate(df)

                # Step 3: Save to disk
                self.logger.info("Step 3/3 — Saving...")
                output_path = self._resolve_output_path()
                self.save(df, output_path)

                result.rows_collected = len(df)
                result.output_path = output_path
                result.status = CollectorStatus.SUCCESS

                self.logger.info(
                    "✓ Collection successful: %d rows saved → %s",
                    len(df), output_path,
                )

        except Exception as exc:
            self.logger.exception(
                "✗ Collection FAILED '%s': %s", self.source_name, exc
            )
            result.status = CollectorStatus.FAILED
            result.errors.append(str(exc))

        result.finished_at = datetime.now()
        self.logger.info(
            "Finished '%s' in %.1fs — status: %s",
            self.source_name,
            result.duration_seconds or 0,
            result.status.value,
        )
        return result

    # --- Abstract methods (subclass contract) -----------------------------

    @abstractmethod
    def collect(self) -> pd.DataFrame:
        """Retrieve data from the external source.

        This method contains all the source-specific logic:
        API calls, file downloads, parsing, etc.

        Returns:
            Pandas DataFrame with the collected raw data.

        Raises:
            requests.RequestException: On network errors.
            ValueError: If the data format is unexpected.
        """
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and lightly clean the collected DataFrame.

        Expected checks:
        - Presence of required columns
        - Correct data types
        - No critical null values
        - Row count within expected bounds

        Args:
            df: Raw DataFrame from `collect()`.

        Returns:
            Validated (and possibly cleaned) DataFrame.

        Raises:
            ValueError: If validation fails critically.
        """
        ...

    # --- Save (overridable) -----------------------------------------------

    def save(self, df: pd.DataFrame, path: Path) -> None:
        """Persist the DataFrame to disk.

        Default implementation: saves as CSV.
        Subclasses can override to use Parquet, etc.

        Args:
            df: Validated DataFrame to save.
            path: Full path of the output file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

        structured_path = self._resolve_output_path(self.config.processed_data_dir)
        structured_path.parent.mkdir(parents=True, exist_ok=True)
        if structured_path.exists():
            existing = pd.read_csv(structured_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(keep="last")
            combined.to_csv(structured_path, index=False)
        else:
            df.drop_duplicates(keep="last").to_csv(structured_path, index=False)
        self.logger.debug("Saved %d rows → %s", len(df), path)

    # --- HTTP helpers (shared by all collectors) --------------------------

    @property
    def session(self) -> requests.Session:
        """HTTP session with automatic retry (lazily initialized).

        Configures automatic retries on transient HTTP status codes:
        - 429: Too Many Requests (rate limiting)
        - 500, 502, 503, 504: Server errors

        The delay between retries follows an exponential progression:
        retry_backoff_factor * (2 ** (retry_number - 1))

        Returns:
            Reusable requests session with built-in retry.
        """
        if self._session is None:
            self._session = requests.Session()
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("https://", adapter)
        return self._session

    def fetch_json(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """HTTP GET -> JSON with comprehensive error handling.

        Args:
            url: Endpoint URL.
            params: Query string parameters (optional).

        Returns:
            Python object parsed from JSON (dict, list, etc.).

        Raises:
            requests.HTTPError: On 4xx/5xx status codes after retries.
            ValueError: If the response body is not valid JSON.
        """
        self.logger.debug("GET %s | params=%s", url, params)
        response = self.session.get(
            url, params=params, timeout=self.config.request_timeout
        )
        response.raise_for_status()
        return response.json()

    def fetch_xml(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> etree._Element:
        """HTTP GET -> parsed XML with error handling.

        Primarily used for the INSEE SDMX API.

        Args:
            url: Endpoint URL.
            params: Query string parameters (optional).

        Returns:
            Root element of the parsed XML tree (lxml).

        Raises:
            requests.HTTPError: On 4xx/5xx status codes.
            etree.XMLSyntaxError: If the content is not valid XML.
        """
        self.logger.debug("GET (XML) %s | params=%s", url, params)
        response = self.session.get(
            url, params=params, timeout=self.config.request_timeout
        )
        response.raise_for_status()
        return etree.fromstring(response.content)

    def fetch_bytes(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """HTTP GET -> raw binary content (ZIP, large files).

        Args:
            url: Endpoint URL.
            params: Query string parameters (optional).

        Returns:
            Raw response content as bytes.

        Raises:
            requests.HTTPError: On 4xx/5xx status codes.
        """
        self.logger.debug("GET (bytes) %s", url)
        response = self.session.get(
            url, params=params, timeout=self.config.request_timeout
        )
        response.raise_for_status()
        return response.content

    def rate_limit_pause(self) -> None:
        """Courtesy pause between two API calls.

        Respects the delay configured in `rate_limit_delay`.
        Prevents overloading free APIs.
        """
        if self.config.rate_limit_delay > 0:
            time.sleep(self.config.rate_limit_delay)

    # --- Private methods --------------------------------------------------

    def _resolve_output_path(self, base_dir: Optional[Path] = None) -> Path:
        """Compute the full output file path.

        Uses class overrides (output_subdir, output_filename)
        or default values based on source_name.

        Returns:
            Absolute or relative path of the output file.
        """
        subdir = self.output_subdir or self.source_name
        filename = self.output_filename or f"{self.source_name}.csv"
        root = base_dir or self.config.raw_data_dir
        return root / subdir / filename

    def _setup_logging(self) -> None:
        """Configure the logger for this collector instance.

        Format: YYYY-MM-DD HH:MM:SS | collectors.source_name | LEVEL | message
        """
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        # Avoid duplicate handlers if the module is reloaded
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        # Prevent duplicate output: root logger (via basicConfig in pipeline)
        # already has a handler, so disable propagation for this logger.
        self.logger.propagate = False


# =============================================================================
# Collector registry (plugin system)
# =============================================================================

class CollectorRegistry:
    """Central registry of all available collectors.

    Collectors are automatically registered via the `__init_subclass__`
    hook of BaseCollector. Simply importing the module containing a
    collector makes it available.

    This "plugin" pattern allows adding new sources without
    modifying any existing file -- just create a new module.

    Usage:
        >>> CollectorRegistry.available()
        ['dpe', 'eurostat', 'insee', 'sitadel', 'weather']

        >>> result = CollectorRegistry.run("weather", config)
        >>> print(result)
        [SUCCESS] weather | 12345 rows | 5.2s

        >>> results = CollectorRegistry.run_all(config)
    """

    # Storage of collector classes (not instances)
    _collectors: ClassVar[Dict[str, Type[BaseCollector]]] = {}

    @classmethod
    def _register(cls, collector_class: Type[BaseCollector]) -> None:
        """Register a collector class (called automatically).

        Args:
            collector_class: Concrete subclass of BaseCollector.
        """
        name = collector_class.source_name
        if name in cls._collectors:
            logging.getLogger("collectors.registry").warning(
                "Overwriting existing collector '%s' with %s",
                name, collector_class.__name__,
            )
        cls._collectors[name] = collector_class
        logging.getLogger("collectors.registry").debug(
            "Collector '%s' registered (%s)", name, collector_class.__name__
        )

    @classmethod
    def available(cls) -> List[str]:
        """List the names of all registered collectors.

        Returns:
            Sorted list of available source_name values.
        """
        return sorted(cls._collectors.keys())

    @classmethod
    def get(cls, name: str) -> Type[BaseCollector]:
        """Retrieve a collector class by its name.

        Args:
            name: The source_name of the desired collector.

        Returns:
            The collector class (not an instance).

        Raises:
            KeyError: If no collector is registered with this name.
        """
        if name not in cls._collectors:
            raise KeyError(
                f"Unknown collector: '{name}'. "
                f"Available: {cls.available()}"
            )
        return cls._collectors[name]

    @classmethod
    def run(cls, name: str, config: CollectorConfig) -> CollectorResult:
        """Instantiate and execute a collector by its name.

        Args:
            name: The source_name of the collector.
            config: Shared configuration.

        Returns:
            CollectorResult with the collection summary.
        """
        collector_class = cls.get(name)
        collector = collector_class(config)
        return collector.run()

    @classmethod
    def run_all(
        cls,
        config: CollectorConfig,
        order: Optional[List[str]] = None,
    ) -> List[CollectorResult]:
        """Execute multiple collectors in sequence.

        Args:
            config: Shared configuration.
            order: Explicit execution order. If None, runs
                   all collectors in alphabetical order.

        Returns:
            List of CollectorResult, one per collector.
        """
        logger = logging.getLogger("collectors.registry")
        names = order or cls.available()
        results: List[CollectorResult] = []

        logger.info(
            "Launching %d collectors: %s", len(names), names
        )

        for name in names:
            try:
                result = cls.run(name, config)
            except KeyError as exc:
                logger.error("Collector '%s' not found: %s", name, exc)
                result = CollectorResult(
                    name=name,
                    status=CollectorStatus.FAILED,
                    errors=[str(exc)],
                )
            results.append(result)

            # Log progress if a collector fails
            if result.status == CollectorStatus.FAILED:
                logger.warning(
                    "⚠ Collector '%s' FAILED — moving to next.", name
                )

        # Final summary
        succeeded = sum(1 for r in results if r.status == CollectorStatus.SUCCESS)
        logger.info(
            "\n" + "="*60 + "\n"
            "  COLLECTION SUMMARY: %d/%d succeeded\n" +
            "="*60,
            succeeded, len(results),
        )
        for result in results:
            logger.info("  %s", result)

        return results
