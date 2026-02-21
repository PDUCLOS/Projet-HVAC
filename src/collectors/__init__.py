# -*- coding: utf-8 -*-
"""
Data Collectors Package
========================

Data collectors for the HVAC Market Analysis project.
Each data source has its own module inheriting from BaseCollector.

Extensible architecture:
    To add a new source, create a file in this package
    with a class inheriting from BaseCollector. It will be automatically
    registered in the CollectorRegistry.

Example:
    >>> from src.collectors.base import CollectorRegistry, CollectorConfig
    >>> config = CollectorConfig.from_env()
    >>> results = CollectorRegistry.run_all(config)

Available sources:
    - weather   : Open-Meteo (historical weather data for France)
    - insee     : INSEE BDM (household confidence, business climate, IPI)
    - eurostat  : Eurostat (HVAC industrial production)
    - sitadel   : SITADEL (building permits)
    - dpe       : ADEME DPE (energy performance diagnostics)
"""

# Import all collectors to trigger auto-registration
from src.collectors.base import BaseCollector, CollectorConfig, CollectorRegistry
from src.collectors.weather import WeatherCollector
from src.collectors.insee import InseeCollector
from src.collectors.eurostat_col import EurostatCollector
from src.collectors.sitadel import SitadelCollector
from src.collectors.dpe import DpeCollector

__all__ = [
    "BaseCollector",
    "CollectorConfig",
    "CollectorRegistry",
    "WeatherCollector",
    "InseeCollector",
    "EurostatCollector",
    "SitadelCollector",
    "DpeCollector",
]
