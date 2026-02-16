# -*- coding: utf-8 -*-
"""
Data Collectors Package
========================

Collecteurs de données pour le projet HVAC Market Analysis.
Chaque source de données a son propre module héritant de BaseCollector.

Architecture extensible :
    Pour ajouter une nouvelle source, créer un fichier dans ce package
    avec une classe héritant de BaseCollector. Elle sera automatiquement
    enregistrée dans le CollectorRegistry.

Exemple:
    >>> from src.collectors.base import CollectorRegistry, CollectorConfig
    >>> config = CollectorConfig.from_env()
    >>> results = CollectorRegistry.run_all(config)

Sources disponibles:
    - weather   : Open-Meteo (météo historique AURA)
    - insee     : INSEE BDM (confiance ménages, climat affaires, IPI)
    - eurostat  : Eurostat (production industrielle HVAC)
    - sitadel   : SITADEL (permis de construire)
    - dpe       : ADEME DPE (diagnostics énergétiques)
"""

# Importer tous les collecteurs pour déclencher l'auto-enregistrement
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
