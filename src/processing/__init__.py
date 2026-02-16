# -*- coding: utf-8 -*-
"""
Data Processing Package — Nettoyage, fusion et feature engineering.
===================================================================

Ce package contient les modules de traitement des données (Phase 2-3) :

    - clean_data.py           : Nettoyage source par source
    - merge_datasets.py       : Fusion multi-sources → dataset ML-ready
    - feature_engineering.py  : Construction des features avancées

Usage :
    >>> from src.processing.clean_data import DataCleaner
    >>> from src.processing.merge_datasets import DatasetMerger
    >>> from src.processing.feature_engineering import FeatureEngineer
"""

from src.processing.clean_data import DataCleaner
from src.processing.merge_datasets import DatasetMerger
from src.processing.feature_engineering import FeatureEngineer

__all__ = ["DataCleaner", "DatasetMerger", "FeatureEngineer"]
