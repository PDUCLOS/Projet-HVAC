# -*- coding: utf-8 -*-
"""
Data Processing Package — Cleaning, merging, and feature engineering.
=====================================================================

This package contains the data processing modules (Phase 2-3):

    - clean_data.py           : Source-by-source data cleaning
    - merge_datasets.py       : Multi-source merge → ML-ready dataset
    - feature_engineering.py  : Advanced feature construction

Usage:
    >>> from src.processing.clean_data import DataCleaner
    >>> from src.processing.merge_datasets import DatasetMerger
    >>> from src.processing.feature_engineering import FeatureEngineer
"""

from src.processing.clean_data import DataCleaner
from src.processing.merge_datasets import DatasetMerger
from src.processing.feature_engineering import FeatureEngineer

__all__ = ["DataCleaner", "DatasetMerger", "FeatureEngineer"]
