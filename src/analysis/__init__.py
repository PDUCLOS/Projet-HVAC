# -*- coding: utf-8 -*-
"""
Analysis Package — Exploratory analysis and correlation studies.
================================================================

This package contains the analysis modules (Phase 3):

    - eda.py          : Exploratory data analysis (distributions, time series,
                        seasonality, geographic comparison)
    - correlation.py  : Correlation studies (matrix, by dept/season,
                        multicollinearity, lags)

Usage:
    >>> from src.analysis.eda import EDAAnalyzer
    >>> from src.analysis.correlation import CorrelationAnalyzer
"""

from src.analysis.eda import EDAAnalyzer
from src.analysis.correlation import CorrelationAnalyzer

__all__ = ["EDAAnalyzer", "CorrelationAnalyzer"]
