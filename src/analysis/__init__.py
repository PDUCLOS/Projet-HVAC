# -*- coding: utf-8 -*-
"""
Analysis Package — Analyse exploratoire et études de corrélation.
=================================================================

Ce package contient les modules d'analyse (Phase 3) :

    - eda.py          : Analyse exploratoire (distributions, séries temporelles,
                        saisonnalité, comparaison géographique)
    - correlation.py  : Études de corrélation (matrice, par dept/saison,
                        multicolinéarité, lags)

Usage :
    >>> from src.analysis.eda import EDAAnalyzer
    >>> from src.analysis.correlation import CorrelationAnalyzer
"""

from src.analysis.eda import EDAAnalyzer
from src.analysis.correlation import CorrelationAnalyzer

__all__ = ["EDAAnalyzer", "CorrelationAnalyzer"]
