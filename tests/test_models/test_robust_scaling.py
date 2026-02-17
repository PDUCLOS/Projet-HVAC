# -*- coding: utf-8 -*-
"""Tests pour verifier l'utilisation de RobustScaler et HuberLoss."""

from __future__ import annotations

import pytest


class TestRobustScaler:
    """Verifie que train.py utilise RobustScaler au lieu de StandardScaler."""

    def test_imports_robust_scaler(self):
        """Le module train importe RobustScaler."""
        from src.models import train
        import inspect
        source = inspect.getsource(train)
        assert "RobustScaler" in source
        # Verify RobustScaler is used for actual scaling (not StandardScaler)
        assert "scaler = RobustScaler()" in source

    def test_robust_scaler_available(self):
        """RobustScaler est bien disponible dans sklearn."""
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        assert scaler is not None


class TestHuberLoss:
    """Verifie que deep_learning.py utilise HuberLoss au lieu de MSELoss."""

    def test_uses_huber_loss(self):
        """Le module deep_learning utilise HuberLoss."""
        from src.models import deep_learning
        import inspect
        source = inspect.getsource(deep_learning)
        assert "HuberLoss" in source
        # MSELoss ne doit plus etre le criterion principal
        assert "criterion = nn.MSELoss()" not in source
