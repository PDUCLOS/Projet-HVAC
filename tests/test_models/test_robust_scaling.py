# -*- coding: utf-8 -*-
"""Tests to verify the use of RobustScaler and HuberLoss."""

from __future__ import annotations

import pytest


class TestRobustScaler:
    """Verify that train.py uses RobustScaler instead of StandardScaler."""

    def test_imports_robust_scaler(self):
        """The train module imports RobustScaler."""
        from src.models import train
        import inspect
        source = inspect.getsource(train)
        assert "RobustScaler" in source
        # Verify RobustScaler is used for actual scaling (not StandardScaler)
        assert "scaler = RobustScaler()" in source

    def test_robust_scaler_available(self):
        """RobustScaler is available in sklearn."""
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        assert scaler is not None


class TestHuberLoss:
    """Verify that deep_learning.py uses HuberLoss instead of MSELoss."""

    def test_uses_huber_loss(self):
        """The deep_learning module uses HuberLoss."""
        from src.models import deep_learning
        import inspect
        source = inspect.getsource(deep_learning)
        assert "HuberLoss" in source
        # MSELoss should no longer be the main criterion
        assert "criterion = nn.MSELoss()" not in source
