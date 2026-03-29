"""Shared test fixtures for SPR biosensor tests."""

import sys
import os

import pytest
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def wavelength():
    """He-Ne laser wavelength in meters."""
    return 633e-9


@pytest.fixture
def angle_grid():
    """Standard angular scan grid."""
    return np.arange(50.0, 90.0, 0.01)
