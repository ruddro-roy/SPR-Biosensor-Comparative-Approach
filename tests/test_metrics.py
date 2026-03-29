"""Tests for SPR analysis metrics (resonance finding, FWHM, sensitivity, FOM)."""

import numpy as np
import pytest

from spr.analysis.metrics import (
    find_resonance,
    compute_fwhm,
    compute_sensitivity_deg_per_RIU,
    compute_fom,
    analyze_curve,
)


def _make_lorentzian_dip(theta_0=75.0, width=2.0, depth=0.8, n=5000):
    """Create a synthetic Lorentzian SPR-like reflectance curve."""
    angles = np.linspace(60, 90, n)
    R = 1.0 - depth / (1.0 + ((angles - theta_0) / (width / 2)) ** 2)
    return angles, R


class TestFindResonance:
    """Tests for resonance angle finding with refinement."""

    def test_synthetic_exact(self):
        """Finds exact minimum of a symmetric Lorentzian."""
        angles, R = _make_lorentzian_dip(theta_0=75.0, width=3.0, depth=0.9)
        theta_spr, r_min = find_resonance(angles, R)
        assert abs(theta_spr - 75.0) < 0.02
        assert abs(r_min - 0.1) < 0.02

    def test_asymmetric_dip(self):
        """Handles slightly asymmetric dip."""
        angles = np.linspace(60, 90, 5000)
        # Asymmetric dip
        R = 1.0 - 0.8 / (1.0 + ((angles - 76.0) / 1.5) ** 2) * (
            1.0 + 0.1 * (angles - 76.0)
        )
        R = np.clip(R, 0, 1)
        theta_spr, r_min = find_resonance(angles, R)
        # Should be near 76 but slightly shifted due to asymmetry
        assert 74 < theta_spr < 78

    def test_returns_finite(self):
        """Always returns finite values for valid input."""
        angles, R = _make_lorentzian_dip()
        theta, r = find_resonance(angles, R)
        assert np.isfinite(theta)
        assert np.isfinite(r)

    def test_refinement_better_than_discrete(self):
        """Refined result is at least as good as discrete argmin."""
        angles, R = _make_lorentzian_dip(theta_0=75.123)
        theta_refined, _ = find_resonance(angles, R)
        theta_discrete = angles[np.argmin(R)]
        # Refined should be closer to true minimum
        assert abs(theta_refined - 75.123) <= abs(theta_discrete - 75.123) + 0.001


class TestFWHM:
    """Tests for FWHM computation."""

    def test_lorentzian_fwhm(self):
        """FWHM of a Lorentzian with known width."""
        angles, R = _make_lorentzian_dip(theta_0=75.0, width=3.0, depth=0.8)
        fwhm = compute_fwhm(angles, R)
        # For this inverted Lorentzian, FWHM should be ~3 degrees
        # (The half-maximum definition uses baseline, so not exactly 3.0)
        assert 2.0 < fwhm < 5.0

    def test_narrow_dip(self):
        """Narrow dip gives smaller FWHM."""
        _, R_wide = _make_lorentzian_dip(width=5.0)
        _, R_narrow = _make_lorentzian_dip(width=1.0)
        angles = np.linspace(60, 90, 5000)
        # Regenerate with consistent angles
        R_wide = 1.0 - 0.8 / (1.0 + ((angles - 75) / 2.5) ** 2)
        R_narrow = 1.0 - 0.8 / (1.0 + ((angles - 75) / 0.5) ** 2)
        fwhm_wide = compute_fwhm(angles, R_wide)
        fwhm_narrow = compute_fwhm(angles, R_narrow)
        assert fwhm_narrow < fwhm_wide


class TestSensitivity:
    """Tests for sensitivity and FOM calculations."""

    def test_basic_sensitivity(self):
        """S = delta_theta / delta_n."""
        s = compute_sensitivity_deg_per_RIU(2.0, 0.01)
        assert s == pytest.approx(200.0)

    def test_zero_delta_n(self):
        """Zero RI change gives zero sensitivity."""
        s = compute_sensitivity_deg_per_RIU(1.0, 0.0)
        assert s == 0.0

    def test_fom_basic(self):
        """FOM = sensitivity / FWHM."""
        fom = compute_fom(100.0, 5.0)
        assert fom == pytest.approx(20.0)

    def test_fom_nan_fwhm(self):
        """FOM is NaN when FWHM is NaN."""
        fom = compute_fom(100.0, np.nan)
        assert np.isnan(fom)

    def test_fom_zero_fwhm(self):
        """FOM is NaN when FWHM is zero."""
        fom = compute_fom(100.0, 0.0)
        assert np.isnan(fom)


class TestAnalyzeCurve:
    """Integration test for full curve analysis."""

    def test_synthetic(self):
        """Full analysis of synthetic Lorentzian."""
        angles, R = _make_lorentzian_dip(theta_0=75.5, width=3.0, depth=0.85)
        result = analyze_curve(angles, R)
        assert abs(result.theta_spr - 75.5) < 0.05
        assert result.r_min < 0.2
        assert result.fwhm > 0
