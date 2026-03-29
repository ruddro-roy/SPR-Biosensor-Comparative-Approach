"""Tests for Fresnel coefficient and Snell's law implementations."""

import numpy as np
import pytest

from spr.optics.fresnel import snell_cos, fresnel_rp


class TestSnellCos:
    """Tests for generalized Snell's law."""

    def test_same_medium(self):
        """cos(theta_t) = cos(theta_i) when n_i = n_t."""
        cos_i = np.array([0.5, 0.7, 0.9])
        cos_t = snell_cos(1.5, 1.5, cos_i)
        np.testing.assert_allclose(cos_t, cos_i, atol=1e-14)

    def test_normal_incidence(self):
        """At normal incidence (cos=1), refracted angle is also normal."""
        cos_t = snell_cos(1.0, 1.5, np.array([1.0]))
        np.testing.assert_allclose(cos_t, [1.0], atol=1e-14)

    def test_glass_to_air(self):
        """Snell's law from glass (n=1.5) to air (n=1.0) at 30°."""
        cos_i = np.array([np.cos(np.deg2rad(30.0))])
        cos_t = snell_cos(1.5, 1.0, cos_i)
        # n1*sin(theta1) = n2*sin(theta2)
        sin_t = np.sqrt(1 - cos_t**2)
        np.testing.assert_allclose(
            1.5 * np.sin(np.deg2rad(30.0)),
            1.0 * np.abs(sin_t),
            atol=1e-12,
        )

    def test_total_internal_reflection(self):
        """Beyond critical angle, cos(theta_t) should be purely imaginary."""
        # Critical angle for glass->air: sin(theta_c) = 1/1.5
        theta_c = np.arcsin(1.0 / 1.5)
        theta_i = theta_c + 0.1  # beyond critical angle
        cos_i = np.array([np.cos(theta_i)])
        cos_t = snell_cos(1.5, 1.0, cos_i)
        # cos_t should have significant imaginary part
        assert np.abs(np.imag(cos_t[0])) > 0.01

    def test_complex_index(self):
        """Works with complex refractive index (absorbing medium)."""
        n_metal = 0.056 + 4.278j
        cos_i = np.array([np.cos(np.deg2rad(70.0))])
        cos_t = snell_cos(1.7786, n_metal, cos_i)
        # Should return a complex number without error
        assert np.isfinite(cos_t[0])

    def test_evanescent_branch(self):
        """Imaginary part of cos(theta_t) should be >= 0."""
        # At angles beyond critical, check branch cut
        cos_i = np.cos(np.deg2rad(np.array([50.0, 60.0, 70.0, 80.0])))
        cos_t = snell_cos(1.7786, 1.0 + 0.0j, cos_i)
        assert all(np.imag(cos_t) >= -1e-15)


class TestFresnelRp:
    """Tests for p-polarization Fresnel coefficient."""

    def test_normal_incidence(self):
        """At normal incidence, r_p = (n_t - n_i)/(n_t + n_i)."""
        n_i, n_t = 1.0, 1.5
        r = fresnel_rp(n_i, n_t, np.array([1.0]), np.array([1.0]))
        expected = (n_t - n_i) / (n_t + n_i)
        np.testing.assert_allclose(np.real(r), expected, atol=1e-14)

    def test_brewster_angle(self):
        """At Brewster's angle, |r_p| ≈ 0 for real dielectrics."""
        n_i, n_t = 1.0, 1.5
        theta_B = np.arctan(n_t / n_i)
        cos_i = np.array([np.cos(theta_B)])
        cos_t = snell_cos(n_i, n_t, cos_i)
        r = fresnel_rp(n_i, n_t, cos_i, cos_t)
        assert np.abs(r[0]) < 1e-12

    def test_reflectance_bounded(self):
        """R = |r_p|^2 should be in [0, 1] for real dielectrics."""
        n_i, n_t = 1.0, 1.5
        thetas = np.deg2rad(np.linspace(0, 89, 100))
        cos_i = np.cos(thetas)
        cos_t = snell_cos(n_i, n_t, cos_i)
        r = fresnel_rp(n_i, n_t, cos_i, cos_t)
        R = np.abs(r) ** 2
        assert np.all(R >= -1e-10)
        assert np.all(R <= 1.0 + 1e-10)
