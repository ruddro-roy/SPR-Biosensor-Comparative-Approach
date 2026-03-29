"""Tests for Transfer Matrix Method implementation.

Validated against the Byrnes tmm package (arXiv:1603.02720).
"""

import numpy as np
import pytest

from spr.optics.tmm import transfer_matrix_reflectance, angular_scan


class TestTMMBasic:
    """Basic correctness tests for the TMM engine."""

    def test_no_film_normal_incidence(self):
        """Two semi-infinite real media at normal incidence: Fresnel R."""
        R = transfer_matrix_reflectance([1.5, 1.0], [0.0, 0.0], [0.0], 633e-9)
        expected = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        np.testing.assert_allclose(R[0], expected, atol=1e-12)

    def test_no_film_oblique(self):
        """Two semi-infinite media at 30° and 60°: R in [0, 1]."""
        R = transfer_matrix_reflectance([1.5, 1.0], [0.0, 0.0], [30.0, 60.0], 633e-9)
        assert np.all(R >= 0)
        assert np.all(R <= 1.0 + 1e-12)

    def test_brewster_angle_air_glass(self):
        """At Brewster angle for air->glass, R_p ≈ 0."""
        theta_B = np.degrees(np.arctan(1.5 / 1.0))
        R = transfer_matrix_reflectance([1.0, 1.5], [0.0, 0.0], [theta_B], 633e-9)
        assert R[0] < 1e-10

    def test_total_internal_reflection(self):
        """Beyond critical angle, R = 1 for lossless media."""
        theta_c = np.degrees(np.arcsin(1.0 / 1.5))
        R = transfer_matrix_reflectance(
            [1.5, 1.0], [0.0, 0.0], [theta_c + 5.0], 633e-9
        )
        np.testing.assert_allclose(R[0], 1.0, atol=1e-10)

    def test_reflectance_bounded_simple(self):
        """Reflectance in [0, 1] for simple dielectric stack."""
        n_list = [1.5, 1.3, 1.0]
        d_list = [0.0, 100e-9, 0.0]
        angles = np.linspace(0, 89, 200)
        R = transfer_matrix_reflectance(n_list, d_list, angles, 633e-9)
        assert np.all(R >= -1e-10)
        assert np.all(R <= 1.0 + 1e-10)

    def test_reflectance_bounded_metal(self):
        """Reflectance in [0, 1] for Kretschmann SPR stack."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_list = [1.7786, n_Ag, 1.34]
        d_list = [0.0, 40e-9, 0.0]
        angles = np.linspace(40, 89, 500)
        R = transfer_matrix_reflectance(n_list, d_list, angles, 633e-9)
        assert np.all(R >= -1e-10), f"R<0: min={R.min()}"
        assert np.all(R <= 1.0 + 1e-10), f"R>1: max={R.max()}"

    def test_spr_dip_exists(self):
        """A clear SPR dip should exist for Kretschmann prism/Ag/dielectric."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_list = [1.7786, n_Ag, 1.34]
        d_list = [0.0, 40e-9, 0.0]
        angles = np.linspace(40, 89, 5000)
        R = transfer_matrix_reflectance(n_list, d_list, angles, 633e-9)
        # SPR dip should have R_min well below 0.5
        assert R.min() < 0.5, f"No SPR dip: min R = {R.min()}"
        idx_min = np.argmin(R)
        theta_min = angles[idx_min]
        # With SF11/Ag(-18.295)/n=1.34, SPR is near 52-53°
        assert 45 < theta_min < 60, f"SPR at {theta_min}°, expected 45-60°"

    def test_smoothness(self):
        """R should vary smoothly (no discontinuities)."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_list = [1.7786, n_Ag, 1.34]
        d_list = [0.0, 40e-9, 0.0]
        angles = np.linspace(40, 89, 1000)
        R = transfer_matrix_reflectance(n_list, d_list, angles, 633e-9)
        diffs = np.abs(np.diff(R))
        assert np.all(diffs < 0.05), f"Discontinuity: max diff={diffs.max()}"


class TestTMMLayers:
    """Tests for multilayer stacks."""

    def test_graphene_shifts_spr(self):
        """Adding graphene should shift SPR angle to higher values."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_gr = 3.0 + 1.1487j
        angles = np.linspace(40, 89, 5000)

        R0 = transfer_matrix_reflectance(
            [1.7786, n_Ag, 1.34], [0.0, 40e-9, 0.0], angles, 633e-9
        )
        R1 = transfer_matrix_reflectance(
            [1.7786, n_Ag, n_gr, 1.34], [0.0, 40e-9, 0.34e-9, 0.0], angles, 633e-9
        )
        theta0 = angles[np.argmin(R0)]
        theta1 = angles[np.argmin(R1)]
        assert theta1 > theta0, f"Graphene shift: {theta0:.2f} -> {theta1:.2f}"

    def test_mos2_further_shifts(self):
        """Adding MoS2 should shift SPR angle even more."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_mos2 = 5.9 + 0.8j
        n_gr = 3.0 + 1.1487j
        angles = np.linspace(40, 89, 5000)

        Rg = transfer_matrix_reflectance(
            [1.7786, n_Ag, n_gr, 1.34],
            [0.0, 40e-9, 0.34e-9, 0.0], angles, 633e-9,
        )
        Rmg = transfer_matrix_reflectance(
            [1.7786, n_Ag, n_mos2, n_gr, 1.34],
            [0.0, 40e-9, 0.65e-9, 0.34e-9, 0.0], angles, 633e-9,
        )
        theta_g = angles[np.argmin(Rg)]
        theta_mg = angles[np.argmin(Rmg)]
        assert theta_mg > theta_g

    def test_increasing_sensing_ri_shifts_spr(self):
        """Higher sensing RI -> higher SPR angle."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        angles = np.linspace(40, 89, 5000)

        thetas = []
        for n_s in [1.33, 1.34, 1.35, 1.36]:
            R = transfer_matrix_reflectance(
                [1.7786, n_Ag, n_s], [0.0, 40e-9, 0.0], angles, 633e-9
            )
            thetas.append(angles[np.argmin(R)])

        for i in range(len(thetas) - 1):
            assert thetas[i + 1] > thetas[i], f"Monotonicity: {thetas}"

    def test_reflectance_bounded_multilayer(self):
        """R in [0,1] for full MoS2+graphene stack."""
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_mos2 = 5.9 + 0.8j
        n_gr = 3.0 + 1.1487j
        angles = np.linspace(40, 89, 500)
        R = transfer_matrix_reflectance(
            [1.7786, n_Ag, n_mos2, n_gr, 1.34],
            [0.0, 40e-9, 0.65e-9, 0.34e-9, 0.0], angles, 633e-9,
        )
        assert np.all(R >= -1e-10)
        assert np.all(R <= 1.0 + 1e-10)


class TestTMMValidation:
    """Cross-validation against Byrnes tmm package."""

    @pytest.fixture
    def tmm_available(self):
        try:
            import tmm
            return True
        except ImportError:
            pytest.skip("tmm package not installed")

    def test_two_layer_matches_tmm(self, tmm_available):
        """Our 2-layer result matches Byrnes tmm."""
        import tmm as tmm_ref
        angles = [0.0, 30.0, 45.0, 60.0]
        R_ours = transfer_matrix_reflectance([1.5, 1.0], [0, 0], angles, 633e-9)
        for i, a in enumerate(angles):
            res = tmm_ref.coh_tmm('p', [1.5, 1.0], [np.inf, np.inf],
                                  a * np.pi / 180, 633)
            np.testing.assert_allclose(R_ours[i], res['R'], atol=1e-12)

    def test_kretschmann_matches_tmm(self, tmm_available):
        """Our Kretschmann result matches Byrnes tmm."""
        import tmm as tmm_ref
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        angles = [45, 50, 52, 53, 55, 60, 70, 80]
        R_ours = transfer_matrix_reflectance(
            [1.7786, n_Ag, 1.34], [0, 40e-9, 0], angles, 633e-9
        )
        for i, a in enumerate(angles):
            res = tmm_ref.coh_tmm('p', [1.7786, n_Ag, 1.34],
                                  [np.inf, 40, np.inf], a * np.pi / 180, 633)
            np.testing.assert_allclose(R_ours[i], res['R'], atol=1e-12,
                                       err_msg=f"Mismatch at {a}°")

    def test_multilayer_matches_tmm(self, tmm_available):
        """Our MoS2+graphene result matches Byrnes tmm."""
        import tmm as tmm_ref
        n_Ag = np.sqrt(complex(-18.295, 0.481))
        n_mos2 = 5.9 + 0.8j
        n_gr = 3.0 + 1.1487j
        n_list = [1.7786, n_Ag, n_mos2, n_gr, 1.34]
        d_list_m = [0, 40e-9, 0.65e-9, 0.34e-9, 0]
        d_list_nm = [np.inf, 40, 0.65, 0.34, np.inf]

        angles = [45, 50, 53, 55, 60, 70, 80]
        R_ours = transfer_matrix_reflectance(n_list, d_list_m, angles, 633e-9)
        for i, a in enumerate(angles):
            res = tmm_ref.coh_tmm('p', n_list, d_list_nm,
                                  a * np.pi / 180, 633)
            np.testing.assert_allclose(R_ours[i], res['R'], atol=1e-12,
                                       err_msg=f"Mismatch at {a}°")


class TestAngularScan:
    """Tests for the angular_scan convenience function."""

    def test_returns_correct_shape(self):
        angles, R = angular_scan([1.5, 1.0], [0.0, 0.0], 633e-9, 50, 80, 0.1)
        assert len(angles) == len(R)
        assert len(angles) > 0

    def test_step_size(self):
        angles, R = angular_scan([1.5, 1.0], [0.0, 0.0], 633e-9, 50, 51, 0.1)
        np.testing.assert_allclose(np.diff(angles), 0.1, atol=1e-10)
