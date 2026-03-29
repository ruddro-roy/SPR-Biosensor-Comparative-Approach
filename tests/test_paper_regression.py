"""Regression tests for SPR simulation physics.

These tests verify that:
1. The TMM produces physically correct SPR behavior
2. Qualitative trends from the paper are preserved
3. Key physical invariants hold

NOTE: The paper's reported SPR angles (74-77°) are NOT reproducible
with the stated material constants using a correct TMM. Our validated
implementation finds SPR at ~52-54°. The tests verify correct physics,
not exact paper number matching.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spr.optics.tmm import transfer_matrix_reflectance, angular_scan
from spr.models.sensor import (
    build_layer_stack,
    sensing_medium_ri_reproduction,
    WAVELENGTH_M,
)
from spr.analysis.metrics import find_resonance, analyze_curve


SCAN = (40.0, 89.0, 0.01)


def _run(n_mos2, n_graphene, n_sensing):
    stack = build_layer_stack(n_mos2, n_graphene, n_sensing)
    angles, R = angular_scan(
        stack.n_list, stack.d_list, WAVELENGTH_M,
        SCAN[0], SCAN[1], SCAN[2],
    )
    return angles, R


class TestQualitativeTrends:
    """Verify the paper's qualitative findings hold with correct TMM."""

    def test_spr_ordering(self):
        """θ_conv < θ_graphene < θ_mos2_graphene."""
        thetas = []
        for m, l in [(0, 0), (0, 1), (1, 1)]:
            angles, R = _run(m, l, 1.34)
            theta, _ = find_resonance(angles, R)
            thetas.append(theta)
        assert thetas[0] < thetas[1] < thetas[2], f"Wrong ordering: {thetas}"

    def test_graphene_shifts_spr_up(self):
        """Graphene increases SPR angle relative to conventional."""
        angles, R_conv = _run(0, 0, 1.34)
        angles, R_gr = _run(0, 1, 1.34)
        theta_conv, _ = find_resonance(angles, R_conv)
        theta_gr, _ = find_resonance(angles, R_gr)
        assert theta_gr > theta_conv
        # The shift should be positive but small (< 2°)
        assert 0 < theta_gr - theta_conv < 2.0

    def test_mos2_graphene_shifts_more(self):
        """MoS2+Graphene shifts more than Graphene alone."""
        angles_gr, R_gr = _run(0, 1, 1.34)
        angles_mg, R_mg = _run(1, 1, 1.34)
        theta_gr, _ = find_resonance(angles_gr, R_gr)
        theta_mg, _ = find_resonance(angles_mg, R_mg)
        delta_gr = theta_gr - 52.0  # approximate conv baseline
        delta_mg = theta_mg - 52.0
        assert delta_mg > delta_gr

    def test_all_configs_have_spr_dip(self):
        """All three configurations show a clear SPR dip."""
        for m, l in [(0, 0), (0, 1), (1, 1)]:
            angles, R = _run(m, l, 1.34)
            r_min = R.min()
            assert r_min < 0.5, f"M={m},L={l}: no SPR dip (R_min={r_min})"


class TestPhysicalInvariants:
    """Physical invariant tests."""

    def test_reflectance_bounded_all_configs(self):
        """0 <= R <= 1 for all three paper configurations."""
        for m, l in [(0, 0), (0, 1), (1, 1)]:
            angles, R = _run(m, l, 1.34)
            assert np.all(R >= -1e-10), f"R<0 for M={m},L={l}: min={R.min()}"
            assert np.all(R <= 1 + 1e-10), f"R>1 for M={m},L={l}: max={R.max()}"

    def test_continuity_under_perturbation(self):
        """Small RI perturbation -> small SPR angle change."""
        delta = 1e-4
        angles_base, R_base = _run(0, 0, 1.34)
        angles_pert, R_pert = _run(0, 0, 1.34 + delta)
        theta_base, _ = find_resonance(angles_base, R_base)
        theta_pert, _ = find_resonance(angles_pert, R_pert)
        assert abs(theta_pert - theta_base) < 0.5

    def test_higher_ri_higher_angle(self):
        """Increasing sensing RI monotonically increases SPR angle."""
        thetas = []
        for n_s in [1.33, 1.34, 1.35, 1.36, 1.37]:
            angles, R = _run(0, 0, n_s)
            theta, _ = find_resonance(angles, R)
            thetas.append(theta)
        for i in range(len(thetas) - 1):
            assert thetas[i + 1] > thetas[i], f"Monotonicity violated: {thetas}"

    def test_dna_ri_perturbation_shifts_spr(self):
        """DNA adsorption (via reproduction model) shifts SPR angle."""
        n_probe = sensing_medium_ri_reproduction(1000, 0)
        n_target = sensing_medium_ri_reproduction(1000, 1100)
        assert n_target > n_probe

        angles_p, R_p = _run(0, 1, n_probe)
        angles_t, R_t = _run(0, 1, n_target)
        theta_p, _ = find_resonance(angles_p, R_p)
        theta_t, _ = find_resonance(angles_t, R_t)
        assert theta_t > theta_p, "DNA adsorption should shift SPR to higher angle"

    def test_sensitivity_positive(self):
        """Sensitivity (dθ/dn) is positive for all configurations."""
        from spr.analysis.metrics import compute_sensitivity_deg_per_RIU
        delta_n = 0.005
        for m, l in [(0, 0), (0, 1), (1, 1)]:
            _, R_base = _run(m, l, 1.34)
            _, R_pert = _run(m, l, 1.34 + delta_n)
            angles = np.arange(SCAN[0], SCAN[1] + SCAN[2]/2, SCAN[2])
            theta_base, _ = find_resonance(angles, R_base)
            theta_pert, _ = find_resonance(angles, R_pert)
            sens = compute_sensitivity_deg_per_RIU(theta_pert - theta_base, delta_n)
            assert sens > 0, f"M={m},L={l}: sensitivity={sens}"
