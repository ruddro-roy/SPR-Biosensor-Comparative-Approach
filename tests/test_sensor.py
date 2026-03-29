"""Tests for sensor layer stack and DNA perturbation models."""

import numpy as np
import pytest

from spr.models.sensor import (
    build_layer_stack,
    build_paper_stacks,
    sensing_medium_ri_reproduction,
    sensing_medium_ri_physics,
    LayerStack,
    SensorConfig,
)


class TestLayerStack:
    """Tests for layer stack construction."""

    def test_conventional_stack(self):
        """Conventional stack: prism/Ag/sensing."""
        stack = build_layer_stack(0, 0)
        assert len(stack) == 3
        assert stack.names[0] == "SF11 prism"
        assert stack.names[1] == "Ag"
        assert stack.names[2] == "Sensing medium"
        # Boundary layers have d=0
        assert stack.d_list[0] == 0.0
        assert stack.d_list[-1] == 0.0
        # Ag thickness
        assert stack.d_list[1] == pytest.approx(40e-9)

    def test_graphene_stack(self):
        """Graphene stack: prism/Ag/graphene/sensing."""
        stack = build_layer_stack(0, 1)
        assert len(stack) == 4
        assert "Graphene" in stack.names[2]
        assert stack.d_list[2] == pytest.approx(0.34e-9)

    def test_mos2_graphene_stack(self):
        """MoS2+Graphene stack: prism/Ag/MoS2/graphene/sensing."""
        stack = build_layer_stack(1, 1)
        assert len(stack) == 5
        assert "MoS2" in stack.names[2]
        assert "Graphene" in stack.names[3]

    def test_multiple_layers(self):
        """Multiple MoS2/graphene layers scale thickness correctly."""
        stack = build_layer_stack(3, 5)
        # MoS2: 3 * 0.65 nm
        assert stack.d_list[2] == pytest.approx(3 * 0.65e-9)
        # Graphene: 5 * 0.34 nm
        assert stack.d_list[3] == pytest.approx(5 * 0.34e-9)

    def test_custom_sensing_ri(self):
        """Custom sensing medium RI is applied."""
        stack = build_layer_stack(0, 0, n_sensing=1.38)
        assert stack.n_list[-1] == complex(1.38, 0)

    def test_paper_stacks(self):
        """build_paper_stacks returns all three configs."""
        stacks = build_paper_stacks()
        assert "conventional" in stacks
        assert "graphene" in stacks
        assert "mos2_graphene" in stacks
        assert len(stacks["conventional"]) == 3
        assert len(stacks["graphene"]) == 4
        assert len(stacks["mos2_graphene"]) == 5


class TestSensingMediumRI:
    """Tests for DNA concentration -> RI models."""

    def test_reproduction_baseline(self):
        """Zero concentration gives baseline PBS RI."""
        n = sensing_medium_ri_reproduction(0, 0)
        assert n == pytest.approx(1.34, abs=1e-10)

    def test_reproduction_increases_with_concentration(self):
        """Higher DNA concentration -> higher RI."""
        n1 = sensing_medium_ri_reproduction(1000, 0)
        n2 = sensing_medium_ri_reproduction(1000, 1000)
        n3 = sensing_medium_ri_reproduction(1000, 1100)
        assert n1 < n2 < n3

    def test_reproduction_positive_shift(self):
        """DNA adsorption always increases RI from baseline."""
        n = sensing_medium_ri_reproduction(1000, 1000)
        assert n > 1.34

    def test_physics_baseline(self):
        """Physics mode at zero concentration gives PBS RI."""
        n = sensing_medium_ri_physics(0, 0)
        assert n == pytest.approx(1.34, abs=1e-10)

    def test_physics_increases(self):
        """Physics mode: RI increases with concentration."""
        n1 = sensing_medium_ri_physics(1000, 0)
        n2 = sensing_medium_ri_physics(1000, 1000)
        assert n2 > n1

    def test_physics_dimensional_analysis(self):
        """Physics mode delta_n has correct order of magnitude.

        For 2000 nM DNA (MW~6500):
            c_mass = 2000 * 6500 * 1e-12 = 1.3e-5 g/cm^3
            delta_n = 0.182 * 1.3e-5 ≈ 2.4e-6
        This is very small, consistent with bulk dn/dc.
        """
        n = sensing_medium_ri_physics(1000, 1000, mw_da=6500, dn_dc=0.182)
        delta_n = n - 1.34
        assert 1e-7 < delta_n < 1e-4

    def test_reproduction_vs_physics_magnitude(self):
        """Reproduction mode gives much larger delta_n than physics mode.

        This is expected and documented — the paper's shifts are too large
        to be explained by bulk dn/dc alone.
        """
        n_repro = sensing_medium_ri_reproduction(1000, 1000)
        n_phys = sensing_medium_ri_physics(1000, 1000)
        delta_repro = n_repro - 1.34
        delta_phys = n_phys - 1.34
        # Reproduction mode should give at least 100x larger delta_n
        assert delta_repro > 100 * delta_phys
