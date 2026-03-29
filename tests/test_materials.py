"""Tests for material database."""

import numpy as np
import pytest

from spr.materials.database import get_material, MATERIAL_DB, Material


class TestMaterialDB:
    """Tests for material property database."""

    def test_all_materials_exist(self):
        """All expected materials are in the database."""
        for name in ["SF11", "Ag", "MoS2", "graphene", "PBS"]:
            m = get_material(name)
            assert isinstance(m, Material)

    def test_sf11_ri(self):
        """SF11 prism RI matches paper value."""
        sf11 = get_material("SF11")
        assert sf11.n == complex(1.7786, 0)

    def test_ag_permittivity(self):
        """Ag permittivity matches paper values."""
        ag = get_material("Ag")
        eps = ag.epsilon
        np.testing.assert_allclose(eps.real, -18.295, atol=0.01)
        np.testing.assert_allclose(eps.imag, 0.481, atol=0.01)

    def test_mos2_ri(self):
        """MoS2 RI matches paper."""
        mos2 = get_material("MoS2")
        assert mos2.n == complex(5.9, 0.8)
        assert mos2.monolayer_thickness_m == pytest.approx(0.65e-9)

    def test_graphene_ri(self):
        """Graphene RI matches paper."""
        gr = get_material("graphene")
        assert gr.n == complex(3.0, 1.1487)
        assert gr.monolayer_thickness_m == pytest.approx(0.34e-9)

    def test_pbs_ri(self):
        """PBS sensing medium RI."""
        pbs = get_material("PBS")
        assert pbs.n.real == pytest.approx(1.34)
        assert pbs.n.imag == 0.0

    def test_unknown_material_raises(self):
        """Requesting unknown material raises KeyError."""
        with pytest.raises(KeyError):
            get_material("unobtanium")

    def test_all_wavelengths_633(self):
        """All materials reference 633 nm."""
        for name, m in MATERIAL_DB.items():
            assert m.wavelength_nm == 633.0, f"{name} not at 633 nm"

    def test_all_have_sources(self):
        """All materials have documented sources."""
        for name, m in MATERIAL_DB.items():
            assert len(m.source) > 0, f"{name} has no source documentation"
