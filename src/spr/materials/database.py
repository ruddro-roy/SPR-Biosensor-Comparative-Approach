"""Material optical constants database for SPR simulation.

Provides refractive indices at 633 nm for the materials used in the
Habib et al. (2019) Graphene-MoS2 SPR biosensor study.

Two modes are supported:
- "paper": Uses exact values from the paper for reproduction.
- Default: Same values (no alternative dispersion data available in this repo).

If spectral dispersion is needed in future work, add wavelength-dependent
models here and document the source.

Material constants and their sources
-------------------------------------
SF11 prism : n = 1.7786 at 633 nm
    Source: Schott glass catalog / paper Table parameters
Silver (Ag) : epsilon = -18.295 + 0.481j at 633 nm
    Source: Johnson & Christy (1972), Palik handbook; paper Ref [24]
    n_Ag = sqrt(epsilon) ≈ 0.0562 + 4.2776j
MoS2 : n = 5.9 + 0.8j at 633 nm
    Source: Mak et al., PRL 105, 136805 (2010); monolayer thickness 0.65 nm
Graphene : n = 3.0 + 1.1487j at 633 nm
    Source: Bruna & Borini, APL 94, 031901 (2009); monolayer thickness 0.34 nm
PBS sensing medium : n = 1.34 (real)
    Source: Standard PBS buffer RI near visible wavelengths
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Material:
    """Optical material with refractive index at a reference wavelength.

    Attributes
    ----------
    name : str
        Human-readable material name.
    n : complex
        Complex refractive index at the reference wavelength.
    wavelength_nm : float
        Reference wavelength in nm.
    monolayer_thickness_m : float or None
        Thickness of one monolayer in meters (for 2D materials).
    source : str
        Literature source for the optical constants.
    """

    name: str
    n: complex
    wavelength_nm: float = 633.0
    monolayer_thickness_m: float | None = None
    source: str = ""

    @property
    def epsilon(self) -> complex:
        """Complex dielectric permittivity: epsilon = n^2."""
        return self.n**2


# ---------------------------------------------------------------------------
# Material database at 633 nm (paper values)
# ---------------------------------------------------------------------------

_AG_EPSILON = complex(-18.295, 0.481)
_AG_N = np.sqrt(_AG_EPSILON)

MATERIAL_DB: dict[str, Material] = {
    "SF11": Material(
        name="SF11 glass prism",
        n=complex(1.7786, 0),
        source="Schott glass catalog; Habib et al. (2019) Table params",
    ),
    "Ag": Material(
        name="Silver (Ag)",
        n=_AG_N,
        source="Johnson & Christy (1972); epsilon=-18.295+0.481j at 633 nm",
    ),
    "MoS2": Material(
        name="Molybdenum disulfide (MoS2)",
        n=complex(5.9, 0.8),
        monolayer_thickness_m=0.65e-9,
        source="Mak et al., PRL 105, 136805 (2010)",
    ),
    "graphene": Material(
        name="Graphene",
        n=complex(3.0, 1.1487),
        monolayer_thickness_m=0.34e-9,
        source="Bruna & Borini, APL 94, 031901 (2009)",
    ),
    "PBS": Material(
        name="PBS buffer (sensing medium)",
        n=complex(1.34, 0),
        source="Standard PBS RI near visible wavelengths",
    ),
}


def get_material(name: str) -> Material:
    """Look up a material by name.

    Parameters
    ----------
    name : str
        Material key (case-sensitive). One of: SF11, Ag, MoS2, graphene, PBS.

    Returns
    -------
    Material
        The material dataclass.

    Raises
    ------
    KeyError
        If the material is not in the database.
    """
    return MATERIAL_DB[name]
