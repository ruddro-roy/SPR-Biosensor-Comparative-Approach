"""Sensor layer-stack models and DNA sensing-medium perturbation.

Defines the Kretschmann-configuration SPR sensor stacks from Habib et al.
(2019) and two modes of mapping DNA concentration to refractive-index change:

- reproduction_mode: Infers effective delta_n from the paper's reported
  angular shifts.  This uses an inverse-calibrated scale factor and is
  explicitly labeled as such.

- physics_mode: Maps concentration -> mass concentration -> delta_n
  using dn/dc with documented assumptions and dimensional analysis.

Caution
-------
Both modes are approximations.  The dn/dc for DNA varies with sequence,
ionic strength, and surface coverage.  The reproduction mode is tuned to
match the paper and should not be interpreted as a physical prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spr.materials.database import get_material, Material


# ---------------------------------------------------------------------------
# Physical constants for the paper
# ---------------------------------------------------------------------------
WAVELENGTH_M = 633e-9  # He-Ne laser wavelength
AG_THICKNESS_M = 40e-9  # Silver film thickness

# DNA sensing parameters
DN_DC = 0.182  # cm^3/g, refractive index increment for DNA
# Approximate MW for short ssDNA oligonucleotide (~20-mer): ~6500 Da
DNA_MW_DA = 6500.0


@dataclass
class LayerStack:
    """Ordered list of layers for TMM computation.

    Attributes
    ----------
    names : list[str]
        Human-readable layer names.
    n_list : list[complex]
        Refractive indices of each layer.
    d_list : list[float]
        Thicknesses in meters (0 for semi-infinite boundary layers).
    """

    names: list[str] = field(default_factory=list)
    n_list: list[complex] = field(default_factory=list)
    d_list: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.n_list)

    def add_layer(self, name: str, n: complex, d: float) -> None:
        self.names.append(name)
        self.n_list.append(n)
        self.d_list.append(d)

    def summary(self) -> str:
        lines = []
        for i, (name, n, d) in enumerate(
            zip(self.names, self.n_list, self.d_list)
        ):
            if d == 0:
                lines.append(f"  {i}: {name}  n={n}  (semi-infinite)")
            else:
                lines.append(f"  {i}: {name}  n={n}  d={d*1e9:.2f} nm")
        return "\n".join(lines)


@dataclass
class SensorConfig:
    """Configuration for one SPR sensor variant.

    Parameters
    ----------
    name : str
        Configuration label (e.g., "Conventional", "Graphene", "MoS2-Graphene").
    n_mos2_layers : int
        Number of MoS2 monolayers.
    n_graphene_layers : int
        Number of graphene monolayers.
    ag_thickness_m : float
        Silver film thickness in meters.
    """

    name: str
    n_mos2_layers: int = 0
    n_graphene_layers: int = 0
    ag_thickness_m: float = AG_THICKNESS_M


def sensing_medium_ri_reproduction(
    ca_probe_nM: float, ca_target_nM: float = 0.0
) -> float:
    """Compute sensing-medium RI using the paper's inverse-calibrated model.

    This is the **reproduction mode**: a scale factor is calibrated so that
    the simulation matches the paper's reported SPR angle shifts for DNA
    hybridization at the concentrations listed in Tables 1-2.

    The effective formula is:
        n_sensing = n_PBS + dn/dc * (c_probe + c_target) * scale_factor

    where scale_factor = 1.87e-5 was determined by matching the paper's
    resonance angles.  This has no independent physical derivation and
    is purely an inverse calibration.

    Parameters
    ----------
    ca_probe_nM : float
        Probe DNA concentration in nM.
    ca_target_nM : float
        Complementary target DNA concentration in nM.

    Returns
    -------
    float
        Modified refractive index of sensing medium.
    """
    n_base = get_material("PBS").n.real
    total_nM = ca_probe_nM + ca_target_nM
    # Inverse-calibrated scale factor (dimensionless fudge)
    REPRODUCTION_SCALE = 1.87e-5
    delta_n = DN_DC * total_nM * REPRODUCTION_SCALE
    return n_base + delta_n


def sensing_medium_ri_physics(
    ca_probe_nM: float,
    ca_target_nM: float = 0.0,
    mw_da: float = DNA_MW_DA,
    dn_dc: float = DN_DC,
) -> float:
    """Compute sensing-medium RI using a physics-based dn/dc model.

    Maps nM concentration to mass concentration, then to delta_n:
        c_mass [g/cm^3] = c_nM * 1e-9 [mol/L] * MW [g/mol] * 1e-3 [L/cm^3]
        delta_n = (dn/dc) * c_mass

    Dimensional analysis:
        [nM] = [nmol/L] = 1e-9 [mol/L]
        c_mass = c_nM * 1e-9 * MW_Da * 1e-3 = c_nM * MW_Da * 1e-12 [g/cm^3]
        delta_n = dn_dc [cm^3/g] * c_mass [g/cm^3]  ->  dimensionless

    Caution
    -------
    This gives delta_n ~ 1e-7 for typical nM concentrations, which produces
    very small angle shifts.  The paper's reported shifts are much larger,
    suggesting either (a) surface accumulation/amplification effects not
    captured by bulk dn/dc, or (b) the paper used a different model.
    This mode is provided for physical reference; use reproduction_mode
    to match the paper.

    Parameters
    ----------
    ca_probe_nM : float
        Probe DNA concentration in nM.
    ca_target_nM : float
        Complementary target concentration in nM.
    mw_da : float
        Molecular weight in Daltons.
    dn_dc : float
        Refractive index increment in cm^3/g.

    Returns
    -------
    float
        Modified refractive index of sensing medium.
    """
    n_base = get_material("PBS").n.real
    total_nM = ca_probe_nM + ca_target_nM
    # Convert nM -> g/cm^3
    c_mass = total_nM * mw_da * 1e-12  # g/cm^3
    delta_n = dn_dc * c_mass
    return n_base + delta_n


def build_layer_stack(
    n_mos2_layers: int = 0,
    n_graphene_layers: int = 0,
    n_sensing: float | None = None,
    ag_thickness_m: float = AG_THICKNESS_M,
) -> LayerStack:
    """Build a Kretschmann SPR layer stack.

    Stack order: prism / Ag / [MoS2 x M] / [graphene x L] / sensing medium

    Parameters
    ----------
    n_mos2_layers : int
        Number of MoS2 monolayers (0 to omit).
    n_graphene_layers : int
        Number of graphene monolayers (0 to omit).
    n_sensing : float or None
        Refractive index of sensing medium.  If None, uses PBS baseline.
    ag_thickness_m : float
        Silver film thickness.

    Returns
    -------
    LayerStack
    """
    if n_sensing is None:
        n_sensing = get_material("PBS").n.real

    stack = LayerStack()

    # Layer 0: Prism (semi-infinite)
    prism = get_material("SF11")
    stack.add_layer("SF11 prism", prism.n, 0.0)

    # Layer 1: Silver film
    ag = get_material("Ag")
    stack.add_layer("Ag", ag.n, ag_thickness_m)

    # Optional MoS2 layers
    if n_mos2_layers > 0:
        mos2 = get_material("MoS2")
        d_mos2 = n_mos2_layers * mos2.monolayer_thickness_m
        stack.add_layer(
            f"MoS2 ({n_mos2_layers}L)", mos2.n, d_mos2
        )

    # Optional graphene layers
    if n_graphene_layers > 0:
        gr = get_material("graphene")
        d_gr = n_graphene_layers * gr.monolayer_thickness_m
        stack.add_layer(
            f"Graphene ({n_graphene_layers}L)", gr.n, d_gr
        )

    # Final layer: Sensing medium (semi-infinite)
    stack.add_layer("Sensing medium", complex(n_sensing, 0), 0.0)

    return stack


def build_paper_stacks(
    n_sensing: float | None = None,
) -> dict[str, LayerStack]:
    """Build the three standard stacks from the paper.

    Returns
    -------
    dict
        Keys: "conventional", "graphene", "mos2_graphene"
        Values: LayerStack instances
    """
    return {
        "conventional": build_layer_stack(0, 0, n_sensing),
        "graphene": build_layer_stack(0, 1, n_sensing),
        "mos2_graphene": build_layer_stack(1, 1, n_sensing),
    }
