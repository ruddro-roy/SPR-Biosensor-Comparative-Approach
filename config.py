"""
config.py
=========
Physical constants, material optical parameters, and sensor geometry
for the Graphene-MoS2 SPR biosensor simulation.

All values are taken directly from:
  Habib et al., Int. J. Natural Sciences Research, 7(1), 1-9, 2019.
    DOI: 10.18488/journal.63.2019.71.1.9

    References for material parameters:
      - SF11 prism RI: Schott glass catalog at 633 nm
        - Ag dielectric function: Lorentz-Drude model (Ref [24] in paper)
          - MoS2 complex RI: Mak et al., PRL 105, 136805 (2010)
            - Graphene complex RI: Bruna & Borini, APL 94, 031901 (2009)
            """

import numpy as np


# =============================================================================
# Operating Wavelength
# =============================================================================
WAVELENGTH = 633e-9          # Operating wavelength: 633 nm (He-Ne laser)

# =============================================================================
# Prism Parameters (SF11 glass)
# =============================================================================
N_PRISM = 1.7786             # Refractive index of SF11 glass at 633 nm

# =============================================================================
# Silver (Ag) Film Parameters
# =============================================================================
AG_THICKNESS = 40e-9         # Silver film thickness: 40 nm

# Ag complex dielectric function at 633 nm via Lorentz-Drude model
# epsilon_Ag = epsilon_real + j * epsilon_imag
# At 633 nm: n_Ag ~ 0.056206 + 4.2776j  (from Palik / Johnson & Christy data)
# Corresponding permittivity: epsilon = n^2
# Using standard tabulated values for Ag at 633 nm:
AG_EPSILON_REAL = -18.295    # Real part of Ag permittivity at 633 nm
AG_EPSILON_IMAG = 0.481      # Imaginary part of Ag permittivity at 633 nm
N_AG = np.sqrt(complex(AG_EPSILON_REAL, AG_EPSILON_IMAG))

# =============================================================================
# MoS2 Layer Parameters
# =============================================================================
MOS2_THICKNESS_PER_LAYER = 0.65e-9   # Thickness per MoS2 monolayer: 0.65 nm

# Complex refractive index of MoS2 at 633 nm
# From paper: n_MoS2 = 5.9 + 0.8i (Ref [24]: Mak et al.)
N_MOS2_REAL = 5.9
N_MOS2_IMAG = 0.8
N_MOS2 = complex(N_MOS2_REAL, N_MOS2_IMAG)

# MoS2 permittivity (epsilon = n^2)
EPSILON_MOS2 = N_MOS2 ** 2

# =============================================================================
# Graphene Layer Parameters
# =============================================================================
GRAPHENE_THICKNESS_PER_LAYER = 0.34e-9  # Thickness per graphene layer: 0.34 nm

# Complex refractive index of graphene at 633 nm
# From paper: n_graphene = 3.0 + 1.1487i
N_GRAPHENE_REAL = 3.0
N_GRAPHENE_IMAG = 1.1487
N_GRAPHENE = complex(N_GRAPHENE_REAL, N_GRAPHENE_IMAG)

# Graphene permittivity
EPSILON_GRAPHENE = N_GRAPHENE ** 2

# =============================================================================
# Sensing Medium Parameters
# =============================================================================
N_SENSING_BASE = 1.34        # RI of PBS (phosphate-buffered saline) sensing medium

# DNA adsorption parameters (Eq. 9 in paper)
# dn/dc = 0.182 cm^3/g  (refractive index increment for DNA in water)
DN_DC = 0.182                # cm^3/g

# =============================================================================
# Sensor Geometry
# =============================================================================
PRISM_DIAMETER = 50e-6       # Prism diameter: 50 um
SENSING_REGION = 5e-3        # Sensing region length: 5 mm

# =============================================================================
# Simulation Parameters
# =============================================================================
ANGLE_START = 50.0           # Start angle for angular scan (degrees)
ANGLE_END = 90.0             # End angle for angular scan (degrees)
ANGLE_STEP = 0.01            # Angular resolution (degrees)

# =============================================================================
# DNA Concentrations (nM) from Tables 1 & 2
# =============================================================================
DNA_CONCENTRATIONS = {
      "probe": 1000,           # Probe DNA concentration (nM)
      "comp_1000": 1000,       # Complementary target 1000 nM
      "comp_1001": 1001,       # Complementary target 1001 nM
      "comp_1010": 1010,       # Complementary target 1010 nM
      "comp_1100": 1100,       # Complementary target 1100 nM
}

# =============================================================================
# Reference Values from Paper (for validation)
# =============================================================================
# Table 3: Rmin, theta_SPR, delta_theta_SPR for different layer configs
REFERENCE_TABLE3 = {
      "conventional": {"Rmin": 0.3484, "theta_SPR": 74.60, "delta_theta": 0.00},
      "graphene":     {"Rmin": 0.1883, "theta_SPR": 74.95, "delta_theta": 0.35},
      "mos2_graphene": {"Rmin": 0.0293, "theta_SPR": 76.70, "delta_theta": 2.10},
}

# Table 1: Ag-Graphene hybrid SPR sensor
REFERENCE_TABLE1 = {
      "probe_1000":  {"Rmin": 0.1777, "theta_SPR": 76.40},
      "comp_1000":   {"Rmin": 0.0943, "theta_SPR": 81.80},
      "comp_1001":   {"Rmin": 0.0849, "theta_SPR": 82.20},
      "comp_1010":   {"Rmin": 0.0682, "theta_SPR": 82.85},
      "comp_1100":   {"Rmin": 0.0321, "theta_SPR": 84.25},
}

# Table 2: Ag-MoS2-Graphene hybrid SPR sensor
REFERENCE_TABLE2 = {
      "probe_1000":  {"Rmin": 0.0204, "theta_SPR": 78.40},
      "comp_1000":   {"Rmin": 0.0324, "theta_SPR": 85.15},
      "comp_1001":   {"Rmin": 0.0573, "theta_SPR": 85.65},
      "comp_1010":   {"Rmin": 0.1001, "theta_SPR": 86.20},
      "comp_1100":   {"Rmin": 0.1718, "theta_SPR": 86.65},
}

# Figure 2 reference SPR angles
FIG2_GRAPHENE_SPR_ANGLE = 74.4      # degrees (probe DNA, graphene only)
FIG2_MOS2_GRAPHENE_SPR_ANGLE = 76.4  # degrees (probe DNA, MoS2-graphene)

# =============================================================================
# Sensitivity Enhancement Factors (from Eq. 11 and 12)
# =============================================================================
SENSITIVITY_GRAPHENE_PERCENT = 35     # 35% improvement with graphene
SENSITIVITY_MOS2_GRAPHENE_PERCENT = 210  # 210% improvement with MoS2-graphene
SENSITIVITY_RELATIVE_PERCENT = 175    # 175% more than graphene-only
