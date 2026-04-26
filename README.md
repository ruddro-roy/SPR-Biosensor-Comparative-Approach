# SPR Biosensor Comparative Approach

**Transfer-matrix simulation of Graphene and MoS₂-Graphene SPR biosensors**

**Companion site:** [ruddro-roy.github.io/SPR-Biosensor-Comparative-Approach](https://ruddro-roy.github.io/SPR-Biosensor-Comparative-Approach/) — a publication-style summary of the computed results, figures, parameter sweeps, and methods.

A research-grade computational optics codebase implementing the Kretschmann-configuration
Surface Plasmon Resonance (SPR) simulation from:

> Habib, M. M., Roy, R., Islam, M. M., Hassan, M., Islam, M. M., & Hossain, M. B. (2019).
> *Study of Graphene-MoS₂ Based SPR Biosensor with Graphene Based SPR Biosensor: Comparative Approach.*
> International Journal of Natural Sciences Research, 7(1), 1–9.
> DOI: [10.18488/journal.63.2019.71.1.9](https://doi.org/10.18488/journal.63.2019.71.1.9)

## What This Repository Does

- Implements a **multilayer transfer-matrix method (TMM)** for p-polarized angular interrogation at 633 nm
- Compares three Kretschmann SPR configurations:
  1. Conventional: Prism / Ag / Sensing medium
  2. Graphene-enhanced: Prism / Ag / Graphene / Sensing medium
  3. MoS₂-Graphene: Prism / Ag / MoS₂ / Graphene / Sensing medium
- Generates publication-quality figures and metrics tables
- Provides parameter sweeps over Ag thickness, layer counts, and sensing-medium RI
- Includes a comprehensive test suite validated against the [Byrnes tmm package](https://github.com/sbyrnes321/tmm)

## Key Scientific Finding

**The TMM implementation in this repository is validated against the Byrnes tmm package
(arXiv:1603.02720) to machine precision.** Using the paper's stated material constants
(SF11 n=1.7786, Ag ε=−18.295+0.481j at 633 nm, PBS n=1.34), the SPR resonance occurs
at approximately **52–54°**, not the **74–77°** reported in the paper.

This ~22° discrepancy is fundamental and cannot be resolved by adjusting film thickness
or sensing-medium RI. It indicates the paper likely used different Ag optical constants
than those stated, or a different TMM convention. The qualitative trends (graphene shifts
SPR to higher angle, MoS₂+graphene shifts further, sensitivity increases with 2D layers)
are correctly reproduced.

### Computed Results (this repo)

| Configuration | θ_SPR (°) | R_min | Δθ (°) | FWHM (°) | S (°/RIU) |
|---|---|---|---|---|---|
| Conventional (Ag only) | 52.67 | 0.315 | 0.00 | 1.11 | 62.0 |
| Ag + Graphene (1L) | 52.80 | 0.164 | 0.13 | 1.26 | 62.5 |
| Ag + MoS₂ + Graphene | 53.49 | 0.026 | 0.82 | 1.87 | 65.0 |

### Paper-reported values (Table 3)

| Configuration | θ_SPR (°) | R_min | Δθ (°) |
|---|---|---|---|
| Conventional | 74.60 | 0.3484 | 0.00 |
| Graphene | 74.95 | 0.1883 | 0.35 |
| MoS₂-Graphene | 76.70 | 0.0293 | 2.10 |

## Physics

### Transfer Matrix Method

The simulation computes p-polarized reflectance for an N-layer stack using:

1. **Snell's law** generalized for complex refractive indices:
   cos(θ_k) = √(1 − (n₀/n_k)² sin²(θ₀))

2. **Fresnel coefficients** at each interface:
   r_p = (n_f cos θ_i − n_i cos θ_f) / (n_f cos θ_i + n_i cos θ_f)

3. **Propagation phase** through each film:
   δ_k = 2π n_k d_k cos(θ_k) / λ

4. **Transfer matrix product** using interface + propagation matrices (Byrnes convention)

5. **Reflection coefficient**: r = M₂₁/M₁₁, R = |r|²

### Material Constants (at 633 nm)

| Material | Refractive Index | Source |
|---|---|---|
| SF11 prism | 1.7786 | Schott catalog |
| Silver (Ag) | √(−18.295 + 0.481j) ≈ 0.056 + 4.278j | Johnson & Christy (1972) |
| MoS₂ | 5.9 + 0.8j | Mak et al., PRL 105 (2010) |
| Graphene | 3.0 + 1.1487j | Bruna & Borini, APL 94 (2009) |
| PBS buffer | 1.34 | Standard visible-range value |

### DNA Sensing Model

Two modes are provided:

- **Reproduction mode**: Uses an inverse-calibrated scale factor to match the paper's
  reported angular shifts. The formula `δn = dn/dc × c_total × 1.87×10⁻⁵` has no
  independent physical derivation.

- **Physics mode**: Maps nM concentration → mass concentration → δn using:
  `c_mass = c_nM × MW × 10⁻¹²` g/cm³, then `δn = (dn/dc) × c_mass`.
  This gives δn ~ 10⁻⁶, much smaller than the paper implies, consistent with
  the known limitation that bulk dn/dc does not capture surface accumulation effects.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd SPR-Biosensor-Comparative-Approach
pip install -e ".[dev]"

# Optional: install Byrnes tmm for cross-validation
pip install -e ".[validation]"
```

**Requirements**: Python ≥ 3.9, NumPy, SciPy, Matplotlib

## Usage

```bash
# Run all tests
pytest tests/ -v

# Reproduce paper results and generate figures
python scripts/reproduce_paper.py

# Run extended parameter sweeps
python scripts/extended_analysis.py
```

All output figures are saved to `results/`.

## Repository Structure

```
├── src/spr/                  # Main package
│   ├── optics/
│   │   ├── fresnel.py        # Fresnel coefficients and Snell's law
│   │   └── tmm.py            # Transfer Matrix Method engine
│   ├── materials/
│   │   └── database.py       # Optical constants database
│   ├── models/
│   │   └── sensor.py         # Layer stack builder, DNA models
│   ├── analysis/
│   │   └── metrics.py        # Resonance finder, FWHM, sensitivity, FOM
│   └── plotting/
│       └── figures.py        # Publication-quality figure generation
├── tests/                    # 69 tests
│   ├── test_fresnel.py       # Fresnel coefficient tests
│   ├── test_tmm.py           # TMM tests + Byrnes cross-validation
│   ├── test_materials.py     # Material database tests
│   ├── test_sensor.py        # Layer stack + DNA model tests
│   ├── test_metrics.py       # Metrics analysis tests
│   └── test_paper_regression.py  # Physics regression tests
├── scripts/
│   ├── reproduce_paper.py    # Paper reproduction + tables + figures
│   └── extended_analysis.py  # Parameter sweeps
├── results/                  # Generated figures (from actual runs)
├── pyproject.toml            # Package configuration
└── README.md
```

## Tests

The test suite covers:

- **Fresnel physics**: Snell's law, Brewster angle, total internal reflection, complex media
- **TMM correctness**: Reflectance bounds, SPR dip existence, smoothness, layer effects
- **Cross-validation**: Machine-precision match against Byrnes tmm package at multiple angles
- **Materials**: All constants match paper values, all sources documented
- **Sensor models**: Layer stack construction, DNA RI models, dimensional analysis
- **Metrics**: Resonance refinement, FWHM, sensitivity, FOM
- **Physics regression**: Qualitative trends, physical invariants, monotonicity

## Known Limitations

1. **Absolute SPR angles do not match the paper** — see Key Scientific Finding above.
   The qualitative comparative analysis is preserved.

2. **DNA sensing model is approximate** — the reproduction mode uses an empirical
   calibration factor; the physics mode gives much smaller RI changes than the paper
   implies, because bulk dn/dc does not capture surface accumulation.

3. **Fixed wavelength only** — material constants are provided at 633 nm only.
   Spectral dispersion models are not included due to lack of sourced data.

4. **Coherent TMM only** — assumes coherent light and perfectly flat interfaces.
   Roughness, incoherent effects, and beam divergence are not modeled.

5. **dn/dc for DNA** — the value 0.182 cm³/g is an approximation. Actual dn/dc
   varies with DNA sequence, ionic strength, and surface coverage conditions.

## References

- Habib et al. (2019), Int. J. Natural Sciences Research, 7(1), 1–9
- Byrnes, "Multilayer optical calculations", [arXiv:1603.02720](https://arxiv.org/abs/1603.02720)
- Johnson & Christy (1972), PRB 6, 4370 — Ag optical constants
- Mak et al. (2010), PRL 105, 136805 — MoS₂ optical constants
- Bruna & Borini (2009), APL 94, 031901 — Graphene optical constants
- Homola (2003), Anal. Bioanal. Chem. 377, 528 — SPR biosensor review

## Authors

Original paper: Md. Mortuza Habib, Ruddro Roy, Md. Mojidul Islam, Mehedi Hassan,
Md. Muztahidul Islam, Md. Biplob Hossain
