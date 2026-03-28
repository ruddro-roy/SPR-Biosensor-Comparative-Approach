# SPR-Biosensor-Comparative-Approach

## Study of Graphene-MoS₂ Based SPR Biosensor with Graphene Based SPR Biosensor: Comparative Approach

A complete numerical simulation codebase for Surface Plasmon Resonance (SPR) biosensors comparing Ag-Graphene and Ag-MoS₂-Graphene hybrid configurations for DNA hybridization detection.

**Based on:**  
Habib, M. M., Roy, R., Islam, M. M., Hassan, M., Islam, M. M., & Hossain, M. B. (2019).  
*International Journal of Natural Sciences Research*, 7(1), 1–9.  
DOI: [10.18488/journal.63.2019.71.1.9](https://doi.org/10.18488/journal.63.2019.71.1.9)

---

## Project Structure

```
SPR-Biosensor-Comparative-Approach/
├── README.md
├── requirements.txt
├── config.py                  # Physical constants & material parameters
├── spr_simulation.py          # Core 4-layer Fresnel model engine
├── main.py                    # Reproduce all paper results (Tables & Figures)
├── plot_results.py            # Publication-quality figure generation
└── results/                   # Output directory (auto-created)
```

## Physics Overview

The simulation implements the **Kretschmann configuration** for SPR biosensing using a **4-layer Fresnel reflection model**:

1. **Layer 1 – SF11 Glass Prism** (n = 1.7786)  
2. **Layer 2 – Silver (Ag) film** (40 nm, Lorentz-Drude model at λ = 633 nm)  
3. **Layer 3 – MoS₂ layer** (thickness = M × 0.65 nm per layer)  
4. **Layer 4 – Graphene layer** (thickness = L × 0.34 nm per layer)  
5. **Sensing Medium** – PBS + DNA molecules  

The reflected power for p-polarized (TM) light is computed via the **transfer matrix method (TMM)**, and SPR angle shifts are used to quantify sensor sensitivity for DNA hybridization detection.

### Key Equations Implemented

- **Fresnel 4-layer reflectance** (Eq. 1–4)
- **Snell's law generalization** for multilayer angles (Eq. 5)
- **Light wave vector** and **SPW wave vector matching** (Eq. 6–8)
- **Refractive index change** due to DNA adsorption (Eq. 9)
- **Sensitivity analysis** for Graphene vs MoS₂-Graphene (Eq. 10–12)

## Key Results Reproduced

| Configuration | Rmin | θ_SPR (°) | Δθ_SPR (°) |
|---|---|---|---|
| Conventional (L=0, M=0) | 0.3484 | 74.60 | 0.00 |
| Graphene (L=1, M=0) | 0.1883 | 74.95 | 0.35 |
| MoS₂-Graphene (L=1, M=1) | 0.0293 | 76.70 | 2.10 |

**Sensitivity Enhancement:**  
- Single graphene layer → **35% improvement** over conventional  
- MoS₂-Graphene hybrid → **210% improvement** over conventional (**175% more** than graphene-only)

## Installation

```bash
# Clone the repository
git clone https://github.com/ruddro-roy/SPR-Biosensor-Comparative-Approach.git
cd SPR-Biosensor-Comparative-Approach

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the full simulation and reproduce all paper results
python main.py

# Generate publication-quality figures only
python plot_results.py
```

## Output

Running `main.py` produces:
- **Console output**: All tables from the paper (Tables 1–3) with computed values
- **Figures saved to `results/` directory**:
  - `figure2_spr_curves_probe.png` – SPR curves with/without MoS₂-Graphene
    - `figure3a_graphene_dna.png` – Ag-Graphene SPR curves for DNA concentrations
      - `figure3b_mos2_graphene_dna.png` – Ag-MoS₂-Graphene SPR curves for DNA concentrations
        - `sensitivity_comparison.png` – Sensitivity bar chart comparison

        ## Dependencies

        - Python ≥ 3.8
        - NumPy
        - SciPy
        - Matplotlib

        ## License

        This project is for academic and educational purposes.

        ## Authors

        Md. Mortuza Habib, Ruddro Roy, Md. Mojidul Islam, Mehedi Hassan, Md. Muztahidul Islam, Md. Biplob Hossain
