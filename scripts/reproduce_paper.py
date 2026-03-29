#!/usr/bin/env python3
"""Reproduce and analyze results from Habib et al. (2019).

This script runs the transfer-matrix simulation using the paper's stated
material constants and compares results to the paper's reported values.

IMPORTANT SCIENTIFIC NOTE
-------------------------
Our TMM implementation (validated against the Byrnes tmm package,
arXiv:1603.02720) finds the SPR resonance at approximately 52-54 degrees
for the paper's material constants (SF11 n=1.7786, Ag eps=-18.295+0.481j,
sensing medium n=1.34).

The paper reports SPR at ~74-77 degrees. This ~22-degree discrepancy
cannot be resolved by any reasonable parameter variation and indicates
either:
1. The paper used different Ag optical constants than stated
2. There is an error in the paper's reported material parameters
3. The original simulation had a different TMM convention

We report both our physically consistent results AND the paper's
claimed values for comparison. The qualitative trends (graphene shifts
SPR to higher angle, MoS2+graphene shifts further, adding layers
increases sensitivity) are preserved.

Usage:
    python scripts/reproduce_paper.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spr.optics.tmm import angular_scan
from spr.models.sensor import (
    build_layer_stack,
    sensing_medium_ri_reproduction,
    WAVELENGTH_M,
)
from spr.analysis.metrics import (
    find_resonance,
    compute_fwhm,
    compute_sensitivity_deg_per_RIU,
    compute_fom,
    analyze_curve,
)
from spr.plotting.figures import (
    setup_matplotlib,
    plot_reflectance_curves,
    plot_sensitivity_bars,
    COLORS,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Angular scan range — covers both the physically correct SPR (~50-55°)
# and has enough range for parameter sweeps
SCAN = (40.0, 89.0, 0.01)

# Paper reference values (Table 3) — note these are NOT reproducible
# with the stated material constants
PAPER_TABLE3 = {
    "Conventional": {"theta_SPR": 74.60, "Rmin": 0.3484, "delta": 0.00},
    "Graphene": {"theta_SPR": 74.95, "Rmin": 0.1883, "delta": 0.35},
    "MoS2-Graphene": {"theta_SPR": 76.70, "Rmin": 0.0293, "delta": 2.10},
}


def _run_scan(n_mos2: int, n_graphene: int, n_sensing: float):
    """Build stack, run angular scan, analyze."""
    stack = build_layer_stack(n_mos2, n_graphene, n_sensing)
    angles, R = angular_scan(
        stack.n_list, stack.d_list, WAVELENGTH_M,
        angle_start=SCAN[0], angle_end=SCAN[1], angle_step=SCAN[2],
    )
    analysis = analyze_curve(angles, R)
    return angles, R, analysis


def header(title: str) -> None:
    w = 72
    print(f"\n{'='*w}\n  {title}\n{'='*w}")


def reproduce_table3():
    """Table 3: Layer configuration comparison."""
    header("TABLE 3: Layer Configuration Comparison")

    n_s = 1.34  # PBS baseline
    configs = [
        ("Conventional (L=0, M=0)", 0, 0),
        ("Graphene (L=1, M=0)", 0, 1),
        ("MoS2-Graphene (L=1, M=1)", 1, 1),
    ]

    print(f"  Sensing medium RI: {n_s:.4f}")
    print(f"  {'Config':<30s} {'R_min':>8s} {'θ_SPR':>10s} {'Δθ':>8s} {'FWHM':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    results = {}
    base_theta = None
    for name, m, l in configs:
        angles, R, a = _run_scan(m, l, n_s)
        if base_theta is None:
            base_theta = a.theta_spr
        delta = a.theta_spr - base_theta
        results[name] = (angles, R, a, delta)
        fwhm_str = f"{a.fwhm:.2f}" if not np.isnan(a.fwhm) else "N/A"
        print(f"  {name:<30s} {a.r_min:>8.4f} {a.theta_spr:>8.2f}° {delta:>7.2f}° {fwhm_str:>7s}°")

    print(f"\n  {'--- Paper Reference (Table 3) ---'}")
    print(f"  NOTE: Paper values are NOT reproducible with stated material constants.")
    print(f"  The ~22° discrepancy is discussed in README.md.")
    for name, ref in PAPER_TABLE3.items():
        print(f"  {name:<30s} {ref['Rmin']:>8.4f} {ref['theta_SPR']:>8.2f}° {ref['delta']:>7.2f}°")

    return results


def reproduce_dna_series():
    """Tables 1 & 2: DNA concentration series with reproduction mode RI."""
    header("DNA HYBRIDIZATION SERIES (Reproduction Mode)")

    configs = [
        ("Ag-Graphene (Table 1)", 0, 1),
        ("Ag-MoS2-Graphene (Table 2)", 1, 1),
    ]

    cases = [
        ("Probe 1000 nM", 1000, 0),
        ("Comp. 1000 nM", 1000, 1000),
        ("Comp. 1001 nM", 1000, 1001),
        ("Comp. 1010 nM", 1000, 1010),
        ("Comp. 1100 nM", 1000, 1100),
    ]

    all_results = {}

    for config_name, m, l in configs:
        header(f"{config_name}")
        print(f"  {'Case':<22s} {'n_sens':>10s} {'R_min':>8s} {'θ_SPR':>10s} {'FWHM':>8s}")
        print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

        results = []
        for label, probe, target in cases:
            n_s = sensing_medium_ri_reproduction(probe, target)
            angles, R, a = _run_scan(m, l, n_s)
            results.append((label, angles, R, a, n_s))
            fwhm_str = f"{a.fwhm:.2f}" if not np.isnan(a.fwhm) else "N/A"
            print(f"  {label:<22s} {n_s:>10.6f} {a.r_min:>8.4f} {a.theta_spr:>8.2f}° {fwhm_str:>7s}°")

        # Sensitivity between probe and highest target
        if len(results) >= 2:
            theta_base = results[0][3].theta_spr
            theta_last = results[-1][3].theta_spr
            dn = results[-1][4] - results[0][4]
            d_theta = theta_last - theta_base
            if abs(dn) > 1e-10:
                sens = compute_sensitivity_deg_per_RIU(d_theta, dn)
                print(f"\n  Sensitivity (probe -> 1100nM): {sens:.1f} °/RIU")
                print(f"  Δθ = {d_theta:.2f}°, Δn = {dn:.6f}")

        all_results[config_name] = results

    return all_results


def generate_figure2(table3_results):
    """Figure 2: SPR curves for three configurations."""
    header("FIGURE 2: SPR Reflectance Curves")

    curves = []
    markers = []
    color_map = {
        "Conventional (L=0, M=0)": COLORS["conventional"],
        "Graphene (L=1, M=0)": COLORS["graphene"],
        "MoS2-Graphene (L=1, M=1)": COLORS["mos2_graphene"],
    }
    label_map = {
        "Conventional (L=0, M=0)": "Conventional (Ag only)",
        "Graphene (L=1, M=0)": "Ag + Graphene",
        "MoS2-Graphene (L=1, M=1)": "Ag + MoS$_2$ + Graphene",
    }

    for name, (angles, R, a, delta) in table3_results.items():
        color = color_map[name]
        curves.append((angles, R, label_map[name], color))
        markers.append((a.theta_spr, a.r_min, label_map[name]))
        print(f"  {name}: θ_SPR={a.theta_spr:.2f}°, R_min={a.r_min:.4f}, Δθ={delta:.2f}°")

    savepath = os.path.join(RESULTS_DIR, "figure2_spr_curves.png")
    plot_reflectance_curves(
        curves,
        title="SPR Reflectance: Conventional vs Graphene vs MoS$_2$-Graphene\n"
              "(633 nm, SF11/Ag(40nm)/layers/PBS)",
        xlim=(45, 70),
        markers=markers,
        savepath=savepath,
    )
    print(f"  Saved: {savepath}")


def generate_figure3(dna_results):
    """Figure 3a/3b: DNA concentration series."""
    header("FIGURE 3: DNA Hybridization SPR Curves")

    dna_colors = [
        COLORS["dna_probe"],
        COLORS["dna_1000"],
        COLORS["dna_1001"],
        COLORS["dna_1010"],
        COLORS["dna_1100"],
    ]

    for config_name, results in dna_results.items():
        curves = []
        for (label, angles, R, a, n_s), color in zip(results, dna_colors):
            curves.append((angles, R, label, color))

        if "Graphene" in config_name and "MoS2" not in config_name:
            fname = "figure3a_graphene_dna.png"
            title = "Figure 3a: Ag-Graphene SPR — DNA Hybridization"
        else:
            fname = "figure3b_mos2_graphene_dna.png"
            title = "Figure 3b: Ag-MoS$_2$-Graphene SPR — DNA Hybridization"

        savepath = os.path.join(RESULTS_DIR, fname)
        plot_reflectance_curves(
            curves, title=title,
            xlim=(45, 75), savepath=savepath,
        )
        print(f"  Saved: {savepath}")


def generate_sensitivity_chart(table3_results):
    """Sensitivity bar chart."""
    header("SENSITIVITY COMPARISON")

    names = list(table3_results.keys())
    deltas = [table3_results[n][3] for n in names]

    labels = ["Conventional\n(L=0, M=0)", "Graphene\n(L=1, M=0)", "MoS$_2$-Graphene\n(L=1, M=1)"]
    colors = [COLORS["conventional"], COLORS["graphene"], COLORS["mos2_graphene"]]

    if deltas[1] > 0:
        print(f"  Graphene Δθ: {deltas[1]:.2f}°")
        print(f"  MoS2-Graphene Δθ: {deltas[2]:.2f}°")
        enh = (deltas[2] / deltas[1] - 1) * 100
        print(f"  MoS2-Gr enhancement over Graphene: {enh:.0f}%")
        print(f"  (Paper claims: graphene 35%, MoS2-Gr 210% over conventional)")

    savepath = os.path.join(RESULTS_DIR, "sensitivity_comparison.png")
    plot_sensitivity_bars(labels, deltas, colors, savepath=savepath)
    print(f"  Saved: {savepath}")


def generate_metrics_table(table3_results):
    """Extended metrics: sensitivity in °/RIU, FWHM, FOM."""
    header("EXTENDED METRICS (Sensitivity in °/RIU)")

    delta_n = 0.005  # RI perturbation for sensitivity measurement
    n_base = 1.34

    print(f"  Sensitivity measured with Δn = {delta_n}")
    print(f"  {'Config':<28s} {'S (°/RIU)':>10s} {'FWHM (°)':>10s} {'FOM (1/RIU)':>12s} {'R_min':>8s}")
    print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    for name, m, l in [
        ("Conventional", 0, 0),
        ("Graphene (1L)", 0, 1),
        ("MoS2+Graphene (1L each)", 1, 1),
    ]:
        _, _, a_base = _run_scan(m, l, n_base)
        _, _, a_pert = _run_scan(m, l, n_base + delta_n)
        d_theta = a_pert.theta_spr - a_base.theta_spr
        sens = compute_sensitivity_deg_per_RIU(d_theta, delta_n)
        fom = compute_fom(sens, a_base.fwhm)
        fwhm_str = f"{a_base.fwhm:.2f}" if not np.isnan(a_base.fwhm) else "N/A"
        fom_str = f"{fom:.1f}" if not np.isnan(fom) else "N/A"
        print(f"  {name:<28s} {sens:>10.1f} {fwhm_str:>10s} {fom_str:>12s} {a_base.r_min:>8.4f}")


def main():
    print("\n" + "#" * 72)
    print("#  SPR Biosensor Simulation — Paper Reproduction & Analysis")
    print("#  Habib et al. (2019), Int. J. Natural Sciences Research")
    print("#  TMM validated against Byrnes tmm package (arXiv:1603.02720)")
    print("#" * 72)
    print(f"\n  Wavelength: 633 nm | Scan: {SCAN[0]}°–{SCAN[1]}° | Step: {SCAN[2]}°")

    table3 = reproduce_table3()
    dna = reproduce_dna_series()

    generate_figure2(table3)
    generate_figure3(dna)
    generate_sensitivity_chart(table3)
    generate_metrics_table(table3)

    header("COMPLETE")
    print(f"  Results in: {os.path.abspath(RESULTS_DIR)}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        sz = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1024
        print(f"    {f} ({sz:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
