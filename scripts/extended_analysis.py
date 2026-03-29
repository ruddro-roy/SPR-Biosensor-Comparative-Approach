#!/usr/bin/env python3
"""Extended analysis: parameter sweeps and optimization for SPR biosensor.

Goes beyond paper reproduction to explore:
1. Ag thickness optimization (20–60 nm)
2. Number of MoS2 layers (0–5)
3. Number of graphene layers (0–10)
4. Sensing-medium RI sweep with sensitivity in °/RIU
5. Comparative sensitivity across configurations

Usage:
    python scripts/extended_analysis.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from spr.optics.tmm import angular_scan
from spr.models.sensor import build_layer_stack, WAVELENGTH_M
from spr.analysis.metrics import (
    find_resonance,
    analyze_curve,
    compute_sensitivity_deg_per_RIU,
    compute_fom,
)
from spr.plotting.figures import (
    plot_parameter_sweep,
    plot_reflectance_curves,
    COLORS,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
SCAN = (40.0, 89.0, 0.01)
N_SENSING_BASE = 1.34


def _scan(n_mos2, n_graphene, n_sensing, ag_thick=40e-9):
    stack = build_layer_stack(n_mos2, n_graphene, n_sensing, ag_thick)
    angles, R = angular_scan(
        stack.n_list, stack.d_list, WAVELENGTH_M,
        SCAN[0], SCAN[1], SCAN[2],
    )
    return angles, R


def header(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


def sweep_ag_thickness():
    """Sweep Ag film thickness for each configuration."""
    header("SWEEP: Ag Thickness (20–60 nm)")

    thicknesses_nm = np.arange(20, 62, 2)
    configs = [
        ("Conventional", 0, 0),
        ("Graphene (1L)", 0, 1),
        ("MoS2+Graphene", 1, 1),
    ]

    theta_data = {}
    rmin_data = {}
    for name, m, l in configs:
        thetas = []
        rmins = []
        for t_nm in thicknesses_nm:
            angles, R = _scan(m, l, N_SENSING_BASE, t_nm * 1e-9)
            th, rm = find_resonance(angles, R)
            thetas.append(th)
            rmins.append(rm)
        theta_data[name] = np.array(thetas)
        rmin_data[name] = np.array(rmins)
        opt_idx = np.argmin(rmins)
        print(f"  {name}: deepest dip at {thicknesses_nm[opt_idx]} nm "
              f"(R_min={rmins[opt_idx]:.4f}, θ_SPR={thetas[opt_idx]:.2f}°)")

    plot_parameter_sweep(
        thicknesses_nm, theta_data,
        xlabel="Ag Thickness (nm)", ylabel="$\\theta_{SPR}$ (degrees)",
        title="Effect of Ag Film Thickness on Resonance Angle",
        savepath=os.path.join(RESULTS_DIR, "sweep_ag_thickness_theta.png"),
    )
    plot_parameter_sweep(
        thicknesses_nm, rmin_data,
        xlabel="Ag Thickness (nm)", ylabel="Minimum Reflectance",
        title="Effect of Ag Film Thickness on Dip Depth",
        savepath=os.path.join(RESULTS_DIR, "sweep_ag_thickness_rmin.png"),
    )
    print(f"  Saved: sweep_ag_thickness_theta.png, sweep_ag_thickness_rmin.png")


def sweep_mos2_layers():
    """Sweep number of MoS2 monolayers (0–5) with 1 graphene layer."""
    header("SWEEP: MoS2 Layers (0–5, with 1 graphene layer)")

    n_layers = np.arange(0, 6)
    thetas, rmins, fwhms = [], [], []
    for m in n_layers:
        angles, R = _scan(int(m), 1, N_SENSING_BASE)
        a = analyze_curve(angles, R)
        thetas.append(a.theta_spr)
        rmins.append(a.r_min)
        fwhms.append(a.fwhm)
        fwhm_s = f"{a.fwhm:.2f}" if not np.isnan(a.fwhm) else "N/A"
        print(f"  M={m}: θ={a.theta_spr:.2f}°, R_min={a.r_min:.4f}, FWHM={fwhm_s}°")

    plot_parameter_sweep(
        n_layers,
        {"$\\theta_{SPR}$": np.array(thetas)},
        xlabel="Number of MoS$_2$ Monolayers",
        ylabel="$\\theta_{SPR}$ (degrees)",
        title="Effect of MoS$_2$ Layer Count on SPR Angle (with 1 graphene layer)",
        savepath=os.path.join(RESULTS_DIR, "sweep_mos2_layers.png"),
    )
    print(f"  Saved: sweep_mos2_layers.png")


def sweep_graphene_layers():
    """Sweep number of graphene monolayers (0–10, no MoS2)."""
    header("SWEEP: Graphene Layers (0–10, no MoS2)")

    n_layers = np.arange(0, 11)
    thetas, rmins = [], []
    for lg in n_layers:
        angles, R = _scan(0, int(lg), N_SENSING_BASE)
        th, rm = find_resonance(angles, R)
        thetas.append(th)
        rmins.append(rm)
        print(f"  L={lg}: θ={th:.2f}°, R_min={rm:.4f}")

    plot_parameter_sweep(
        n_layers,
        {"$\\theta_{SPR}$": np.array(thetas)},
        xlabel="Number of Graphene Monolayers",
        ylabel="$\\theta_{SPR}$ (degrees)",
        title="Effect of Graphene Layer Count on SPR Angle",
        savepath=os.path.join(RESULTS_DIR, "sweep_graphene_layers.png"),
    )
    print(f"  Saved: sweep_graphene_layers.png")


def sweep_sensing_ri():
    """Sweep sensing-medium RI and compute sensitivity in °/RIU."""
    header("SWEEP: Sensing Medium RI (1.33–1.40)")

    ri_values = np.arange(1.33, 1.401, 0.005)
    configs = [
        ("Conventional", 0, 0),
        ("Graphene (1L)", 0, 1),
        ("MoS2+Graphene", 1, 1),
    ]

    theta_data = {}
    for name, m, l in configs:
        thetas = []
        for ri in ri_values:
            angles, R = _scan(m, l, ri)
            th, _ = find_resonance(angles, R)
            thetas.append(th)
        theta_data[name] = np.array(thetas)

        # Sensitivity from linear fit
        coeffs = np.polyfit(ri_values, thetas, 1)
        print(f"  {name}: sensitivity ≈ {coeffs[0]:.1f} °/RIU (linear fit over 1.33–1.40)")

    plot_parameter_sweep(
        ri_values, theta_data,
        xlabel="Sensing Medium Refractive Index",
        ylabel="$\\theta_{SPR}$ (degrees)",
        title="SPR Angle vs Sensing Medium RI",
        savepath=os.path.join(RESULTS_DIR, "sweep_sensing_ri.png"),
    )
    print(f"  Saved: sweep_sensing_ri.png")


def main():
    print("\n" + "#" * 72)
    print("#  SPR Biosensor — Extended Parameter Analysis")
    print("#  TMM at 633 nm, SF11 prism, Ag film")
    print("#" * 72)

    sweep_ag_thickness()
    sweep_mos2_layers()
    sweep_graphene_layers()
    sweep_sensing_ri()

    header("COMPLETE")
    print(f"  Results in: {os.path.abspath(RESULTS_DIR)}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if "sweep" in f:
            sz = os.path.getsize(os.path.join(RESULTS_DIR, f)) / 1024
            print(f"    {f} ({sz:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
