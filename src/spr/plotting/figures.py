"""Publication-quality figure generation for SPR biosensor study.

Provides consistent Matplotlib styling and reusable plotting functions
for reflectance curves, sensitivity comparisons, and parameter sweeps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Color palette (colorblind-friendly, publication-ready)
# ---------------------------------------------------------------------------
COLORS = {
    "conventional": "#1f77b4",  # blue
    "graphene": "#2ca02c",  # green
    "mos2_graphene": "#d62728",  # red
    "dna_probe": "#000000",  # black
    "dna_1000": "#1f77b4",
    "dna_1001": "#2ca02c",
    "dna_1010": "#ff7f0e",
    "dna_1100": "#d62728",
}


def setup_matplotlib() -> None:
    """Configure Matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "axes.grid": False,
        }
    )


def plot_reflectance_curves(
    curves: list[tuple[np.ndarray, np.ndarray, str, str]],
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (0, 1.05),
    markers: list[tuple[float, float, str]] | None = None,
    savepath: str | Path | None = None,
    figsize: tuple[float, float] = (8, 5.5),
) -> plt.Figure:
    """Plot one or more reflectance-vs-angle curves.

    Parameters
    ----------
    curves : list of (angles, reflectance, label, color)
    title : str
    xlim, ylim : axis limits
    markers : list of (angle, reflectance, label) for SPR angle annotations
    savepath : path to save figure (PNG)
    figsize : figure dimensions

    Returns
    -------
    matplotlib Figure
    """
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for angles, refl, label, color in curves:
        ax.plot(angles, refl, color=color, label=label)

    if markers:
        for theta, r_val, label in markers:
            ax.axvline(x=theta, color="gray", linestyle=":", alpha=0.4, linewidth=0.7)
            ax.annotate(
                f"$\\theta_{{SPR}}$={theta:.1f}°",
                xy=(theta, r_val),
                xytext=(theta + 0.5, r_val + 0.12),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    ax.set_xlabel("Angle of Incidence (degrees)")
    ax.set_ylabel("Reflectance")
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="best", framealpha=0.9)

    fig.tight_layout()
    if savepath:
        fig.savefig(str(savepath))
        plt.close(fig)
    return fig


def plot_sensitivity_bars(
    labels: list[str],
    deltas: list[float],
    colors: list[str],
    title: str = "SPR Angle Shift Comparison",
    savepath: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of SPR angle shifts."""
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    bars = ax.bar(labels, deltas, color=colors, width=0.45, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}°",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("$\\Delta\\theta_{SPR}$ (degrees)")
    ax.set_title(title)
    ax.set_ylim(0, max(deltas) * 1.35 if max(deltas) > 0 else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(str(savepath))
        plt.close(fig)
    return fig


def plot_parameter_sweep(
    x_values: np.ndarray,
    y_dict: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str = "",
    savepath: str | Path | None = None,
) -> plt.Figure:
    """Line plot for parameter sweep results."""
    setup_matplotlib()
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    for label, y_vals in y_dict.items():
        ax.plot(x_values, y_vals, "o-", markersize=4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    if savepath:
        fig.savefig(str(savepath))
        plt.close(fig)
    return fig
