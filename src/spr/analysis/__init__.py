"""Analysis and metrics modules."""

from spr.analysis.metrics import (
    find_resonance,
    compute_fwhm,
    compute_sensitivity_deg_per_RIU,
    compute_fom,
    analyze_curve,
)

__all__ = [
    "find_resonance",
    "compute_fwhm",
    "compute_sensitivity_deg_per_RIU",
    "compute_fom",
    "analyze_curve",
]
