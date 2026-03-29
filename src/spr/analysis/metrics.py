"""SPR performance metrics: resonance angle, FWHM, sensitivity, FOM.

Provides refined resonance-angle estimation (parabolic fit around minimum),
FWHM computation, sensitivity in deg/RIU, and figure of merit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


@dataclass
class CurveAnalysis:
    """Results of analyzing an SPR reflectance curve.

    Attributes
    ----------
    theta_spr : float
        Resonance angle in degrees (refined).
    r_min : float
        Minimum reflectance.
    fwhm : float
        Full width at half maximum in degrees, or NaN if not computable.
    """

    theta_spr: float
    r_min: float
    fwhm: float


def find_resonance(
    angles_deg: np.ndarray,
    reflectance: np.ndarray,
) -> tuple[float, float]:
    """Find SPR resonance angle with sub-grid refinement.

    Uses two-stage approach:
    1. Discrete argmin to locate approximate minimum.
    2. Cubic spline interpolation + bounded scalar minimization for
       sub-step refinement.

    Parameters
    ----------
    angles_deg : ndarray
        Angle array in degrees.
    reflectance : ndarray
        Reflectance values.

    Returns
    -------
    theta_spr : float
        Refined resonance angle in degrees.
    r_min : float
        Minimum reflectance value.
    """
    idx_min = int(np.argmin(reflectance))

    # Bracket for refinement: ±2 degrees around discrete minimum
    step = angles_deg[1] - angles_deg[0] if len(angles_deg) > 1 else 0.01
    bracket_half = max(2.0, 20 * step)
    lo = max(angles_deg[0], angles_deg[idx_min] - bracket_half)
    hi = min(angles_deg[-1], angles_deg[idx_min] + bracket_half)

    # Select data in bracket
    mask = (angles_deg >= lo) & (angles_deg <= hi)
    a_bracket = angles_deg[mask]
    r_bracket = reflectance[mask]

    if len(a_bracket) < 4:
        # Fallback: return discrete minimum
        return float(angles_deg[idx_min]), float(reflectance[idx_min])

    # Cubic spline interpolation
    cs = CubicSpline(a_bracket, r_bracket)
    result = minimize_scalar(
        lambda x: float(cs(x)),
        bounds=(float(a_bracket[0]), float(a_bracket[-1])),
        method="bounded",
    )
    theta_spr = float(result.x)
    r_min = float(cs(theta_spr))

    return theta_spr, r_min


def compute_fwhm(
    angles_deg: np.ndarray,
    reflectance: np.ndarray,
    theta_spr: float | None = None,
    r_min: float | None = None,
) -> float:
    """Compute full width at half maximum of the SPR dip.

    The "half maximum" level is defined as:
        R_half = R_min + 0.5 * (R_baseline - R_min)
    where R_baseline is the maximum reflectance near the dip.

    Parameters
    ----------
    angles_deg : ndarray
    reflectance : ndarray
    theta_spr : float, optional
        Pre-computed resonance angle; found if not given.
    r_min : float, optional
        Pre-computed minimum reflectance.

    Returns
    -------
    float
        FWHM in degrees, or NaN if the half-max crossings cannot be found.
    """
    if theta_spr is None or r_min is None:
        theta_spr, r_min = find_resonance(angles_deg, reflectance)

    # Use reflectance at the scan boundaries as baseline
    r_baseline = max(reflectance[0], reflectance[-1])
    r_half = r_min + 0.5 * (r_baseline - r_min)

    idx_spr = int(np.argmin(np.abs(angles_deg - theta_spr)))

    # Find left crossing
    left_angle = np.nan
    for i in range(idx_spr, 0, -1):
        if reflectance[i - 1] >= r_half >= reflectance[i]:
            # Linear interpolation
            frac = (r_half - reflectance[i]) / (
                reflectance[i - 1] - reflectance[i]
            )
            left_angle = angles_deg[i] - frac * (
                angles_deg[i] - angles_deg[i - 1]
            )
            break

    # Find right crossing
    right_angle = np.nan
    for i in range(idx_spr, len(reflectance) - 1):
        if reflectance[i] <= r_half <= reflectance[i + 1]:
            frac = (r_half - reflectance[i]) / (
                reflectance[i + 1] - reflectance[i]
            )
            right_angle = angles_deg[i] + frac * (
                angles_deg[i + 1] - angles_deg[i]
            )
            break

    if np.isnan(left_angle) or np.isnan(right_angle):
        return np.nan

    return right_angle - left_angle


def compute_sensitivity_deg_per_RIU(
    delta_theta_deg: float, delta_n: float
) -> float:
    """Sensitivity in degrees per RIU.

    S = delta_theta_SPR / delta_n

    Parameters
    ----------
    delta_theta_deg : float
        Change in resonance angle (degrees).
    delta_n : float
        Change in sensing-medium refractive index.

    Returns
    -------
    float
        Sensitivity in deg/RIU.
    """
    if abs(delta_n) < 1e-15:
        return 0.0
    return delta_theta_deg / delta_n


def compute_fom(sensitivity: float, fwhm: float) -> float:
    """Figure of Merit = Sensitivity / FWHM.

    Parameters
    ----------
    sensitivity : float
        In deg/RIU.
    fwhm : float
        In degrees.

    Returns
    -------
    float
        FOM in 1/RIU, or NaN if FWHM is zero/NaN.
    """
    if np.isnan(fwhm) or fwhm < 1e-10:
        return np.nan
    return sensitivity / fwhm


def analyze_curve(
    angles_deg: np.ndarray,
    reflectance: np.ndarray,
) -> CurveAnalysis:
    """Full analysis of an SPR reflectance curve.

    Returns resonance angle, minimum reflectance, and FWHM.
    """
    theta_spr, r_min = find_resonance(angles_deg, reflectance)
    fwhm = compute_fwhm(angles_deg, reflectance, theta_spr, r_min)
    return CurveAnalysis(theta_spr=theta_spr, r_min=r_min, fwhm=fwhm)
