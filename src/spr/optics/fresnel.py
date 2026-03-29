"""Fresnel coefficients and Snell's law for multilayer optics.

Implements p-polarized (TM) Fresnel reflection/transmission at a single
interface, and the generalized Snell's law for complex refractive indices.

Reference:
    Born & Wolf, "Principles of Optics", 7th ed.
    Byrnes, "Multilayer optical calculations", arXiv:1603.02720
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def snell_cos(n_i: complex, n_t: complex, cos_theta_i: ArrayLike) -> np.ndarray:
    """Compute cos(theta_t) in transmission medium via Snell's law.

    Uses the relation: n_i * sin(theta_i) = n_t * sin(theta_t)
    Rewritten as: cos(theta_t) = sqrt(1 - (n_i/n_t)^2 * sin^2(theta_i))

    For complex refractive indices, the branch cut of sqrt is chosen so
    that Im(cos_theta_t) >= 0, ensuring evanescent waves decay.

    Parameters
    ----------
    n_i : complex
        Refractive index of incident medium.
    n_t : complex
        Refractive index of transmission medium.
    cos_theta_i : array_like
        Cosine of incidence angle (may be complex).

    Returns
    -------
    np.ndarray
        Cosine of refracted angle (complex).
    """
    cos_theta_i = np.asarray(cos_theta_i, dtype=complex)
    sin2_theta_i = 1.0 - cos_theta_i**2
    sin2_theta_t = (n_i / n_t) ** 2 * sin2_theta_i
    cos_theta_t = np.sqrt(1.0 - sin2_theta_t)
    # Ensure correct branch: Im(cos_theta_t) >= 0 for evanescent decay
    cos_theta_t = np.where(
        np.imag(cos_theta_t) < 0, -cos_theta_t, cos_theta_t
    )
    return cos_theta_t


def fresnel_rp(
    n_i: complex, n_t: complex, cos_theta_i: ArrayLike, cos_theta_t: ArrayLike
) -> np.ndarray:
    """Fresnel reflection coefficient for p-polarization (TM).

    r_p = (n_t * cos_theta_i - n_i * cos_theta_t) /
          (n_t * cos_theta_i + n_i * cos_theta_t)

    Parameters
    ----------
    n_i, n_t : complex
        Refractive indices of incident and transmission media.
    cos_theta_i, cos_theta_t : array_like
        Cosines of incidence and refraction angles.

    Returns
    -------
    np.ndarray
        Complex p-polarization reflection coefficient.
    """
    cos_i = np.asarray(cos_theta_i, dtype=complex)
    cos_t = np.asarray(cos_theta_t, dtype=complex)
    num = n_t * cos_i - n_i * cos_t
    den = n_t * cos_i + n_i * cos_t
    return num / den
