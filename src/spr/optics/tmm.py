"""Transfer Matrix Method for multilayer thin-film optics (p-polarization).

Computes p-polarized (TM) reflectance for a multilayer stack in the
Kretschmann (ATR) configuration using the interface/propagation transfer
matrix approach.

This implementation follows the algorithm of the Byrnes tmm package
(arXiv:1603.02720) and has been validated against it.

For N layers indexed 0..N-1:
  - Layer 0: semi-infinite incident medium (prism)
  - Layer N-1: semi-infinite substrate (sensing medium)
  - Layers 1..N-2: thin films with given thicknesses

Each intermediate layer i (1 <= i <= N-2) has a combined matrix:

  M_i = (1/t_{i,i+1}) * [[exp(-j*delta_i), 0           ],
                          [0,               exp(j*delta_i)]]
        @ [[1,         r_{i,i+1}],
           [r_{i,i+1}, 1        ]]

where:
  delta_i = (2*pi/lambda) * n_i * d_i * cos(theta_i)
  r_{i,i+1} and t_{i,i+1} are p-pol Fresnel coefficients at the i/(i+1) interface

The total system matrix is:
  Mtilde = I_{0,1} * M_1 * M_2 * ... * M_{N-2}

where I_{0,1} = (1/t_{0,1}) * [[1, r_{0,1}], [r_{0,1}, 1]]

Reflection coefficient: r = Mtilde[1,0] / Mtilde[0,0]

References
----------
Byrnes, "Multilayer optical calculations", arXiv:1603.02720
Born & Wolf, "Principles of Optics", 7th ed.
Habib et al., Int. J. Natural Sciences Research, 7(1), 1-9, 2019
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from spr.optics.fresnel import snell_cos


def _interface_r_p(n_i, n_f, cos_th_i, cos_th_f):
    """p-polarization Fresnel reflection coefficient."""
    return (n_f * cos_th_i - n_i * cos_th_f) / (n_f * cos_th_i + n_i * cos_th_f)


def _interface_t_p(n_i, n_f, cos_th_i, cos_th_f):
    """p-polarization Fresnel transmission coefficient."""
    return 2 * n_i * cos_th_i / (n_f * cos_th_i + n_i * cos_th_f)


def transfer_matrix_reflectance(
    n_list: list[complex],
    d_list: list[float],
    theta_i_deg: ArrayLike,
    wavelength: float,
) -> np.ndarray:
    """Compute p-polarized reflectance for a multilayer stack via TMM.

    Validated against Byrnes tmm package (arXiv:1603.02720).

    Parameters
    ----------
    n_list : list of complex
        Refractive indices [n_0, n_1, ..., n_{N-1}].
        n_0 = incident medium (prism), n_{N-1} = substrate (sensing medium).
    d_list : list of float
        Thicknesses in meters. d_0 and d_{N-1} are ignored (semi-infinite).
    theta_i_deg : array_like
        Incidence angles in degrees (in the prism).
    wavelength : float
        Free-space wavelength in meters.

    Returns
    -------
    ndarray
        Reflectance R = |r_p|^2 at each angle.
    """
    theta_i_deg = np.atleast_1d(np.asarray(theta_i_deg, dtype=float))
    N_angles = len(theta_i_deg)
    theta_i_rad = np.deg2rad(theta_i_deg)

    N = len(n_list)
    assert len(d_list) == N
    assert N >= 2

    # Convert to complex
    n_arr = [complex(n) for n in n_list]
    n_0 = n_arr[0]

    # Compute cos(theta) in each layer via Snell's law
    cos_th = [np.cos(theta_i_rad).astype(complex)]
    for k in range(1, N):
        cos_th.append(snell_cos(n_0, n_arr[k], cos_th[0]))

    # Compute phase thickness delta[k] = 2*pi*n_k*d_k*cos(th_k)/lambda
    delta = []
    for k in range(N):
        d_k = float(d_list[k])
        delta_k = 2.0 * np.pi * n_arr[k] * d_k * cos_th[k] / wavelength
        # Clamp imaginary part to avoid overflow for very opaque layers
        delta_k = np.where(
            np.imag(delta_k) > 35,
            np.real(delta_k) + 35j,
            delta_k,
        )
        delta.append(delta_k)

    # Build the combined matrices for intermediate layers (1..N-2)
    # Each M_i = (1/t_{i,i+1}) * Prop_i @ Interface_{i,i+1}
    # Then Mtilde = Interface_{0,1} * M_1 * M_2 * ... * M_{N-2}

    # Start with interface 0->1
    r_01 = _interface_r_p(n_arr[0], n_arr[1], cos_th[0], cos_th[1])
    t_01 = _interface_t_p(n_arr[0], n_arr[1], cos_th[0], cos_th[1])
    inv_t_01 = 1.0 / t_01

    # Mtilde = (1/t_01) * [[1, r_01], [r_01, 1]]
    Mt00 = inv_t_01 * np.ones(N_angles, dtype=complex)
    Mt01 = inv_t_01 * r_01
    Mt10 = inv_t_01 * r_01
    Mt11 = inv_t_01 * np.ones(N_angles, dtype=complex)

    # Multiply by M_i for i = 1..N-2
    for i in range(1, N - 1):
        # Propagation through layer i
        exp_neg = np.exp(-1j * delta[i])
        exp_pos = np.exp(1j * delta[i])

        # Interface i -> i+1
        r_ip1 = _interface_r_p(n_arr[i], n_arr[i + 1], cos_th[i], cos_th[i + 1])
        t_ip1 = _interface_t_p(n_arr[i], n_arr[i + 1], cos_th[i], cos_th[i + 1])
        inv_t_ip1 = 1.0 / t_ip1

        # M_i = (1/t_{i,i+1}) * [[exp(-j*d_i), 0], [0, exp(j*d_i)]]
        #                       @ [[1, r_{i,i+1}], [r_{i,i+1}, 1]]
        # = (1/t) * [[exp(-jd)*1,        exp(-jd)*r],
        #            [exp(jd)*r,          exp(jd)*1 ]]
        Mi00 = inv_t_ip1 * exp_neg
        Mi01 = inv_t_ip1 * exp_neg * r_ip1
        Mi10 = inv_t_ip1 * exp_pos * r_ip1
        Mi11 = inv_t_ip1 * exp_pos

        # Mtilde = Mtilde @ M_i
        new_00 = Mt00 * Mi00 + Mt01 * Mi10
        new_01 = Mt00 * Mi01 + Mt01 * Mi11
        new_10 = Mt10 * Mi00 + Mt11 * Mi10
        new_11 = Mt10 * Mi01 + Mt11 * Mi11
        Mt00, Mt01, Mt10, Mt11 = new_00, new_01, new_10, new_11

    # Reflection coefficient
    r_p = Mt10 / Mt00
    R = np.abs(r_p) ** 2
    return R


def angular_scan(
    n_list: list[complex],
    d_list: list[float],
    wavelength: float,
    angle_start: float = 40.0,
    angle_end: float = 90.0,
    angle_step: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Run angular scan and return (angles_deg, reflectance).

    Parameters
    ----------
    n_list : list of complex
        Refractive indices of all layers.
    d_list : list of float
        Thicknesses of all layers in meters.
    wavelength : float
        Free-space wavelength in meters.
    angle_start, angle_end : float
        Scan range in degrees.
    angle_step : float
        Angular resolution in degrees.

    Returns
    -------
    angles : ndarray
        Angle array in degrees.
    R : ndarray
        Reflectance at each angle.
    """
    angles = np.arange(angle_start, angle_end + angle_step / 2, angle_step)
    R = transfer_matrix_reflectance(n_list, d_list, angles, wavelength)
    return angles, R
