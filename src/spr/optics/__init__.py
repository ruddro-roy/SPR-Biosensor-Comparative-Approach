"""Optical simulation modules."""

from spr.optics.tmm import transfer_matrix_reflectance, angular_scan
from spr.optics.fresnel import fresnel_rp, snell_cos

__all__ = [
    "transfer_matrix_reflectance",
    "angular_scan",
    "fresnel_rp",
    "snell_cos",
]
