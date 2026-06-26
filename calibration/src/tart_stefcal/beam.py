"""Airy primary beam for the TART GPS patch antenna.

Evaluated per source direction (rather than the full HEALPix grid used in kremetart).
The GPS patch antenna's ground plane (125 mm diameter) produces an Airy diffraction
pattern; this module computes the power beam at each source direction.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j1 as _scipy_j1

LIGHTSPEED = 299792458.0
GROUND_PLANE_DIAMETER = 0.125  # metres


def airy_power_beam(s_enu, boresight, freqs, *, diameter: float = GROUND_PLANE_DIAMETER):
    """Airy power beam evaluated at source directions.

    Computes ``B(theta) = [2 J1(x) / x]**2`` with ``x = (pi * D / lambda) * sin(theta)``
    and ``cos(theta) = s_enu · boresight``, normalised to 1 at boresight, zeroed below the
    local horizon.

    Args:
        s_enu: ``(nsrc, 3)`` source ENU unit vectors.
        boresight: ``(3,)`` antenna-zenith unit vector (ENU zenith is ``(0, 0, 1)``).
        freqs: ``(nchan,)`` frequencies in Hz.
        diameter: aperture (ground plane) diameter in metres.

    Returns:
        ``(nchan, nsrc)`` real power beam, peak 1 at boresight, 0 below the horizon.
    """
    s_enu = np.asarray(s_enu)
    boresight = np.asarray(boresight)
    boresight = boresight / np.linalg.norm(boresight)

    mu = np.clip(s_enu @ boresight, -1.0, 1.0)  # cos(theta), (nsrc,)
    sinth = np.sqrt(1.0 - mu**2)  # sin(theta), (nsrc,)

    inv_wl = np.asarray(freqs) / LIGHTSPEED  # (nchan,) cycles per metre
    x = np.pi * diameter * inv_wl[:, None] * sinth[None, :]  # (nchan, nsrc)

    # Airy voltage 2 J1(x) / x, with x -> 0 limit A(0) = 1
    safe_x = np.where(x == 0.0, 1.0, x)
    amp = np.where(x == 0.0, 1.0, 2.0 * _scipy_j1(safe_x) / safe_x)
    beam = amp**2

    return np.where(mu[None, :] >= 0.0, beam, 0.0)
