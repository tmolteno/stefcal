"""Unit-flux sky model in the local ENU frame.

Both functions are ``xp``-injectable (``xp=numpy`` on CPU, ``xp=cupy`` on GPU).
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

LIGHTSPEED = 299792458.0


def enu_direction_cosines(az, el, *, xp: ModuleType = np):
    """ENU unit vectors for source azimuth/elevation.

    Args:
        az: azimuth in radians, measured from North toward East. Scalar or ``(nsrc,)``.
        el: elevation (altitude above the horizon) in radians. Scalar or ``(nsrc,)``.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(..., 3)`` array of ``(East, North, Up)`` unit vectors; ``(3,)`` for scalar inputs.
    """
    az = xp.asarray(az)
    el = xp.asarray(el)
    cos_el = xp.cos(el)
    east = xp.sin(az) * cos_el
    north = xp.cos(az) * cos_el
    up = xp.sin(el)
    return xp.stack([east, north, up], axis=-1)


def model_visibilities(s_enu, bl_enu, freqs, *, beam=None, xp: ModuleType = np):
    """Model visibilities ``M_pq = sum_s B_s exp(2*pi*i*(nu/c)*b_pq . s_s)``.

    Every source is at unit flux; the optional ``beam`` supplies a per-source (per-channel) real
    weight that stands in for the apparent-flux modulation of the primary beam.

    Args:
        s_enu: ``(nsrc, 3)`` source ENU unit vectors (e.g. from :func:`enu_direction_cosines`).
        bl_enu: ``(nbl, 3)`` ENU baseline vectors in metres.
        freqs: ``(nchan,)`` frequencies in Hz.
        beam: optional ``(nchan, nsrc)`` real per-source weight; ``None`` -> all ones.
        xp: array module.

    Returns:
        ``(nbl, nchan)`` complex model visibilities.
    """
    s_enu = xp.asarray(s_enu)
    bl_enu = xp.asarray(bl_enu)
    inv_wl = xp.asarray(freqs) / LIGHTSPEED  # (nchan,) cycles per metre
    delay = bl_enu @ s_enu.T  # (nbl, nsrc) metres
    phase = 2.0 * xp.pi * inv_wl[None, :, None] * delay[:, None, :]  # (nbl, nchan, nsrc)
    kernel = xp.exp(1j * phase)  # (nbl, nchan, nsrc)
    if beam is not None:
        kernel = kernel * xp.asarray(beam)[None, :, :]
    return kernel.sum(axis=-1)  # (nbl, nchan)
