"""Acquisition StEFCal: alternating per-antenna complex-gain least squares.

The cold-start solver of the Stage-1 calibration operator. Holding all other antenna
gains fixed, ``V_pq = g_p M_pq conj(g_q)`` is linear in ``g_p``; alternating that antenna-by-antenna
gives a robust, phase-wrap-free cold start. The per-antenna normal equations are diagonal (a scalar
Hessian ``H_p = sum_q w |z_pq|^2`` per antenna, with ``z_pq = M_pq conj(g_q)``), so the "inversion"
is a scalar divide -- no matrix solve. A solution pools ``t_int`` consecutive integrations
(the sources move between them, so the model varies per integration while one gain per antenna is
solved). ``xp``-injectable (``xp=numpy`` CPU / ``xp=cupy`` GPU); the per-antenna reduction is a
segment sum over the bipartite baseline-antenna incidence.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np

_NAN_C = complex(float("nan"), float("nan"))


def _seg_sum_complex(idx, vals, n: int, *, xp: ModuleType):
    """Per-segment complex sum: ``out[k] = sum(vals[idx == k])`` over ``n`` segments."""
    real = xp.bincount(idx, weights=vals.real, minlength=n)
    imag = xp.bincount(idx, weights=vals.imag, minlength=n)
    return real + 1j * imag


def _broadcast_weight(weight, ntime: int, nbl: int, nchan: int, *, xp: ModuleType):
    """Broadcast a weight (``None`` / ``(nbl,)`` / ``(nbl, nchan)`` / ``(ntime, nbl, nchan)``)."""
    if weight is None:
        return xp.ones((ntime, nbl, nchan))
    w = xp.asarray(weight)
    if w.ndim == 1:  # (nbl,)
        w = w[None, :, None]
    elif w.ndim == 2:  # (nbl, nchan)
        w = w[None, :, :]
    return xp.broadcast_to(w, (ntime, nbl, nchan))


def _solve_interval(vis, model, p_idx, q_idx, n_ant, ref_ant, weight, g0, max_iter, tol, *, xp):
    """Solve one gain vector pooling all integrations in ``vis``/``model`` (each ``(nt, nbl, nchan)``).

    Returns ``(gains (n_ant,), iterations, converged, max_change)``. Gains are gauged to
    ``gains[ref_ant] == 1`` with dead antennas ``NaN``; an interval in which ``ref_ant`` is itself
    dead yields all-``NaN`` gains and ``converged=False``.
    """
    vdir = xp.concatenate([vis, xp.conj(vis)], axis=1)  # (nt, 2*nbl, nchan)
    mdir = xp.concatenate([model, xp.conj(model)], axis=1)
    wdir = xp.concatenate([weight, weight], axis=1)

    deg = xp.bincount(p_idx, weights=wdir.sum(axis=(0, 2)), minlength=n_ant)  # total weight per antenna
    live = deg > 0
    if not bool(live[ref_ant]):
        return xp.full(n_ant, _NAN_C), 0, False, float("inf")

    g = xp.where(xp.isfinite(g0), g0, 1.0).astype(complex)
    converged = False
    change = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        z = mdir * xp.conj(g[q_idx])[None, :, None]  # (nt, 2*nbl, nchan): V_dir = g[p] * z
        num = _seg_sum_complex(p_idx, (wdir * xp.conj(z) * vdir).sum(axis=(0, 2)), n_ant, xp=xp)
        den = xp.bincount(p_idx, weights=(wdir * (z.real**2 + z.imag**2)).sum(axis=(0, 2)), minlength=n_ant)
        g_new = xp.where(den > 0, num / xp.where(den > 0, den, 1.0), g)  # scalar Hessian inverse
        if it % 2 == 0:
            g_new = 0.5 * (g_new + g)  # StEFCal even-iteration stabiliser
        delta = xp.abs(g_new - g)
        scale = xp.where(xp.abs(g) > 1e-12, xp.abs(g), 1.0)
        change = float(xp.max(xp.where(live, delta / scale, 0.0)))
        g = g_new
        if change < tol:
            converged = True
            break

    g = xp.where(live, g, _NAN_C)
    g = g / g[ref_ant]  # gauge: g_ref -> 1 (live, finite)
    return g, it, converged, change


def stefcal_solve(
    vis,
    model,
    a1,
    a2,
    n_ant: int,
    *,
    t_int: int | None = None,
    ref_ant: int = 0,
    weight=None,
    g0=None,
    max_iter: int = 100,
    tol: float = 1e-8,
    xp: ModuleType = np,
):
    """Solve per-antenna complex gains by alternating least squares (StEFCal), at a ``t_int`` cadence.

    Args:
        vis: ``(ntime, nbl, nchan)`` observed visibilities (a single ``(nbl, nchan)`` frame is
            accepted and treated as ``ntime == 1``).
        model: model visibilities, same shape as ``vis`` (per-integration, since sources move).
        a1: ``(nbl,)`` int antenna index of the first antenna of each baseline.
        a2: ``(nbl,)`` int antenna index of the second antenna of each baseline.
        n_ant: number of antennas.
        t_int: number of consecutive integrations pooled into one gain solution. ``None`` pools all
            ``ntime`` (one solution); ``1`` solves every integration; the trailing partial interval
            is solved from whatever frames remain.
        ref_ant: reference antenna pinned to 1 (fixes both gauges). Must be live over the series.
        weight: optional ``(nbl,)`` / ``(nbl, nchan)`` / ``(ntime, nbl, nchan)`` real weights;
            ``0`` flags a baseline out.
        g0: optional ``(n_ant,)`` complex initial gains (warm start); defaults to unity. Each
            interval also warm-starts from the previous interval's solution.
        max_iter: maximum alternating iterations per interval.
        tol: convergence threshold on the max relative gain change over live antennas.
        xp: array module.

    Returns:
        ``(gains, info)``: ``gains`` is ``(n_sol, n_ant)`` complex (``n_sol = ceil(ntime / t_int)``)
        with ``gains[:, ref_ant] == 1`` and dead antennas ``NaN``; ``info`` has ``iterations``,
        ``converged`` and ``max_change``, each an ``(n_sol,)`` array.

    Raises:
        ValueError: if ``ref_ant`` has no live baselines over the whole series, or ``t_int < 1``.
    """
    vis = xp.asarray(vis)
    model = xp.asarray(model)
    if vis.ndim == 2:  # (nbl, nchan) -> single integration
        vis = vis[None]
        model = model[None]
    ntime, nbl, nchan = vis.shape
    weight = _broadcast_weight(weight, ntime, nbl, nchan, xp=xp)
    a1 = xp.asarray(a1)
    a2 = xp.asarray(a2)
    p_idx = xp.concatenate([a1, a2])  # antenna being solved (forward then reverse direction)
    q_idx = xp.concatenate([a2, a1])  # its partner

    # ref must be live somewhere in the series (per-interval dead-ref is handled by _solve_interval).
    wdir_all = xp.concatenate([weight, weight], axis=1)
    deg = xp.bincount(p_idx, weights=wdir_all.sum(axis=(0, 2)), minlength=n_ant)
    if not bool(deg[ref_ant] > 0):
        raise ValueError(f"ref_ant {ref_ant} has no live baselines")

    t_int = ntime if t_int is None else int(t_int)
    if t_int < 1:
        raise ValueError(f"t_int must be >= 1, got {t_int}")

    g_warm = xp.ones(n_ant, dtype=complex) if g0 is None else xp.asarray(g0).astype(complex)
    gains, iters, conv, chg = [], [], [], []
    for start in range(0, ntime, t_int):
        sl = slice(start, start + t_int)
        g, it, c, ch = _solve_interval(
            vis[sl], model[sl], p_idx, q_idx, n_ant, ref_ant, weight[sl], g_warm, max_iter, tol, xp=xp
        )
        gains.append(g)
        iters.append(it)
        conv.append(c)
        chg.append(ch)
        g_warm = xp.where(xp.isfinite(g), g, 1.0)  # carry the solution forward (dead antennas -> unity)

    info = {"iterations": xp.asarray(iters), "converged": xp.asarray(conv), "max_change": xp.asarray(chg)}
    return xp.stack(gains), info


def referenced_phases(gains, ref_ant: int, *, xp: ModuleType = np):
    """Gauge-referenced gain phases (amplitudes discarded).

    Args:
        gains: ``(..., n_ant)`` complex gains (works for one solution or a ``(n_sol, n_ant)`` stack).
        ref_ant: reference antenna; its phase is subtracted from all phases.
        xp: array module.

    Returns:
        ``(..., n_ant)`` real phases ``angle(g_p) - angle(g_ref)`` in radians (``NaN`` for dead antennas).
    """
    gains = xp.asarray(gains)
    return xp.angle(gains) - xp.angle(gains[..., ref_ant])[..., None]
