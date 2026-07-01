"""Unit tests for the acquisition StEFCal solver — adapted from tests/test_stefcal.py."""

import numpy as np
import pytest
from tart_stefcal import enu_direction_cosines, model_visibilities, referenced_phases, stefcal_solve

_FREQS = np.array([1.575e9])
_N_ANT = 24


def _layout(seed=0):
    """A planar 24-antenna ENU layout and its baseline antenna-index arrays."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1.0, 1.0, size=(_N_ANT, 3))
    pos[:, 2] = 0.0  # TART elements are coplanar (Up = 0)
    a1, a2 = np.triu_indices(_N_ANT, k=1)
    bl = pos[a1] - pos[a2]
    return a1, a2, bl


def _sky(seed=1, nsrc=30):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0.0, 2.0 * np.pi, nsrc)
    el = rng.uniform(np.radians(20.0), np.radians(90.0), nsrc)
    return enu_direction_cosines(az, el)


def _true_gains(seed=2):
    rng = np.random.default_rng(seed)
    amp = rng.uniform(0.5, 2.0, _N_ANT)
    phase = rng.uniform(-np.pi, np.pi, _N_ANT)
    return amp * np.exp(1j * phase)


def _synth(g, model, a1, a2, noise=0.0, seed=3):
    """One integration: (nbl, nchan) visibilities from a model and constant gains."""
    vis = (g[a1] * model[:, 0] * np.conj(g[a2]))[:, None]
    if noise:
        rng = np.random.default_rng(seed)
        vis = vis + noise * (rng.standard_normal(vis.shape) + 1j * rng.standard_normal(vis.shape))
    return vis


def _model_series(bl, ntime, seed=5, nsrc=30):
    """(ntime, nbl, nchan) model with the sources at a DIFFERENT position each integration."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(ntime):
        az = rng.uniform(0.0, 2.0 * np.pi, nsrc)
        el = rng.uniform(np.radians(20.0), np.radians(90.0), nsrc)
        out.append(model_visibilities(enu_direction_cosines(az, el), bl, _FREQS))
    return np.stack(out)  # (ntime, nbl, nchan)


# ---------------------------------------------------------------------------
# StEFCal solver tests
# ---------------------------------------------------------------------------


def test_stefcal_recovers_gains_noiseless():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    gains, info = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)
    assert gains.shape == (1, _N_ANT)
    assert bool(info["converged"][0])
    g_hat = gains[0]
    g_true_ref = g_true / g_true[0]
    np.testing.assert_allclose(g_hat, g_true_ref, atol=1e-6)
    np.testing.assert_allclose(g_hat[0], 1.0 + 0.0j, atol=1e-12)


def test_referenced_phases_match_truth_up_to_gauge():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    g_hat = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)[0][0]
    got = referenced_phases(g_hat, 0)
    want = np.angle(g_true) - np.angle(g_true[0])
    np.testing.assert_allclose(np.exp(1j * got), np.exp(1j * want), atol=1e-6)


def test_gauge_invariant_to_global_complex_scale():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    c = 0.7 * np.exp(1j * 1.1)
    vis_a = _synth(g_true, model, a1, a2)
    vis_b = _synth(c * g_true, model, a1, a2)
    pa = referenced_phases(stefcal_solve(vis_a, model, a1, a2, _N_ANT, ref_ant=0)[0][0], 0)
    pb = referenced_phases(stefcal_solve(vis_b, model, a1, a2, _N_ANT, ref_ant=0)[0][0], 0)
    np.testing.assert_allclose(np.exp(1j * pa), np.exp(1j * pb), atol=1e-6)


def test_stefcal_flags_dead_antenna():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2)
    dead = 5
    touch = (a1 == dead) | (a2 == dead)
    weight = np.ones((a1.size, 1))
    weight[touch] = 0.0
    vis[touch] = 0.0
    g_hat = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0, weight=weight)[0][0]
    assert np.isnan(g_hat[dead])
    live = np.arange(_N_ANT) != dead
    assert np.all(np.isfinite(g_hat[live]))
    np.testing.assert_allclose(g_hat[live], (g_true / g_true[0])[live], atol=1e-6)


def test_stefcal_recovers_with_noise():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    g_true = _true_gains()
    vis = _synth(g_true, model, a1, a2, noise=1e-3)
    g_hat = stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0)[0][0]
    np.testing.assert_allclose(g_hat, g_true / g_true[0], atol=5e-3)


def test_stefcal_ref_dead_raises():
    a1, a2, bl = _layout()
    model = model_visibilities(_sky(), bl, _FREQS)
    vis = _synth(_true_gains(), model, a1, a2)
    touch = (a1 == 0) | (a2 == 0)
    weight = np.ones((a1.size, 1))
    weight[touch] = 0.0
    with pytest.raises(ValueError):
        stefcal_solve(vis, model, a1, a2, _N_ANT, ref_ant=0, weight=weight)


def test_stefcal_t_int_pools_moving_sources():
    a1, a2, bl = _layout()
    ntime = 8
    models = _model_series(bl, ntime)
    g_true = _true_gains()
    vis = g_true[a1][None, :, None] * models * np.conj(g_true[a2])[None, :, None]
    gains, info = stefcal_solve(vis, models, a1, a2, _N_ANT, t_int=ntime, ref_ant=0)
    assert gains.shape == (1, _N_ANT)
    np.testing.assert_allclose(gains[0], g_true / g_true[0], atol=1e-6)
    np.testing.assert_allclose(stefcal_solve(vis, models, a1, a2, _N_ANT, ref_ant=0)[0], gains, atol=1e-10)


def test_stefcal_t_int_sets_solution_cadence():
    a1, a2, bl = _layout()
    ntime = 6
    models = _model_series(bl, ntime, seed=9)
    g_true = _true_gains()
    vis = g_true[a1][None, :, None] * models * np.conj(g_true[a2])[None, :, None]
    assert stefcal_solve(vis, models, a1, a2, _N_ANT, t_int=1, ref_ant=0)[0].shape == (6, _N_ANT)
    assert stefcal_solve(vis, models, a1, a2, _N_ANT, t_int=2, ref_ant=0)[0].shape == (3, _N_ANT)
    assert stefcal_solve(vis, models, a1, a2, _N_ANT, t_int=4, ref_ant=0)[0].shape == (2, _N_ANT)
    assert stefcal_solve(vis, models, a1, a2, _N_ANT, t_int=None, ref_ant=0)[0].shape == (1, _N_ANT)
    g_each = stefcal_solve(vis, models, a1, a2, _N_ANT, t_int=1, ref_ant=0)[0]
    for k in range(ntime):
        np.testing.assert_allclose(g_each[k], g_true / g_true[0], atol=1e-6)
