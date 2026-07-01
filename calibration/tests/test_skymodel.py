"""Unit tests for the ENU sky model — adapted from tests/test_skymodel.py and tests/test_beam.py."""

import numpy as np
from tart_stefcal import airy_power_beam, enu_direction_cosines, model_visibilities
from tart_stefcal.skymodel import LIGHTSPEED

_ZENITH = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# enu_direction_cosines
# ---------------------------------------------------------------------------


def test_enu_direction_cosines_cardinal_points():
    np.testing.assert_allclose(enu_direction_cosines(0.0, np.pi / 2), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(enu_direction_cosines(np.pi / 2, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(enu_direction_cosines(0.0, 0.0), [0.0, 1.0, 0.0], atol=1e-12)


def test_enu_direction_cosines_unit_norm_and_shape():
    az = np.array([0.1, 1.0, 2.0, 3.0])
    el = np.array([0.2, 0.5, 0.9, 1.2])
    s = enu_direction_cosines(az, el)
    assert s.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(s, axis=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# model_visibilities
# ---------------------------------------------------------------------------


def test_model_visibilities_single_source_analytic_fringe():
    bl = np.array([[10.0, 0.0, 0.0]])
    s = enu_direction_cosines(np.array([np.pi / 2]), np.array([0.0]))
    freqs = np.array([1.5e9])
    mvis = model_visibilities(s, bl, freqs)
    expected = np.exp(2j * np.pi * (1.5e9 / LIGHTSPEED) * 10.0)
    assert mvis.shape == (1, 1)
    np.testing.assert_allclose(mvis[0, 0], expected, rtol=1e-10)


def test_model_visibilities_is_flux_one_superposition():
    rng = np.random.default_rng(0)
    bl = rng.uniform(-5, 5, size=(7, 3))
    az = rng.uniform(0, 2 * np.pi, 4)
    el = rng.uniform(0.1, 1.5, 4)
    s = enu_direction_cosines(az, el)
    freqs = np.array([1.575e9])
    mvis = model_visibilities(s, bl, freqs)
    per_src = np.stack([model_visibilities(s[i : i + 1], bl, freqs)[:, 0] for i in range(4)], axis=0).sum(axis=0)
    np.testing.assert_allclose(mvis[:, 0], per_src, rtol=1e-10)


def test_model_visibilities_beam_weights_each_source():
    rng = np.random.default_rng(2)
    bl = rng.uniform(-5, 5, size=(5, 3))
    s = enu_direction_cosines(np.array([0.3, 1.2, 2.5]), np.array([0.4, 0.8, 1.3]))
    freqs = np.array([1.575e9])
    beam = np.array([[0.0, 1.0, 0.5]])
    got = model_visibilities(s, bl, freqs, beam=beam)
    want = sum(beam[0, i] * model_visibilities(s[i : i + 1], bl, freqs)[:, 0] for i in range(3))
    np.testing.assert_allclose(got[:, 0], want, rtol=1e-10)


def test_model_visibilities_beam_none_equals_ones():
    rng = np.random.default_rng(3)
    bl = rng.uniform(-5, 5, size=(4, 3))
    s = enu_direction_cosines(np.array([0.1, 2.0]), np.array([0.5, 1.0]))
    freqs = np.array([1.575e9])
    np.testing.assert_allclose(
        model_visibilities(s, bl, freqs),
        model_visibilities(s, bl, freqs, beam=np.ones((1, 2))),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# airy_power_beam
# ---------------------------------------------------------------------------


def test_beam_peak_at_zenith():
    s = enu_direction_cosines(np.array([0.0]), np.array([np.pi / 2]))
    freqs = np.array([1.575e9])
    beam = airy_power_beam(s, _ZENITH, freqs)
    assert beam.shape == (1, 1)
    np.testing.assert_allclose(beam[0, 0], 1.0, atol=1e-12)


def test_beam_below_horizon_is_zero():
    s = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    freqs = np.array([1.575e9])
    beam = airy_power_beam(s, _ZENITH, freqs)
    assert beam[0, 0] == 0.0
    assert beam[0, 1] == 0.0
    assert beam[0, 2] > 0.0
    assert beam[0, 3] == 1.0


def test_beam_low_elevation_downweighted():
    s = enu_direction_cosines(np.array([0.0, 0.0]), np.array([np.radians(85.0), np.radians(20.0)]))
    freqs = np.array([1.575e9])
    beam = airy_power_beam(s, _ZENITH, freqs)
    assert beam[0, 0] > beam[0, 1]


def test_beam_boresight_normalised_internally():
    s = enu_direction_cosines(np.array([0.1, 0.5]), np.array([0.3, 0.8]))
    freqs = np.array([1.575e9])
    unit = np.array([0.0, 0.0, 1.0])
    scaled = 7.3 * unit
    np.testing.assert_allclose(
        airy_power_beam(s, scaled, freqs),
        airy_power_beam(s, unit, freqs),
        rtol=1e-12,
        atol=1e-12,
    )
