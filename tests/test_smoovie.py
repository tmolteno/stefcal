"""Tests for the ``smoovie`` command and its Holoscan imaging app.

``smoovie`` is inherently GPU/Holoscan-driven (mirrors :mod:`kremetart.core.stream_msv4`): there is
no CPU fallback, so the whole suite is gated on a CUDA device plus the cupy/holoscan/healpy stack
(assumed installed via the ``[full]`` extra). On GitHub CPU runners these skip cleanly; run them
locally on a GPU box. If a test segfaults during the holoscan import, raise the stack limit first:
``ulimit -s 32768``. Test data comes from the shared ``hdf_paths`` / ``hdf_dir`` fixtures
(``conftest.py``).

The host-wiring tests use the ``sm_stub`` fixture, which stubs the imaging seam
(``image_via_app``) and the web server (``FrameServer`` + ``_wait_for_interrupt``) so they exercise
``smoovie()``'s orchestration (phase resolution, track computation, serve lifecycle, cache-path
default, nframes, profiling) without running the GPU pipeline or binding a port; the end-to-end
tests run the real Holoscan app.
"""

from pathlib import Path

import pytest


def _gpu() -> bool:
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import healpy  # noqa: F401
        import holoscan  # noqa: F401

        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu(), reason="requires a CUDA device + cupy/holoscan/healpy")


@pytest.fixture
def sm():
    """The smoovie module (imported lazily so collection never needs the GPU stack)."""
    import kremetart.core.smoovie as sm

    return sm


@pytest.fixture
def sm_stub(sm, monkeypatch):
    """smoovie module with the GPU imaging seam + web server stubbed, for host-wiring tests.

    ``image_via_app`` becomes a no-op returning the output path; ``FrameServer`` is a fake that
    records its constructor kwargs (read via ``sm.FrameServer.last``) and never binds a port;
    ``_wait_for_interrupt`` returns immediately so the frozen-session block never hangs the suite.
    Tests add their own tracks / phase patches via ``monkeypatch.setattr(sm_stub, ...)``.
    """

    monkeypatch.setattr(sm, "image_via_app", lambda paths, nside, **k: k.get("output_zarr"))

    class FakeServer:
        last: dict = {}

        def __init__(self, holder, **kwargs):
            FakeServer.last = {"holder": holder, **kwargs}
            self.holder = holder

        def start(self) -> str:
            return "http://localhost:8080/"

        def stop(self) -> None:
            FakeServer.last["stopped"] = True

    monkeypatch.setattr(sm, "FrameServer", FakeServer)
    monkeypatch.setattr(sm, "_wait_for_interrupt", lambda: None)
    return sm


# --- the GPU imaging app -------------------------------------------------------------------------


def test_gpu_operators_import():
    from kremetart.operators.dft_healpix import HealpixDFTOperator
    from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator
    from kremetart.operators.iwp_kalman import IWPKalmanOperator
    from kremetart.operators.web_sink import WebStreamSinkOperator

    assert HealpixDFTOperator and HealpixZarrReaderOperator and HealpixWriterOperator
    assert IWPKalmanOperator and WebStreamSinkOperator


def test_image_via_app_end_to_end(hdf_paths, tmp_path):
    """The real Holoscan app images + filters one frame per sub-integration into a durable zarr."""
    import numpy as np
    import xarray as xr

    from kremetart.core.smoovie import image_via_app

    nside = 8
    npix = 12 * nside * nside
    out = tmp_path / "imaging.zarr"
    result = image_via_app(hdf_paths[:1], nside, output_zarr=out, correct_gains=True, nframes=3)
    assert Path(result) == out and out.exists()
    ds = xr.open_zarr(out)
    for var in ("dirty", "tikhonov", "l1", "filtered", "znorm"):
        assert ds[var].shape == (3, npix)
        assert np.all(np.isfinite(ds[var].values))


# --- smoovie() orchestration (imaging + server stubbed) ------------------------------------------


def test_smoovie_requires_both_phase_components(tmp_path, sm):
    # Validation happens before any data access or server start, so no HDFs are needed.
    with pytest.raises(ValueError, match="both or neither"):
        sm.smoovie(hdf_dir=tmp_path, output=tmp_path / "out.zarr", phase_ra_deg=10.0)


def test_smoovie_honors_explicit_phase_direction(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    def fake_image(paths, nside, **k):
        captured.update(k)
        return k.get("output_zarr")

    def fail_cpd(paths):
        raise AssertionError("common_phase_direction must not be called when phase is supplied")

    monkeypatch.setattr(sm_stub, "image_via_app", fake_image)
    monkeypatch.setattr(sm_stub, "common_phase_direction", fail_cpd)

    sm_stub.smoovie(
        hdf_dir=hdf_dir,
        output=tmp_path / "out.zarr",
        nside=1,
        phase_ra_deg=12.0,
        phase_dec_deg=-20.0,
        serve=False,
    )
    assert captured["phase_ra_deg"] == 12.0 and captured["phase_dec_deg"] == -20.0


def test_smoovie_auto_phase_direction_used(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (123.0, 45.0))

    def fake_image(paths, nside, **k):
        captured.update(k)
        return k.get("output_zarr")

    monkeypatch.setattr(sm_stub, "image_via_app", fake_image)

    sm_stub.smoovie(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, serve=False)
    assert captured["phase_ra_deg"] == 123.0 and captured["phase_dec_deg"] == 45.0


def test_smoovie_overlay_passes_tracks_to_server(tmp_path, hdf_dir, sm_stub, monkeypatch):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    # Patch the name smoovie calls (sm.satellite_tracks) so no API/catalogue read happens.
    monkeypatch.setattr(sm_stub, "satellite_tracks", lambda paths, elev, **k: {"SAT": [(0, 1.0, 2.0, 1.0)]})

    sm_stub.smoovie(
        hdf_dir=hdf_dir,
        output=tmp_path / "out.zarr",
        nside=1,
        overlay_catalog=True,
        catalog_elevation_deg=30.0,
        serve=True,
    )
    assert sm_stub.FrameServer.last["tracks"] == {"SAT": [(0, 1.0, 2.0, 1.0)]}


def test_smoovie_no_overlay_passes_none_tracks(tmp_path, hdf_dir, sm_stub, monkeypatch):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    sm_stub.smoovie(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1)
    assert sm_stub.FrameServer.last["tracks"] is None


def test_smoovie_serve_starts_finishes_and_stops_server(tmp_path, hdf_dir, sm_stub, monkeypatch):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    waited = {"n": 0}
    monkeypatch.setattr(sm_stub, "_wait_for_interrupt", lambda: waited.__setitem__("n", waited["n"] + 1))

    sm_stub.smoovie(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1)

    assert sm_stub.FrameServer.last["holder"].finished is True  # frozen-session signal sent
    assert sm_stub.FrameServer.last.get("stopped") is True  # server torn down
    assert waited["n"] == 1  # blocked for inspection exactly once


def test_smoovie_no_serve_skips_server(tmp_path, hdf_dir, sm_stub, monkeypatch):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("FrameServer must not be constructed when serve=False")

    monkeypatch.setattr(sm_stub, "FrameServer", Boom)
    sm_stub.smoovie(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, serve=False)


def test_smoovie_nframes_flows_to_imaging(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    def fake_image(paths, nside, **k):
        captured.update(k)
        return k.get("output_zarr")

    monkeypatch.setattr(sm_stub, "image_via_app", fake_image)
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))

    sm_stub.smoovie(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, nframes=7, serve=False)
    assert captured.get("nframes") == 7


def test_smoovie_default_catalog_cache_path(tmp_path, hdf_dir, sm_stub, monkeypatch):
    captured = {}

    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))

    def fake_tracks(paths, elevation_deg, **k):
        captured.update(k)
        return {}

    monkeypatch.setattr(sm_stub, "satellite_tracks", fake_tracks)

    output = tmp_path / "out.zarr"
    sm_stub.smoovie(hdf_dir=hdf_dir, output=output, nside=1, overlay_catalog=True, nframes=3, serve=True)
    assert captured["cache_path"] == str(output) + ".catalog.zarr"
    assert captured["nframes"] == 3


def test_smoovie_profile_prints(tmp_path, hdf_dir, sm_stub, monkeypatch, capsys):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))

    sm_stub.smoovie(hdf_dir=hdf_dir, output=tmp_path / "out.zarr", nside=1, profile=True, serve=False)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out
    assert "render" not in out and "encode" not in out


def test_smoovie_overwrite_fail_fast(tmp_path, hdf_dir, sm_stub, monkeypatch):
    monkeypatch.setattr(sm_stub, "common_phase_direction", lambda paths: (0.0, 0.0))
    output = tmp_path / "out.zarr"
    output.mkdir()  # pre-existing output

    with pytest.raises(FileExistsError):
        sm_stub.smoovie(hdf_dir=hdf_dir, output=output, nside=1, serve=False)

    # With overwrite set, the guard passes and orchestration proceeds (imaging stubbed).
    sm_stub.smoovie(hdf_dir=hdf_dir, output=output, nside=1, overwrite=True, serve=False)


# --- full end-to-end (real Holoscan app) ---------------------------------------------------------


def test_smoovie_writes_zarr(tmp_path, hdf_dir):
    """Short end-to-end (no serving): the durable zarr holds the five named maps."""
    import numpy as np
    import xarray as xr

    from kremetart.core.smoovie import smoovie

    out = tmp_path / "live.zarr"
    smoovie(hdf_dir=hdf_dir, output=out, nside=16, nframes=4, serve=False)
    assert out.exists()
    ds = xr.open_zarr(out)
    for var in ("dirty", "tikhonov", "l1", "filtered", "znorm"):
        assert ds[var].shape[0] == 4
        assert np.all(np.isfinite(ds[var].values))
