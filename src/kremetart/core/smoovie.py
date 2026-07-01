"""
Image a TART HDF sequence into HEALPix all-sky maps and stream them to a live web viewer.

Concatenates the HDFs into a single MSv4 zarr (host prepare-step), then streams it through the GPU
Holoscan app: the HEALPix imager produces a per-sub-integration dirty map; Tikhonov (CG) and
reweighted-L1 (FISTA) deconvolvers run in parallel on every frame; a per-pixel IWP-Kalman filter
smooths the Tikhonov stream and emits the filtered flux and normalised innovation; a writer persists
``dirty``/``tikhonov``/``l1``/``filtered``/``znorm`` to a durable ``(TIME, npix)`` zarr; and (when
serving) a terminal sink fans the five named maps out to a FastAPI+uvicorn web layer for a live
interactive sphere, freezing for scrub-back inspection when the pipeline finishes. The PNG+ffmpeg
movie path has been removed (the parked Mollweide renderer lives in
``kremetart.utils.visualisation`` for a future sub-command).
"""

from __future__ import annotations

import tempfile
import threading
import webbrowser
from pathlib import Path

import healpy as hp
import holoscan as hs
import xarray as xr
from holoscan.conditions import CountCondition

from kremetart.operators.dft_healpix import HealpixDFTOperator
from kremetart.operators.io import HealpixWriterOperator, HealpixZarrReaderOperator
from kremetart.operators.iwp_kalman import IWPKalmanOperator
from kremetart.operators.l1_reweight import L1ReweightOperator
from kremetart.operators.tikhonov import TikhonovOperator
from kremetart.operators.web_sink import WebStreamSinkOperator
from kremetart.utils.beam import GROUND_PLANE_DIAMETER
from kremetart.utils.healpix_viz import NAMES, UNITS, LatestFrameHolder
from kremetart.utils.profiling import print_profile, stage_timer
from kremetart.utils.read_tart_hdf import prepare_msv4_zarr
from kremetart.utils.rephasing import common_phase_direction
from kremetart.utils.satellites import satellite_tracks
from kremetart.utils.web_server import FrameServer


class SmooviePipeline(hs.core.Application):
    """Stream a prepared imaging zarr through the GPU HEALPix imager into a ``(TIME, npix)`` zarr.

    When ``holder`` is given, a terminal :class:`WebStreamSinkOperator` is added and the ``iwp``
    outputs fan out to both the writer and the sink (Holoscan broadcasts the shared output ports).
    """

    def __init__(
        self,
        prepared_zarr: Path | str,
        output_zarr: Path | str,
        nside: int,
        *args,
        nest: bool = True,
        sigma2: float = 1e-3,
        noise: float = 1e-2,
        apply_beam: bool = True,
        ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
        l2: float = 0.01,
        l1: float = 0.01,
        holder: LatestFrameHolder | None = None,
        **kwargs,
    ):
        self.prepared_zarr = str(prepared_zarr)
        self.output_zarr = str(output_zarr)
        self.nside = nside
        self.nest = nest
        self.sigma2 = sigma2
        self.noise = noise
        self.apply_beam = apply_beam
        self.ground_plane_diameter = ground_plane_diameter
        self.l2 = l2
        self.l1 = l1
        self.holder = holder
        super().__init__(*args, **kwargs)

        ds = xr.open_zarr(self.prepared_zarr)
        self.ntime = int(ds.time.size)
        self.out_times = ds.time.values
        self.freqs = ds.frequency.values
        self.npix = hp.nside2npix(nside)

    def compose(self):
        reader = HealpixZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.prepared_zarr,
        )
        imager = HealpixDFTOperator(
            self,
            self.nside,
            self.freqs,
            name="imager",
            nest=self.nest,
            apply_beam=self.apply_beam,
            ground_plane_diameter=self.ground_plane_diameter,
        )
        tikhonov = TikhonovOperator(
            self,
            self.nside,
            self.freqs,
            self.l2,
            name="tikhonov",
            nest=self.nest,
            apply_beam=self.apply_beam,
            ground_plane_diameter=self.ground_plane_diameter,
        )
        l1 = L1ReweightOperator(
            self,
            self.nside,
            self.freqs,
            self.l1,
            name="l1reweight",
            nest=self.nest,
            apply_beam=self.apply_beam,
            ground_plane_diameter=self.ground_plane_diameter,
        )
        iwp = IWPKalmanOperator(self, self.npix, name="iwp", sigma2=self.sigma2, noise=self.noise)
        writer = HealpixWriterOperator(
            self,
            self.ntime,
            self.npix,
            name="writer",
            output_dataset=self.output_zarr,
            out_times=self.out_times,
            var_specs=(
                ("dirty", "dirty"),
                ("tikhonov", "tikhonov"),
                ("l1", "l1"),
                ("filtered", "filtered"),
                ("znorm", "znorm"),
            ),
        )

        self.add_flow(
            reader,
            imager,
            {
                ("VISIBILITY", "VISIBILITY"),
                ("WEIGHT", "WEIGHT"),
                ("B_ROT", "B_ROT"),
                ("BORESIGHT", "BORESIGHT"),
                ("time", "time"),
            },
        )
        # Both deconvolvers share the imager dirty (the RHS) and the reader geometry that builds H.
        self.add_flow(imager, tikhonov, {("cube", "cube"), ("time_out", "time_out")})
        self.add_flow(imager, l1, {("cube", "cube")})
        self.add_flow(reader, tikhonov, {("WEIGHT", "WEIGHT"), ("B_ROT", "B_ROT"), ("BORESIGHT", "BORESIGHT")})
        self.add_flow(reader, l1, {("WEIGHT", "WEIGHT"), ("B_ROT", "B_ROT"), ("BORESIGHT", "BORESIGHT")})
        # Only the Tikhonov stream is smoothed by the IWP (its passthrough cube is the tikhonov channel).
        self.add_flow(tikhonov, iwp, {("cube", "cube"), ("time_out", "time_out")})

        # Durable writer: dirty (tikhonov passthrough), tikhonov model (iwp passthrough), l1 model,
        # filtered (smoothed tikhonov), znorm.
        self.add_flow(
            iwp,
            writer,
            {("cube", "tikhonov"), ("filtered", "filtered"), ("znorm", "znorm"), ("time_out", "time_out")},
        )
        self.add_flow(tikhonov, writer, {("dirty", "dirty")})
        self.add_flow(l1, writer, {("cube", "l1")})

        if self.holder is not None:
            sink = WebStreamSinkOperator(self, name="websink", holder=self.holder)
            self.add_flow(
                iwp,
                sink,
                {("cube", "tikhonov"), ("filtered", "smooth"), ("znorm", "znorm"), ("time_out", "time_out")},
            )
            self.add_flow(tikhonov, sink, {("dirty", "dirty")})
            self.add_flow(l1, sink, {("cube", "l1")})


def image_via_app(
    hdf_paths,
    nside: int,
    *,
    output_zarr,
    holder: LatestFrameHolder | None = None,
    correct_gains: bool = False,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    nframes: int | None = None,
    nest: bool = True,
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
    apply_beam: bool = True,
    ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
    l2: float = 0.01,
    l1: float = 0.01,
) -> Path:
    """Image the HDF sequence through the GPU Holoscan app; write a durable ``(TIME, PIX)`` zarr.

    Runs the host prepare-step into a temp zarr, then streams it through :class:`SmooviePipeline`
    (reader -> imager -> tikhonov + l1 in parallel -> IWP (Tikhonov stream only) -> writer, plus a
    web sink when ``holder`` is given). The pipeline always images dirty / Tikhonov / L1 in every
    frame. Persists ``dirty``/``tikhonov``/``l1``/``filtered``/``znorm`` to ``output_zarr`` and
    returns its path. This is the single imaging seam :func:`smoovie` drives; host-wiring tests
    stub it.

    Args:
        hdf_paths: ordered iterable of TART HDF paths.
        nside: HEALPix resolution.
        output_zarr: durable output zarr path; the caller does the fail-fast existence check.
        holder: optional live-frame holder; when set, frames also stream to the web layer.
        correct_gains: apply the inverse per-antenna gain solution in the prepare-step.
        phase_ra_deg, phase_dec_deg: common phase direction (deg, ICRS), stored as zarr metadata.
        nframes: optional cap on the number of frames imaged.
        nest: NESTED HEALPix ordering (default True).
        iwp_sigma: IWP driving variance sigma^2.
        iwp_noise: measurement-noise variance R.
        apply_beam: fold the Airy primary beam into the measurement operator (default True).
        ground_plane_diameter: Airy aperture (ground plane) diameter in metres.
        l2: Tikhonov regularisation strength as a fraction of Σw (lambda = l2 * Σw, default 0.01).
            See ``scripts/compare_regularisers.py`` for guidance on tuning.
        l1: Reweighted-L1 (FISTA) regularisation strength as a fraction of Σw (default 0.01).
            Both ``l2`` and ``l1`` are independent; the IWP smooths the Tikhonov stream.

    Returns:
        The ``output_zarr`` path.
    """
    output = Path(output_zarr)
    with tempfile.TemporaryDirectory() as td:
        prepared = Path(td) / "prepared.zarr"
        config = Path(td) / "config.yaml"
        config.touch()  # an empty Holoscan config is valid

        prepare_msv4_zarr(
            hdf_paths,
            prepared,
            correct_gains=correct_gains,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            nframes=nframes,
        )

        app = SmooviePipeline(
            prepared,
            output,
            nside,
            nest=nest,
            sigma2=iwp_sigma,
            noise=iwp_noise,
            apply_beam=apply_beam,
            ground_plane_diameter=ground_plane_diameter,
            l2=l2,
            l1=l1,
            holder=holder,
        )
        app.config(str(config))
        app.run()

    return output


def _wait_for_interrupt() -> None:
    """Block until the user interrupts (Ctrl-C), keeping the frozen session served."""
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass


def smoovie(
    hdf_dir,
    output,
    nside: int = 128,
    phase_ra_deg: float | None = None,
    phase_dec_deg: float | None = None,
    correct_gains: bool = False,
    overlay_catalog: bool = False,
    catalog_elevation_deg: float = 45.0,
    catalog_cache: str | None = None,
    profile: bool = False,
    iwp_sigma: float = 1e-3,
    iwp_noise: float = 1e-2,
    apply_beam: bool = True,
    ground_plane_diameter: float = GROUND_PLANE_DIAMETER,
    l2: float = 0.01,
    l1: float = 0.01,
    overwrite: bool = False,
    nframes: int | None = None,
    serve: bool = True,
    port: int = 8080,
    open_browser: bool = False,
) -> Path:
    """Image the HDF sequence in ``hdf_dir`` into the durable ``output`` zarr; serve a live view.

    Every sub-integration is imaged into a common ICRS HEALPix frame centered on the shared phase
    direction. If ``phase_ra_deg``/``phase_dec_deg`` are unset they default to the local zenith
    RA/Dec at the global mid-time (:func:`common_phase_direction`); supplying only one raises
    ``ValueError``. Writes ``output`` (a ``(TIME, PIX)`` zarr holding ``dirty``/``tikhonov``/
    ``l1``/``filtered``/``znorm``); raises ``FileExistsError`` if it exists unless
    ``overwrite=True``.

    The pipeline **always** images dirty / Tikhonov / L1 in every frame. ``l2`` and ``l1`` set the
    Tikhonov (CG) and reweighted-L1 (FISTA) regularisation strengths respectively, each as a
    fraction of the per-frame weight sum (``lambda = strength * Σw``; defaults ``0.01`` — see
    ``scripts/compare_regularisers.py`` for guidance). The IWP-Kalman filter smooths the Tikhonov
    stream; ``filtered`` and ``znorm`` therefore track Tikhonov.

    With ``serve`` (the default), a FastAPI+uvicorn server starts on ``port``, the view URL is
    printed (and opened if ``open_browser``), frames stream live to an interactive sphere, and when
    the pipeline finishes the session freezes for scrub-back inspection until interrupted (Ctrl-C).
    ``--no-serve`` images headlessly (batch / Stimela-cab use). ``overlay_catalog`` (serve only)
    overlays each catalogue satellite above ``catalog_elevation_deg`` as 3D tracks; it requires
    network access to the TART catalogue API, cached at ``catalog_cache`` (``None`` ->
    ``<output>.catalog.zarr``). ``profile`` prints the imaging wall-time; ``nframes`` caps the frames
    imaged; ``iwp_sigma``/``iwp_noise`` set the per-pixel filter's σ²/R. ``apply_beam`` (default on)
    folds the Airy primary beam of a ``ground_plane_diameter``-metre aperture into the measurement
    operator so the maps are oriented toward the intrinsic sky.

    Returns:
        The ``output`` zarr path.
    """
    if hdf_dir is None or output is None:
        raise ValueError("hdf_dir and output are required")
    if (phase_ra_deg is None) != (phase_dec_deg is None):
        raise ValueError("phase_ra_deg and phase_dec_deg must be given together (both or neither)")
    hdf_dir = Path(hdf_dir)
    output = Path(output)
    hdf_paths = sorted(hdf_dir.glob("*.hdf"))
    if not hdf_paths:
        raise FileNotFoundError(f"no .hdf files found in {hdf_dir}")

    if phase_ra_deg is None:
        phase_ra_deg, phase_dec_deg = common_phase_direction(hdf_paths)

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass overwrite=True to replace it.")

    tracks = None
    if serve and overlay_catalog:
        cache_path = catalog_cache if catalog_cache is not None else str(output) + ".catalog.zarr"
        tracks = satellite_tracks(hdf_paths, catalog_elevation_deg, cache_path=cache_path, nframes=nframes)

    holder = None
    server = None
    url = None
    if serve:
        holder = LatestFrameHolder(NAMES)
        server = FrameServer(holder, nside=nside, nest=True, names=NAMES, units=UNITS, port=port, tracks=tracks)
        url = server.start()
        print(f"kremetart live view: {url}")
        if open_browser:
            webbrowser.open(url)

    timings: list = []
    try:
        with stage_timer("imaging", timings):
            image_via_app(
                hdf_paths,
                nside,
                output_zarr=output,
                holder=holder,
                correct_gains=correct_gains,
                phase_ra_deg=phase_ra_deg,
                phase_dec_deg=phase_dec_deg,
                nframes=nframes,
                iwp_sigma=iwp_sigma,
                iwp_noise=iwp_noise,
                apply_beam=apply_beam,
                ground_plane_diameter=ground_plane_diameter,
                l2=l2,
                l1=l1,
            )
        if serve:
            holder.finish()
            print(f"pipeline finished — serving frozen session at {url} (Ctrl-C to exit)")
            _wait_for_interrupt()
    finally:
        if serve and server is not None:
            # Flush a clean end even if imaging raised, so connected clients freeze
            # (and stop reconnecting) instead of looping against the stopped server.
            if holder is not None:
                holder.finish()
            server.stop()

    if profile:
        print_profile(timings, nframes=nframes)

    return output
