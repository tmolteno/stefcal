from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="smoovie",
    info="Image a TART HDF sequence into HEALPix maps and stream them to a live web viewer.",
)
@stimela_output(
    dtype="Directory",
    name="output",
    info="Output HEALPix (TIME, PIX) zarr",
)
def smoovie(
    hdf_dir: Annotated[
        str | None,
        typer.Option(
            help="Directory of input TART HDF snapshots",
        ),
        StimelaMeta(
            type="Directory",
        ),
    ] = None,
    nside: Annotated[
        int,
        typer.Option(
            help="HEALPix nside resolution.",
        ),
    ] = 128,
    phase_ra_deg: Annotated[
        float | None,
        typer.Option(
            help="Common phase-direction RA (deg, ICRS); auto from global mid-time zenith if unset.",
        ),
    ] = None,
    phase_dec_deg: Annotated[
        float | None,
        typer.Option(
            help="Common phase-direction Dec (deg, ICRS); auto from global mid-time zenith if unset.",
        ),
    ] = None,
    correct_gains: Annotated[
        bool,
        typer.Option(
            help="Apply the inverse per-antenna gain solution before imaging.",
        ),
    ] = False,
    overlay_catalog: Annotated[
        bool,
        typer.Option(
            help="Overlay TART catalog satellite tracks on the live sphere (requires network).",
        ),
    ] = False,
    catalog_elevation_deg: Annotated[
        float,
        typer.Option(
            help="Elevation cutoff (deg) for catalog sources to overlay.",
        ),
    ] = 45.0,
    catalog_cache: Annotated[
        str | None,
        typer.Option(
            help="Catalog cache zarr path; defaults to <output>.catalog.zarr.",
        ),
    ] = None,
    profile: Annotated[
        bool,
        typer.Option(
            help="Print a per-stage timing summary.",
        ),
    ] = False,
    iwp_sigma: Annotated[
        float,
        typer.Option(
            help="IWP driving variance (sigma^2) for the per-pixel Kalman filter.",
        ),
    ] = 0.001,
    iwp_noise: Annotated[
        float,
        typer.Option(
            help="Measurement-noise variance (R) for the per-pixel Kalman filter.",
        ),
    ] = 0.01,
    apply_beam: Annotated[
        bool,
        typer.Option(
            help="Fold the Airy primary beam into the measurement operator (image the intrinsic sky).",
        ),
    ] = True,
    ground_plane_diameter: Annotated[
        float,
        typer.Option(
            help="Airy aperture (ground plane) diameter in metres.",
        ),
    ] = 0.125,
    l2: Annotated[
        float,
        typer.Option(
            help="Tikhonov (L2) deconvolution strength as a fraction of weight sum (lambda = l2 * sum w).",
        ),
    ] = 0.01,
    l1: Annotated[
        float,
        typer.Option(
            help="Reweighted-L1 deconvolution strength as a fraction of weight sum (lambda = l1 * sum w).",
        ),
    ] = 0.01,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Overwrite the output zarr if it already exists.",
        ),
    ] = False,
    nframes: Annotated[
        int | None,
        typer.Option(
            help="Cap the number of frames imaged (profiling/preview aid).",
        ),
    ] = None,
    serve: Annotated[
        bool,
        typer.Option(
            help="Serve the live web viewer (use --no-serve for headless/batch runs).",
        ),
    ] = True,
    port: Annotated[
        int,
        typer.Option(
            help="Port for the live web viewer.",
        ),
    ] = 8080,
    open_browser: Annotated[
        bool,
        typer.Option(
            help="Open the live web viewer in a browser on startup.",
        ),
    ] = False,
    output: Annotated[
        Directory | None,
        typer.Option(
            parser=parse_upath,
            help="Output HEALPix (TIME, PIX) zarr",
        ),
    ] = None,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Image a TART HDF sequence into HEALPix maps and stream them to a live web viewer.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                smoovie,
                dict(
                    hdf_dir=hdf_dir,
                    nside=nside,
                    phase_ra_deg=phase_ra_deg,
                    phase_dec_deg=phase_dec_deg,
                    correct_gains=correct_gains,
                    overlay_catalog=overlay_catalog,
                    catalog_elevation_deg=catalog_elevation_deg,
                    catalog_cache=catalog_cache,
                    profile=profile,
                    iwp_sigma=iwp_sigma,
                    iwp_noise=iwp_noise,
                    apply_beam=apply_beam,
                    ground_plane_diameter=ground_plane_diameter,
                    l2=l2,
                    l1=l1,
                    overwrite=overwrite,
                    nframes=nframes,
                    serve=serve,
                    port=port,
                    open_browser=open_browser,
                    output=output,
                ),
            )

            # Lazy import the core implementation
            from kremetart.core.smoovie import smoovie as smoovie_core  # noqa: E402

            # Call the core function with all parameters
            smoovie_core(
                hdf_dir=hdf_dir,
                nside=nside,
                phase_ra_deg=phase_ra_deg,
                phase_dec_deg=phase_dec_deg,
                correct_gains=correct_gains,
                overlay_catalog=overlay_catalog,
                catalog_elevation_deg=catalog_elevation_deg,
                catalog_cache=catalog_cache,
                profile=profile,
                iwp_sigma=iwp_sigma,
                iwp_noise=iwp_noise,
                apply_beam=apply_beam,
                ground_plane_diameter=ground_plane_diameter,
                l2=l2,
                l1=l1,
                overwrite=overwrite,
                nframes=nframes,
                serve=serve,
                port=port,
                open_browser=open_browser,
                output=output,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("kremetart")
    if image is None:
        raise RuntimeError("No Container URL in kremetart metadata.")

    run_in_container(
        smoovie,
        dict(
            hdf_dir=hdf_dir,
            nside=nside,
            phase_ra_deg=phase_ra_deg,
            phase_dec_deg=phase_dec_deg,
            correct_gains=correct_gains,
            overlay_catalog=overlay_catalog,
            catalog_elevation_deg=catalog_elevation_deg,
            catalog_cache=catalog_cache,
            profile=profile,
            iwp_sigma=iwp_sigma,
            iwp_noise=iwp_noise,
            apply_beam=apply_beam,
            ground_plane_diameter=ground_plane_diameter,
            l2=l2,
            l1=l1,
            overwrite=overwrite,
            nframes=nframes,
            serve=serve,
            port=port,
            open_browser=open_browser,
            output=output,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
