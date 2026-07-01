"""CLI for the StEFCal solver — local solve, phases extraction, and end-to-end TART calibration."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
import urllib.parse
from pathlib import Path

import numpy as np

from .beam import airy_power_beam
from .skymodel import enu_direction_cosines, model_visibilities
from .stefcal import referenced_phases, stefcal_solve

_ZENITH = np.array([0.0, 0.0, 1.0])  # antenna boresight in the ENU frame

# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _load_array(path: str, key: str | None = None, fallback_keys: list[str] | None = None) -> np.ndarray:
    """Load a NumPy array from a .npy or .npz file."""
    if path.endswith(".npz"):
        data = np.load(path)
        if key is not None:
            return data[key]
        if fallback_keys:
            for fk in fallback_keys:
                if fk in data:
                    return data[fk]
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(f"NPZ file {path} contains multiple arrays; specify a key. Available: {keys}")
        return data[keys[0]]
    return np.load(path)


# ---------------------------------------------------------------------------
# TART archive / API helpers
# ---------------------------------------------------------------------------


def _tart_name_from_input(name: str) -> str:
    """Extract the bare telescope name from a name or URL."""
    if name.startswith("http://") or name.startswith("https://"):
        return name.rstrip("/").rsplit("/", 1)[-1]
    return name


def _resolve_api_url(name: str) -> str:
    """Resolve a telescope name or URL to its API base URL."""
    if name.startswith("http://") or name.startswith("https://"):
        return name.rstrip("/")
    return f"https://api.elec.ac.nz/tart/{name}"


def _download_from_archive(target: str, out_dir: str, n: int, start: str, duration: str) -> list[Path]:
    """Download visibility HDF5 files from the TART S3 archive.

    Returns a list of downloaded file paths (``obs_00000.hdf``, ...).
    """
    from tart_tools.archive_handler import handle_archive_request

    handle_archive_request(
        target=target,
        num_observations=n,
        output_dir=out_dir,
        start_str=start,
        duration_str=duration,
    )
    paths = sorted(Path(out_dir).glob("obs_*.hdf"))
    if not paths:
        raise RuntimeError(f"No HDF files downloaded from archive for target={target}")
    return paths


def _download_latest_vis(api_url: str, out_dir: str, n: int = 1) -> list[Path]:
    """Download the latest visibility HDF5 file(s) from a TART API.

    Returns a list of downloaded file paths.
    """
    from tart_tools.api_handler import APIhandler, download_file

    api = APIhandler(api_url)
    entries = api.get("vis/data")
    if not entries:
        raise RuntimeError(f"No visibility data available at {api_url}")

    downloaded: list[Path] = []
    for entry in entries[:n]:
        fname = entry["filename"]
        data_url = urllib.parse.urljoin(api_url + "/", fname)
        file_name = fname.split("/")[-1]
        file_path = os.path.join(out_dir, file_name)
        download_file(data_url, entry["checksum"], file_path)
        downloaded.append(Path(file_path))

    return downloaded


def _fetch_catalog(lat: float, lon: float, dt, elevation: float = 45.0) -> list[dict]:
    """Fetch satellite positions for a given time from the TART catalogue client.

    Returns a list of dicts with keys ``az``, ``el`` (degrees), filtered to
    sources above ``elevation`` degrees.
    """
    from tart_client import CatalogueClient

    client = CatalogueClient()
    positions = client.horizontal_positions(lat=lat, lon=lon, dt=dt)
    return [{"az": s["azimuth_deg"], "el": s["elevation_deg"]} for s in positions if s["elevation_deg"] >= elevation]


def _gains_to_json_dict(gains: np.ndarray) -> dict:
    """Convert complex gains to the TART API JSON format.

    Returns ``{"gain": [...], "phase_offset": [...]}``, rounding to 4 decimal places.
    Dead antennas (NaN) are written as ``null``.
    """
    g = np.asarray(gains)
    amp = np.round(np.abs(g), 4)
    phase = np.round(np.angle(g), 4)
    dead = ~np.isfinite(g)
    return {
        "gain": [None if dead[i] else float(amp[i]) for i in range(len(g))],
        "phase_offset": [None if dead[i] else float(phase[i]) for i in range(len(g))],
    }


def _upload_gains_json(api_url: str, password: str, gains: np.ndarray, *, negate_phases: bool = False) -> None:
    """Upload complex gains to the TART telescope API."""
    from tart_tools.api_handler import AuthorizedAPIhandler, upload_gain

    api_dict = _gains_to_json_dict(gains)
    if negate_phases:
        api_dict["phase_offset"] = [None if v is None else -v for v in api_dict["phase_offset"]]
    api = AuthorizedAPIhandler(api_url, password)
    upload_gain(api, api_dict)
    print(f"Gains uploaded to {api_url}")


# ---------------------------------------------------------------------------
# HDF5 reader
# ---------------------------------------------------------------------------


def _read_tart_hdf(hdf_path: str) -> dict:
    """Read a TART visibility HDF5 file and return the key arrays.

    Returns a dict with keys:
        vis: (ntime, nbl) complex
        baselines: (nbl, 2) int [ant1, ant2]
        timestamps: (ntime,) str
        config: dict with lat, lon, num_antenna, frequency, antenna_positions, name
        gains: optional (n_ant,) complex from the HDF5 (zero = dead antenna)
    """
    import h5py

    with h5py.File(hdf_path, "r") as f:
        vis = f["vis"][:]  # (ntime, nbl) complex
        baselines = f["baselines"][:]  # (nbl, 2) int
        timestamps = [t.decode() if isinstance(t, bytes) else t for t in f["timestamp"][:]]
        raw_config = f["config"][()]
        # config may be a scalar or an array of one element
        if hasattr(raw_config, "flat"):
            raw_config = raw_config.flat[0]
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode()
        config = json.loads(raw_config)
        # Gains may or may not be present; zero gain = dead antenna
        hdf_gains = None
        if "gains" in f:
            hdf_gains = f["gains"][:]

    return {
        "vis": vis,
        "baselines": baselines,
        "timestamps": timestamps,
        "config": config,
        "gains": hdf_gains,
    }


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _write_gains_json(gains: np.ndarray, path: str, *, ref_ant: int | None = None) -> None:
    """Write gains as TART-compatible JSON.

    If ``ref_ant`` is given, a per-solution entry ``"ref_ant"`` is included.
    """
    g = np.asarray(gains)
    if g.ndim == 2:
        # (n_sol, n_ant) — write all solutions (last one is the final)
        solutions = [_gains_to_json_dict(g[i]) for i in range(g.shape[0])]
        out: dict = {"solutions": solutions}
        if ref_ant is not None:
            out["ref_ant"] = ref_ant
        out["gain"] = solutions[-1]["gain"]
        out["phase_offset"] = solutions[-1]["phase_offset"]
    else:
        out = _gains_to_json_dict(g)
        if ref_ant is not None:
            out["ref_ant"] = ref_ant
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Gains JSON written to {path}")


def _cmd_solve(args: argparse.Namespace) -> None:
    vis = _load_array(args.vis)
    model = _load_array(args.model)
    a1 = np.load(args.a1)
    a2 = np.load(args.a2)

    weight = None
    if args.weight is not None:
        weight = np.load(args.weight)

    g0 = None
    if args.g0 is not None:
        g0 = np.load(args.g0)

    gains, info = stefcal_solve(
        vis,
        model,
        a1,
        a2,
        args.n_ant,
        t_int=args.t_int,
        ref_ant=args.ref_ant,
        weight=weight,
        g0=g0,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    save = {"gains": np.asarray(gains)}
    for k, v in info.items():
        save[k] = np.asarray(v)

    np.savez(args.output, **save)
    print(f"Gains written to {args.output}  ({save['gains'].shape[0]} solution(s), {save['gains'].shape[1]} antennas)")

    if args.phases:
        phase_path = str(Path(args.output).with_name(Path(args.output).stem + "_phases.npy"))
        phases = referenced_phases(gains, args.ref_ant)
        np.save(phase_path, np.asarray(phases))
        print(f"Phases written to {phase_path}")

    if args.json:
        json_path = str(Path(args.output).with_name(Path(args.output).stem + ".json"))
        _write_gains_json(gains, json_path, ref_ant=args.ref_ant)


def _cmd_phases(args: argparse.Namespace) -> None:
    gains = _load_array(args.gains, fallback_keys=["gains"])
    phases = referenced_phases(gains, args.ref_ant)
    np.save(args.output, phases)
    print(f"Phases written to {args.output}  ({phases.shape})")


def _cmd_run(args: argparse.Namespace) -> None:
    """End-to-end: download vis data, run StEFCal, optionally upload gains."""
    if args.phases_only:
        args.upload = True
    api_url = _resolve_api_url(args.tart_name)
    tart_name = _tart_name_from_input(args.tart_name)
    print(f"Telescope: {tart_name}  →  API: {api_url}")

    # --- step 1: download visibility data ---
    dl_dir = args.dir or tempfile.mkdtemp(prefix="tart_stefcal_")
    os.makedirs(dl_dir, exist_ok=True)
    # Remove stale HDF5 files so every run downloads fresh data
    for stale in Path(dl_dir).glob("*.hdf"):
        stale.unlink()
    if args.archive:
        n_archive = args.n if args.n != 1 else -1
        start = args.start if args.start is not None else f"-{args.duration}"
        print(
            f"Downloading from S3 archive (target={tart_name}, start={start}, duration={args.duration}, n={n_archive}) ..."
        )
        hdf_paths = _download_from_archive(tart_name, dl_dir, n=n_archive, start=start, duration=args.duration)
    else:
        print(f"Downloading latest visibility data from API to {dl_dir} ...")
        hdf_paths = _download_latest_vis(api_url, dl_dir, n=args.n)
    print(f"Downloaded {len(hdf_paths)} file(s)")

    # --- step 2: read and concatenate all HDF5 files ---
    all_vis = []
    all_timestamps = []
    config = None
    baselines = None
    ant_positions = None
    telescope_name = tart_name

    for hdf_path in hdf_paths:
        data = _read_tart_hdf(str(hdf_path))
        if config is None:
            config = data["config"]
            baselines = data["baselines"]
            ant_positions = np.array(config.get("antenna_positions", []))
            telescope_name = config.get("name", tart_name)
        all_vis.append(data["vis"])
        all_timestamps.extend(data["timestamps"])

    n_ant = config["num_antenna"]
    freq_hz = config.get("frequency", 1.57542e9)
    lat = config["lat"]
    lon = config["lon"]
    a1 = baselines[:, 0].astype(np.int64)
    a2 = baselines[:, 1].astype(np.int64)
    nbl = len(a1)

    vis = np.concatenate(all_vis, axis=0)
    ntime = len(all_timestamps)
    if vis.ndim == 2:
        vis = vis[:, :, None]

    print(f"  Telescope: {telescope_name}  ({n_ant} antennas, {freq_hz / 1e6:.1f} MHz)")
    print(f"  Location:  lat={lat:.4f}  lon={lon:.4f}")
    print(
        f"  Visibilities: {ntime} integrations x {nbl} baselines x {vis.shape[2]} channel(s) ({len(hdf_paths)} file(s))"
    )

    # Baseline ENU vectors (metres)
    if len(ant_positions) >= n_ant:
        bl_enu = ant_positions[a1] - ant_positions[a2]  # (nbl, 3)
    else:
        # Fallback: zero baselines (model will be all ones)
        bl_enu = np.zeros((nbl, 3))

    # --- step 3: build model visibilities ---
    use_beam = not args.no_beam
    freqs = np.array([freq_hz])
    models = []
    source_counts = []

    for t_idx in range(ntime):
        ts = all_timestamps[t_idx]
        try:
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            catalog = _fetch_catalog(lat, lon, dt, elevation=args.elevation)
        except Exception as e:
            print(f"Warning: could not get catalogue for {ts}: {e}", file=sys.stderr)
            models.append(np.ones((nbl, 1), dtype=complex))
            source_counts.append(0)
            continue

        if not catalog:
            models.append(np.ones((nbl, 1), dtype=complex))
            source_counts.append(0)
            continue

        source_counts.append(len(catalog))
        az = np.radians([s["az"] for s in catalog])
        el = np.radians([s["el"] for s in catalog])
        s_enu = enu_direction_cosines(az, el)
        beam = airy_power_beam(s_enu, _ZENITH, freqs) if use_beam else None
        mvis = model_visibilities(s_enu, bl_enu, freqs, beam=beam)
        models.append(mvis)

    models = np.stack(models)  # (ntime, nbl, 1)
    sc = np.array(source_counts)
    print(f"  Model sources: {sc.min()}-{sc.max()} above {args.elevation} deg elevation (mean {sc.mean():.1f})")

    # --- step 4: auto-detect reference antenna ---
    if args.ref_ant is None:
        hdf_gains = data.get("gains")
        if hdf_gains is not None and hdf_gains.ndim >= 1:
            dead = np.asarray(hdf_gains).flat[:n_ant] == 0
        else:
            dead = np.zeros(n_ant, dtype=bool)
        live = ~dead
        live_from_bl = np.zeros(n_ant, dtype=bool)
        live_from_bl[a1] = True
        live_from_bl[a2] = True
        live = live & live_from_bl
        ref_ant = int(np.where(live)[0][0]) if live.any() else 0
        print(f"  Auto-detected reference antenna: {ref_ant}")
    else:
        ref_ant = args.ref_ant

    # --- step 5: run StEFCal ---
    print(f"Running StEFCal (max_iter={args.max_iter}, tol={args.tol}) ...")
    gains, info = stefcal_solve(
        vis,
        models,
        a1,
        a2,
        n_ant,
        t_int=args.t_int,
        ref_ant=ref_ant,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    n_sol = gains.shape[0]
    g_final = np.asarray(gains[-1])  # last solution
    conv = bool(np.asarray(info["converged"])[-1])
    n_iter = int(np.asarray(info["iterations"])[-1])
    print(f"  Converged: {conv}  in {n_iter} iteration(s)  ({n_sol} solution interval(s))")

    # --- step 5: save ---
    save = {"gains": np.asarray(gains)}
    for k, v in info.items():
        save[k] = np.asarray(v)
    np.savez(args.output, **save)
    print(f"Gains written to {args.output}")

    if args.phases:
        phase_path = str(Path(args.output).with_name(Path(args.output).stem + "_phases.npy"))
        phases_run = referenced_phases(gains, ref_ant)
        np.save(phase_path, np.asarray(phases_run))
        print(f"Phases written to {phase_path}")

    if args.json:
        json_path = str(Path(args.output).with_name(Path(args.output).stem + ".json"))
        _write_gains_json(gains, json_path, ref_ant=ref_ant)

    # --- step 7: upload ---
    if args.upload:
        if args.phases_only:
            g_upload = g_final / np.abs(g_final)
            g_upload = np.where(np.isfinite(g_upload), g_upload, g_final)
            print(f"Uploading phases only (unity amplitudes) to {api_url} ...")
        else:
            g_upload = g_final
            print(f"Uploading gains to {api_url} ...")
        _upload_gains_json(api_url, args.pw, g_upload, negate_phases=args.negate_phases)
    else:
        print("Skipping upload (use --upload to push gains to the telescope).")

    if args.dir is None:
        import shutil

        shutil.rmtree(dl_dir)


# ---------------------------------------------------------------------------
# Parser & main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tart-stefcal",
        description="StEFCal: alternating per-antenna complex-gain least squares solver.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run (end-to-end) ---
    p_run = sub.add_parser("run", help="Download TART vis data, run StEFCal, upload gains.")
    p_run.add_argument("--tart-name", required=True, help="Telescope name (e.g. mu-udm, signal) or API URL.")
    p_run.add_argument("--pw", default="password", help="API password for uploading gains.")
    p_run.add_argument("--dir", default=None, help="Download directory (default: temp dir).")
    p_run.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of HDF files to download. With --archive, default becomes all files in the time window.",
    )
    p_run.add_argument("--archive", action="store_true", help="Download from S3 archive instead of the API.")
    p_run.add_argument(
        "--start",
        default=None,
        help="Start time for archive download (default: -duration; ISO-8601 also accepted).",
    )
    p_run.add_argument("--duration", default="10", help="Duration in minutes for archive download (default: 10).")
    p_run.add_argument("--elevation", type=float, default=45.0, help="Elevation cutoff for source catalog (deg).")
    p_run.add_argument("--t-int", type=int, default=None, help="Integrations per solution interval (None=all).")
    p_run.add_argument("--ref-ant", type=int, default=None, help="Reference antenna (default: auto-detect first live).")
    p_run.add_argument("--no-beam", action="store_true", help="Disable Airy primary beam weighting in the model.")
    p_run.add_argument("--max-iter", type=int, default=100, help="Max StEFCal iterations (default: 100).")
    p_run.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance (default: 1e-8).")
    p_run.add_argument("--output", "-o", default="gains.npz", help="Output file (default: gains.npz).")
    p_run.add_argument("--phases", action="store_true", help="Also output gauge-referenced phases.")
    p_run.add_argument("--json", action="store_true", help="Also output gains as TART-compatible JSON.")
    p_run.add_argument("--upload", action="store_true", help="Upload gains to the telescope after solving.")
    p_run.add_argument(
        "--phases-only",
        action="store_true",
        help="Upload phase offsets only (unity amplitudes). Implies --upload.",
    )
    p_run.add_argument(
        "--negate-phases",
        action="store_true",
        help="Negate phase offsets before upload.",
    )

    # --- solve (local) ---
    p_solve = sub.add_parser("solve", help="Solve per-antenna complex gains from local files.")
    p_solve.add_argument("vis", help="Path to observed visibilities (.npy or .npz).")
    p_solve.add_argument("model", help="Path to model visibilities (.npy or .npz).")
    p_solve.add_argument("a1", help="Path to first-antenna indices (.npy).")
    p_solve.add_argument("a2", help="Path to second-antenna indices (.npy).")
    p_solve.add_argument("--n-ant", type=int, required=True, help="Number of antennas.")
    p_solve.add_argument("--t-int", type=int, default=None, help="Integrations per solution interval (None=all).")
    p_solve.add_argument("--ref-ant", type=int, default=0, help="Reference antenna (default: 0).")
    p_solve.add_argument("--weight", default=None, help="Optional weight array (.npy).")
    p_solve.add_argument("--g0", default=None, help="Optional initial gains (.npy).")
    p_solve.add_argument("--max-iter", type=int, default=100, help="Max iterations per interval (default: 100).")
    p_solve.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance (default: 1e-8).")
    p_solve.add_argument("--output", "-o", default="gains.npz", help="Output file (default: gains.npz).")
    p_solve.add_argument("--phases", action="store_true", help="Also output gauge-referenced phases.")
    p_solve.add_argument("--json", action="store_true", help="Also output gains as TART-compatible JSON.")

    # --- phases ---
    p_phases = sub.add_parser("phases", help="Extract gauge-referenced phases from gains.")
    p_phases.add_argument("gains", help="Path to gains file (.npy or .npz).")
    p_phases.add_argument("--ref-ant", type=int, default=0, help="Reference antenna (default: 0).")
    p_phases.add_argument("--output", "-o", default="phases.npy", help="Output file (default: phases.npy).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "solve":
        _cmd_solve(args)
    elif args.command == "phases":
        _cmd_phases(args)
