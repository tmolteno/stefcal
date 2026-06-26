# CALIBRATION.md — `tart-stefcal` Package

Standalone StEFCal solver extracted from `kremetart` for the calibration branch.

## What it is

`tart-stefcal` is a `uv`-managed Python package implementing the StEFCal
(Statistically Efficient and Fast Calibration) algorithm: alternating
per-antenna complex-gain least squares. It solves

```
V_pq ≈ g_p * M_pq * conj(g_q)
```

for per-antenna complex gains `g_p` given observed visibilities `V_pq` and a
model `M_pq`. The solver runs on CPU with NumPy (CuPy GPU support via the
Python API's `xp` parameter).

## Package layout

```
calibration/
├── .gitignore                  # Ignores *.npy / *.npz test artefacts
├── pyproject.toml              # uv-managed: deps, entry point, build config
├── README.md                   # Quick-start install & usage
├── CALIBRATION.md              # This file
└── src/tart_stefcal/
    ├── __init__.py             # Public API
    ├── stefcal.py              # Core solver (ported from kremetart/utils/stefcal.py)
    ├── skymodel.py             # ENU model visibilities (ported from kremetart/utils/skymodel.py)
    ├── beam.py                 # Airy primary beam (ported from kremetart/utils/beam.py)
    └── cli.py                  # argparse CLI: run, solve, phases subcommands
```

## Key differences from `kremetart`

| Aspect | kremetart | tart-stefcal |
|---|---|---|
| `xp.complex128` | Used (breaks on NumPy >= 2.0) | Replaced with `complex` — works on all NumPy versions |
| GPU | `cupy` via lazy import | Python API supports `xp=cupy`; no GPU in CLI |
| CLI wrapper | Typer/Stimela (cab system) | Standalone `argparse` CLI with `run`, `solve`, `phases` |
| Package manager | `uv` (workspace member) | `uv` (workspace member under `calibration/`) |
| Stimela/cab integration | Yes (via `cli/core/cabs` layers) | No — standalone tool |
| TART HDF5 I/O | Via `kremetart.utils.read_tart_hdf` (xarray MSv4) | Direct `h5py` from raw HDF5 |
| Source catalogue | Offline Zarr cache + TART API | `tart-catalogue-client` (TLE propagation) |
| Model beam | `airy_power_beam` applied | `airy_power_beam` applied (disable with `--no-beam`) |
| Reference antenna | Auto-detect from `ANTENNA_FLAG` | Auto-detect from HDF5 gains + baseline incidence |
| Upload | Separate `tart_upload_gains.py` | Built-in `--upload` and `--phases-only` |

## Install

```bash
pip install tart-stefcal
```

## CLI usage

### End-to-end: download -> solve -> upload

```bash
tart-stefcal run --tart-name mu-udm --upload
```

Pipeline:
1. Downloads visibility HDF5 from the TART API (or S3 archive with `--archive`)
2. Concatenates multiple files along the time axis
3. Fetches satellite positions via `tart-catalogue-client` for each timestamp
4. Builds model visibilities with Airy primary beam weighting
5. Auto-detects the reference antenna (or uses `--ref-ant`)
6. Runs StEFCal on the pooled integrations
7. Optionally uploads gains to the telescope

Key options:

| Option | Default | Description |
|---|---|---|
| `--tart-name` | *(required)* | Telescope name or API URL |
| `--upload` | off | Push gains to the telescope |
| `--phases-only` | off | Upload phases only (unity amplitudes). Implies `--upload` |
| `--archive` | off | Download from S3 archive instead of API |
| `--start` | `-duration` | Archive start time (negative offset in min, or ISO-8601) |
| `--duration` | `10` | Archive time window in minutes |
| `--n` | `1` (API) / all (archive) | Number of HDF files to download |
| `--elevation` | `45` | Source catalogue elevation cutoff (deg) |
| `--no-beam` | off | Disable Airy primary beam weighting |
| `--ref-ant` | auto-detect | Reference antenna for phase gauge |
| `--t-int` | `None` (all) | Integrations per solution interval |
| `--max-iter` | `100` | Max StEFCal iterations |
| `--tol` | `1e-8` | Convergence tolerance |
| `--output` / `-o` | `gains.npz` | Output file path |
| `--phases` | off | Also write `_phases.npy` |
| `--json` | off | Also write `.json` (TART API format) |

### Archive mode

```bash
# Last 90 minutes from the S3 archive
tart-stefcal run --tart-name mu-udm --archive --duration 90

# Specific time window
tart-stefcal run --tart-name signal --archive \
    --start "2026-06-25T00:00:00+00:00" --duration 60
```

`--start` defaults to `-duration` (e.g. `--duration 90` gives `--start -90`).
When `--n` is not set, all files in the time window are downloaded. Stale HDF5
files in the download directory are removed before each run so data is always
fresh.

### Output formats

| Flag | File | Contents |
|---|---|---|
| (always) | `gains.npz` | Complex gains `(n_sol, n_ant)` + convergence info |
| `--phases` | `gains_phases.npy` | Gauge-referenced phases `(n_sol, n_ant)` in radians |
| `--json` | `gains.json` | Final solution as `{"gain": [...], "phase_offset": [...]}` |

The JSON format is TART API-compatible. Dead antennas (NaN) are written as
`null`. The `--phases-only` flag normalizes amplitudes to 1.0 before upload
while keeping the full solution on disk.

### Local solve (from files)

```bash
tart-stefcal solve vis.npy model.npy a1.npy a2.npy \
    --n-ant 24 --t-int 10 --ref-ant 0 \
    --max-iter 100 --tol 1e-8 \
    --output gains.npz --phases --json
```

### Extract phases from existing gains

```bash
tart-stefcal phases gains.npz --ref-ant 0 --output phases.npy
```

## Python API

```python
from tart_stefcal import (
    stefcal_solve, referenced_phases,
    enu_direction_cosines, model_visibilities,
    airy_power_beam,
)

gains, info = stefcal_solve(vis, model, a1, a2, n_ant, ref_ant=0, max_iter=100)
phases = referenced_phases(gains, ref_ant=0)
```

Full signatures match the kremetart version (including `t_int`, `weight`, `g0`, `tol`, `xp`).

## Dependencies

| Dependency | Required | Note |
|---|---|---|
| `numpy>=1.24` | Always | Core array ops |
| `scipy>=1.9` | Always | Bessel function for Airy beam |
| `tart-tools>=1.4.5` | Always | TART API client (download, upload) |
| `tart-catalogue-client>=0.4.1` | Always | Satellite position catalogue (TLE propagation) |
| `h5py>=3.0` | Always | HDF5 visibility file I/O |
