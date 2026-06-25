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
model `M_pq`. The solver is `xp`-injectable — it runs on CPU with NumPy or on
GPU with CuPy.

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
    └── cli.py                  # argparse CLI: run, solve, phases subcommands
```

## Key differences from `kremetart`

| Aspect | kremetart | tart-stefcal |
|---|---|---|
| `xp.complex128` | Used (breaks on NumPy >= 2.0) | Replaced with `complex` — works on all NumPy versions |
| GPU deps | Lazy import per `python-standards.md` S2 | `cupy` is an optional extra `[gpu]`; CLI flag `--gpu` |
| CLI wrapper | Typer/Stimela (cab system) | Standalone `argparse` CLI with `run`, `solve`, `phases` |
| Package manager | `uv` (workspace member) | `uv` (workspace member under `calibration/`) |
| Stimela/cab integration | Yes (via `cli/core/cabs` layers) | No — standalone tool |
| TART integration | Via `kremetart.utils.read_tart_hdf` | Direct `tart-tools` API: download HDF5, fetch catalog, upload gains |

## Install

```bash
# CPU only
pip install tart-stefcal

# With GPU support
pip install tart-stefcal[gpu]
```

## CLI usage

### End-to-end: download -> solve -> upload

```bash
tart-stefcal run --tart-name mu-udm --upload
```

This is the primary workflow:
1. Downloads the latest visibility HDF5 from the TART API
2. Fetches the source catalog for each integration timestamp
3. Builds model visibilities via `enu_direction_cosines` + `model_visibilities`
4. Runs StEFCal (`xp=numpy` or `xp=cupy` with `--gpu`)
5. Optionally uploads gains to the telescope (`--upload`)

Key options:
- `--tart-name` — telescope name (`mu-udm`, `signal`, `za-dias`, `mu-rhodes`) or API URL
- `--upload` — push gains to the telescope after solving
- `--n` — number of HDF files to download (default: 1)
- `--elevation` — source catalog elevation cutoff in degrees (default: 45)
- `--output` / `-o` — output `.npz` file (default: `gains.npz`)
- `--phases` — also write `_phases.npy` (gauge-referenced phases in radians)
- `--json` — also write `.json` (TART API-compatible gain/phase_offset format)

### Output formats

The solver always writes a `.npz` archive containing `gains`, `iterations`,
`converged`, and `max_change`. Optional sidecar files:

| Flag | File | Contents |
|---|---|---|
| (always) | `gains.npz` | Complex gains `(n_sol, n_ant)` + convergence info |
| `--phases` | `gains_phases.npy` | Gauge-referenced phases `(n_sol, n_ant)` in radians |
| `--json` | `gains.json` | Final solution as `{"gain": [...], "phase_offset": [...]}` |

The JSON format is TART API-compatible — it can be uploaded directly via
`tart_upload_gains.py` or used with the `--upload` flag.

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
from tart_stefcal import stefcal_solve, referenced_phases

gains, info = stefcal_solve(vis, model, a1, a2, n_ant, ref_ant=0, max_iter=100)
phases = referenced_phases(gains, ref_ant=0)
```

Full signatures match the kremetart version (including `t_int`, `weight`, `g0`, `tol`, `xp`).

## Dependencies

| Dependency | Required | Note |
|---|---|---|
| `numpy>=1.24` | Always | Core array ops |
| `tart-tools>=1.4.5` | Always | TART API client (download, catalog, upload) |
| `h5py>=3.0` | Always | HDF5 visibility file I/O |
| `cupy>=13.0` | Optional `[gpu]` | GPU backend via `--gpu` flag |
