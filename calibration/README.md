# tart-stefcal

StEFCal: alternating per-antenna complex-gain least squares solver for TART.

## Install

```bash
pip install tart-stefcal           # CPU only
pip install tart-stefcal[gpu]      # with CuPy GPU support
```

## Quick start

Calibrate a TART telescope in one command:

```bash
tart-stefcal run --tart-name mu-udm --upload
```

This downloads the latest visibility data, builds a sky model, solves for
per-antenna complex gains, and uploads the result to the telescope.

---

## Commands

`tart-stefcal` provides three subcommands:

| Command | Purpose |
|---|---|
| `run` | End-to-end: download visibilities, solve gains, optionally upload |
| `solve` | Solve gains from local NumPy files |
| `phases` | Extract gauge-referenced phases from a gains file |

---

## `tart-stefcal run` — End-to-end calibration

Download visibility data from a TART telescope, run StEFCal, and (optionally)
upload the gains.

### Basic usage

```bash
# Solve and save locally (no upload)
tart-stefcal run --tart-name mu-udm

# Solve and upload gains to the telescope
tart-stefcal run --tart-name signal --upload

# Full options: GPU, tight tolerance, multiple outputs
tart-stefcal run --tart-name mu-udm \
    --n 3 --elevation 30 \
    --max-iter 200 --tol 1e-10 \
    --phases --json \
    --output cal_2026-06-25.npz \
    --gpu
```

### Options

| Option | Default | Description |
|---|---|---|
| `--tart-name` | *(required)* | Telescope name (`mu-udm`, `signal`, `za-dias`, `mu-rhodes`) or full API URL |
| `--pw` | `password` | API password for uploading gains |
| `--dir` | temp dir | Directory to download HDF5 files into |
| `--n` | `1` | Number of HDF5 visibility files to download |
| `--elevation` | `45.0` | Source catalog elevation cutoff in degrees |
| `--t-int` | `None` | Integrations per solution interval (`None` = pool all) |
| `--ref-ant` | `0` | Reference antenna for phase gauge |
| `--max-iter` | `100` | Maximum StEFCal iterations per interval |
| `--tol` | `1e-8` | Convergence tolerance on relative gain change |
| `--output` / `-o` | `gains.npz` | Output file path |
| `--phases` | off | Also write `_phases.npy` (radians) |
| `--json` | off | Also write `.json` (TART API format) |
| `--upload` | off | Upload gains to the telescope after solving |
| `--gpu` | off | Use CuPy GPU backend |

### Output files

| File | When | Contents |
|---|---|---|
| `gains.npz` | always | `gains` `(n_sol, n_ant)` complex + `iterations`, `converged`, `max_change` |
| `gains_phases.npy` | `--phases` | Gauge-referenced phases in radians `(n_sol, n_ant)` |
| `gains.json` | `--json` | TART-compatible JSON: `{"gain": [...], "phase_offset": [...]}` |

---

## `tart-stefcal solve` — Local solve from files

Run StEFCal with visibilities and model data already on disk.

### Basic usage

```bash
tart-stefcal solve vis.npy model.npy a1.npy a2.npy --n-ant 24
```

### Full example

```bash
tart-stefcal solve obs_vis.npy sky_model.npy ant1_idx.npy ant2_idx.npy \
    --n-ant 24 --t-int 60 --ref-ant 3 \
    --max-iter 150 --tol 1e-9 \
    --weight per_bl_weight.npy \
    --g0 warm_start_gains.npy \
    --output solution.npz --phases --json \
    --gpu
```

### Options

| Option | Default | Description |
|---|---|---|
| `vis` | *(required)* | Observed visibilities `.npy` or `.npz` — `(ntime, nbl, nchan)` complex |
| `model` | *(required)* | Model visibilities `.npy` or `.npz` — same shape as `vis` |
| `a1` | *(required)* | First antenna indices `.npy` — `(nbl,)` int |
| `a2` | *(required)* | Second antenna indices `.npy` — `(nbl,)` int |
| `--n-ant` | *(required)* | Number of antennas |
| `--t-int` | `None` | Integrations per solution interval |
| `--ref-ant` | `0` | Reference antenna |
| `--weight` | `None` | Per-baseline/channel weights `.npy` |
| `--g0` | `None` | Initial gains (warm start) `.npy` |
| `--max-iter` | `100` | Max iterations per interval |
| `--tol` | `1e-8` | Convergence tolerance |
| `--output` / `-o` | `gains.npz` | Output file path |
| `--phases` | off | Also write `_phases.npy` |
| `--json` | off | Also write `.json` (TART API format) |
| `--gpu` | off | Use CuPy GPU backend |

Input array shapes:
- `vis` / `model`: `(ntime, nbl, nchan)` complex — a 2D `(nbl, nchan)` frame is auto-promoted to `ntime=1`
- `a1` / `a2`: `(nbl,)` integer antenna indices
- `weight`: `(nbl,)` or `(nbl, nchan)` or `(ntime, nbl, nchan)` float
- `g0`: `(n_ant,)` complex

---

## `tart-stefcal phases` — Extract phases from gains

Read a gains file and extract gauge-referenced phases.

```bash
tart-stefcal phases gains.npz --ref-ant 0 --output phases.npy
```

The command auto-detects the `gains` key inside multi-array `.npz` files
(including files written by `run` or `solve`).

| Option | Default | Description |
|---|---|---|
| `gains` | *(required)* | Gains `.npy` or `.npz` file |
| `--ref-ant` | `0` | Reference antenna |
| `--output` / `-o` | `phases.npy` | Output `.npy` file |

---

## Python API

```python
from tart_stefcal import stefcal_solve, referenced_phases
from tart_stefcal import enu_direction_cosines, model_visibilities
import numpy as np

# Build model visibilities from source positions
s_enu = enu_direction_cosines(az_rad, el_rad)    # (nsrc, 3)
model = model_visibilities(s_enu, bl_enu, freqs)  # (nbl, nchan)

# Solve gains
gains, info = stefcal_solve(vis, model, a1, a2, n_ant, ref_ant=0, max_iter=100)

# gains:       (n_sol, n_ant) complex, gains[:, ref_ant] == 1
# info:        dict with keys "iterations", "converged", "max_change" (each (n_sol,))

# Extract phases
phases = referenced_phases(gains, ref_ant=0)  # (..., n_ant) radians
```
