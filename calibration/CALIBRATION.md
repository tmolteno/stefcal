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
    ├── __init__.py             # Public API: stefcal_solve, referenced_phases, main
    ├── stefcal.py              # Core solver (same algorithm as kremetart/utils/stefcal.py)
    └── cli.py                  # argparse CLI: solve and phases subcommands
```

## Key differences from `kremetart/utils/stefcal.py`

| Aspect | kremetart | tart-stefcal |
|---|---|---|
| `xp.complex128` | Used (breaks on NumPy ≥ 2.0) | Replaced with `complex` — works on all NumPy versions |
| GPU deps | Lazy import per `python-standards.md` §2 | `cupy` is an optional extra `[gpu]`; CLI flag `--gpu` |
| CLI wrapper | None (library only) | Full `argparse` CLI with `solve` and `phases` subcommands |
| Package manager | `uv` (workspace member) | `uv` (workspace member under `calibration/`) |
| Stimela/cab integration | Yes (via `cli/core/cabs` layers) | No — standalone tool |

## Install

```bash
# CPU only
pip install tart-stefcal

# With GPU support
pip install tart-stefcal[gpu]
```

## CLI usage

### Solve gains

```bash
tart-stefcal solve vis.npy model.npy a1.npy a2.npy \
    --n-ant 24 --t-int 10 --ref-ant 0 \
    --max-iter 100 --tol 1e-8 \
    --output gains.npz --phases
```

Inputs:
- `vis` / `model`: `(ntime, nbl, nchan)` complex visibilities (`.npy` or single-array `.npz`)
- `a1` / `a2`: `(nbl,)` integer antenna indices per baseline (`.npy`)
- `--n-ant`: number of antennas (required)
- `--gpu`: use CuPy backend

Outputs:
- `gains.npz`: `gains` `(n_sol, n_ant)` complex, plus `iterations`, `converged`, `max_change`
- `gains_phases.npy`: gauge-referenced phases `(n_sol, n_ant)` if `--phases` given

### Extract phases from existing gains

```bash
tart-stefcal phases gains.npz --ref-ant 0 --output phases.npy
```

The `phases` command auto-detects the `gains` key inside a multi-array `.npz`.

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
| `cupy>=13.0` | Optional `[gpu]` | GPU backend via `--gpu` flag |
