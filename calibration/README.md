# tart-stefcal

StEFCal: alternating per-antenna complex-gain least squares solver.

## Install

```bash
pip install tart-stefcal           # CPU only
pip install tart-stefcal[gpu]      # with CuPy GPU support
```

## Usage

```bash
# Solve gains
tart-stefcal solve vis.npy model.npy a1.npy a2.npy --n-ant 24 --output gains.npz

# Extract phases
tart-stefcal phases gains.npz --ref-ant 0 --output phases.npy
```
