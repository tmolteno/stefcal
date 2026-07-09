# CAL_FIX.md — Gain Amplitude Inversion for TART Upload

## Problem

The gains and phases calculated by `tart-stefcal` did not work when uploaded
to the TART telescope. Images showed no improvement after calibration.

## Root cause

The TART telescope applies uploaded gains as a **multiplicative correction**
to measured visibilities, not a division. From
`tart.imaging.calibration.CalibratedVisibility.get_visibility`:

```python
v * gain[i] * gain[j] * exp(-1j * (phase_offset[i] - phase_offset[j]))
```

This is equivalent to:

```
V_corrected = V_raw * G_corr_p * conj(G_corr_q)
```

where `G_corr = gain * exp(-1j * phase_offset)`.

The measured visibility contains the true gain corruption:

```
V_raw = G_true_p * conj(G_true_q) * V_sky
```

For `V_corrected = V_sky`, the correction must be `G_corr = 1 / G_true`,
which decomposes as:

| Quantity | StEFCal estimates | Telescope needs |
|---|---|---|
| Amplitude | `\|g\|` | `1 / \|g\|` (inverted) |
| Phase | `angle(g)` radians | `angle(g)` radians (same) |

**Before the fix**, `_gains_to_json_dict` uploaded `|g|` (the raw estimated
amplitude). The telescope then multiplied by `|g|` instead of `1/|g|`,
amplifying the corruption rather than correcting it.

## Fix

`_gains_to_json_dict` now accepts an `invert_gain` parameter:

```python
def _gains_to_json_dict(gains, *, invert_gain=False):
    amp = np.abs(gains)
    if invert_gain:
        amp = np.where(amp > 0, 1.0 / amp, 0.0)
    ...
```

`_upload_gains_json` always passes `invert_gain=True`:

```python
api_dict = _gains_to_json_dict(gains, invert_gain=True)
```

On-disk outputs (`.npz`, `.json`, `_phases.npy`) still contain the raw
estimated gains — only the upload payload inverts the amplitude.

## Verification

The test `test_upload_format_matches_tart_api` in
`calibration/tests/test_cli.py` verifies both paths:

- On-disk JSON: `gain = [2.0, 1.0]` (raw amplitudes)
- Upload payload: `gain = [0.5, 1.0]` (inverted amplitudes)
- Phase: unchanged in both cases

## Files changed

- `calibration/src/tart_stefcal/cli.py` — `_gains_to_json_dict` gains
  `invert_gain` parameter; `_upload_gains_json` passes `invert_gain=True`
- `calibration/tests/test_cli.py` — `test_upload_format_matches_tart_api`
  updated to verify both raw and inverted amplitudes
