# Three-Stream Comparison Viewer (dirty / Tikhonov / L1) — Design

**Date:** 2026-06-26
**Status:** Design — approved for planning
**Supersedes (partially):** the `--eta` / off-by-default smoovie wiring from
`docs/superpowers/specs/2026-06-23-tikhonov-cg-regularisation-design.md` and the
`--regulariser {tikhonov,l1}` selector from
`docs/superpowers/specs/2026-06-25-reweighted-l1-deconvolution-design.md`.

## 1. Goal

Make `smoovie` always image **all three** maps per frame and stream them as
separate, switchable channels in the live web viewer so they can be compared at
the same instant:

- **`dirty`** — the natural-weighted dirty image (Jy/beam).
- **`tikhonov`** — the L2 (Tikhonov-CG) deconvolved model (Jy/pixel).
- **`l1`** — the reweighted-L1 (FISTA) deconvolved model (Jy/pixel).
- **`smooth`** — the IWP-Kalman-filtered Tikhonov model (Jy/pixel).
- **`znorm`** — the Tikhonov stream's normalised innovation (dimensionless).

Each channel carries a **colorbar** labelled in its unit. The Tikhonov and L1
solvers get **independent strength knobs** (`--l2`, `--l1`), both on by default.

## 2. Motivation

The deconvolution feature (PR #12) already wired a *single* regulariser into
`smoovie` via `--regulariser`, and the IWP passed its input through as a
mislabelled `raw` channel — so the viewer never showed the genuine dirty image,
and Tikhonov vs L1 could only be compared across separate runs. The goal now is
a side-by-side comparison in one run: dirty (the data), both deconvolved models,
and the smoothed Tikhonov model, with correct units and a colorbar.

The FISTA stream currently runs slower than real time. **Optimising it is a
separate, later task.** This design accepts the current cost (running both
solvers per frame; the Tikhonov CG is cheap relative to the L1 FISTA) and is
explicitly scoped to the comparison viewer, not performance.

## 3. Units (label-only; no numeric rescale)

The unit requirement is satisfied by *labelling*, not by rescaling the streamed
values — the numbers are already in the right units:

- The imager emits `img.real / Σw` (`utils/healpix_dft.py::dirty_map`, line 119)
  — the **natural-weighted dirty image**, peak-normalised by the central PSF
  value `Σw`. A point source of flux `S` peaks at `≈ S`. This is **Jy/beam**.
- The Tikhonov/L1 operators solve `H x = b` (with `H = B Mᴴ W M B`,
  `b = dirty·Σw`) for the deconvolved sky model `x`. By construction the forward
  operator `M` maps a per-pixel sky to visibilities, so `x` is flux **per
  HEALPix pixel** — **Jy/pixel**.
- `smooth` is the IWP-Kalman filter applied to the Tikhonov model, so it is also
  **Jy/pixel**.
- `znorm` is a normalised innovation — **dimensionless** (empty unit string).

No value in any channel is rescaled. Units are attached as metadata and drawn on
the colorbar.

## 4. Channel definitions — `src/kremetart/utils/healpix_viz.py`

Because both solvers now run unconditionally, the channel set is **constant** —
the old `NAMES = ("raw","smooth","znorm")` is replaced (and the `raw`→`dirty`
mislabel fixed):

```python
NAMES: tuple[str, ...] = ("dirty", "tikhonov", "l1", "smooth", "znorm")
UNITS: dict[str, str] = {
    "dirty": "Jy/beam",
    "tikhonov": "Jy/pixel",
    "l1": "Jy/pixel",
    "smooth": "Jy/pixel",
    "znorm": "",
}
SYMMETRIC: frozenset[str] = frozenset({"znorm"})  # unchanged
```

`geometry_message(name, nside, nest, *, unit="")` gains a `unit` field in its
returned dict (sent once per name on connect), so the frontend can label each
channel's colorbar without a per-frame unit message. `frame_header`,
`encode_frame`, `LatestFrameHolder`, `tracks_payload` are unchanged.

No `viewer_channels(regularise)` helper is introduced — there is only one mode.

## 5. Pipeline wiring — `src/kremetart/core/smoovie.py` `compose()`

The `eta is None` un-regularised branch is **deleted**. The graph is now
single-mode and always builds both deconvolvers off the shared imager dirty +
reader geometry. The Tikhonov stream is the only one fed to the IWP (so
`smooth`/`znorm` track Tikhonov, as required).

| viewer/writer channel | source operator + port |
|---|---|
| `dirty`    | `tikhonov.dirty` (raw-dirty passthrough) |
| `tikhonov` | `iwp.cube` (Tikhonov model passed through the filter) |
| `l1`       | `l1.cube` |
| `smooth`   | `iwp.filtered` |
| `znorm`    | `iwp.znorm` |

Flows (`add_flow`):

```
reader  → imager   {VISIBILITY, WEIGHT, B_ROT, BORESIGHT, time}
imager  → tikhonov {(cube,cube), (time_out,time_out)}
imager  → l1       {(cube,cube), (time_out,time_out)}
reader  → tikhonov {WEIGHT, B_ROT, BORESIGHT}
reader  → l1       {WEIGHT, B_ROT, BORESIGHT}
tikhonov → iwp     {(cube,cube), (time_out,time_out)}

# durable writer (always, serve or not)
iwp      → writer  {(cube,tikhonov), (filtered,filtered), (znorm,znorm), (time_out,time_out)}
tikhonov → writer  {(dirty,dirty)}
l1       → writer  {(cube,l1)}

# web sink (only when holder is not None)
iwp      → sink    {(cube,tikhonov), (filtered,smooth), (znorm,znorm), (time_out,time_out)}
tikhonov → sink    {(dirty,dirty)}
l1       → sink    {(cube,l1)}
```

Writer `var_specs` (input-port → stored-variable):
`(("dirty","dirty"), ("tikhonov","tikhonov"), ("l1","l1"), ("filtered","filtered"), ("znorm","znorm"))`.

`l1` exposes only its `cube` output. (During implementation its previously-unused
`dirty`/`time_out` output ports were removed: Holoscan refuses to schedule an
operator whose transmitter has no receiver — `[E00070] No receiver connected to
transmitter` — so "leaving them unconnected" is not possible. `L1ReweightOperator`
is therefore no longer port-identical to `TikhonovOperator`; it is used only here.)
`tikhonov.cube` goes only to the IWP; the `tikhonov` channel is sourced from
`iwp.cube` (the filter passes its input through unchanged), keeping fan-out low.

**Operators are unchanged.** `TikhonovOperator` and `L1ReweightOperator` keep
their `eta` constructor argument (meaning "strength as a fraction of Σw");
`compose()` passes `eta=self.l2` and `eta=self.l1` respectively.

## 6. CLI / core signature — `cli/smoovie.py` + `core/smoovie.py`

- **Remove** `eta: float | None = None` and `regulariser: str = "tikhonov"`.
- **Add** `l2: Annotated[float, typer.Option(help=...)] = 1e-2` — Tikhonov
  (L2) strength as a fraction of Σw (`λ_tik = l2·Σw`).
- **Add** `l1: Annotated[float, typer.Option(help=...)] = 1e-2` — reweighted-L1
  strength as a fraction of Σw (`λ_l1 = l1·Σw`).

`l1`/`l2` are valid identifiers (no `lambda` keyword clash) and ruff-clean (not
the single-char `l`), so they round-trip with no `typer.Option("--name", …)`
override. Defaults `1e-2` match `scripts/compare_regularisers.py`'s established
matched-strength default; both are independent and tunable, and the docstrings
say so and point at the offline comparison script for refinement.

The signature change must be mirrored in `core/smoovie.smoovie`,
`core/smoovie.image_via_app`, and `SmooviePipeline.__init__` (the
`StimelaMeta(skip=True)` flags `backend`/`always_pull_images` stay CLI-only).
`tests/test_structure.py` enforces the cli↔core mirror; `tests/test_roundtrip.py`
enforces the cli↔cab round trip — the cab regenerates with `l1`/`l2` inputs and
without `eta`/`regulariser`.

Strengths are plain positive floats (no `None`, no per-stream off switch): the
pipeline always produces both streams. Setting a strength to `0` is a degenerate
solve (un-regularised), not an error and not special-cased; docstrings advise
strengths `> 0`.

## 7. Viewer plumbing — `web_server.py`, `web_sink.py`

- `WebStreamSinkOperator` iterates the module-level `NAMES` (now five), so its
  declared inputs become `dirty/tikhonov/l1/smooth/znorm`; no signature change.
  It keeps using `SYMMETRIC` for the per-name scale mode.
- `FrameServer` gains a `units: dict[str, str]` parameter; `create_app` builds
  the per-name geometry messages with `unit=units[name]`. `smoovie()` passes
  `UNITS`. `stream_handler` is unchanged (frame headers already carry
  `vmin/vmax/seq/t`).
- `smoovie()` constructs `LatestFrameHolder(NAMES)` and
  `FrameServer(holder, nside=…, nest=True, names=NAMES, units=UNITS, …)`.

## 8. Colorbar — `src/kremetart/static/index.html`

A fixed-position colorbar widget, styled to match the existing dark viewer
(`#0a0e17` background, `#cfe0f5` text, monospace):

- A vertical colormap gradient strip, painted once from the same inline ramp
  used in `paint()` (so the bar matches the sphere's colours).
- Three numeric tick labels — `vmax` (top), midpoint (centre), `vmin` (bottom) —
  formatted compactly (e.g. `toExponential(2)`), updated per displayed frame.
- The channel's unit string (from the geometry `unit` field).

Wiring:

- The geometry handler stores `geom[msg.name].unit = msg.unit || ""`.
- A new `updateColorbar(vmin, vmax, unit)` is called from `showSeq(...)` (which
  already holds the displayed frame's `vmin/vmax`) and on channel switch
  (`selectName`), so the bar tracks the frame currently on screen, including
  during scrub-back. The scale spans the frame's own min/max (or symmetric for
  `znorm`), exactly as the sphere colouring already does.

The frontend is not unit-tested (consistent with the existing renderer).

## 9. Runtime unknown (verify on first GPU run)

Operators are GPU-only and not pytested; `compose()` wiring is exercised only by
the GPU end-to-end test. The one new structural risk is **Holoscan implicit
3-way port broadcast**: the reader's `WEIGHT`/`B_ROT`/`BORESIGHT` outputs now
fan to `imager` + `tikhonov` + `l1` (the current code already relies on 2-way
fan-out; this escalates to 3-way). This is expected to work (Holoscan inserts a
broadcast for one-output-to-many-inputs), but it is the thing to confirm when
the pipeline first runs on a GPU box. Fallback if it misbehaves: an explicit
broadcast/duplicate, noted in the plan but not implemented pre-emptively.

## 10. Test plan

CPU-unit-testable (run on CI):

1. `NAMES == ("dirty","tikhonov","l1","smooth","znorm")` and `UNITS` maps each
   to the expected unit string; `SYMMETRIC == {"znorm"}`.
2. `geometry_message(..., unit="Jy/beam")` includes `"unit": "Jy/beam"`; default
   `unit` is `""`.
3. `LatestFrameHolder(NAMES)` snapshot has the five keys; existing latest-wins /
   thread-safety / finish tests updated to the new names.
4. `stream_handler` over the five names emits geometry (with `unit`) → one
   header+binary pair per name → `end` (extend the existing fake-WebSocket test).
5. `FrameServer(..., names=NAMES, units=UNITS)` builds an app with `/` and
   `/stream`; geometry messages carry the units.
6. `tests/test_structure.py` — cli↔core signature mirror holds with `l1`/`l2`
   and without `eta`/`regulariser`.
7. `tests/test_roundtrip.py` — cab regenerates and round-trips.

GPU-gated (skip on CI; run on the user's box — the real smoke test):

8. `image_via_app(...)` (defaults) writes a zarr holding **all five** vars
   `dirty/tikhonov/l1/filtered/znorm`, each `(nframes, npix)` and finite. Update
   the existing `test_image_via_app_end_to_end` and `test_smoovie_writes_zarr`
   accordingly.

## 11. Out of scope

- FISTA performance optimisation (the explicit next task).
- Any per-stream "disable for speed" switch (contradicts always-both).
- The downstream self-cal operator.
- `scripts/compare_regularisers.py` changes (it keeps its single `--eta`).
- Frontend unit tests / visual regression.
- Numeric unit conversion (units are already correct; this is label-only).
```
