# Three-Stream Comparison Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `smoovie` always image dirty / Tikhonov / L1 per frame and stream them as five separate, colorbar-labelled viewer channels, with independent strength knobs `--l2` (Tikhonov) and `--l1` (reweighted-L1).

**Architecture:** The pipeline drops its single-regulariser branch and always wires the imager dirty into *both* a `TikhonovOperator` and an `L1ReweightOperator`; the IWP smooths only the Tikhonov stream. The viewer's channel set becomes a constant five-tuple with a per-channel unit map; the colorbar is a frontend widget driven by the per-frame vmin/vmax already streamed plus the per-name unit sent once on connect.

**Tech Stack:** Python 3.10+, Typer + hip-cargo (CLI/cab round-trip), Holoscan + cupy (GPU operators), FastAPI + uvicorn (web server), healpy/numpy, three.js (vendored, frontend), pytest.

**Spec:** `docs/superpowers/specs/2026-06-26-three-stream-comparison-viewer-design.md`

## Global Constraints

- **Units are label-only — never rescale a streamed value.** dirty = `Jy/beam`, tikhonov/l1/smooth = `Jy/pixel`, znorm = `""` (dimensionless).
- **Channel set is the exact constant** `NAMES = ("dirty", "tikhonov", "l1", "smooth", "znorm")`; `SYMMETRIC = frozenset({"znorm"})`; `UNITS = {"dirty":"Jy/beam","tikhonov":"Jy/pixel","l1":"Jy/pixel","smooth":"Jy/pixel","znorm":""}`.
- **Strength knobs are plain positive floats**, both default `0.01`: `l2` (Tikhonov) and `l1` (reweighted-L1), each a fraction of `Σw`. No `None`, no per-stream off switch, no `--eta`, no `--regulariser`.
- **`l2` → `TikhonovOperator(eta=…)`, `l1` → `L1ReweightOperator(eta=…)`** — the operators' `eta` constructor arg (strength as a fraction of `Σw`) is unchanged; **do not edit the operator files.**
- **The cli↔core signature must mirror** (minus `backend`/`always_pull_images`); the cli↔cab round-trip must stay byte-identical after `ruff format`. `cabs/smoovie.yml` regenerates via the pre-commit hook — never hand-edit it.
- **Never edit `cabs/*.yml` by hand.** After every code change run `uv run ruff format . && uv run ruff check . --fix`.
- **The IWP smooths the Tikhonov stream only** (so `smooth`/`znorm` track Tikhonov).
- **This machine has a GPU** — GPU-gated tests (`tests/test_smoovie.py`, `tests/test_serve_integration.py::test_fanout_populates_holder`) actually run here; do not treat them as skip-only.

---

### Task 1: Viewer channel metadata + server units

**Files:**
- Modify: `src/kremetart/utils/healpix_viz.py` (NAMES, add UNITS, `geometry_message` unit field)
- Modify: `src/kremetart/utils/web_server.py` (`FrameServer` gains `units`)
- Test: `tests/test_web_viz.py` (update channel/snapshot/geometry tests; add UNITS + unit + server-units tests)
- Test: `tests/test_serve_integration.py` (`served` fixture passes `units`; geometry assertion checks units)

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces: `NAMES = ("dirty","tikhonov","l1","smooth","znorm")`; `UNITS: dict[str,str]`; `geometry_message(name, nside, nest, *, unit: str = "") -> dict` (adds `"unit"`); `FrameServer(holder, *, nside, nest, names, port, tracks=None, units=None, host="127.0.0.1")`.

- [ ] **Step 1: Update the failing tests in `tests/test_web_viz.py`**

Replace `test_names_and_symmetric` and `test_holder_put_and_snapshot_latest_wins`, and add two new tests. Add `UNITS` to the existing import from `kremetart.utils.healpix_viz`.

```python
def test_names_symmetric_and_units():
    assert NAMES == ("dirty", "tikhonov", "l1", "smooth", "znorm")
    assert SYMMETRIC == frozenset({"znorm"})
    assert UNITS == {
        "dirty": "Jy/beam",
        "tikhonov": "Jy/pixel",
        "l1": "Jy/pixel",
        "smooth": "Jy/pixel",
        "znorm": "",
    }


def test_holder_put_and_snapshot_latest_wins():
    h = LatestFrameHolder(NAMES)
    assert h.snapshot() == {n: None for n in NAMES}
    h.put("dirty", 0, 1.0, 0.0, 1.0, b"a")
    h.put("dirty", 1, 2.0, 0.0, 1.0, b"b")  # latest wins
    snap = h.snapshot()
    assert isinstance(snap["dirty"], FrameSlot)
    assert snap["dirty"].seq == 1 and snap["dirty"].data == b"b"
    assert h.current_seq == 1


def test_geometry_message_includes_unit():
    assert geometry_message("dirty", 2, nest=True, unit="Jy/beam")["unit"] == "Jy/beam"
    assert geometry_message("dirty", 2, nest=True)["unit"] == ""  # default empty


def test_frame_server_forwards_units():
    from kremetart.utils.web_server import FrameServer

    holder = LatestFrameHolder(NAMES)
    server = FrameServer(holder, nside=2, nest=True, names=NAMES, units=UNITS, port=8080)
    assert server.units == UNITS
```

Also update `test_geometry_message_shape`: change `geometry_message("raw", nside, nest=True)` to `geometry_message("dirty", nside, nest=True)` and `msg["name"] == "raw"` to `msg["name"] == "dirty"`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_web_viz.py -v`
Expected: FAIL — `test_names_symmetric_and_units` (NAMES still `("raw","smooth","znorm")`, no `UNITS`), import error for `UNITS`, `geometry_message` has no `unit`, `FrameServer` has no `units`.

- [ ] **Step 3: Update `src/kremetart/utils/healpix_viz.py`**

Replace the `NAMES`/`SYMMETRIC` block (lines 16-17) and the `geometry_message` body:

```python
NAMES: tuple[str, ...] = ("dirty", "tikhonov", "l1", "smooth", "znorm")
SYMMETRIC: frozenset[str] = frozenset({"znorm"})
UNITS: dict[str, str] = {
    "dirty": "Jy/beam",
    "tikhonov": "Jy/pixel",
    "l1": "Jy/pixel",
    "smooth": "Jy/pixel",
    "znorm": "",
}
```

```python
def geometry_message(name: str, nside: int, nest: bool, *, unit: str = "") -> dict:
    """Per-name pixel-corner geometry + unit label, sent once on connect."""
    npix = hp.nside2npix(nside)
    vecs = hp.boundaries(nside, np.arange(npix), step=1, nest=nest)  # (npix, 3, 4)
    corners = np.transpose(vecs, (0, 2, 1)).astype(np.float32)  # (npix, 4, 3)
    return {
        "type": "geometry",
        "name": name,
        "nside": int(nside),
        "order": _order(nest),
        "npix": int(npix),
        "unit": unit,
        "corners": corners.reshape(-1).tolist(),
    }
```

- [ ] **Step 4: Update `src/kremetart/utils/web_server.py`**

In `FrameServer.__init__`, add the `units` parameter and store it:

```python
    def __init__(self, holder: LatestFrameHolder, *, nside, nest, names, port, tracks=None, units=None, host="127.0.0.1"):
        self.holder = holder
        self.nside = nside
        self.nest = nest
        self.names = tuple(names)
        self.port = port
        self.host = host
        self.units = dict(units) if units is not None else {}
        self._tracks_msg = tracks_payload(tracks) if tracks is not None else None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
```

In `create_app`, build the geometry messages with units:

```python
        geom_msgs = [geometry_message(n, self.nside, self.nest, unit=self.units.get(n, "")) for n in self.names]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_web_viz.py -v`
Expected: PASS (all, including the new tests).

- [ ] **Step 6: Extend `tests/test_serve_integration.py` to assert units travel end-to-end**

Import `UNITS` (add to the existing `from kremetart.utils.healpix_viz import ...` line). In the `served` fixture, pass units to the server:

```python
    server = FrameServer(holder, nside=2, nest=True, names=NAMES, units=UNITS, port=port, host="127.0.0.1")
```

In `test_stream_protocol_geometry_frames_end`, after the existing geometry assertion, add:

```python
    assert {g["name"]: g["unit"] for g in geo} == UNITS  # each geometry carries its unit label
```

- [ ] **Step 7: Run the serving integration tests (CPU; no GPU needed)**

Run: `uv run pytest tests/test_serve_integration.py -v -k "not fanout"`
Expected: PASS — `test_serves_renderer_html`, `test_serves_vendored_threejs`, `test_stream_protocol_geometry_frames_end` (now asserting units).

- [ ] **Step 8: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
git add src/kremetart/utils/healpix_viz.py src/kremetart/utils/web_server.py tests/test_web_viz.py tests/test_serve_integration.py
git commit -m "feat: constant 5-channel viewer set with per-channel units"
```

---

### Task 2: smoovie wiring — always image dirty / Tikhonov / L1

**Files:**
- Modify: `src/kremetart/cli/smoovie.py` (add `l2`/`l1`, remove `eta`/`regulariser`; update the 3 param-dict/call sites)
- Modify: `src/kremetart/core/smoovie.py` (`SmooviePipeline.__init__` + `compose`; `image_via_app`; `smoovie`; import `UNITS`; pass `units=UNITS` to `FrameServer`)
- Regenerated by pre-commit (do NOT hand-edit): `src/kremetart/cabs/smoovie.yml`
- Test: `tests/test_smoovie.py` (GPU end-to-end zarr var lists)

**Interfaces:**
- Consumes from Task 1: `NAMES`, `UNITS`, `FrameServer(..., units=…)`. The web sink iterates the module-level `NAMES` (now five) and needs no edit; `WebStreamSinkOperator`, `TikhonovOperator`, `L1ReweightOperator`, `IWPKalmanOperator`, `HealpixWriterOperator` are unchanged.
- Produces: `smoovie(..., l2: float = 0.01, l1: float = 0.01, ...)` and the same on `image_via_app` and `SmooviePipeline.__init__`; a pipeline that writes/streams `dirty/tikhonov/l1/filtered/znorm`.

- [ ] **Step 1: Update the GPU end-to-end zarr assertions (failing test)**

In `tests/test_smoovie.py`, update both end-to-end tests' var loops to the new five-variable schema.

In `test_image_via_app_end_to_end` (currently `for var in ("dirty", "filtered", "znorm"):`):

```python
    for var in ("dirty", "tikhonov", "l1", "filtered", "znorm"):
        assert ds[var].shape == (3, npix)
        assert np.all(np.isfinite(ds[var].values))
```

In `test_smoovie_writes_zarr` (currently `for var in ("dirty", "filtered", "znorm"):`):

```python
    for var in ("dirty", "tikhonov", "l1", "filtered", "znorm"):
        assert ds[var].shape[0] == 4
        assert np.all(np.isfinite(ds[var].values))
```

- [ ] **Step 2: Run the end-to-end test to verify it fails**

Run: `uv run pytest tests/test_smoovie.py::test_image_via_app_end_to_end -v`
Expected: FAIL — the zarr has only `dirty/filtered/znorm` (no `tikhonov`/`l1` yet), so `ds["tikhonov"]` raises `KeyError`. (This is a GPU test; it runs on this machine.)

- [ ] **Step 3: Edit `src/kremetart/cli/smoovie.py` — replace the `eta` and `regulariser` options with `l2` and `l1`**

Replace the two `Annotated` blocks for `eta` (lines ~101-106) and `regulariser` (lines ~107-112) with:

```python
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
```

Then in all **three** parameter blocks — the `preflight_remote_must_exist(...)` dict, the `smoovie_core(...)` call, and the `run_in_container(...)` dict — replace the two lines `eta=eta,` and `regulariser=regulariser,` with:

```python
                    l2=l2,
                    l1=l1,
```

(The `run_in_container` block is indented 12 spaces; match the surrounding indentation in each site.)

- [ ] **Step 4: Edit `src/kremetart/core/smoovie.py` — import, signatures, and rewire `compose`**

(a) Update the healpix_viz import (line ~32):

```python
from kremetart.utils.healpix_viz import NAMES, UNITS, LatestFrameHolder
```

(b) In `SmooviePipeline.__init__`, change the two signature lines `eta: float | None = None,` and `regulariser: str = "tikhonov",` to:

```python
        l2: float = 0.01,
        l1: float = 0.01,
```

and change the two assignment lines `self.eta = eta` and `self.regulariser = regulariser` to:

```python
        self.l2 = l2
        self.l1 = l1
```

(c) Replace the **entire** `compose` method body with the always-both wiring:

```python
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
        self.add_flow(imager, l1, {("cube", "cube"), ("time_out", "time_out")})
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
```

- [ ] **Step 5: Edit `src/kremetart/core/smoovie.py` — `image_via_app` and `smoovie` signatures + server units**

(a) In `image_via_app`, change the `eta: float | None = None,` and `regulariser: str = "tikhonov",` keyword-only params to:

```python
    l2: float = 0.01,
    l1: float = 0.01,
```

and in the `SmooviePipeline(...)` construction inside `image_via_app`, replace `eta=eta,` and `regulariser=regulariser,` with:

```python
            l2=l2,
            l1=l1,
```

(b) In `smoovie`, change the `eta: float | None = None,` and `regulariser: str = "tikhonov",` params to:

```python
    l2: float = 0.01,
    l1: float = 0.01,
```

(c) In `smoovie`, the `FrameServer(...)` construction (line ~343) gains units:

```python
        server = FrameServer(holder, nside=nside, nest=True, names=NAMES, units=UNITS, port=port, tracks=tracks)
```

(d) In `smoovie`, the `image_via_app(...)` call (lines ~352-367) replaces `eta=eta,` and `regulariser=regulariser,` with:

```python
                l2=l2,
                l1=l1,
```

(e) Update the `eta`/`regulariser` prose in the `smoovie` and `image_via_app` docstrings: state that the pipeline always images dirty / Tikhonov / L1, that `l2`/`l1` are the Tikhonov/reweighted-L1 strengths as fractions of `Σw` (defaults `0.01`, independent and tunable — point at `scripts/compare_regularisers.py`), and that the IWP smooths the Tikhonov stream.

- [ ] **Step 6: Format and lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: clean (no errors).

- [ ] **Step 7: Run the structure + round-trip tests (CPU)**

Run: `uv run pytest tests/test_structure.py tests/test_roundtrip.py -v`
Expected: PASS — `test_core_signature_mirrors_cli` confirms `cli.smoovie` and `core.smoovie` now share `{l1, l2}` and neither has `eta`/`regulariser`; `test_roundtrip.py` confirms `cabs/smoovie.yml` regenerated with `l1`/`l2` inputs and round-trips byte-identical after `ruff format`.

> If the round-trip fails, the cab is stale: re-run `git add -u` after the pre-commit hook regenerates `cabs/smoovie.yml`, and fix the **CLI source** (never the cab).

- [ ] **Step 8: Run the full smoovie + fan-out suite (GPU; this machine)**

`tests/test_smoovie.py` is GPU-gated as a whole (a module-level `skipif`); on this machine it runs. This covers the `sm_stub` host-wiring tests (which ride the new `l2`/`l1` defaults, never passing `eta`/`regulariser`) **and** the end-to-end zarr-schema tests.

Run: `uv run pytest tests/test_smoovie.py tests/test_serve_integration.py::test_fanout_populates_holder -v`
Expected: PASS — host-wiring orchestration unchanged; the durable zarr now holds `dirty/tikhonov/l1/filtered/znorm` (each finite); the serve-mode fan-out populates the holder for all five `NAMES`. (`test_fanout_populates_holder` self-skips if `tests/data/*.hdf` is absent; confirm the HDFs are present so it actually runs.)

> This is the live confirmation of the §9 runtime unknown — the reader's `WEIGHT`/`B_ROT`/`BORESIGHT` ports now fan **3-way** (imager + tikhonov + l1). If Holoscan errors on the broadcast here, stop and report: the fallback is an explicit duplicate/broadcast, which is a plan change to discuss, not a silent workaround.

- [ ] **Step 9: Commit**

```bash
git add src/kremetart/cli/smoovie.py src/kremetart/core/smoovie.py src/kremetart/cabs/smoovie.yml tests/test_smoovie.py
git commit -m "feat: always image dirty/Tikhonov/L1; --l2/--l1 strengths"
```

---

### Task 3: Colorbar in the live viewer

**Files:**
- Modify: `src/kremetart/static/index.html` (store the per-name unit; add a colorbar widget; update it per displayed frame)

**Interfaces:**
- Consumes from Tasks 1-2: the `unit` field on each `geometry` message and a running five-channel pipeline.
- Produces: a colorbar on the rendered page. No new JS module/function is exported; there is no JS test harness in this repo, so verification is a real GPU run + browser screenshot (Step 5).

This task has no unit test (the renderer is not unit-tested, consistent with the existing frontend). It is verified by running the live pipeline and inspecting the page.

- [ ] **Step 1: Store the per-name unit when geometry arrives**

In `src/kremetart/static/index.html`, in the WebSocket `onmessage` geometry branch, record the unit on the per-name geometry state. Change:

```javascript
      if(msg.type==="geometry"){
        if(!geom[msg.name]){ geom[msg.name]=buildMesh(msg);
          bytesPerFrame=Math.max(bytesPerFrame, msg.npix*4);
          updateCacheReadout(); }
        refreshNames();
```

to:

```javascript
      if(msg.type==="geometry"){
        if(!geom[msg.name]){ geom[msg.name]=buildMesh(msg);
          geom[msg.name].unit = msg.unit || "";
          bytesPerFrame=Math.max(bytesPerFrame, msg.npix*4);
          updateCacheReadout(); }
        refreshNames();
```

- [ ] **Step 2: Add the colorbar widget to the layout**

After the `root.appendChild(bar);` line (end of the control-bar block, ~line 37), add the colorbar DOM and a one-time ramp paint. It reuses `mkLabel` (defined just below in the source — JS function declarations hoist, so referencing it here is fine; place this block *after* the `mkLabel` definition to keep reading order clean — i.e. insert it right after the `nameSel`/control widgets are created, just before `bar.append(...)`). Insert immediately before the `bar.append(mkLabel("output"), ...)` line:

```javascript
// ---- colorbar --------------------------------------------------------------
const cbar = document.createElement("div");
cbar.style.cssText = `position:fixed;right:16px;top:50%;transform:translateY(-50%);
  display:flex;flex-direction:column;gap:6px;align-items:flex-end;pointer-events:none;
  background:rgba(10,14,23,.55);padding:8px 10px;border-radius:8px;border:1px solid #1d2940;`;
const cbUnit = mkLabel(""); cbUnit.style.color = "#cfe0f5"; cbUnit.style.fontSize = "12px";
const cbRow = document.createElement("div");
cbRow.style.cssText = "display:flex;gap:6px;align-items:stretch;";
const cbarLabels = document.createElement("div");
cbarLabels.style.cssText = `display:flex;flex-direction:column;justify-content:space-between;
  font-size:11px;color:#7fa8d8;text-align:right;text-shadow:0 1px 2px #000;`;
const cbMax = mkLabel(""), cbMid = mkLabel(""), cbMin = mkLabel("");
cbarLabels.append(cbMax, cbMid, cbMin);
const cbarCanvas = document.createElement("canvas");
cbarCanvas.width = 16; cbarCanvas.height = 180;
cbarCanvas.style.cssText = "border:1px solid #2a3a55;border-radius:3px;";
cbRow.append(cbarLabels, cbarCanvas);
cbar.append(cbUnit, cbRow);
root.appendChild(cbar);

// Paint the fixed colormap ramp once (top = max, bottom = min) — same inline ramp as paint().
(function paintRamp(){
  const ctx = cbarCanvas.getContext("2d"), h = cbarCanvas.height, w = cbarCanvas.width;
  for(let y=0;y<h;y++){ const tt = 1 - y/(h-1);
    const r = Math.min(1, 1.5*tt + 0.1*Math.sin(3.1*tt));
    const g = Math.max(0, 1.1*tt*tt - 0.05);
    const b = Math.max(0, 0.6*Math.sin(Math.PI*tt) + 0.15*(1-tt));
    ctx.fillStyle = `rgb(${r*255|0},${g*255|0},${b*255|0})`; ctx.fillRect(0, y, w, 1);
  }
})();

function fmtTick(v){ if(!isFinite(v)) return "—"; const a = Math.abs(v);
  return (a !== 0 && (a < 1e-2 || a >= 1e4)) ? v.toExponential(2) : v.toFixed(3); }
function updateColorbar(vmin, vmax, unit){
  cbMax.textContent = fmtTick(vmax);
  cbMid.textContent = fmtTick((vmin + vmax) / 2);
  cbMin.textContent = fmtTick(vmin);
  cbUnit.textContent = unit || "";
}
```

- [ ] **Step 3: Drive the colorbar from the displayed frame**

In `showSeq`, update the colorbar from the frame's stored scale and the channel's unit. Change:

```javascript
function showSeq(name, seq){
  const m=rings[name]; if(!m) return;
  const f=m.get(seq); if(!f) return;     // aged out of window
  paint(name, f.values, f.vmin, f.vmax);
  viewSeq=seq;
  rebuildOverlay(seq);
}
```

to:

```javascript
function showSeq(name, seq){
  const m=rings[name]; if(!m) return;
  const f=m.get(seq); if(!f) return;     // aged out of window
  paint(name, f.values, f.vmin, f.vmax);
  updateColorbar(f.vmin, f.vmax, geom[name]?.unit || "");
  viewSeq=seq;
  rebuildOverlay(seq);
}
```

(`selectName` already calls `showSeq`, so switching channels refreshes the bar; scrub-back also refreshes it because `timeInput.oninput` calls `showSeq`.)

- [ ] **Step 4: Sanity-check the served HTML still loads (CPU)**

Run: `uv run pytest tests/test_serve_integration.py -v -k "renderer or vendored"`
Expected: PASS — `test_serves_renderer_html` and `test_serves_vendored_threejs` (the page still serves; no CDN references introduced).

- [ ] **Step 5: Live browser smoke test (GPU)**

Image a few frames and serve, then inspect the page:

```bash
uv run kremetart smoovie --hdf-dir tests/data --output /tmp/cbar_smoke.zarr \
  --nside 16 --nframes 6 --serve --port 8080 --overwrite
```

(`tests/data` must contain `*.hdf`; otherwise point `--hdf-dir` at a directory that does.) The command prints `kremetart live view: http://localhost:8080/` and freezes the session for inspection. Open that URL in a browser and confirm:
- the `output` dropdown lists **dirty, tikhonov, l1, smooth, znorm**;
- a colorbar is visible on the right with `vmin`/`mid`/`vmax` ticks and a unit label;
- switching the dropdown to `dirty` shows unit **Jy/beam**; `tikhonov`/`l1`/`smooth` show **Jy/pixel**; `znorm` shows no unit and a symmetric (zero-centred) range;
- the tick values change when scrubbing the time slider.

Capture a screenshot for the record, then stop the server (Ctrl-C).

- [ ] **Step 6: Commit**

```bash
git add src/kremetart/static/index.html
git commit -m "feat: add unit-labelled colorbar to the live viewer"
```

---

## Self-Review

**Spec coverage:**
- §3 units (label-only) → Task 1 `UNITS`, Task 3 colorbar labels; no value rescaled anywhere. ✓
- §4 channel metadata (`NAMES`/`UNITS`/`geometry_message` unit) → Task 1. ✓
- §5 compose rewiring (always both, IWP smooths Tikhonov, writer/sink 5 channels) → Task 2 Step 4c. ✓
- §6 `--l2`/`--l1` defaults `0.01`, remove `--eta`/`--regulariser`, signature mirror + round-trip → Task 2 Steps 3-5, 7. ✓
- §7 sink picks up `NAMES` (no edit) + `FrameServer` units → Task 1 Step 4, Task 2 Step 5c. ✓
- §8 colorbar → Task 3. ✓
- §9 3-way broadcast verification → Task 2 Step 8 (live GPU run). ✓
- §10 test plan items 1-8 → Task 1 Steps 1/6, Task 2 Steps 1/7/8. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**Type consistency:** `l2`/`l1` are plain `float` everywhere (cli, `smoovie`, `image_via_app`, `SmooviePipeline`); `TikhonovOperator(eta=self.l2)`, `L1ReweightOperator(eta=self.l1)`; `NAMES`/`UNITS` names match exactly across Tasks 1-3 and the writer `var_specs`/sink port map. ✓
