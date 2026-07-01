"""Integration tests for the live web-viewer serving path.

Closes two risks the unit tests leave open, with real execution:

1. The Holoscan fan-out: in serve mode SmooviePipeline wires iwp -> writer AND iwp -> web sink off
   the same iwp output ports. ``test_fanout_populates_holder`` runs the real imaging app with a
   holder and asserts the sink populated it and the durable zarr was still written.
2. The HTTP/WebSocket serving: FrameServer serves the renderer, the vendored three.js, and the
   /stream protocol on a real uvicorn thread. The serving tests start the server and drive it with
   a stdlib HTTP client and a websockets client.
"""

import asyncio
import json
import socket
import time
import urllib.request
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from kremetart.utils.healpix_viz import NAMES, UNITS, LatestFrameHolder, encode_frame
from kremetart.utils.web_server import FrameServer


def _free_port() -> int:
    """Grab an ephemeral port the OS just confirmed is free."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gpu() -> bool:
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        import healpy  # noqa: F401
        import holoscan  # noqa: F401

        return True
    except Exception:
        return False


# --- serving: HTTP renderer, vendored three.js, /stream protocol (no GPU) ------------------------


@pytest.fixture
def served():
    """A FrameServer on an ephemeral port with one finished frame per name; torn down after."""
    npix = 48  # nside=2
    holder = LatestFrameHolder(NAMES)
    for name in NAMES:  # one frame each, shared seq 0
        vmin, vmax, data = encode_frame(np.zeros(npix, dtype=np.float32), symmetric=(name == "znorm"))
        holder.put(name, 0, 1.0, vmin, vmax, data)
    holder.finish()

    port = _free_port()
    server = FrameServer(holder, nside=2, nest=True, names=NAMES, units=UNITS, port=port, host="127.0.0.1")
    url = server.start()

    deadline = time.time() + 10.0  # wait for uvicorn to bind
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.5).read()
            break
        except Exception:
            time.sleep(0.1)

    yield url, port
    server.stop()


def test_serves_renderer_html(served):
    url, _port = served
    body = urllib.request.urlopen(url, timeout=5).read().decode()
    assert "importmap" in body
    assert "/static/vendor/three.module.min.js" in body  # vendored path, no CDN
    assert "cdnjs" not in body


def test_serves_vendored_threejs(served):
    url, _port = served
    body = urllib.request.urlopen(url + "static/vendor/three.module.min.js", timeout=5).read()
    assert len(body) > 100_000  # the real minified three.js is ~670 KB
    assert b"Three.js Authors" in body[:400]  # license banner of the genuine library


def test_stream_protocol_geometry_frames_end(served):
    _url, port = served
    import websockets

    async def collect():
        texts, nbin = [], 0
        uri = f"ws://127.0.0.1:{port}/stream"
        async with websockets.connect(uri, max_size=None) as ws:
            while True:
                m = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(m, (bytes, bytearray)):
                    nbin += 1
                    continue
                msg = json.loads(m)
                texts.append(msg)
                if msg.get("type") == "end":
                    break
        return texts, nbin

    texts, nbin = asyncio.run(collect())
    types = [t["type"] for t in texts]
    geo = [t for t in texts if t["type"] == "geometry"]
    frames = [t for t in texts if t["type"] == "frame"]

    assert {g["name"] for g in geo} == set(NAMES)  # geometry once per name
    assert {f["name"] for f in frames} == set(NAMES)  # one frame per name
    assert nbin == len(NAMES)  # one binary payload per frame header
    assert types[-1] == "end"  # clean terminator
    assert types.index("geometry") < types.index("frame") < types.index("end")
    assert {g["name"]: g["unit"] for g in geo} == UNITS  # each geometry carries its unit label


# --- fan-out: real GPU pipeline populates the holder AND writes the zarr -------------------------


@pytest.mark.skipif(not _gpu(), reason="requires a CUDA device + cupy/holoscan/healpy")
def test_fanout_populates_holder(tmp_path):
    """Serve-mode pipeline (iwp -> writer AND iwp -> web sink) runs; sink fills holder, writer writes."""
    data_dir = Path(__file__).resolve().parent / "data"
    hdf_paths = sorted(data_dir.glob("*.hdf"))
    if not hdf_paths:
        pytest.skip("no test HDFs present in tests/data")

    from kremetart.core.smoovie import image_via_app

    nside = 16
    npix = 12 * nside * nside
    holder = LatestFrameHolder(NAMES)
    out = tmp_path / "fanout.zarr"

    image_via_app(hdf_paths, nside, output_zarr=out, holder=holder, nframes=3)

    snap = holder.snapshot()
    for name in NAMES:
        assert snap[name] is not None, f"sink never populated {name!r} -- fan-out broken"
        assert len(snap[name].data) == npix * 4  # float32 little-endian, full map
    assert holder.current_seq == 2  # nframes - 1, shared seq advanced once per integration
    assert out.exists()  # writer still wrote the durable zarr -> both broadcast branches ran
