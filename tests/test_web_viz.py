import asyncio
import json
import threading

import healpy as hp
import numpy as np

from kremetart.utils.healpix_viz import (
    NAMES,
    SYMMETRIC,
    UNITS,
    FrameSlot,
    LatestFrameHolder,
    encode_frame,
    frame_header,
    geometry_message,
    tracks_payload,
)
from kremetart.utils.web_server import FrameServer


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


def test_encode_frame_plain():
    values = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    vmin, vmax, data = encode_frame(values, symmetric=False)
    assert (vmin, vmax) == (0.0, 3.0)
    assert np.frombuffer(data, dtype="<f4").tolist() == [0.0, 1.0, 2.0, 3.0]


def test_encode_frame_symmetric_centers_on_zero():
    values = np.array([-2.0, 0.5, 1.0], dtype=np.float32)
    vmin, vmax, _ = encode_frame(values, symmetric=True)
    assert vmax == 2.0 and vmin == -2.0


def test_encode_frame_empty():
    vmin, vmax, data = encode_frame(np.array([], dtype=np.float32), symmetric=False)
    assert (vmin, vmax) == (0.0, 0.0) and data == b""
    vmin, vmax, data = encode_frame(np.array([], dtype=np.float32), symmetric=True)
    assert (vmin, vmax) == (0.0, 0.0) and data == b""


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


def test_holder_finish_flag():
    h = LatestFrameHolder(NAMES)
    assert h.finished is False
    h.finish()
    assert h.finished is True


def test_holder_is_thread_safe_under_concurrent_puts():
    h = LatestFrameHolder(("raw",))

    def worker(start):
        for seq in range(start, start + 500):
            h.put("raw", seq, float(seq), 0.0, 1.0, b"x")

    threads = [threading.Thread(target=worker, args=(i * 500,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No crash, slot holds a valid FrameSlot, current_seq is the max seen.
    assert h.snapshot()["raw"] is not None
    expected_max = max(i * 500 + 499 for i in range(4))  # 1999, derived from the worker ranges
    assert h.current_seq == expected_max


def test_geometry_message_shape():
    nside = 2
    npix = hp.nside2npix(nside)  # 48
    msg = geometry_message("dirty", nside, nest=True)
    assert msg["type"] == "geometry"
    assert msg["name"] == "dirty"
    assert msg["order"] == "NESTED"
    assert msg["npix"] == npix
    assert len(msg["corners"]) == npix * 4 * 3  # (npix,4,3) flattened
    assert all(isinstance(v, float) for v in msg["corners"][:6])  # native floats for JSON, not numpy scalars


def test_frame_header_is_static_template():
    h = frame_header("znorm", 2, nest=False)
    assert h == {"type": "frame", "name": "znorm", "order": "RING", "nside": 2, "npix": 48}


def test_tracks_payload_converts_radec_to_unit_vectors():
    tracks = {"SAT-A": [(0, 10.0, -20.0, 1.5), (1, 12.0, -19.0, 1.6)]}
    payload = tracks_payload(tracks)
    assert payload["type"] == "tracks"
    sat = payload["sats"][0]
    assert sat["name"] == "SAT-A"
    assert sat["points"][0]["seq"] == 0
    assert sat["points"][0]["flux"] == 1.5
    x, y, z = sat["points"][0]["xyz"]
    expected = hp.ang2vec(10.0, -20.0, lonlat=True)
    assert np.allclose([x, y, z], expected, atol=1e-6)
    assert np.isclose(x * x + y * y + z * z, 1.0, atol=1e-6)  # unit vector
    assert sat["points"][1]["seq"] == 1
    assert sat["points"][1]["flux"] == 1.6


# ---------------------------------------------------------------------------
# Task 3: stream_handler + FrameServer tests
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Records the ordered (kind, payload) of accept/send/close for handler assertions."""

    def __init__(self):
        self.log: list[tuple[str, object]] = []

    async def accept(self):
        self.log.append(("accept", None))

    async def send_text(self, text):
        self.log.append(("text", json.loads(text)))

    async def send_bytes(self, data):
        self.log.append(("bytes", data))

    async def close(self):
        self.log.append(("close", None))


def test_stream_handler_emits_geometry_frames_then_end():
    from kremetart.utils.web_server import stream_handler

    names = ("raw", "smooth", "znorm")
    holder = LatestFrameHolder(names)
    geom_msgs = [geometry_message(n, 2, nest=True) for n in names]
    headers = {n: frame_header(n, 2, nest=True) for n in names}
    # Pre-fill one frame per name, then mark finished (frozen session).
    for n in names:
        _vmin, _vmax, data = encode_frame(np.zeros(48, dtype=np.float32), symmetric=(n == "znorm"))
        holder.put(n, 0, 1.0, _vmin, _vmax, data)
    holder.finish()

    ws = FakeWebSocket()
    asyncio.run(stream_handler(ws, holder, geom_msgs, headers, None, poll=0.0))

    kinds = [k for k, _ in ws.log]
    assert kinds[0] == "accept"
    # three geometry messages first
    geo = [p for k, p in ws.log if k == "text" and p.get("type") == "geometry"]
    assert {g["name"] for g in geo} == set(names)
    # one header+binary pair per name
    frames = [p for k, p in ws.log if k == "text" and p.get("type") == "frame"]
    assert {f["name"] for f in frames} == set(names)
    assert sum(1 for k, _ in ws.log if k == "bytes") == 3
    # ends cleanly
    end = [p for k, p in ws.log if k == "text" and p.get("type") == "end"]
    assert len(end) == 1
    assert kinds[-1] == "close"
    assert kinds.index("close") > kinds.index("accept")


def test_stream_handler_sends_tracks_after_geometry():
    from kremetart.utils.web_server import stream_handler

    names = ("raw",)
    holder = LatestFrameHolder(names)
    holder.finish()  # no frames; just geometry + tracks + end
    geom_msgs = [geometry_message("raw", 2, nest=True)]
    headers = {"raw": frame_header("raw", 2, nest=True)}
    tracks_msg = tracks_payload({"SAT": [(0, 1.0, 2.0, 1.0)]})

    ws = FakeWebSocket()
    asyncio.run(stream_handler(ws, holder, geom_msgs, headers, tracks_msg, poll=0.0))

    texts = [p for k, p in ws.log if k == "text"]
    types = [p["type"] for p in texts]
    assert types.index("tracks") == types.index("geometry") + 1  # tracks right after geometry
    assert types[-1] == "end"


def test_frame_server_create_app_has_routes():
    from kremetart.utils.web_server import FrameServer

    holder = LatestFrameHolder(NAMES)
    server = FrameServer(holder, nside=2, nest=True, names=NAMES, port=8080)
    app = server.create_app()
    paths = {r.path for r in app.routes}
    assert "/" in paths
    assert "/stream" in paths


def test_frame_server_defaults_to_localhost():
    holder = LatestFrameHolder(NAMES)
    server = FrameServer(holder, nside=2, nest=True, names=NAMES, port=8080)
    assert server.host == "127.0.0.1"  # localhost-only by default; no LAN exposure
