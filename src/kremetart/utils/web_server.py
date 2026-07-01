"""FastAPI + uvicorn frame server: serves the renderer and the /stream WebSocket on one port.

Runs in a daemon thread so the Holoscan app keeps the main thread. The connection handler
polls the LatestFrameHolder (latest-wins) and pushes the demo's geometry/frame protocol,
plus the additive ``tracks`` (on connect) and ``end`` (on pipeline finish) control messages.
A slow client only misses intermediate frames — it never backpressures the sink.
"""

from __future__ import annotations

import asyncio
import importlib.resources as resources
import json
import threading

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from kremetart.utils.healpix_viz import (
    LatestFrameHolder,
    frame_header,
    geometry_message,
    tracks_payload,
)


async def stream_handler(
    websocket: WebSocket,
    holder: LatestFrameHolder,
    geom_msgs: list[dict],
    headers: dict[str, dict],
    tracks_msg: dict | None,
    *,
    poll: float = 0.03,
) -> None:
    """Drive one /stream connection through the geometry → frames → end lifecycle."""
    await websocket.accept()
    for msg in geom_msgs:
        await websocket.send_text(json.dumps(msg))
    if tracks_msg is not None:
        await websocket.send_text(json.dumps(tracks_msg))

    last_sent = {name: -1 for name in headers}
    try:
        while True:
            sent_something = False
            snapshot = holder.snapshot()
            # Iterate the advertised names (those we sent geometry for), not the holder's keys,
            # so a stray/mismatched holder entry can never KeyError on last_sent[name].
            for name in headers:
                slot = snapshot.get(name)
                if slot is not None and slot.seq > last_sent[name]:
                    header = {
                        **headers[name],
                        "vmin": slot.vmin,
                        "vmax": slot.vmax,
                        "seq": slot.seq,
                        "t": slot.t,
                    }
                    await websocket.send_text(json.dumps(header))
                    await websocket.send_bytes(slot.data)
                    last_sent[name] = slot.seq
                    sent_something = True
            if holder.finished and not sent_something:
                await websocket.send_text(json.dumps({"type": "end"}))
                await websocket.close()
                return
            await asyncio.sleep(poll)
    except WebSocketDisconnect:
        return


class FrameServer:
    """FastAPI app (renderer + /static + /stream) on a uvicorn daemon thread."""

    def __init__(
        self, holder: LatestFrameHolder, *, nside, nest, names, port, tracks=None, units=None, host="127.0.0.1"
    ):
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

    def create_app(self) -> FastAPI:
        static_dir = resources.files("kremetart") / "static"
        index_html = (static_dir / "index.html").read_text()
        geom_msgs = [geometry_message(n, self.nside, self.nest, unit=self.units.get(n, "")) for n in self.names]
        headers = {n: frame_header(n, self.nside, self.nest) for n in self.names}

        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return index_html

        @app.websocket("/stream")
        async def stream(websocket: WebSocket) -> None:
            await stream_handler(websocket, self.holder, geom_msgs, headers, self._tracks_msg)

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        return app

    def start(self) -> str:
        config = uvicorn.Config(
            self.create_app(), host=self.host, port=self.port, log_level="warning", ws_max_size=None
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        shown = "localhost" if self.host in ("0.0.0.0", "127.0.0.1", "") else self.host
        return f"http://{shown}:{self.port}/"

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
