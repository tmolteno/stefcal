# src/kremetart/operators/web_sink.py
"""Holoscan terminal sink: stream named float32 HEALPix maps to the web layer.

Receives the last compute stage's named maps (one input per name in ``NAMES`` + time_out), copies
device→host with ``cp.asnumpy`` when GPU-resident, and drops one frame per name (sharing a
per-integration sequence number) into a LatestFrameHolder. ``compute`` never touches the
socket and returns immediately, so a slow/disconnected browser cannot backpressure the GXF
scheduler.
"""

import cupy as cp
from holoscan.core import Operator, OperatorSpec

from kremetart.utils.healpix_viz import NAMES, SYMMETRIC, LatestFrameHolder, encode_frame


class WebStreamSinkOperator(Operator):
    """Terminal sink: named HEALPix maps → LatestFrameHolder (one shared seq per integration)."""

    def __init__(self, fragment, *args, holder: LatestFrameHolder, **kwargs):
        self.holder = holder
        self.seq = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        for name in NAMES:
            spec.input(name)
        spec.input("time_out")

    def compute(self, op_input, op_output, context):
        t = float(cp.asnumpy(cp.asarray(op_input.receive("time_out")))[0])
        for name in NAMES:
            values = cp.asnumpy(cp.asarray(op_input.receive(name)))[0]  # (npix,)
            vmin, vmax, data = encode_frame(values, symmetric=name in SYMMETRIC)
            self.holder.put(name, self.seq, t, vmin, vmax, data)
        self.seq += 1
