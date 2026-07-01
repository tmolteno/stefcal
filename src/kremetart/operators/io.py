import cupy as cp
import dask.array as da
import holoscan as hs
import numpy as np
import xarray as xr
from holoscan.core import Operator, OperatorSpec
from numpy.typing import NDArray


class XarrayZarrReaderOperator(Operator):
    """
    Reads zarr slices via xarray, transfers to GPU, emits pointer.
    """

    def __init__(
        self, fragment, *args, zarr_path: str, data_column: str = "VISIBILITY", stokes: str = "IQUV", **kwargs
    ):
        self.zarr_path = zarr_path
        self.data_column = data_column
        self.stokes = stokes
        self.current_index = 0
        self.ntime = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("UVW")
        spec.output("VISIBILITY")
        spec.output("WEIGHT")
        spec.output("FLAG")
        spec.output("FREQ")
        spec.output("time")
        # TODO - pass Jones solutions

    def start(self):
        dz = xr.open_datatree(self.zarr_path, engine="zarr", chunks="auto")
        # only the first data group for now (will need to iterate)
        self.dataset = dz[list(dz.children)[0]].ds
        self.ntime = self.dataset["time"].size
        self.current_index = 0

        # get additional metadata
        self.out_times = np.mean(self.dataset.time.values, keepdims=True)
        self.times = hs.as_tensor(cp.asarray(self.dataset.time.values))
        self.out_freqs = np.mean(self.dataset.frequency.values, keepdims=True)
        self.frequencies = hs.as_tensor(cp.asarray(self.dataset.frequency.values))

        # will need this eventually
        mask = ["antenna_xds" in g for g in list(dz.groups)]
        ant_idx = next(i for i, x in enumerate(mask) if x)
        ds_ant = dz[dz.groups[ant_idx]].ds
        self.nant = ds_ant["antenna_name"].size

    def compute(self, op_input, op_output, context):
        if self.current_index >= self.ntime:
            return

        # Read slice from zarr to CPU
        ds_slice = self.dataset.isel(
            {"time": slice(self.current_index, self.current_index + 1)},  # could use larger chunks
        )

        # Emit individual tensors (dict did not work?)
        op_output.emit(hs.as_tensor(cp.asarray(ds_slice["UVW"].values)), "UVW")
        op_output.emit(hs.as_tensor(cp.asarray(ds_slice["VISIBILITY"].values)), "VISIBILITY")
        op_output.emit(hs.as_tensor(cp.asarray(ds_slice["WEIGHT"].values)), "WEIGHT")
        op_output.emit(hs.as_tensor(cp.asarray(ds_slice["FLAG"].values)), "FLAG")
        op_output.emit(self.frequencies, "FREQ")
        op_output.emit(hs.as_tensor(cp.asarray(ds_slice["time"].values)), "time")

        self.current_index += 1

    def stop(self):
        pass


class ResultWriterOperator(Operator):
    """
    The output format for tron actually looks like this

    https://github.com/ratt-ru/pfb-imaging/blob/bb5af97cd4b887cf26560c6c02cad7a34612aad4/pfb/workers/hci.py#L503
    """

    def __init__(
        self,
        fragment,
        nstokes,
        nfreq_out,
        ntime,
        ny,
        nx,
        *args,
        output_dataset: str = None,
        out_stokes: NDArray = None,
        out_freqs: NDArray = None,
        out_times: NDArray = None,
        out_ras: NDArray = None,
        out_decs: NDArray = None,
        **kwargs,
    ):
        self.output_dataset = output_dataset
        self.nstokes = nstokes
        self.nfreq_out = nfreq_out
        self.ntime = ntime
        self.ny = ny
        self.nx = nx
        # x and y is swapped to avoid transpose when writing to fits
        self.cube_dims = (nstokes, nfreq_out, ntime, ny, nx)
        self.cube_chunks = (nstokes, 1, 1, ny, nx)
        self.mean_dims = (nstokes, nfreq_out, ny, nx)
        self.mean_chunks = (nstokes, 1, ny, nx)
        self.out_stokes = out_stokes if out_stokes is not None else np.array(["I", "Q", "U", "V"])[0:nstokes]
        self.out_ras = out_ras if out_ras is not None else np.arange(nx)
        self.out_decs = out_decs if out_decs is not None else np.arange(ny)
        self.coords = {
            "TIME": (("TIME",), out_times if out_times is not None else np.arange(ntime)),
            "STOKES": (("STOKES",), self.out_stokes),
            "FREQ": (("FREQ",), out_freqs if out_freqs is not None else np.arange(nfreq_out)),
            "X": (("X",), self.out_ras),
            "Y": (("Y",), self.out_decs),
        }
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # here we use dask to create a xarray scaffold to write data to
        # note the use of da.empty to avoid actually writing the data

        data_vars = {
            "cube": (
                ("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(self.cube_dims, chunks=self.cube_chunks, dtype=np.float32),
            ),
            "cube_mean": (
                ("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(self.cube_dims, chunks=self.cube_chunks, dtype=np.float32),
            ),
        }
        attrs = {"just": 1.0, "testing": 2.0}
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=self.coords,
            attrs=attrs,
        )
        ds.to_zarr(self.output_dataset, mode="w", compute=True)

    def setup(self, spec: OperatorSpec):
        spec.input("cube")
        spec.input("time_out")
        spec.input("freq_out")

    def compute(self, op_input, op_output, context):
        cube = cp.asnumpy(cp.asarray(op_input.receive("cube")))
        freq_out = cp.asnumpy(cp.asarray(op_input.receive("freq_out")))
        time_out = cp.asnumpy(cp.asarray(op_input.receive("time_out")))

        # these allow us to use region=auto
        coords = {
            "FREQ": (("FREQ",), freq_out),
            "TIME": (("TIME",), time_out),
            "STOKES": (("STOKES",), self.out_stokes),
            "X": (("X",), self.out_ras),
            "Y": (("Y",), self.out_decs),
        }

        dso = xr.Dataset(data_vars={"cube": (("STOKES", "FREQ", "TIME", "Y", "X"), cube)}, coords=coords)

        dso.to_zarr(
            self.output_dataset,
            region="auto",
        )

    def stop(self):
        # need to compute the mean here
        pass


class HealpixZarrReaderOperator(Operator):
    """Stream a prepared imaging zarr (VISIBILITY, WEIGHT, B_ROT, time) one frame at a time to the GPU.

    Reads the host prepare-step output (:func:`kremetart.utils.read_tart_hdf.prepare_msv4_zarr`).
    Unlike :class:`XarrayZarrReaderOperator` it carries the precomputed ``B_ROT`` and drops UVW/FLAG
    (the HEALPix imager does not need them).
    """

    def __init__(self, fragment, *args, zarr_path: str, **kwargs):
        self.zarr_path = zarr_path
        self.current_index = 0
        self.ntime = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("VISIBILITY")
        spec.output("WEIGHT")
        spec.output("B_ROT")
        spec.output("BORESIGHT")
        spec.output("time")

    def start(self):
        self.dataset = xr.open_zarr(self.zarr_path)
        self.ntime = self.dataset["time"].size
        self.out_times = self.dataset["time"].values  # symmetry with XarrayZarrReaderOperator
        self.current_index = 0

    def compute(self, op_input, op_output, context):
        if self.current_index >= self.ntime:
            return
        s = self.dataset.isel({"time": slice(self.current_index, self.current_index + 1)})
        op_output.emit(hs.as_tensor(cp.asarray(s["VISIBILITY"].values)), "VISIBILITY")
        op_output.emit(hs.as_tensor(cp.asarray(s["WEIGHT"].values)), "WEIGHT")
        op_output.emit(hs.as_tensor(cp.asarray(s["B_ROT"].values)), "B_ROT")
        op_output.emit(hs.as_tensor(cp.asarray(s["BORESIGHT"].values)), "BORESIGHT")
        op_output.emit(hs.as_tensor(cp.asarray(s["time"].values)), "time")
        self.current_index += 1

    def stop(self):
        pass


class HealpixWriterOperator(Operator):
    """Write per-frame HEALPix maps to a ``(TIME, PIX)`` zarr (dask scaffold + region="auto").

    The stored variables are configured by ``var_specs`` (one ``(input_port, variable_name)`` pair
    per map). Mirrors :class:`ResultWriterOperator`'s scaffold-then-region-write pattern, but for
    flat ``(TIME, npix)`` HEALPix variables rather than a ``(STOKES, FREQ, TIME, Y, X)`` image cube.

    ``out_times`` MUST equal the streamed frames' ``time`` values (the prepared zarr's ``time``
    coordinate): ``region="auto"`` locates each frame by matching its emitted ``time_out`` against
    the scaffold's ``TIME`` coordinate, so a mismatch raises ``KeyError`` at write time. The app
    builds both the scaffold (via this ``out_times``) and the reader's stream from the same zarr, so
    they agree by construction.
    """

    def __init__(
        self,
        fragment,
        ntime,
        npix,
        *args,
        output_dataset: str | None = None,
        out_times: NDArray = None,
        var_specs: tuple[tuple[str, str], ...] = (("cube", "dirty"), ("filtered", "filtered"), ("znorm", "znorm")),
        **kwargs,
    ):
        self.output_dataset = output_dataset
        self.ntime = ntime
        self.npix = npix
        self.out_times = out_times if out_times is not None else np.arange(ntime)
        self.pix = np.arange(npix)
        # var_specs maps (input_port -> stored_variable_name); the default below is a minimal schema.
        # smoovie's pipeline overrides it with dirty/tikhonov/l1/filtered/znorm sourced from the
        # imager, both deconvolvers and the IWP (see kremetart.core.smoovie.SmooviePipeline.compose).
        self.var_specs = tuple(var_specs)
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # dask scaffold: da.empty allocates no data, only the zarr structure to write regions into.
        data_vars = {
            name: (("TIME", "PIX"), da.empty((self.ntime, self.npix), chunks=(1, self.npix), dtype=np.float32))
            for _, name in self.var_specs
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"TIME": (("TIME",), self.out_times), "PIX": (("PIX",), self.pix)},
        )
        ds.to_zarr(self.output_dataset, mode="w", compute=True)

    def setup(self, spec: OperatorSpec):
        for port, _ in self.var_specs:
            spec.input(port)
        spec.input("time_out")

    def compute(self, op_input, op_output, context):
        time_out = cp.asnumpy(cp.asarray(op_input.receive("time_out")))  # (1,)
        data_vars = {
            name: (("TIME", "PIX"), cp.asnumpy(cp.asarray(op_input.receive(port))).astype(np.float32))
            for port, name in self.var_specs
        }
        dso = xr.Dataset(
            data_vars=data_vars,
            coords={"TIME": (("TIME",), time_out), "PIX": (("PIX",), self.pix)},
        )
        dso.to_zarr(self.output_dataset, region="auto")

    def stop(self):
        pass
