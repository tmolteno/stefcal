"""Holoscan operator: per-frame reweighted-L1 deconvolution via FISTA (GPU-resident, xp=cupy).

The reweighted-L1 counterpart of :class:`kremetart.operators.tikhonov.TikhonovOperator`: it shares
the same per-frame inputs (the imager dirty plus the weights/geometry/beam that build ``H``) but
exposes a single output, ``cube`` (the L1 deconvolved image), wired to the writer/sink ``l1`` port in
:class:`kremetart.core.smoovie.SmooviePipeline`. It solves ``min ½⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ|``
(non-negative, sparse) with :func:`kremetart.opt.fista.fista_quadratic` instead of the Tikhonov CG
normal-equation solve. ``H`` is the image-space Hessian
(:func:`kremetart.utils.healpix_dft.hessian_healpix`), ``b`` the un-normalised dirty image (the
imager's normalised dirty times ``Σw``), and ``λ = eta·Σw`` makes ``eta`` a frame-invariant fraction
of the central PSF value ``Σw`` (matching the Tikhonov knob; ``smoovie`` drives it via ``--l1``). The
Lipschitz step is seeded from the closed-form ``diag(H).max()`` that ``hessian_healpix`` returns, so
backtracking almost never fires. See
docs/superpowers/specs/2026-06-25-reweighted-l1-deconvolution-design.md.
"""

import cupy as cp
import holoscan as hs
from holoscan.core import Operator, OperatorSpec

from kremetart.opt.fista import fista_quadratic
from kremetart.utils.beam import GROUND_PLANE_DIAMETER, airy_power_beam
from kremetart.utils.healpix_dft import hessian_healpix, make_pixel_grid


class L1ReweightOperator(Operator):
    """Per-frame reweighted-L1 deconvolution (sparse non-negative image) via FISTA.

    Args:
        fragment: Holoscan fragment.
        nside: HEALPix resolution.
        freqs: ``(nchan,)`` frequencies in Hz.
        eta: regularisation strength as a fraction of ``Σw`` (``λ = eta·Σw``); must be > 0.
        nest: NESTED HEALPix ordering (default True).
        apply_beam: build the Airy beam into ``H`` (must match the imager's setting).
        ground_plane_diameter: Airy aperture diameter in metres.
        max_iter: maximum inner FISTA iterations per reweight round.
        tol: inner relative-change tolerance.
        max_reweight: outer Candès–Wakin–Boyd reweighting rounds.
        reweight_eps: ``ε`` in ``wᵢ = 1/(|xᵢ| + ε)``.
        positive: enforce ``x >= 0`` (the sky is non-negative).
        use_warm_start: seed each frame's FISTA with the previous frame's solution.
    """

    def __init__(
        self,
        fragment,
        nside,
        freqs,
        eta,
        *args,
        nest=True,
        apply_beam=True,
        ground_plane_diameter=GROUND_PLANE_DIAMETER,
        max_iter=200,
        tol=1e-5,
        max_reweight=2,
        reweight_eps=1e-3,
        positive=True,
        use_warm_start=True,
        **kwargs,
    ):
        self.nside = nside
        self.freqs = cp.asarray(freqs)
        self.eta = float(eta)
        self.nest = nest
        self.apply_beam = apply_beam
        self.ground_plane_diameter = ground_plane_diameter
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.max_reweight = int(max_reweight)
        self.reweight_eps = float(reweight_eps)
        self.positive = positive
        self.use_warm_start = use_warm_start
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        self.pix_vec = make_pixel_grid(self.nside, nest=self.nest, xp=cp)
        self.x_prev = None  # device-resident warm-start state

    def setup(self, spec: OperatorSpec):
        spec.input("cube")  # imager dirty map = RHS (normalised)
        spec.input("WEIGHT")
        spec.input("B_ROT")
        spec.input("BORESIGHT")
        spec.output("cube")  # regularised image -> writer "l1" port

    def compute(self, op_input, op_output, context):
        dirty = cp.asarray(op_input.receive("cube"))  # (1, npix)
        weights = cp.asarray(op_input.receive("WEIGHT"))  # (1, nbl, nchan)
        b_rot = cp.asarray(op_input.receive("B_ROT"))  # (1, nbl, 3)
        boresight = cp.asarray(op_input.receive("BORESIGHT"))  # (1, 3)

        w = weights[0]  # (nbl, nchan)
        wsum = w.sum()
        # Fully-flagged frame: the imager's dirty is the all-zero no-data map and λ = eta·Σw = 0, so
        # there is nothing to solve. Pass the zero map straight through (the IWP reads it as a no-data
        # frame and coasts); leave the warm-start untouched so the next live frame still seeds well.
        if float(wsum) == 0.0:
            zeros = cp.zeros(self.pix_vec.shape[0], dtype=cp.float64)
            op_output.emit(hs.as_tensor(zeros[None, :]), "cube")
            return

        beam = None
        if self.apply_beam:
            beam = airy_power_beam(
                self.pix_vec, boresight[0], self.freqs, diameter=self.ground_plane_diameter, xp=cp
            )  # (nchan, npix)

        rows = b_rot[0]  # (nbl, 3)
        hmv, hdiag = hessian_healpix(rows, self.pix_vec, self.freqs, w, beam=beam, xp=cp)
        b = dirty[0] * wsum  # un-normalise the imager's normalised dirty to the Hessian RHS
        lam = self.eta * wsum

        x0 = None
        if self.use_warm_start and self.x_prev is not None and bool(cp.all(cp.isfinite(self.x_prev))):
            x0 = self.x_prev

        x, _info = fista_quadratic(
            hmv,
            b,
            lam=float(lam),
            x0=x0,
            positive=self.positive,
            L0=float(hdiag.max()),
            max_iter=self.max_iter,
            tol=self.tol,
            max_reweight=self.max_reweight,
            reweight_eps=self.reweight_eps,
            xp=cp,
        )
        self.x_prev = x

        op_output.emit(hs.as_tensor(x[None, :]), "cube")
