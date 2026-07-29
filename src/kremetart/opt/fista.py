"""Reweighted-L1 FISTA solver (xp-injectable; CPU via numpy, GPU via cupy).

Generic and operator-agnostic, like :mod:`kremetart.opt.cg`: the forward operator ``A`` and its
Hermitian adjoint ``AH`` are plain callables ``x -> A @ x`` and ``r -> Aᴴ @ r``. Minimises, over a
**real** vector ``x``, ``0.5 ||W^½ (A x - y)||² + λ Σ wᵢ |xᵢ|`` with FISTA + backtracking and an
outer Candès–Wakin–Boyd reweighting loop. Used at the outset of self-calibration to recover
non-negative source fluxes from calibrated visibilities; the caller wires ``A`` to the imaging DFT
or the per-source sky model (out of scope here, mirroring how the imager wires up ``cg``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType

import numpy as np


def _soft_threshold(z, tau, *, positive: bool, xp: ModuleType = np):
    """Proximal operator of the (optionally non-negative) weighted-L1 penalty.

    Args:
        z: real array, the point at which to evaluate the prox.
        tau: scalar or array (broadcastable to ``z``) of non-negative thresholds.
        positive: if True, clamp at 0 (prox of L1 + non-negativity indicator).
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``max(z - tau, 0)`` if ``positive`` else ``sign(z) * max(|z| - tau, 0)``.
    """
    if positive:
        return xp.maximum(z - tau, 0.0)
    return xp.sign(z) * xp.maximum(xp.abs(z) - tau, 0.0)


def _fista_single(value_grad, value, x_start, threshold_w, *, positive, L0, eta, max_iter, tol, xp):
    """FISTA with backtracking for one fixed prox threshold and smooth term.

    Args:
        value_grad: callable ``x -> (f(x), grad(x))`` for the smooth data term (momentum point).
        value: callable ``x -> f(x)`` for the smooth data term (backtracking trials; no gradient).
        x_start: real warm-start iterate.
        threshold_w: per-element prox threshold ``lam * w`` (broadcastable to ``x``).

    Returns:
        ``(x, iters, converged, lipschitz)``.
    """
    lipschitz = 1.0 if L0 is None else float(L0)
    x = xp.asarray(x_start).copy()
    x_prev = x.copy()
    v = x.copy()
    t = 1.0
    iters = 0
    converged = False
    for _ in range(max_iter):
        iters += 1
        f_v, g_v = value_grad(v)
        bt = 0
        while True:
            z = v - g_v / lipschitz
            x_new = _soft_threshold(z, threshold_w / lipschitz, positive=positive, xp=xp)
            diff = x_new - v
            rhs = f_v + float((diff * g_v).sum()) + 0.5 * lipschitz * float((diff * diff).sum())
            # +1e-12: numerical slack so float round-off near equality doesn't force a needless backtrack
            if value(x_new) <= rhs + 1e-12 or bt >= 100:
                break
            lipschitz *= eta
            bt += 1
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        v = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x_prev = x
        x = x_new
        t = t_new
        change = float(xp.linalg.norm(x - x_prev))
        scale = max(float(xp.linalg.norm(x)), 1e-12)
        if change / scale < tol:
            converged = True
            break
    return x, iters, converged, lipschitz


def _reweighted_fista(
    value_grad,
    value,
    x_init,
    *,
    lam: float,
    positive: bool,
    L0,
    eta: float,
    max_iter: int,
    tol: float,
    max_reweight: int,
    reweight_eps: float,
    reweight_tol: float,
    xp: ModuleType,
):
    """Outer Candès–Wakin–Boyd reweighting loop shared by ``fista`` and ``fista_quadratic``."""
    x = xp.asarray(x_init)
    w = xp.ones(x.shape, dtype=x.dtype)
    iterations: list[int] = []
    converged = False
    lipschitz = 1.0 if L0 is None else float(L0)
    for ell in range(max(max_reweight, 0) + 1):  # negative count -> a single plain-L1 solve
        x_prev_round = x.copy()
        x, iters, converged, lipschitz = _fista_single(
            value_grad, value, x, lam * w, positive=positive, L0=L0, eta=eta, max_iter=max_iter, tol=tol, xp=xp
        )
        iterations.append(iters)
        if ell == max_reweight:
            break
        w = 1.0 / (xp.abs(x) + reweight_eps)  # Candès–Wakin–Boyd reweighting
        denom = max(float(xp.linalg.norm(x_prev_round)), 1e-12)
        if float(xp.linalg.norm(x - x_prev_round)) / denom < reweight_tol:
            break  # support / values have stabilised
    info = {
        "iterations": iterations,
        "reweights": len(iterations) - 1,
        "objective": value(x) + lam * float((w * xp.abs(x)).sum()),
        "lipschitz": lipschitz,
        "converged": converged,
    }
    return x, info


def fista(
    A: Callable,
    AH: Callable,
    y,
    *,
    lam: float,
    weight=None,
    x0=None,
    positive: bool = True,
    L0: float | None = None,
    eta: float = 2.0,
    max_iter: int = 500,
    tol: float = 1e-5,
    max_reweight: int = 0,
    reweight_eps: float = 1e-3,
    reweight_tol: float = 1e-3,
    xp: ModuleType = np,
):
    """Minimise ``0.5 ||W^½(A x − y)||² + λ Σ wᵢ|xᵢ|`` over real ``x`` via reweighted-L1 FISTA.

    Args:
        A: forward operator, a callable ``x -> A @ x`` (real ``x`` -> complex output).
        AH: Hermitian adjoint of ``A``, a callable ``r -> Aᴴ @ r``.
        y: ``(m,)`` complex data (any shape ``A`` returns).
        lam: L1 strength ``λ`` (``0`` allowed -> plain least squares).
        weight: optional real inverse-variance ``W`` broadcastable to ``y``; ``None`` -> ones.
        x0: optional real warm start; ``None`` -> zeros.
        positive: enforce ``x >= 0`` in the prox.
        L0: initial Lipschitz estimate; ``None`` -> ``1.0``.
        eta: backtracking growth factor (> 1).
        max_iter: max inner FISTA iterations per reweight round.
        tol: inner relative-change stopping tolerance.
        max_reweight: outer reweighting rounds (``0`` -> plain L1).
        reweight_eps: ``ε`` in ``wᵢ = 1/(|xᵢ| + ε)``.
        reweight_tol: outer between-round relative-change stop.
        xp: array module (``numpy`` or ``cupy``).

    Returns:
        ``(x, info)``: real ``x`` and a dict with ``iterations``, ``reweights``, ``objective``,
        ``lipschitz``, ``converged``.
    """
    y = xp.asarray(y)
    weight = None if weight is None else xp.asarray(weight)
    real_dtype = y.real.dtype
    if x0 is None:
        x_init = xp.zeros(AH(y).shape, dtype=real_dtype)
    else:
        x_init = xp.asarray(x0, dtype=real_dtype).copy()

    def fval(r):
        sq = xp.abs(r) ** 2
        if weight is not None:
            sq = weight * sq
        return 0.5 * float(sq.sum())

    def value(x):
        return fval(A(x) - y)

    def value_grad(x):
        r = A(x) - y
        wr = r if weight is None else weight * r
        return fval(r), xp.real(AH(wr))  # x is real -> take the real part of the complex adjoint

    return _reweighted_fista(
        value_grad,
        value,
        x_init,
        lam=lam,
        positive=positive,
        L0=L0,
        eta=eta,
        max_iter=max_iter,
        tol=tol,
        max_reweight=max_reweight,
        reweight_eps=reweight_eps,
        reweight_tol=reweight_tol,
        xp=xp,
    )


def fista_quadratic(
    hess: Callable,
    b,
    *,
    lam: float,
    x0=None,
    positive: bool = True,
    L0: float | None = None,
    eta: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    max_reweight: int = 0,
    reweight_eps: float = 1e-3,
    reweight_tol: float = 1e-3,
    xp: ModuleType = np,
):
    """Minimise ``0.5⟨x,Hx⟩ − ⟨b,x⟩ + λ Σ wᵢ|xᵢ|`` over real ``x`` via reweighted-L1 FISTA.

    For the image-space deconvolution: ``hess`` is the Hessian matvec ``x -> H x`` with
    ``H = B Mᴴ W M B`` (:func:`kremetart.utils.healpix_dft.hessian_healpix`) and ``b`` the
    un-normalised dirty image (the imager's normalised dirty times ``Σw``). The full least-squares
    constant ``½ yᴴ W y`` is dropped — it cancels in the backtracking test and the caller has only
    ``b``, not ``y`` — so ``info["objective"]`` is the data fit up to that additive constant plus the
    L1 penalty. The real part of ``hess(x)`` is taken internally (``H`` is real-symmetric over real
    ``x``), so a complex-returning ``H = Aᴴ W A`` matvec is accepted.

    Args:
        hess: callable ``x -> H x`` (SPD over the reals; real or complex output, real part used).
        b: ``(n,)`` real right-hand side (the un-normalised dirty image).
        lam: L1 strength ``λ`` (``= eta·Σw`` from the operator).
        x0: optional real warm start; ``None`` -> zeros.
        positive: enforce ``x >= 0`` in the prox.
        L0: initial Lipschitz estimate; pass ``diag(H).max()`` for a tight, free seed; ``None`` -> 1.0.
        eta: backtracking growth factor (> 1) — distinct from the regulariser strength.
        max_iter, tol, max_reweight, reweight_eps, reweight_tol, xp: as in :func:`fista`.

    Returns:
        ``(x, info)`` as in :func:`fista`.
    """
    b = xp.asarray(b)
    real_dtype = b.real.dtype
    if x0 is None:
        x_init = xp.zeros(b.shape, dtype=real_dtype)
    else:
        x_init = xp.asarray(x0, dtype=real_dtype).copy()

    # Short-circuit: if b == 0, return the warm start unchanged.
    if float(xp.linalg.norm(b)) == 0.0:
        info = {
            "iterations": [],
            "reweights": 0,
            "objective": lam * float(xp.abs(x_init).sum()),
            "lipschitz": 1.0 if L0 is None else float(L0),
            "converged": True,
        }
        return x_init, info

    def value(x):
        hx = xp.real(hess(x))
        return 0.5 * float((x * hx).sum()) - float((b * x).sum())

    def value_grad(x):
        hx = xp.real(hess(x))
        f = 0.5 * float((x * hx).sum()) - float((b * x).sum())
        return f, hx - b

    return _reweighted_fista(
        value_grad,
        value,
        x_init,
        lam=lam,
        positive=positive,
        L0=L0,
        eta=eta,
        max_iter=max_iter,
        tol=tol,
        max_reweight=max_reweight,
        reweight_eps=reweight_eps,
        reweight_tol=reweight_tol,
        xp=xp,
    )
