"""CLI for the StEFCal solver."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .stefcal import referenced_phases, stefcal_solve


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tart-stefcal",
        description="StEFCal: alternating per-antenna complex-gain least squares solver.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- solve ---
    p_solve = sub.add_parser("solve", help="Solve per-antenna complex gains.")
    p_solve.add_argument("vis", help="Path to observed visibilities (.npy or .npz).")
    p_solve.add_argument("model", help="Path to model visibilities (.npy or .npz).")
    p_solve.add_argument("a1", help="Path to first-antenna indices (.npy).")
    p_solve.add_argument("a2", help="Path to second-antenna indices (.npy).")
    p_solve.add_argument("--n-ant", type=int, required=True, help="Number of antennas.")
    p_solve.add_argument("--t-int", type=int, default=None, help="Integrations per solution interval (None = all).")
    p_solve.add_argument("--ref-ant", type=int, default=0, help="Reference antenna (default: 0).")
    p_solve.add_argument("--weight", default=None, help="Optional weight array (.npy).")
    p_solve.add_argument("--g0", default=None, help="Optional initial gains (.npy).")
    p_solve.add_argument("--max-iter", type=int, default=100, help="Max iterations per interval (default: 100).")
    p_solve.add_argument("--tol", type=float, default=1e-8, help="Convergence tolerance (default: 1e-8).")
    p_solve.add_argument("--output", "-o", default="gains.npz", help="Output file for gains (default: gains.npz).")
    p_solve.add_argument("--phases", action="store_true", help="Also output gauge-referenced phases.")
    p_solve.add_argument("--gpu", action="store_true", help="Use CuPy GPU backend.")

    # --- phases ---
    p_phases = sub.add_parser("phases", help="Extract gauge-referenced phases from gains.")
    p_phases.add_argument("gains", help="Path to gains file (.npy or .npz).")
    p_phases.add_argument("--ref-ant", type=int, default=0, help="Reference antenna (default: 0).")
    p_phases.add_argument("--output", "-o", default="phases.npy", help="Output file (default: phases.npy).")

    return parser


def _load_array(path: str, key: str | None = None, fallback_keys: list[str] | None = None) -> np.ndarray:
    """Load a NumPy array from a .npy or .npz file.

    For NPZ files, uses ``key`` if given, then tries each ``fallback_keys`` in order,
    then expects a single array.
    """
    if path.endswith(".npz"):
        data = np.load(path)
        if key is not None:
            return data[key]
        if fallback_keys:
            for fk in fallback_keys:
                if fk in data:
                    return data[fk]
        keys = list(data.keys())
        if len(keys) != 1:
            raise ValueError(f"NPZ file {path} contains multiple arrays; specify a key. Available: {keys}")
        return data[keys[0]]
    return np.load(path)


def _cmd_solve(args: argparse.Namespace) -> None:
    xp = np
    if args.gpu:
        try:
            import cupy as cp  # noqa: F811

            xp = cp
        except ImportError:
            print("Error: --gpu requested but cupy is not installed.", file=sys.stderr)
            sys.exit(1)

    vis = _load_array(args.vis)
    model = _load_array(args.model)
    a1 = np.load(args.a1)
    a2 = np.load(args.a2)

    weight = None
    if args.weight is not None:
        weight = np.load(args.weight)

    g0 = None
    if args.g0 is not None:
        g0 = np.load(args.g0)

    gains, info = stefcal_solve(
        vis,
        model,
        a1,
        a2,
        args.n_ant,
        t_int=args.t_int,
        ref_ant=args.ref_ant,
        weight=weight,
        g0=g0,
        max_iter=args.max_iter,
        tol=args.tol,
        xp=xp,
    )

    save = {"gains": np.asarray(gains)}
    for k, v in info.items():
        save[k] = np.asarray(v)

    np.savez(args.output, **save)
    print(f"Gains written to {args.output}  ({save['gains'].shape[0]} solution(s), {save['gains'].shape[1]} antennas)")

    if args.phases:
        phase_path = args.output.replace(".npz", "_phases.npy")
        phases = referenced_phases(gains, args.ref_ant, xp=xp)
        np.save(phase_path, np.asarray(phases))
        print(f"Phases written to {phase_path}")


def _cmd_phases(args: argparse.Namespace) -> None:
    gains = _load_array(args.gains, fallback_keys=["gains"])
    phases = referenced_phases(gains, args.ref_ant)
    np.save(args.output, phases)
    print(f"Phases written to {args.output}  ({phases.shape})")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "solve":
        _cmd_solve(args)
    elif args.command == "phases":
        _cmd_phases(args)
