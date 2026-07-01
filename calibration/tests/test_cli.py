"""Smoke tests for the tart-stefcal CLI."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from tart_stefcal.cli import _build_parser, _cmd_solve


def _get_subparser(name: str):
    parser = _build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise KeyError(name)


def test_help():
    parser = _build_parser()
    help_text = parser.format_help()
    assert "run" in help_text
    assert "solve" in help_text
    assert "phases" in help_text


def test_run_help():
    sp = _get_subparser("run")
    help_text = sp.format_help()
    assert "--tart-name" in help_text
    assert "--archive" in help_text
    assert "--no-beam" in help_text
    assert "--phases-only" in help_text


def test_solve_help():
    sp = _get_subparser("solve")
    help_text = sp.format_help()
    assert "--n-ant" in help_text
    assert "--phases" in help_text
    assert "--json" in help_text


def test_phases_help():
    sp = _get_subparser("phases")
    help_text = sp.format_help()
    assert "--ref-ant" in help_text


def test_json_handles_nan_gains():
    from tart_stefcal.cli import _gains_to_json_dict

    g = np.array([1.0 + 0j, np.nan + 0j, 0.8 + 0.1j])
    d = _gains_to_json_dict(g)
    assert d["gain"] == [1.0, None, 0.8062]
    assert d["phase_offset"] == [0.0, None, 0.1244]
    json.dumps(d)


def test_upload_format_matches_tart_api():
    """Output format matches tart_upload_gains: {gain: [amps], phase_offset: [radians]}."""
    from tart_stefcal.cli import _gains_to_json_dict

    g = np.array([2.0 * np.exp(1j * np.pi / 4), np.exp(1j * np.pi)])
    d = _gains_to_json_dict(g)

    assert set(d.keys()) == {"gain", "phase_offset"}
    assert d["gain"] == [2.0, 1.0]
    assert d["phase_offset"] == [0.7854, 3.1416]
    # Phases are in radians, not degrees
    assert np.isclose(d["phase_offset"][0], np.pi / 4, atol=0.001)
    assert np.isclose(d["phase_offset"][1], np.pi, atol=0.001)


def test_referenced_phases_outputs_radians():
    """referenced_phases returns radians, not degrees."""
    from tart_stefcal import referenced_phases

    g = np.array([[1.0 + 0j, np.exp(1j * np.pi / 2), np.exp(1j * np.pi)]])
    phases = referenced_phases(g, 0)
    assert phases.shape == (1, 3)
    assert np.isclose(phases[0, 0], 0.0, atol=1e-10)
    assert np.isclose(phases[0, 1], np.pi / 2, atol=0.001)
    assert np.isclose(phases[0, 2], np.pi, atol=0.001)


def test_tart_name_from_url():
    from tart_stefcal.cli import _resolve_api_url, _tart_name_from_input

    assert _tart_name_from_input("mu-udm") == "mu-udm"
    assert _tart_name_from_input("https://api.elec.ac.nz/tart/mu-udm") == "mu-udm"
    assert _tart_name_from_input("https://api.elec.ac.nz/tart/mu-udm/") == "mu-udm"
    assert _resolve_api_url("mu-udm") == "https://api.elec.ac.nz/tart/mu-udm"


def test_solve_end_to_end():
    n_ant, nbl = 3, 3
    a1 = np.array([0, 0, 1])
    a2 = np.array([1, 2, 2])
    g_true = np.array([1.0 + 0j, 0.8 + 0.1j, 1.2 - 0.3j])
    model = np.ones((1, nbl, 1), dtype=complex)
    vis = np.zeros((1, nbl, 1), dtype=complex)
    np.random.seed(42)
    for b, (p, q) in enumerate(zip(a1, a2)):
        vis[0, b, 0] = g_true[p] * model[0, b, 0] * np.conj(g_true[q]) + 1e-10 * (
            np.random.randn() + 1j * np.random.randn()
        )

    with tempfile.TemporaryDirectory() as d:
        vf = os.path.join(d, "vis.npy")
        mf = os.path.join(d, "model.npy")
        a1f = os.path.join(d, "a1.npy")
        a2f = os.path.join(d, "a2.npy")
        out = os.path.join(d, "gains")
        np.save(vf, vis)
        np.save(mf, model)
        np.save(a1f, a1)
        np.save(a2f, a2)

        parser = _build_parser()
        args = parser.parse_args(
            [
                "solve",
                vf,
                mf,
                a1f,
                a2f,
                "--n-ant",
                str(n_ant),
                "--max-iter",
                "200",
                "--tol",
                "1e-8",
                "--output",
                out,
                "--phases",
                "--json",
            ]
        )
        _cmd_solve(args)

        assert os.path.exists(out + ".npz")
        assert os.path.exists(out + "_phases.npy")
        assert os.path.exists(out + ".json")

        with open(out + ".json") as f:
            j = json.load(f)
        assert "gain" in j
        assert "phase_offset" in j
        assert len(j["gain"]) == n_ant
        assert np.isclose(j["gain"][0], 1.0)

        data = np.load(out + ".npz")
        assert data["gains"].shape == (1, n_ant)
        assert bool(data["converged"][0])
