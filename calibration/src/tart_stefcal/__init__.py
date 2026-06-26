"""tart-stefcal — StEFCal per-antenna complex-gain least squares solver."""

from .beam import airy_power_beam
from .skymodel import enu_direction_cosines, model_visibilities
from .stefcal import referenced_phases, stefcal_solve


def main() -> None:
    """CLI entry point."""
    from .cli import main as _cli_main

    _cli_main()


__all__ = [
    "airy_power_beam",
    "enu_direction_cosines",
    "model_visibilities",
    "referenced_phases",
    "stefcal_solve",
    "main",
]
