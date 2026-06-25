"""tart-stefcal — StEFCal per-antenna complex-gain least squares solver."""

from .skymodel import enu_direction_cosines, model_visibilities
from .stefcal import referenced_phases, stefcal_solve


def main() -> None:
    """CLI entry point."""
    from .cli import main as _cli_main

    _cli_main()


__all__ = ["stefcal_solve", "referenced_phases", "enu_direction_cosines", "model_visibilities", "main"]
