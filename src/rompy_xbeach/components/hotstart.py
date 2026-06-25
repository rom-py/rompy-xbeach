"""XBeach hotstart configuration.

This module contains the Hotstart component for initializing XBeach simulations
from a previous simulation state.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel, XBeachDirectoryBlob


class Hotstart(XBeachBaseModel):
    """Hotstart configuration for initializing from previous simulation state.

    XBeach can read hotstart files from a previous simulation to initialize
    a new simulation, avoiding spin-up time. Hotstart files follow the naming
    convention: hotstart_{varname}{fileno:06d}.dat

    The minimum required files are: zs, zb, uu, vv (always written/read).
    Additional files may be present depending on the simulation options
    (groundwater, wave energy, sediment, etc.).

    Usage in Config:
        hotstart: false  # No hotstart (default)
        hotstart: true   # Enable hotstart, files must exist in run directory
        hotstart:        # Full hotstart configuration
          hotstartfileno: 3
          previous_run:
            source: /path/to/previous/run

    See https://xbeach.readthedocs.io/en/latest/xbeach_manual.html#hotstart-beta
    for more information.
    """

    hotstart: Literal[True] = Field(
        default=True,
        description="Initialize simulation with hotstart files (XBeach default: 0)",
    )
    hotstartfileno: int = Field(
        default=0,
        description=(
            "Hotstart file number to use for initialization. Selects which set "
            "of hotstart files to read when multiple snapshots exist (0-999)."
        ),
        ge=0,
        le=999,
    )
    previous_run: Optional[XBeachDirectoryBlob] = Field(
        default=None,
        description=(
            "Directory containing hotstart files from a previous XBeach simulation. "
            "If not specified, hotstart files must already exist in the run directory."
        ),
    )

    def get(self, destdir: str | Path) -> dict:
        """Fetch hotstart files if previous_run is specified, and return the params dict.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for fetching files.

        Returns
        -------
        dict
            Parameters dictionary with hotstart settings.
        """
        params = {"hotstart": 1, "hotstartfileno": self.hotstartfileno}

        # Fetch hotstart files from previous run directory if specified
        if self.previous_run:
            pattern = f"hotstart_*{self.hotstartfileno:06d}.dat"
            self.previous_run.get(destdir, patterns=[pattern])

        return params
