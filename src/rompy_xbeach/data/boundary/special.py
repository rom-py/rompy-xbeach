"""Special wave boundary condition classes for XBeach.

This module contains special boundary classes:
- BoundaryOff: No wave forcing
- BoundaryReuse: Reuse previous simulation boundary files
"""

from typing import Literal
from pathlib import Path
from pydantic import Field

from rompy.core.time import TimeRange

from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.types import XBeachDirectoryBlob
from rompy_xbeach.data.boundary.base import (
    WaveBoundaryParams,
    SpectralWaveBoundaryParams,
)


# List files written by XBeach that describe the boundary time series to reuse
REUSE_LIST_FILES = ["ebcflist.bcf", "qbcflist.bcf", "esbcflist.bcf"]


class BoundaryOff(WaveBoundaryParams):
    """No wave forcing.

    Use this when you don't want any wave forcing in the model.

    Examples
    --------
    >>> boundary = BoundaryOff()

    """

    id: Literal["off"] = Field(default="off", description="Boundary type identifier")
    model_type: Literal["off"] = Field(
        default="off",
        description="Model type discriminator",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for no wave boundary.

        Parameters
        ----------
        destdir : str | Path
            Destination directory (not used, but required for interface).
        grid : RegularGrid, optional
            Grid instance (not used).
        time : TimeRange, optional
            Time range (not used).

        Returns
        -------
        dict
            XBeach parameters with wbctype='off' and wave boundary settings.

        """
        params = {"wbctype": self.wbctype}
        params.update(self.model_dump(exclude={"model_type", "id"}, exclude_none=True))
        return params


class BoundaryReuse(SpectralWaveBoundaryParams):
    """Reuse previous boundary conditions.

    Makes XBeach reuse wave time series from a previous simulation. The list files
    ebcflist.bcf and qbcflist.bcf (and esbcflist.bcf when the previous run used
    single_dir), and the series files they reference, are copied from the previous
    run directory into the workspace.

    XBeach automatically looks for these files in the run directory - no bcfile
    parameter is needed in params.txt.

    Examples
    --------
    >>> from rompy_xbeach.types import XBeachDirectoryBlob
    >>> boundary = BoundaryReuse(
    ...     previous_run=XBeachDirectoryBlob(source="/path/to/previous/run")
    ... )

    """

    id: Literal["reuse"] = Field(
        default="reuse", description="Boundary type identifier"
    )
    model_type: Literal["reuse"] = Field(
        default="reuse",
        description="Model type discriminator",
    )
    previous_run: XBeachDirectoryBlob = Field(
        description=(
            "Directory containing ebcflist.bcf and qbcflist.bcf files "
            "from a previous XBeach simulation."
        ),
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for reuse wave boundary.

        Parameters
        ----------
        destdir : str | Path
            Destination directory where boundary files will be fetched.
        grid : RegularGrid, optional
            Grid instance (not used).
        time : TimeRange, optional
            Time range (not used).

        Returns
        -------
        dict
            XBeach parameters with wbctype='reuse' and wave boundary settings.

        """
        # Fetch the list files and the series files they reference
        listfiles = self.previous_run.get(destdir, patterns=REUSE_LIST_FILES)
        series = sorted(
            {
                line.split()[-1]
                for listfile in listfiles
                for line in Path(listfile).read_text().splitlines()
                if line.strip().endswith(".bcf")
            }
        )
        if series:
            copied = self.previous_run.get(destdir, patterns=series)
            missing = set(series) - {f.name for f in copied}
            if missing:
                raise FileNotFoundError(
                    f"Boundary series files {sorted(missing)} referenced in "
                    f"{REUSE_LIST_FILES} not found in {self.previous_run.source}"
                )
        params = {"wbctype": self.wbctype}
        params.update(
            self.model_dump(
                exclude={"model_type", "id", "previous_run"},
                exclude_none=True,
            )
        )
        return params
