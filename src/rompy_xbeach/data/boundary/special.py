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


class BoundaryOff(WaveBoundaryParams):
    """No wave forcing.

    Use this when you don't want any wave forcing in the model.

    Examples
    --------
    >>> boundary = BoundaryOff()

    """

    id: Literal["off"] = Field(
        default="off", description="Boundary type identifier"
    )
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
            XBeach parameters with wbctype='off'.

        """
        return {"wbctype": self.id}


class BoundaryReuse(SpectralWaveBoundaryParams):
    """Reuse previous boundary conditions.

    Makes XBeach reuse wave time series from a previous simulation.
    Requires the ebcflist.bcf and qbcflist.bcf files from a previous run.
    The source field should point to the directory containing these files.

    XBeach automatically looks for ebcflist.bcf and qbcflist.bcf in the
    run directory - no bcfile parameter is needed in params.txt.

    .. note::
        TODO: The ebcflist.bcf and qbcflist.bcf files reference additional files
        (typically with E_ and q_ prefixes) that also need to be present in the
        workspace. Currently these referenced files are not automatically fetched.
        Users must ensure all referenced files are available in the source directory.

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
        # Fetch the required bcf files from previous run directory
        self.previous_run.get(destdir, patterns=["ebcflist.bcf", "qbcflist.bcf"])
        params = {"wbctype": self.id}
        params.update(
            self.model_dump(
                exclude={"model_type", "id", "previous_run"},
                exclude_none=True,
            )
        )
        return params
