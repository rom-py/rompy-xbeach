"""Special wave boundary condition classes for XBeach.

This module contains special boundary classes:
- BoundaryOff: No wave forcing
- BoundaryReuse: Reuse previous simulation boundary files
"""

from typing import Literal, Optional
from pathlib import Path
from pydantic import Field

from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange

from rompy_xbeach.grid import RegularGrid


class BoundaryOff(RompyBaseModel):
    """No wave forcing.

    Use this when you don't want any wave forcing in the model.

    Examples
    --------
    >>> boundary = BoundaryOff()

    """

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
        return {"wbctype": "off"}


class BoundaryReuse(RompyBaseModel):
    """Reuse previous boundary conditions.

    Makes XBeach reuse wave time series from a previous simulation.
    Requires copying ebcflist.bcf and qbcflist.bcf files (and referenced files)
    to the current working directory.

    Examples
    --------
    >>> boundary = BoundaryReuse()
    >>> # Or with explicit file path
    >>> boundary = BoundaryReuse(bcfile="path/to/ebcflist.bcf")

    """

    model_type: Literal["reuse"] = Field(
        default="reuse",
        description="Model type discriminator",
    )
    bcfile: Optional[str] = Field(
        default=None,
        description="Path to previous boundary files (optional)",
    )

    def get(
        self, destdir: str | Path, grid: RegularGrid = None, time: TimeRange = None
    ) -> dict:
        """Return XBeach parameters for reuse wave boundary.

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
            XBeach parameters with wbctype='reuse' and optional bcfile.

        """
        params = {"wbctype": "reuse"}
        if self.bcfile:
            params["bcfile"] = self.bcfile
        return params
