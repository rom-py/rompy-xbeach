"""XBeach groundwater flow parameter configurations.

This module contains models for groundwater flow parameters including permeability
coefficients, aquifer properties, and groundwater head modeling.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob


class GroundwaterFlow(XBeachBaseModel):
    """Groundwater flow parameters (XBeach Table 41).

    Controls groundwater flow processes including Darcy flow permeability,
    aquifer properties, and groundwater head modeling.

    The vertical permeability coefficient can be set differently than the
    horizontal using kx, ky, and kz keywords. The initial bed level of the
    aquifer and groundwater head can be specified using external files or
    uniform values.

    References
    ----------
    XBeach manual: Grid and bathymetry section for file format details.
    """

    aquiferbot: Optional[float] = Field(
        default=None,
        description=("Level of uniform aquifer bottom (XBeach default: -10.0 m)"),
        ge=-100.0,
        le=100.0,
    )
    aquiferbotfile: Optional[XBeachDataBlob] = Field(
        default=None,
        description=(
            "Name of the aquifer bottom file. File format same as bathymetry file"
        ),
    )
    dwetlayer: Optional[float] = Field(
        default=None,
        description=(
            "Thickness of the top soil layer interacting more freely with "
            "the surface water (XBeach default: 0.1 m)"
        ),
        ge=0.01,
        le=1.0,
    )
    gw0: Optional[float] = Field(
        default=None,
        description=("Level of initial groundwater level (XBeach default: 0.0 m)"),
        ge=-5.0,
        le=5.0,
    )
    gw0file: Optional[XBeachDataBlob] = Field(
        default=None,
        description=(
            "Name of initial groundwater level file. File format same as bathymetry file"
        ),
    )
    gwReturb: Optional[float] = Field(
        default=None,
        description=(
            "Reynolds number for start of turbulent flow in case of gwscheme = turbulent "
            "(XBeach default: 100.0)"
        ),
        ge=1.0,
        le=600.0,
    )
    gwfastsolve: Optional[bool] = Field(
        default=None,
        description=(
            "Reduce full 2D non-hydrostatic solution to quasi-explicit in longshore direction "
            "(XBeach default: 0)"
        ),
    )
    gwheadmodel: Optional[Literal["parabolic", "exponential"]] = Field(
        default=None,
        description=(
            "Model to use for vertical groundwater head (XBeach default: parabolic)"
        ),
    )
    gwhorinfil: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to include horizontal infiltration from surface water to groundwater "
            "(XBeach default: 0)"
        ),
    )
    gwnonh: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to turn on or off non-hydrostatic pressure for groundwater "
            "(XBeach default: 0)"
        ),
    )
    gwscheme: Optional[Literal["laminar", "turbulent"]] = Field(
        default=None,
        description=("Scheme for momentum equation (XBeach default: laminar)"),
    )
    kx: Optional[float] = Field(
        default=None,
        description=(
            "Darcy-flow permeability coefficient in x-direction "
            "(XBeach default: 0.0001 m/s)"
        ),
        ge=1e-05,
        le=0.1,
    )
    ky: Optional[float] = Field(
        default=None,
        description=(
            "Darcy-flow permeability coefficient in y-direction "
            "(XBeach default: 0.0001 m/s)"
        ),
        ge=1e-05,
        le=0.1,
    )
    kz: Optional[float] = Field(
        default=None,
        description=(
            "Darcy-flow permeability coefficient in z-direction "
            "(XBeach default: 0.0001 m/s)"
        ),
        ge=1e-05,
        le=0.1,
    )

    def get(self, destdir: str | Path) -> dict:
        """Fetch external files if specified, and return the params dict.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for fetching files.

        Returns
        -------
        dict
            Parameters dictionary with file paths updated to workspace directory.
        """
        # Get base params (XBeachDataBlob fields are automatically excluded by serializer)
        params = super().get(destdir)

        # Fetch DataBlob files and add the fetched file paths
        if self.aquiferbotfile:
            params["aquiferbotfile"] = self.aquiferbotfile.get(destdir).name
        if self.gw0file:
            params["gw0file"] = self.gw0file.get(destdir).name

        return params
