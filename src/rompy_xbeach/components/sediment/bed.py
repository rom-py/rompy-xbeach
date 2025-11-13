"""XBeach bed composition parameter configurations.

This module contains models for bed composition, layering, and bed update parameters.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob


class BedUpdate(XBeachBaseModel):
    """Bed update parameters (XBeach Table 40).

    Controls the vertical structure of the bed when using multiple sediment
    fractions and bed layers. Parameters determine when layers split or merge
    based on their thickness relative to nominal values.

    The variable thickness layer (specified by nd_var) can grow or shrink
    during the simulation to accommodate sediment deposition or erosion.

    Pre-defined bed evolution can be specified using setbathy parameters,
    allowing time series of imposed bed levels that override computed
    morphological changes.

    Note
    ----
    It is strongly advised to turn off morphology computation (morphology=0)
    when using setbathy, as computed changes will be overridden.
    """

    frac_dz: Optional[float] = Field(
        default=None,
        description=(
            "Relative thickness to split time step for bed updating "
            "(XBeach default: 0.7)"
        ),
        ge=0.5,
        le=0.98,
    )
    merge: Optional[float] = Field(
        default=None,
        description=(
            "Merge threshold for variable sediment layer (ratio to nominal thickness) "
            "(XBeach default: 0.01)"
        ),
        ge=0.005,
        le=0.1,
    )
    nd_var: Optional[int] = Field(
        default=None,
        description=(
            "Index of layer with variable thickness (XBeach default: 2)"
        ),
        ge=1,
    )
    nsetbathy: Optional[int] = Field(
        default=None,
        description=(
            "Number of prescribed bed updates in setbathyfile (XBeach default: 1)"
        ),
        ge=1,
        le=1000,
    )
    setbathyfile: Optional[XBeachDataBlob] = Field(
        default=None,
        description=(
            "Name of prescribed bed update file. File contains time series of "
            "bed levels: each update starts with time (s), followed by bed level "
            "at every grid point in same format as initial bathymetry file"
        ),
    )
    split: Optional[float] = Field(
        default=None,
        description=(
            "Split threshold for variable sediment layer (ratio to nominal thickness) "
            "(XBeach default: 1.01)"
        ),
        ge=1.005,
        le=1.1,
    )

    @model_validator(mode="after")
    def validate_setbathy_consistency(self) -> "BedUpdate":
        """Validate that nsetbathy is specified if setbathyfile is provided."""
        if self.setbathyfile is not None and self.nsetbathy is None:
            raise ValueError(
                "nsetbathy must be specified when setbathyfile is provided. "
                "It defines the number of bed update time steps in the file."
            )
        return self

    def get(self, destdir: str | Path) -> dict:
        """Fetch external setbathyfile if specified, and return the params dict.

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

        # Fetch DataBlob file and add the fetched file path
        if self.setbathyfile:
            params["setbathyfile"] = self.setbathyfile.get(destdir).name

        return params
