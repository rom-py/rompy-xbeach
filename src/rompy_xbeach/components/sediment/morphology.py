"""XBeach morphology parameter configurations.

This module contains models for morphological evolution parameters including
morphological acceleration, avalanching, and non-erodible structures.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob


class Morphology(XBeachBaseModel):
    """Morphological evolution parameters.

    Controls morphological time acceleration (morfac), the period during which
    morphology is active, and non-erodible structures.

    The morfac parameter allows decoupling of hydrodynamic and morphological time
    scales, enabling faster simulation of slow morphological processes.

    References
    ----------
    Roelvink, D. (2006). Coastal morphodynamic evolution techniques.
    Coastal Engineering, 53(2-3), 277-287.
    """

    morfac: Optional[float] = Field(
        default=None,
        description=(
            "Morphological acceleration factor. Multiplies all morphological "
            "change by this factor, allowing faster simulation of slow processes "
            "(XBeach default: 1.0, no acceleration)"
        ),
        ge=0.0,
        le=1000.0,
    )
    morfacopt: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to adjust output times for morfac. If enabled (1), simulation "
            "time is shortened by morfac. If disabled (0), simulation runs for full "
            "hydrodynamic time (XBeach default: 1)"
        ),
    )
    morstart: Optional[float] = Field(
        default=None,
        description=(
            "Start time for morphology in morphological time (XBeach default: 0.0 s)"
        ),
        ge=0.0,
    )
    morstop: Optional[float] = Field(
        default=None,
        description=(
            "Stop time for morphology in morphological time (XBeach default: 2000.0 s)"
        ),
        ge=0.0,
        le=10000000.0,
    )
    lsgrad: Optional[float] = Field(
        default=None,
        description=(
            "Factor to include longshore transport gradient in 1D simulations. "
            "dSy/dy = lsgrad * Sy; dimension 1/length scale of longshore gradients "
            "(XBeach default: 0.0 m⁻¹)"
        ),
        ge=-0.1,
        le=0.1,
    )
    struct: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for enabling hard structures (non-erodible layers) "
            "(XBeach default: 0)"
        ),
    )
    ne_layer: Optional[XBeachDataBlob] = Field(
        default=None,
        description=(
            "Name of file containing thickness of the erodible layer. "
            "File format same as bathymetry file. Values define thickness "
            "of erodible layer on top of non-erodible layer (m)"
        ),
    )

    def get(self, destdir: str | Path) -> dict:
        """Fetch external ne_layer file if specified, and return the params dict.

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
        if self.ne_layer:
            params["ne_layer"] = self.ne_layer.get(destdir).name

        return params


class Avalanching(XBeachBaseModel):
    """Avalanching parameters.

    Controls critical avalanching slopes above and below water, and limits
    on bed level change due to avalanching.

    When bed slopes exceed the critical slope, the bed collapses and slides
    downward (avalanching). Different critical slopes apply above and below
    water due to different effective friction.
    """

    dryslp: Optional[float] = Field(
        default=None,
        description=(
            "Critical avalanching slope above water (dz/dx and dz/dy) "
            "(XBeach default: 1.0)"
        ),
        ge=0.1,
        le=2.0,
    )
    wetslp: Optional[float] = Field(
        default=None,
        description=(
            "Critical avalanching slope under water (dz/dx and dz/dy) "
            "(XBeach default: 0.3)"
        ),
        ge=0.1,
        le=1.0,
    )
    hswitch: Optional[float] = Field(
        default=None,
        description=(
            "Water depth at which is switched from wetslp to dryslp "
            "(XBeach default: 0.1 m)"
        ),
        ge=0.01,
        le=1.0,
    )
    dzmax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum bed level change due to avalanching per time step "
            "(XBeach default: 0.05 m/s/m)"
        ),
        ge=0.0,
        le=1.0,
    )


class PrescribedBathymetry(XBeachBaseModel):
    """Prescribed bathymetry evolution parameters.

    Allows specification of pre-defined bed evolution time series, useful for
    imposing known morphological changes or testing model response to specific
    bathymetric scenarios.

    When setbathy is enabled, XBeach interpolates bed levels from the time series
    at each computational time step, overriding computed morphological change.

    Note
    ----
    It is strongly advised to turn off morphology computation (morphology=0)
    when using setbathy, as computed changes will be overridden.
    """

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

    @model_validator(mode="after")
    def validate_setbathy_consistency(self) -> "PrescribedBathymetry":
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
