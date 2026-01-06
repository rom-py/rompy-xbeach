"""Wave boundary specification classes.

These classes represent a complete wave boundary specification including:
- wbctype: The XBeach boundary type
- bcfile: Path to boundary file (if needed)
- wbc: Additional wave boundary parameters

This is separate from:
- parameters.py: Wave boundary parameter classes (WaveBoundaryConditions hierarchy)
- data/boundary.py: Data extraction and file generation classes
"""

from typing import Literal, Optional
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel
from rompy_xbeach.components.boundary.parameters import (
    WaveBoundaryConditions,
    SpectralWaveBoundaryConditions,
    NonSpectralWaveBoundaryConditions,
)


class WaveBoundary(XBeachBaseModel):
    """Base class for wave boundary specification.

    A wave boundary specification contains everything XBeach needs to know
    about wave boundary conditions:
    - wbctype: The type of boundary condition
    - bcfile: Path to boundary file (if needed)
    - wbc: Additional parameters (nmax, rt, dtbc, Hrms, etc.)
    """

    model_type: str = Field(
        description="Model type discriminator for wave boundary specification"
    )
    wbctype: str = Field(description="XBeach wave boundary condition type")
    bcfile: Optional[str] = Field(
        default=None, description="Path to boundary condition file"
    )
    wbc: Optional[WaveBoundaryConditions] = Field(
        default=None, description="Wave boundary condition parameters"
    )


class SpectralWaveBoundary(WaveBoundary):
    """Spectral wave boundary specification.

    For spectral boundary types: jons, parametric, swan, vardens, jonstable.
    These always require a boundary file containing spectral information.

    Examples
    --------
    >>> # JONSWAP boundary with pre-existing file
    >>> boundary = SpectralWaveBoundary(
    ...     wbctype="jons",
    ...     bcfile="jonswap.txt",
    ...     wbc=SpectralWaveBoundaryConditions(
    ...         nmax=0.8,
    ...         rt=3600.0,
    ...         dtbc=1.0,
    ...     )
    ... )
    """

    model_type: Literal["spectral"] = Field(
        default="spectral",
        description="Model type discriminator for spectral wave boundaries",
    )
    wbctype: Literal["jons", "parametric", "swan", "vardens", "jonstable"] = Field(
        description="Spectral wave boundary type (jons, parametric, swan, vardens, jonstable)"
    )
    bcfile: str = Field(
        description="Path to spectral boundary file (required for spectral boundaries)"
    )
    wbc: Optional[SpectralWaveBoundaryConditions] = Field(
        default=None, description="Spectral wave boundary parameters"
    )


class NonSpectralWaveBoundary(WaveBoundary):
    """Non-spectral wave boundary specification.

    For non-spectral boundary types: stat, stat_table, ts_1, ts_2, ts_nonh, bichrom.

    File requirements:
    - stat: No file needed (parameters only)
    - bichrom: No file needed (parameters only)
    - stat_table: Requires JONSWAP table format file
    - ts_1, ts_2: Requires bc/gen.ezs file (time, zs, E)
    - ts_nonh: Requires Boun_u.bcf file (t, U, Zs, W)

    Examples
    --------
    >>> # Stationary waves (no file needed)
    >>> boundary = NonSpectralWaveBoundary(
    ...     wbctype="stat",
    ...     wbc=NonSpectralWaveBoundaryConditions(
    ...         Hrms=2.0,
    ...         Trep=12.0,
    ...         dir0=285.0,
    ...         m=10,
    ...     )
    ... )

    >>> # Time series (file required)
    >>> boundary = NonSpectralWaveBoundary(
    ...     wbctype="ts_1",
    ...     bcfile="bc/gen.ezs",
    ...     wbc=NonSpectralWaveBoundaryConditions(
    ...         Hrms=2.0,
    ...         Trep=12.0,
    ...     )
    ... )
    """

    model_type: Literal["nonspectral"] = Field(
        default="nonspectral",
        description="Model type discriminator for non-spectral wave boundaries",
    )
    wbctype: Literal["stat", "stat_table", "ts_1", "ts_2", "ts_nonh", "bichrom"] = (
        Field(
            description="Non-spectral wave boundary type (stat, stat_table, ts_1, ts_2, ts_nonh, bichrom)"
        )
    )
    bcfile: Optional[str] = Field(
        default=None,
        description="Path to boundary file (required for stat_table, ts_1, ts_2, ts_nonh)",
    )
    wbc: Optional[NonSpectralWaveBoundaryConditions] = Field(
        default=None, description="Non-spectral wave boundary parameters"
    )

    @model_validator(mode="after")
    def validate_bcfile(self):
        """Validate that bcfile is provided when required."""
        needs_file = self.wbctype in ["stat_table", "ts_1", "ts_2", "ts_nonh"]
        if needs_file and not self.bcfile:
            file_format = {
                "stat_table": "JONSWAP table format",
                "ts_1": "bc/gen.ezs (time, zs, E)",
                "ts_2": "bc/gen.ezs (time, zs, E)",
                "ts_nonh": "Boun_u.bcf (scalar/vector, t, U, Zs, W)",
            }
            raise ValueError(
                f"wbctype='{self.wbctype}' requires bcfile to be specified. "
                f"Expected file format: {file_format[self.wbctype]}"
            )
        return self


class OffWaveBoundary(WaveBoundary):
    """No wave forcing.

    Use this when you don't want any wave forcing in the model.

    Examples
    --------
    >>> boundary = OffWaveBoundary()
    """

    model_type: Literal["off"] = Field(
        default="off", description="Model type discriminator for no wave forcing"
    )
    wbctype: Literal["off"] = Field(
        default="off", description="Wave boundary type set to 'off' (no wave forcing)"
    )


class ReuseWaveBoundary(WaveBoundary):
    """Reuse previous boundary conditions.

    Makes XBeach reuse wave time series from a previous simulation.
    Requires copying ebcflist.bcf and qbcflist.bcf files (and referenced files)
    to the current working directory.

    Examples
    --------
    >>> boundary = ReuseWaveBoundary()
    >>> # Or with explicit file path
    >>> boundary = ReuseWaveBoundary(bcfile="path/to/ebcflist.bcf")
    """

    model_type: Literal["reuse"] = Field(
        default="reuse",
        description="Model type discriminator for reusing previous boundary conditions",
    )
    wbctype: Literal["reuse"] = Field(
        default="reuse",
        description="Wave boundary type set to 'reuse' (reuse previous simulation)",
    )
    bcfile: Optional[str] = Field(
        default=None, description="Path to previous boundary files (optional)"
    )
