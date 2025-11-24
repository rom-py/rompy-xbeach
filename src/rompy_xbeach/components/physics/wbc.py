"""XBeach wave boundary condition parameter configurations.

This module contains models for wave boundary condition parameters that control
how short and long waves are specified at the offshore boundary.

This is the single source of truth for all wave boundary condition parameters,
including both general parameters and spectral-specific parameters.
"""

from typing import Optional, Literal
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class WaveBoundaryConditions(XBeachBaseModel):
    """Wave boundary condition parameters.

    These parameters control how short waves (wave action balance) and long waves
    (infragravity waves) are specified and handled at the offshore boundary.

    This class contains:
    - General wave boundary parameters (nmax, wbcevarreduce, etc.)
    - Spectral boundary parameters (rt, dtbc, random, fcutoff, etc.)

    The boundary conditions affect wave generation, energy scaling, and the
    treatment of Stokes drift and wave group variance at the boundary.
    """

    # ==================================================================================
    # General Wave Boundary Parameters
    # ==================================================================================

    nmax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum ratio of cg/c for computing long wave boundary conditions "
            "(XBeach default: 0.8)"
        ),
        ge=0.5,
        le=1.0,
    )
    wbcevarreduce: Optional[float] = Field(
        default=None,
        description=(
            "Reduction factor of short-wave group variance at the boundary "
            "(XBeach default: 1.0, no reduction)"
        ),
        ge=0.0,
        le=1.0,
    )
    bclwonly: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to run boundary conditions with long waves only (XBeach default: 0)"
        ),
    )
    swkhmin: Optional[float] = Field(
        default=None,
        description=(
            "Minimum kh value to include in wave action balance. "
            "Waves with lower kh are included in NLSWE instead "
            "(XBeach default: -0.01)"
        ),
        ge=-0.01,
        le=0.35,
    )
    wbcRemoveStokes: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to remove long wave Stokes drift component at the "
            "offshore boundary (XBeach default: 1)"
        ),
    )
    wbcScaleEnergy: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to correct random time series of wave height to exactly "
            "match input Hm0 (XBeach default: 1)"
        ),
    )
    cyclicdiradjust: Optional[bool] = Field(
        default=None,
        description=(
            "Adjust alongshore wave length to fit inside domain with cyclic "
            "boundary conditions (XBeach default: 0)"
        ),
    )

    # ==================================================================================
    # Spectral Boundary Parameters
    # ==================================================================================
    rt: Optional[float] = Field(
        default=None,
        description=(
            "Duration (s) of wave spectrum at offshore boundary, in morphological "
            "time (XBeach default: min(3600.d0, tstop))"
        ),
        ge=1200.0,
        le=7200.0,
        examples=[3600.0],
    )
    dtbc: Optional[float] = Field(
        default=None,
        description=(
            "Timestep (s) used to describe time series of wave energy and long wave "
            "flux at offshore boundary (not affected by morfac) (XBeach default: 1.0)"
        ),
        ge=0.1,
        le=2.0,
        examples=[1.0],
    )
    tm01switch: Optional[bool] = Field(
        default=None,
        description="Switch to enable tm01 rather than tm-10 (XBeach default: 0)",
    )
    correcthm0: Optional[bool] = Field(
        default=None,
        description="Switch to enable hm0 correction (XBeach default: 1)",
    )
    fcutoff: Optional[float] = Field(
        default=None,
        description=(
            "Low-freq cutoff frequency in Hz for jons, swan or vardens boundary "
            "conditions (XBeach default: 0.0)"
        ),
        ge=0.0,
        le=40.0,
    )
    nonhspectrum: Optional[Literal[0, 1]] = Field(
        default=None,
        description=(
            "Spectrum format for wave action balance of nonhydrostatic waves "
            "(XBeach default: 0)"
        ),
    )
    nspectrumloc: Optional[int] = Field(
        default=None,
        description=("Number of input spectrum locations (XBeach default: 1)"),
        ge=1,
    )
    nspr: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable long wave direction forced into centres of short wave "
            "bins (XBeach default: 0)",
        ),
    )
    random: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable random seed for jons, swan or vardens boundary "
            "conditions (XBeach default: 1)",
        ),
    )
    sprdthr: Optional[float] = Field(
        default=None,
        description=(
            "Threshold ratio to maximum value of s above which spectrum densities "
            "are read in (XBeach default: 0.08)"
        ),
        ge=0.0,
        le=1.0,
    )
    trepfac: Optional[float] = Field(
        default=None,
        description=(
            "Compute mean wave period over energy band: par%trepfac*maxval(sf) for "
            "jons, swan or vardens; converges to tm01 for trepfac = 0.0 "
            "(XBeach default: 0.01)",
        ),
        ge=0.0,
        le=1.0,
    )
    wbcversion: Optional[Literal[1, 2, 3]] = Field(
        default=None,
        description="Version of wave boundary conditions (XBeach default: 3)",
    )
