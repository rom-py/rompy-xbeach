"""XBeach wave boundary condition parameter configurations.

This module contains models for wave boundary condition parameters that control
how short and long waves are specified at the offshore boundary.

This is the single source of truth for all wave boundary condition parameters,
organized into:
- Base class: General parameters (apply to all boundary types)
- SpectralWaveBoundaryConditions: Spectral-specific parameters (jons, swan, vardens, jonstable)
- NonSpectralWaveBoundaryConditions: Non-spectral parameters (stat, ts_1, ts_2, ts_nonh, bichrom)
"""

from typing import Optional, Literal
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class WaveBoundaryConditions(XBeachBaseModel):
    """Base wave boundary condition parameters.

    These are general parameters that apply to ALL wave boundary condition types,
    whether spectral (jons, swan, vardens, jonstable) or non-spectral (stat, ts_1,
    ts_2, ts_nonh, bichrom).

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
    taper: Optional[float] = Field(
        default=None,
        description=(
            "Spin-up time of wave boundary conditions, in morphological time "
            "(XBeach default: 100.0)"
        ),
        ge=0.0,
        le=1000.0,
        examples=[100.0],
    )
    ARC: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for active reflection compensation at seaward boundary. "
            "Compensates for spurious long wave reflection (XBeach default: 1)"
        ),
    )
    freewave: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for free wave propagation at the boundary. When enabled, assumes "
            "incoming long waves propagate at sqrt(gh) instead of group velocity cg. "
            "Affects absorbing/radiating boundary calculations (XBeach default: 0)"
        ),
    )
    thetamin: Optional[float] = Field(
        default=None,
        description=(
            "Minimum wave angle (degrees). When thetanaut=0, this is relative to the "
            "grid x-axis (shore-normal); when thetanaut=1, this is in nautical convention "
            "(N=0°, E=90°). Only used when swave=1 (XBeach default: -90.0)"
        ),
        ge=-360.0,
        le=360.0,
    )
    thetamax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum wave angle (degrees). When thetanaut=0, this is relative to the "
            "grid x-axis (shore-normal); when thetanaut=1, this is in nautical convention "
            "(N=0°, E=90°). Only used when swave=1 (XBeach default: 90.0)"
        ),
        ge=-360.0,
        le=360.0,
    )
    dtheta: Optional[float] = Field(
        default=None,
        description=(
            "Wave directional resolution (degrees). Automatically computed from "
            "thetamax-thetamin when single_dir=1. Only used when swave=1 "
            "(XBeach default: 10.0)"
        ),
        ge=0.1,
        le=180.0,
    )
    thetanaut: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for wave direction convention. When 0 (default), wave angles are "
            "relative to the grid x-axis and rotated by alfa internally. When 1, wave "
            "angles are in nautical convention (N=0°, E=90°) using real-world coordinates "
            "and alfa is ignored. Only used when swave=1 (XBeach default: 0)"
        ),
    )
    order: Optional[Literal[1, 2]] = Field(
        default=None,
        description=(
            "Order of wave steering at the boundary. 1 = first-order (short wave energy "
            "only), 2 = second-order (bound long wave corresponding to short wave forcing "
            "is added) (XBeach default: 2)"
        ),
    )


class SpectralWaveBoundaryConditions(WaveBoundaryConditions):
    """Spectral wave boundary condition parameters.

    These parameters are specific to spectral boundary conditions (wbctype = jons,
    swan, vardens, jonstable). They control how wave spectra are generated and
    applied at the offshore boundary.

    Inherits all general wave boundary parameters from WaveBoundaryConditions.
    """

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
    dthetas_xb: Optional[float] = Field(
        default=None,
        description=(
            "The (counter-clockwise) angle in the degrees needed to rotate from the "
            "x-axis in swan to the x-axis pointing east (XBeach default: 0.0)",
        ),
        ge=-360.0,
        le=360.0,
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


class NonSpectralWaveBoundaryConditions(WaveBoundaryConditions):
    """Non-spectral wave boundary condition parameters.

    These parameters are specific to non-spectral boundary conditions (wbctype = stat,
    ts_1, ts_2, ts_nonh, stat_table, bichrom). They define wave conditions without
    full spectral information.

    Inherits all general wave boundary parameters from WaveBoundaryConditions.
    """

    # ==================================================================================
    # Non-Spectral Boundary Parameters
    # ==================================================================================
    Hrms: Optional[float] = Field(
        default=None,
        description=(
            "Hrms wave height for instat = stat, bichrom, ts_1 or ts_2 "
            "(XBeach default: 1.0)"
        ),
        ge=0.0,
        le=10.0,
        examples=[1.0],
    )
    Trep: Optional[float] = Field(
        default=None,
        description=(
            "Representative wave period for instat = stat, bichrom, ts_1 or ts_2 "
            "(XBeach default: 10.0)"
        ),
        ge=1.0,
        le=20.0,
        examples=[10.0],
    )
    Tlong: Optional[float] = Field(
        default=None,
        description=(
            "Wave group period for case instat = bichrom (XBeach default: 80.0)"
        ),
        ge=20.0,
        le=300.0,
        examples=[80.0],
    )
    dir0: Optional[float] = Field(
        default=None,
        description=(
            "Mean wave direction for instat = stat, bichrom, ts_1 or ts_2, "
            "nautical convention (XBeach default: 270.0)"
        ),
        ge=-360.0,
        le=360.0,
        examples=[270.0],
    )
    m: Optional[int] = Field(
        default=None,
        description=(
            "Power in cos^m directional distribution for instat = stat, bichrom, "
            "ts_1 or ts_2 (XBeach default: 10)"
        ),
        ge=2,
        le=128,
        examples=[10],
    )
