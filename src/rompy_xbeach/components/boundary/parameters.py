"""XBeach flow and tide boundary condition parameter configurations.

This module contains models for flow and tide boundary condition parameters.

Flow Boundary Conditions:
- FlowBoundaryConditions: Flow boundary types and parameters for shallow water equations

Tide Boundary Conditions:
- TideBoundaryConditions: Tide/surge boundary parameters (tideloc, tidetype, zs0, paulrevere)

Note: Wave boundary condition parameters are now defined in data/boundary/base.py
and integrated directly into the wave boundary data classes.
"""

from typing import Optional, Literal
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


FrontType = Literal["abs_1d", "abs_2d", "wall", "wlevel", "nonh_1d", "waveflume"]
BackType = Literal["wall", "abs_1d", "abs_2d", "wlevel"]
LeftRightType = Literal["neumann", "wall", "no_advec", "neumann_v", "abs_1d"]
LateralWaveType = Literal["neumann", "wavecrest", "cyclic"]


class FlowBoundaryConditions(XBeachBaseModel):
    """Flow boundary condition parameters for shallow water equations.

    Controls boundary conditions at all domain boundaries: offshore (front),
    bay side (back), and lateral (left/right). These parameters determine how
    flow, water levels, and waves interact with the domain boundaries.

    The default absorbing-generating (abs_2d) boundary condition is recommended
    for most applications as it allows waves to pass through with minimal reflection.
    """

    front: Optional[FrontType] = Field(
        default=None,
        description=(
            "Seaward boundary condition type. abs_1d/abs_2d = absorbing-generating "
            "(weakly-reflective), wall = no flux, wlevel = water level specification, "
            "nonh_1d = non-hydrostatic, waveflume = flume experiments "
            "(XBeach default: abs_2d)"
        ),
    )
    back: Optional[BackType] = Field(
        default=None,
        description=(
            "Bay side boundary condition type. wall = no flux, abs_1d/abs_2d = "
            "absorbing-generating, wlevel = water level specification "
            "(XBeach default: abs_2d)"
        ),
    )
    left: Optional[LeftRightType] = Field(
        default=None,
        description=(
            "Lateral boundary at ny+1. neumann = no gradient, wall = no flux, "
            "no_advec = advective terms only, neumann_v = copy velocity from adjacent "
            "cell, abs_1d = absorbing (XBeach default: neumann)"
        ),
    )
    right: Optional[LeftRightType] = Field(
        default=None,
        description=(
            "Lateral boundary at 0. neumann = no gradient, wall = no flux, "
            "no_advec = advective terms only, neumann_v = copy velocity from adjacent "
            "cell, abs_1d = absorbing (XBeach default: neumann)"
        ),
    )
    lateralwave: Optional[LateralWaveType] = Field(
        default=None,
        description=(
            "Lateral wave boundary type. neumann = zero longshore gradient (may cause "
            "shadow zones), wavecrest = zero gradient along wave crest (better for "
            "surfbeat), cyclic = periodic boundary (XBeach default: neumann)"
        ),
    )
    nc: Optional[int] = Field(
        default=None,
        description=(
            "Smoothing distance for estimating mean current (umean) at the offshore "
            "boundary, defined as number of grid cells (XBeach default: ny+1)"
        ),
        ge=1,
    )
    highcomp: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for high-order compensation at the boundary. Enables additional "
            "correction terms for improved accuracy (XBeach default: 0)"
        ),
    )


# Type definitions for tide boundary options
TideLocType = Literal[0, 1, 2, 4]
TideTypeType = Literal["instant", "velocity", "hybrid"]
PaulRevereType = Literal["land", "sea"]


class TideBoundaryConditions(XBeachBaseModel):
    """Tide and surge boundary condition parameters.

    Controls how tidal and surge water levels are applied at the domain boundaries.
    XBeach supports up to four time-varying tidal signals applied to the four corners
    (offshore-right, offshore-left, backshore-left, backshore-right).

    The tideloc parameter determines how many tide signals are used:
    - 0: Uniform water level (zs0 value applied everywhere)
    - 1: One time-varying signal (applied to offshore boundary)
    - 2: Two time-varying signals (requires paulrevere to specify application)
    - 4: Four time-varying signals (one per corner)
    """

    tideloc: Optional[TideLocType] = Field(
        default=None,
        description=(
            "Number of tide/surge boundary locations. 0 = uniform water level (zs0), "
            "1 = one signal at offshore, 2 = two signals (sea/land corners), "
            "4 = four signals (all corners) (XBeach default: 0 if zs0 set, else 2)"
        ),
    )
    tidetype: Optional[TideTypeType] = Field(
        default=None,
        description=(
            "Type of tide boundary condition. instant = instantaneous water level, "
            "velocity = velocity boundary, hybrid = combination "
            "(XBeach default: velocity)"
        ),
    )
    zs0: Optional[float] = Field(
        default=None,
        description=(
            "Initial/constant water level (m). Used when tideloc=0 for uniform water "
            "level, or as backshore boundary value when tideloc=1 (XBeach default: 0.0)"
        ),
        ge=-5.0,
        le=5.0,
    )
    paulrevere: Optional[PaulRevereType] = Field(
        default=None,
        description=(
            "Specifies which boundary receives tide signals when tideloc=2. "
            "land = one signal to land corners, one to sea corners; "
            "sea = opposite assignment (XBeach default: land)"
        ),
    )
