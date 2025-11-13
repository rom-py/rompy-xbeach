"""XBeach numerical scheme parameter configurations.

This module contains models for numerical scheme parameters used in wave
and non-hydrostatic computations.
"""

from typing import Literal, Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class WaveNumerics(XBeachBaseModel):
    """Wave action balance numerical parameters.

    These parameters control the numerical aspects of the wave action balance
    solver, including the numerical scheme and convergence criteria.

    The Warming and Beam (1976) scheme is used by default to overcome undesired
    effects of steepening of wave groups through a small additional diffusion term.

    References
    ----------
    Beam, R. M., & Warming, R. F. (1976). An implicit finite-difference algorithm
    for hyperbolic systems in conservation-law form. Journal of computational
    physics, 22(1), 87-110.
    """

    scheme: Optional[Literal["upwind_1", "lax_wendroff", "upwind_2", "warmbeam"]] = (
        Field(
            default=None,
            description=(
                "Numerical scheme for wave propagation. Options: "
                "upwind_1 (first-order upwind), lax_wendroff (Lax-Wendroff), "
                "upwind_2 (second-order upwind), warmbeam (Warming-Beam, default) "
                "(XBeach default: warmbeam)"
            ),
        )
    )
    maxiter: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of iterations in wave stationary solver "
            "(XBeach default: 500)"
        ),
        ge=2,
        le=1000,
    )
    maxerror: Optional[float] = Field(
        default=None,
        description=(
            "Maximum wave height error in wave stationary iteration "
            "(XBeach default: 0.0005 m)"
        ),
        ge=1e-05,
        le=0.001,
    )
    wavint: Optional[float] = Field(
        default=None,
        description=(
            "Interval between wave module calls in seconds "
            "(only in stationary wave mode, XBeach default: 600.0 s)"
        ),
        ge=1.0,
        le=3600.0,
    )


class NonHydrostaticNumerics(XBeachBaseModel):
    """Non-hydrostatic solver numerical parameters.

    These parameters control the non-hydrostatic pressure correction solver,
    including the linear solver method, convergence criteria, and wave breaking
    detection in non-hydrostatic mode.

    These parameters only apply when wavemodel='nonh' or nonh=1.
    """

    solver: Optional[Literal["sip", "tridiag"]] = Field(
        default=None,
        description=(
            "Solver used to solve the linear system. Options: "
            "sip (Strongly Implicit Procedure), tridiag (tridiagonal, default)"
        ),
    )
    solver_acc: Optional[float] = Field(
        default=None,
        description=(
            "Accuracy with respect to the right-hand side used in termination "
            "criterion: ||b-Ax|| < acc*||b|| (XBeach default: 0.005)"
        ),
        ge=1e-05,
        le=0.1,
    )
    solver_maxit: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of iterations in the linear SIP solver (XBeach default: 30)"
        ),
        ge=1,
        le=1000,
    )
    solver_urelax: Optional[float] = Field(
        default=None,
        description=(
            "Under-relaxation parameter for SIP solver (XBeach default: 0.92)"
        ),
        ge=0.5,
        le=0.99,
    )
    Topt: Optional[float] = Field(
        default=None,
        description=(
            "Absolute period to optimize coefficient in non-hydrostatic solver "
            "(XBeach default: 10.0 s)"
        ),
        ge=1.0,
        le=20.0,
    )
    dispc: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient in front of the vertical pressure gradient "
            "(XBeach default: -1.0)"
        ),
        ge=0.1,
        le=2.0,
    )
    kdmin: Optional[float] = Field(
        default=None,
        description=("Minimum value of kd (pi/dx > min(kd)) (XBeach default: 0.0)"),
        ge=0.0,
        le=0.05,
    )
    nhlay: Optional[float] = Field(
        default=None,
        description=(
            "Layer distribution in the non-hydrostatic model (XBeach default: 0.33)"
        ),
        ge=0.0,
        le=1.0,
    )
    maxbrsteep: Optional[float] = Field(
        default=None,
        description=(
            "Maximum wave steepness criterion for breaking detection "
            "(XBeach default: 0.4)"
        ),
        ge=0.3,
        le=0.8,
    )
    secbrsteep: Optional[float] = Field(
        default=None,
        description=(
            "Secondary maximum wave steepness criterion "
            "(XBeach default: 0.5 * maxbrsteep)"
        ),
        ge=0.0,
        le=0.8,
    )
    reformsteep: Optional[float] = Field(
        default=None,
        description=(
            "Wave steepness criterion to reform after breaking "
            "(XBeach default: 0.25 * maxbrsteep)"
        ),
        ge=0.0,
        le=0.8,
    )
    nhbreaker: Optional[int] = Field(
        default=None,
        description=(
            "Non-hydrostatic breaker model: 0=no breaking, 1=roelvink, 2=daly "
            "(XBeach default: 2)"
        ),
        ge=0,
        le=2,
    )
