"""XBeach numerical scheme parameter configurations.

This module contains models for numerical scheme parameters used in wave
and non-hydrostatic computations.
"""

import logging
from typing import Literal, Optional
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel


logger = logging.getLogger(__name__)


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
                "upwind_2 (second-order upwind), warmbeam (Warming-Beam) "
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


class FlowNumerics(XBeachBaseModel):
    """Flow numerical parameters.

    These parameters control numerical aspects of the shallow water equations,
    particularly handling of very shallow water and wet/dry transitions.

    The threshold parameters (eps, hmin, umin) prevent unrealistic behavior in
    shallow water by setting minimum values for depth and velocity calculations.
    """

    eps: Optional[float] = Field(
        default=None,
        description=(
            "Threshold water depth above which cells are considered wet "
            "(XBeach default: 0.005 m)"
        ),
        ge=0.001,
        le=0.1,
    )
    epsi: Optional[float] = Field(
        default=None,
        description=(
            "Ratio of mean current to time varying current through offshore boundary "
            "(XBeach default: -1.0)"
        ),
        ge=-1.0,
        le=0.2,
    )
    eps_sd: Optional[float] = Field(
        default=None,
        description=(
            "Threshold velocity difference to determine conservation of "
            "energy head versus momentum (XBeach default: 0.5 m/s)"
        ),
        ge=0.0,
        le=1.0,
    )
    hmin: Optional[float] = Field(
        default=None,
        description=(
            "Threshold water depth above which Stokes drift is included "
            "(XBeach default: 0.0 m). See also deltahmin and oldhmin"
        ),
        ge=0.001,
        le=1.0,
    )
    oldhu: Optional[bool] = Field(
        default=None,
        description=("Switch to enable old hu calculation (XBeach default: 0)"),
    )
    secorder: Optional[bool] = Field(
        default=None,
        description=(
            "Use second order corrections to advection/non-linear terms "
            "based on MacCormack scheme (XBeach default: 0)"
        ),
    )
    umin: Optional[float] = Field(
        default=None,
        description=(
            "Threshold velocity for upwind velocity detection and for vmag2 "
            "in equilibrium sediment concentration (XBeach default: 0.0 m/s)"
        ),
        ge=0.0,
        le=0.2,
    )
    deltahmin: Optional[float] = Field(
        default=None,
        description=(
            "Dimensionless coefficient to determine the threshold water depth "
            "above which Stokes drift is included (XBeach default: 0.1)"
        ),
        ge=0.0,
        le=1.0,
    )
    oldhmin: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to apply the old hmin parameter instead of deltahmin. "
            "If 1, hmin is used directly as minimum water depth "
            "(XBeach default: 0)"
        ),
    )
    defuse: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable diffusion in the flow solver to prevent "
            "numerical instabilities (XBeach default: 1)"
        ),
    )
    dtset: Optional[float] = Field(
        default=None,
        description=(
            "Fixed timestep (s). If > 0, overrides automatic timestep calculation. "
            "Use with caution - may cause instabilities (XBeach default: 0.0, automatic)"
        ),
        ge=0.0,
        le=100.0,
    )
    maxdtfac: Optional[float] = Field(
        default=None,
        description=(
            "Maximum factor for timestep increase in explosion prevention mechanism. "
            "For surfbeat/stationary: 10-200 (default 50). "
            "For nonh: 100-1000 (default 500)"
        ),
        ge=10.0,
        le=1000.0,
    )
    cfl: Optional[float] = Field(
        default=None,
        description=(
            "Maximum Courant-Friedrichs-Lewy number for timestep control. "
            "Lower values give more stability but slower computation "
            "(XBeach default: 0.7)"
        ),
        ge=0.1,
        le=0.9,
    )

    @model_validator(mode="after")
    def warn_if_oldhmin_and_deltahmin(self):
        if self.oldhmin and self.deltahmin:
            logger.warning(
                "You are setting deltahmin, but oldhmin is also set. "
                "The old hmin parameter will be used instead of the deltahmin"
            )
        return self
