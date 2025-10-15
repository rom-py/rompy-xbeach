"""XBeach flow parameter configurations.

This module contains models for flow-related parameters including horizontal viscosity.
"""

import logging
from typing import Optional
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel


logger = logging.getLogger(__name__)


class HorizontalViscosity(XBeachBaseModel):
    """Horizontal viscosity configuration.

    XBeach uses the Smagorinsky (1963) model by default to compute horizontal
    viscosity, accounting for momentum exchange at spatial scales smaller than
    the computational grid. Alternatively, a user-defined constant viscosity
    can be specified.

    The `nuh` parameter has dual meaning depending on `smag`:
    - If `smag=1` (default): `nuh` is the Smagorinsky constant (default: 0.1)
    - If `smag=0`: `nuh` is the horizontal background viscosity in m²/s

    References
    ----------
    Smagorinsky, J. (1963). General circulation experiments with the primitive
    equations: I. The basic experiment. Monthly weather review, 91(3), 99-164.
    """

    smag: Optional[bool] = Field(
        default=None,
        description=(
            "Switch for Smagorinsky subgrid model for viscosity. "
            "If enabled (1), nuh is the Smagorinsky constant. "
            "If disabled (0), nuh is the constant horizontal viscosity (XBeach default: 1)"
        ),
    )
    nuh: Optional[float] = Field(
        default=None,
        description=(
            "Horizontal viscosity parameter. Meaning depends on smag: "
            "If smag=1: Smagorinsky constant (dimensionless, XBeach default: 0.1). "
            "If smag=0: Horizontal background viscosity (m2/s, XBeach default: 0.1)"
        ),
        ge=0.0,
        le=1.0,
    )
    nuhv: Optional[float] = Field(
        default=None,
        description=(
            "Longshore viscosity enhancement factor, following Svendsen "
            "(XBeach default: 1.0)"
        ),
        ge=1.0,
        le=20.0,
    )
    gamma_turb: Optional[float] = Field(
        default=None,
        description=(
            "Calibration factor for turbulence contribution to bed roughness "
            "(XBeach default: 1.0)"
        ),
        ge=0.0,
        le=2.0,
    )

    @model_validator(mode="after")
    def validate_viscosity_consistency(self) -> "HorizontalViscosity":
        """Validate that viscosity parameters are used consistently."""
        if self.smag is False and self.nuh is None:
            logger.warning(
                "The smagorinsky subgrid model for viscosity is disabled, but "
                "horizontal viscosity (nuh) is not specified"
            )

        return self


class WaveCurrentInteraction(XBeachBaseModel):
    """Wave-current interaction parameters.

    Wave-current interaction (WCI) accounts for the feedback of currents on wave
    propagation. When enabled, currents affect the wave celerity and direction
    through Doppler shifting.

    The WCI computation is limited to specific depth ranges (hwci to hwcimax) to
    avoid numerical issues in very shallow or deep water.

    References
    ----------
    Dingemans et al. (1987). Water wave propagation over uneven bottoms.
    """

    cats: Optional[float] = Field(
        default=None,
        description=(
            "Current averaging time scale for wave-current interaction, "
            "in terms of mean wave periods (XBeach default: 4.0)"
        ),
        ge=1.0,
        le=50.0,
    )
    hwci: Optional[float] = Field(
        default=None,
        description=(
            "Minimum depth until which wave-current interaction is used "
            "(XBeach default: 0.1 m)"
        ),
        ge=0.001,
        le=1.0,
    )
    hwcimax: Optional[float] = Field(
        default=None,
        description=(
            "Maximum depth until which wave-current interaction is used "
            "(XBeach default: 100.0 m)"
        ),
        ge=0.01,
        le=100.0,
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
    deltahmin: Optional[float] = Field(
        default=None,
        description=(
            "Dimensionless coefficient to determine the threshold water depth "
            "above which Stokes drift is included. When oldhmin=0 (default), "
            "hmin = h when H < h, otherwise hmin = h + deltahmin*H "
            "(XBeach default: 0.1)"
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
    umin: Optional[float] = Field(
        default=None,
        description=(
            "Threshold velocity for upwind velocity detection and for vmag2 "
            "in equilibrium sediment concentration (XBeach default: 0.0 m/s)"
        ),
        ge=0.0,
        le=0.2,
    )
    secorder: Optional[bool] = Field(
        default=None,
        description=(
            "Use second order corrections to advection/non-linear terms "
            "based on MacCormack scheme (XBeach default: 0)"
        ),
    )
    oldhu: Optional[bool] = Field(
        default=None,
        description=("Switch to enable old hu calculation (XBeach default: 0)"),
    )
