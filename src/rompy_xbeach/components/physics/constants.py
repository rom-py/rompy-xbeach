"""XBeach physical constants and Coriolis force parameter configurations.

This module contains models for physical constants and Coriolis force parameters.
"""

from typing import Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class PhysicalConstants(XBeachBaseModel):
    """Physical constants used in XBeach computations.

    These fundamental physical constants are used throughout XBeach calculations.
    The depthscale parameter allows scaling of various depth-related thresholds
    for laboratory-scale simulations.
    """

    g: Optional[float] = Field(
        default=None,
        description="Gravitational acceleration (m/s²) (XBeach default: 9.81)",
        ge=9.7,
        le=9.9,
    )
    rho: Optional[float] = Field(
        default=None,
        description="Density of water (kgm-3) (XBeach default: 1025.0)",
        ge=1000.0,
        le=1040.0,
    )
    depthscale: Optional[float] = Field(
        default=None,
        description=(
            "Depth scale of (lab) test simulated. Affects eps, hmin, hswitch "
            "and dzmax. A value lower than 1 increases the cut-off values "
            "(XBeach default: 1.0 for field scale)"
        ),
        ge=1.0,
        le=200.0,
    )


class Coriolis(XBeachBaseModel):
    """Coriolis force parameters.

    The Coriolis force affects the shallow water equations and becomes important
    for large-scale coastal applications or long simulation times. The effect
    depends on the latitude of the model location.

    The Coriolis parameter f is calculated as: f = 2 * wearth * sin(lat)
    """

    lat: Optional[float] = Field(
        default=None,
        description=(
            "Latitude at model location for computing Coriolis force "
            "(XBeach default: 0.0 degrees, no Coriolis effect)"
        ),
        ge=-90.0,
        le=90.0,
    )
    wearth: Optional[float] = Field(
        default=None,
        description=(
            "Angular velocity of Earth calculated as 1/rotation_time. "
            "Default is 1/24 hour⁻¹ (XBeach default: 0.04167 hour⁻¹)"
        ),
        ge=0.0,
        le=1.0,
    )
