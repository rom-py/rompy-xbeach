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

    @model_validator(mode="after")
    def validate_viscosity_consistency(self) -> "HorizontalViscosity":
        """Validate that viscosity parameters are used consistently."""
        if self.smag is False and self.nuh is None:
            logger.warning(
                "The smagorinsky subgrid model for viscosity is disabled, but "
                "horizontal viscosity (nuh) is not specified"
            )

        return self
