"""XBeach wave boundary condition parameter configurations.

This module contains models for wave boundary condition parameters that control
how short and long waves are specified at the offshore boundary.
"""

from typing import Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class WaveBoundaryConditions(XBeachBaseModel):
    """Wave boundary condition parameters.

    These parameters control how short waves (wave action balance) and long waves
    (infragravity waves) are specified and handled at the offshore boundary.

    The boundary conditions affect wave generation, energy scaling, and the
    treatment of Stokes drift and wave group variance at the boundary.
    """

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
