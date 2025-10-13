"""XBeach vegetation parameter configurations.

This module contains the Vegetation model used by the Physics.vegetation field.
"""

from typing import Literal, Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class Vegetation(XBeachBaseModel):
    """Vegetation model configuration.

    When used in Physics.vegetation field, this enables vegetation modeling (vegetation=1)
    and allows specification of vegetation-specific parameters.
    """

    model_type: Literal[True] = Field(
        default=True,
        description="Model type discriminator - set to True to enable vegetation",
    )
    nveg: Optional[int] = Field(
        default=None,
        description="Number of vegetation species",
        ge=1,
    )
    veggiefile: Optional[str] = Field(
        default=None,
        description="Name of veggie species list file",
    )
    veggiemapfile: Optional[str] = Field(
        default=None,
        description="Name of veggie species map file",
    )
    vegcanflo: Optional[bool] = Field(
        default=None,
        description="Include incanopy flow (XBeach default: 0)",
    )
    vegnonlin: Optional[bool] = Field(
        default=None,
        description="Include non-linear wave effect (XBeach default: 0)",
    )
    veguntow: Optional[bool] = Field(
        default=None,
        description="Include undertow in phase-averaged vegetation (XBeach default: 1)",
    )
