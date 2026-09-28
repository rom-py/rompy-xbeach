"""XBeach vegetation parameter configurations.

This module contains the Vegetation model used by the Physics.vegetation field.
"""

from typing import Literal, Optional
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel, XBeachDataBlob


class Vegetation(XBeachBaseModel):
    """Vegetation model configuration.

    The presence of aquatic vegetation within the area of wave propagation or wave
    breaking results in an additional dissipation mechanism for short waves. This is
    modeled using the approach of [Mendez and Losada, 2004], which was adjusted by
    [Suzuki et al., 2012] to take into account vertically heterogeneous vegetation,
    see [van Rooijen et al., 2015]. The short wave dissipation due to vegetation is
    calculated as function of the local wave height and several vegetation parameters.
    The vegetation can be schematized in a number of vertical elements with each
    pecific property. In this way the wave damping effect of vegetation such as mangrove
    trees, with a relatively dense root system but sparse stem area, can be modeled.
    The dissipation term is then computed as the sum of the dissipation per vegetation
    layer ([Suzuki et al., 2012])

    See https://xbeach.readthedocs.io/en/latest/xbeach_manual.html#vegetation for more
    information.

    """

    vegetation: Literal[True] = Field(
        default=True,
        description="Enable vegetation model",
    )
    nveg: Optional[int] = Field(
        default=None,
        description="Number of vegetation species",
        ge=1,
    )
    veggiefile: Optional[XBeachDataBlob] = Field(
        default=None,
        description="Name of veggie species list file",
    )
    veggiemapfile: Optional[XBeachDataBlob] = Field(
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
