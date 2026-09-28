"""XBeach wind parameter configurations.

This module contains models for wind-related physics parameters.
Note: This is separate from forcing.py which handles wind data specification.
"""

from typing import Literal, Optional

from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


class Wind(XBeachBaseModel):
    """Wind physics parameters.

    These parameters control how wind forcing affects the hydrodynamics.
    The wind stress is calculated as: tau = rhoa * Cd * |W| * W

    Note
    ----
    This class defines wind *physics parameters*, not wind forcing data.
    Wind velocity and direction are specified via:
    - `Config.input.wind` for data-driven wind (WindGrid, WindStation, WindPoint)
    - Direct `windv`/`windth` parameters for constant wind

    The air density `rhoa` is defined in `PhysicalConstants` as it is a
    material property rather than a tuning parameter.

    Examples
    --------
    >>> from rompy_xbeach.components.physics import Physics
    >>> from rompy_xbeach.components.physics.wind import Wind
    >>>
    >>> # Enable wind with default parameters
    >>> physics = Physics(wind=True)
    >>>
    >>> # Enable wind with custom drag coefficient
    >>> physics = Physics(wind=Wind(Cd=0.003))
    """

    wind: Literal[True] = Field(
        default=True,
        description="Enable wind in flow solver (always True when using Wind class)",
    )
    Cd: Optional[float] = Field(
        default=None,
        description=(
            "Wind drag coefficient for wind stress calculation. "
            "tau = rhoa * Cd * |W| * W (XBeach default: 0.002)"
        ),
        ge=0.0001,
        le=0.01,
    )
