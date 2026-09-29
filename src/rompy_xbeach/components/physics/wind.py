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
    Enable wind with XBeach's default parameters, or set the drag coefficient:

    ```python exec="on" source="above" result="text" session="wind-wind"
    from rompy_xbeach.components.physics import Physics
    from rompy_xbeach.components.physics.wavemodel import Surfbeat
    from rompy_xbeach.components.physics.wind import Wind

    print(Physics(wavemodel=Surfbeat(), wind=True).get("."))
    print(Physics(wavemodel=Surfbeat(), wind=Wind(Cd=0.003)).get("."))
    ```
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
