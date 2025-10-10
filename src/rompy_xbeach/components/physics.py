"""XBeach physical processes configuration."""

import logging
from typing import Literal, Optional, Any
from pydantic import Field, field_serializer, model_serializer, model_validator

from rompy.core.types import RompyBaseModel

logger = logging.getLogger(__name__)


class Physics(RompyBaseModel):
    """XBeach physical processes configuration.

    XBeach supports a variety of physical processes from generic, like waves and flow,
    to very specific, like ship motions and point discharge. Each process can be
    switched on or off. The commonly used processes are turned on by default.

    This class allows configuration of all physical process switches in XBeach. All
    fields default to None, meaning XBeach's default values will be used unless
    explicitly specified.

    See https://xbeach.readthedocs.io/en/latest/xbeach_manual.html#physical-processes
    for more information.

    """

    model_type: Literal["physics"] = Field(
        default="physics",
        description="Model type discriminator",
    )
    wavemodel: Optional[Literal["stationary", "surfbeat", "nonh"]] = Field(
        default=None,
        description=(
            "Wave model type: stationary (0), surfbeat (1) or non-hydrostatic (2) "
            "(XBeach default: surfbeat)"
        ),
    )
    advection: Optional[bool] = Field(
        default=None,
        description="Include advection in flow solver (XBeach default: 1)",
    )
    avalanching: Optional[bool] = Field(
        default=None,
        description="Turn on avalanching (XBeach default: 1)",
    )
    cyclic: Optional[bool] = Field(
        default=None,
        description="Turn on cyclic boundary conditions (XBeach default: 0)",
    )
    flow: Optional[bool] = Field(
        default=None,
        description="Turn on flow calculation (XBeach default: 1)",
    )
    gwflow: Optional[bool] = Field(
        default=None,
        description="Turn on groundwater flow (XBeach default: 0)",
    )
    lwave: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on short wave forcing on NLSW equations and boundary conditions "
            "(XBeach default: 1)"
        ),
    )
    morphology: Optional[bool] = Field(
        default=None,
        description="Turn on morphology (XBeach default: 0)",
    )
    nonh: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on non-hydrostatic pressure: 0 = NSWE, 1 = NSW + non-hydrostatic "
            "pressure compensation Stelling & Zijlema, 2003 (XBeach default: 0)"
        ),
    )
    q3d: Optional[bool] = Field(
        default=None,
        description="Turn on quasi-3D sediment transport (XBeach default: 0)",
    )
    sedtrans: Optional[bool] = Field(
        default=None,
        description="Turn on sediment transport (XBeach default: 1)",
    )
    setbathy: Optional[bool] = Field(
        default=None,
        description="Turn on timeseries of prescribed bathy input (XBeach default: 0)",
    )
    ships: Optional[bool] = Field(
        default=None,
        description="Turn on ship waves (XBeach default: 0)",
    )
    single_dir: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on stationary model for refraction, surfbeat based on mean direction "
            "(XBeach default: 1)"
        ),
    )
    snells: Optional[bool] = Field(
        default=None,
        description="Turn on Snell's law for wave refraction (XBeach default: 0)",
    )
    swave: Optional[bool] = Field(
        default=None,
        description="Turn on short waves (XBeach default: 1)",
    )
    swrunup: Optional[bool] = Field(
        default=None,
        description="Turn on short wave runup (XBeach default: 0)",
    )
    vegetation: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on interaction of waves and flow with vegetation (XBeach default: 0)"
        ),
    )
    viscosity: Optional[bool] = Field(
        default=None,
        description="Include viscosity in flow solver (XBeach default: 1)",
    )
    wind: Optional[bool] = Field(
        default=None,
        description="Include wind in flow solver (XBeach default: 1)",
    )

    @model_validator(mode="after")
    def log_default_enabled_processes(self) -> "Physics":
        """Logging for default-enabled processes that are not explicitly set."""
        default_enabled_fields = [
            "advection",
            "avalanching",
            "flow",
            "lwave",
            "sedtrans",
            "single_dir",
            "swave",
            "viscosity",
            "wind",
        ]

        for field_name in default_enabled_fields:
            value = getattr(self, field_name)
            if value is None:
                logger.debug(
                    f"Parameter '{field_name}' not explicitly set - "
                    f"will be ENABLED by XBeach default."
                )

        return self

    @model_serializer(mode="wrap")
    def _serialize_bools_to_ints(self, serializer: Any) -> dict:
        """Serialize model with automatic bool to int conversion."""
        data = serializer(self)
        for key, value in list(data.items()):
            if isinstance(value, bool):
                data[key] = int(value)
        return data

    @property
    def params(self) -> dict:
        """Return the XBeach parameters for the physics component."""
        return self.model_dump(exclude_none=True, exclude=["model_type"])

    def get(self, destdir=None) -> dict:
        """Return the params dict.

        The destdir parameter is included for consistency with other components
        but is not used by the Physics component as it doesn't fetch external files.

        """
        return self.params
