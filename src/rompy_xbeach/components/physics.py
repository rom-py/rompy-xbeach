"""XBeach physical processes configuration."""

import logging
from typing import Annotated, Literal, Optional, Any, Union
from pydantic import Field, field_serializer, model_serializer, model_validator

from rompy.core.types import RompyBaseModel

logger = logging.getLogger(__name__)


BreakType = Literal["roelvink1", "baldock", "roelvink2", "roelvink_daly", "janssen"]


class Stationary(RompyBaseModel):
    """Stationary wave model configuration.

    Efficiently solves wave-averaged equations but neglects infragravity waves.
    Useful for conditions where incident waves are relatively small and/or short.

    """
    model_type: Literal["stationary"] = Field(
        default="stationary",
        description="Model type discriminator",
    )


class Surfbeat(RompyBaseModel):
    """Surfbeat (instationary) wave model configuration.

    Resolves short wave variations on the wave group scale (short wave envelope)
    and the long waves associated with them. This is the XBeach default mode.

    """
    model_type: Literal["surfbeat"] = Field(
        default="surfbeat",
        description="Model type discriminator",
    )


class Nonh(RompyBaseModel):
    """Non-hydrostatic (wave-resolving) wave model configuration.

    Uses non-linear shallow water equations with a pressure correction term,
    allowing modeling of propagation and decay of individual waves.

    """
    model_type: Literal["nonh"] = Field(
        default="nonh",
        description="Model type discriminator",
    )

    nhq3d: Optional[bool] = Field(
        default=None,
        description=(
            "Turn on reduced two-layer non-hydrostatic model for improved "
            "dispersive behavior (XBeach default: 0)"
        ),
    )

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
    wavemodel: Optional[Union[Stationary, Surfbeat, Nonh]] = Field(
        default=None,
        description=(
            "Wave model configuration: Stationary, Surfbeat or Nonh "
            "(XBeach default: surfbeat)"
        ),
        discriminator="model_type",
    )
    breaktype: Optional[BreakType] = Field(
        default=None,
        description="Type of breaker formulation (XBeach default: roelvink_daly)",
        alias="break",
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
    wci: Optional[bool] = Field(
        default=None,
        description="Turns on wave-current interaction (XBeach default: 0)",
    )
    wind: Optional[bool] = Field(
        default=None,
        description="Include wind in flow solver (XBeach default: 1)",
    )

    @model_validator(mode="after")
    def swave_must_be_false_if_nonh(self) -> "Physics":
        """Swave must be False if nonh is True or wavemodel is Nonh."""
        # Check if nonh parameter is True
        nonh_enabled = self.nonh is True
        
        # Also check if wavemodel is set to Nonh
        if self.wavemodel is not None and isinstance(self.wavemodel, Nonh):
            nonh_enabled = True
        
        if nonh_enabled:
            if self.swave is True:
                raise ValueError(
                    "Parameter 'swave' cannot be True when non-hydrostatic mode is enabled. "
                    "Set swave=False explicitly."
                )
            elif self.swave is None:
                raise ValueError(
                    "Parameter 'swave' must be explicitly set to False when non-hydrostatic "
                    "mode is enabled. XBeach would enable swave by default (swave=1), which "
                    "conflicts with nonh mode. Please set swave=False explicitly."
                )
        return self

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
            "wci",
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
    def _serialize_with_component_flattening(self, serializer: Any) -> dict:
        """Serialize model with automatic component flattening and bool to int conversion.
        
        This serializer:
        1. Detects nested dictionaries (from XBeachParameterComponent serialization)
        2. Flattens them by setting outer key = inner model_type value
        3. Merges remaining inner key-values into the main dict
        4. Converts booleans to integers for XBeach compatibility
        
        Example:
            {'wavemodel': {'model_type': 'nonh', 'nhq3d': True}}
            becomes:
            {'wavemodel': 'nonh', 'nhq3d': True}
        """
        data = serializer(self)
        
        # Flatten any nested dictionaries (parameter components)
        for field_name, field_value in list(data.items()):
            if isinstance(field_value, dict):
                # This is a nested dict - flatten it
                nested = data.pop(field_name)
                # Set the outer key to the model_type value
                data[field_name] = nested.pop("model_type")
                # Merge remaining nested key-values
                data.update(nested)
        
        # Convert booleans to integers
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
