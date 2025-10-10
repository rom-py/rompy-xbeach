"""XBeach physical processes configuration."""

import logging
from typing import Literal, Optional, Union
from pydantic import Field, model_validator

from rompy_xbeach.types import XBeachBaseModel


logger = logging.getLogger(__name__)


class Janssen(XBeachBaseModel):
    """Janssen & Battjes (2007) breaker model configuration."""

    model_type: Literal["janssen"] = Field(
        default="janssen",
        description="Model type discriminator",
    )


class Baldock(XBeachBaseModel):
    """Baldock breaker model configuration."""

    model_type: Literal["baldock"] = Field(
        default="baldock",
        description="Model type discriminator",
    )
    gamma: Optional[float] = Field(
        default=None,
        description="Breaker parameter gamma (XBeach default: 0.46)",
        ge=0.4,
        le=0.9,
    )


class Roelvink1(XBeachBaseModel):
    """Roelvink (1993a) breaker model configuration."""

    model_type: Literal["roelvink1"] = Field(
        default="roelvink1",
        description="Model type discriminator",
    )
    alpha: Optional[float] = Field(
        default=None,
        description="Wave dissipation coefficient (XBeach default: 1.38)",
        ge=0.5,
        le=2.0,
    )
    delta: Optional[float] = Field(
        default=None,
        description="Fraction of wave height to add to water depth (XBeach default: 0.0)",
        ge=0.0,
        le=1.0,
    )
    gamma: Optional[float] = Field(
        default=None,
        description="Breaker parameter gamma (XBeach default: 0.46)",
        ge=0.4,
        le=0.9,
    )
    n: Optional[float] = Field(
        default=None,
        description="Power in roelvink dissipation model (Xbeach default: 10.0)",
        ge=5.0,
        le=20.0,
    )


class Roelvink2(Roelvink1):
    """Roelvink (1993a) extended breaker model configuration."""

    model_type: Literal["roelvink2"] = Field(
        default="roelvink2",
        description="Model type discriminator",
    )


class RoelvinkDaly(XBeachBaseModel):
    """Daly et al. (2010) breaker model configuration."""

    model_type: Literal["roelvink_daly"] = Field(
        default="roelvink_daly",
        description="Model type discriminator",
    )
    gamma2: Optional[float] = Field(
        default=None,
        description="End of breaking parameter (XBeach default: 0.34)",
        ge=0.0,
        le=0.5,
    )


class Stationary(XBeachBaseModel):
    """Stationary wave model configuration.

    Efficiently solves wave-averaged equations but neglects infragravity waves.
    Useful for conditions where incident waves are relatively small and/or short.

    """

    model_type: Literal["stationary"] = Field(
        default="stationary",
        description="Model type discriminator",
    )
    breaktype: Optional[Union[Baldock, Janssen]] = Field(
        default=None,
        description="Type of breaker formulation for the stationary wave model",
        discriminator="model_type",
        alias="break",
    )


class Surfbeat(XBeachBaseModel):
    """Surfbeat (instationary) wave model configuration.

    Resolves short wave variations on the wave group scale (short wave envelope)
    and the long waves associated with them. This is the XBeach default mode.

    """

    model_type: Literal["surfbeat"] = Field(
        default="surfbeat",
        description="Model type discriminator",
    )
    breaktype: Optional[Union[Roelvink1, Roelvink2, RoelvinkDaly]] = Field(
        default=None,
        description="Type of breaker formulation for the surfbeat wave model",
        discriminator="model_type",
        alias="break",
    )


class Nonh(XBeachBaseModel):
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


class Physics(XBeachBaseModel):
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
