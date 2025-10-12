"""XBeach wavemodel parameter configurations.

This module contains all models used by the Physics.wavemodel field, including:

- Breaker formulation models (used by wave models)
- Wave model configurations (Stationary, Surfbeat, Nonh)

"""

from typing import Literal, Optional, Union
from pydantic import Field

from rompy_xbeach.types import XBeachBaseModel


# =============================================================================
# Roller model
# =============================================================================


class Roller(XBeachBaseModel):
    """Roller model configuration.
    
    When used in Physics.roller field, this enables the roller model (roller=1)
    and allows specification of roller-specific parameters.
    """

    model_type: Literal[True] = Field(
        default=True,
        description="Model type discriminator - set to True to enable roller",
    )
    beta: Optional[float] = Field(
        default=None,
        description="Breaker slope coefficient in roller model (XBeach default: 0.08)",
        ge=0.05,
        le=0.3,
    )
    nuhfac: Optional[float] = Field(
        default=None,
        description=(
            "Viscosity switch for roller induced turbulent horizontal viscosity "
            "(XBeach default: 1.0)"
        ),
        ge=0.0,
        le=1.0,
    )
    rfb: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to feed back maximum wave surface slope in roller energy balance "
            "(XBeach default: 0)"
        ),
    )


# =============================================================================
# Breaker Formulation Models
# =============================================================================


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


# =============================================================================
# Wave Model Configurations
# =============================================================================


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
