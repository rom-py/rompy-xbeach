"""XBeach physical processes configuration."""

import logging
from typing import Literal, Optional, Union

from pydantic import Field, model_validator

from rompy_xbeach.components.physics.constants import Coriolis, PhysicalConstants
from rompy_xbeach.components.physics.wci import (
    WaveCurrentInteraction,
)
from rompy_xbeach.components.physics.friction import (
    Viscosity,
    Cf,
    Chezy,
    Manning,
    WhiteColebrook,
    WhiteColebrookGrainsize,
)
from rompy_xbeach.components.physics.numerics import (
    FlowNumerics,
    WaveNumerics,
)
from rompy_xbeach.components.physics.vegetation import Vegetation
from rompy_xbeach.components.physics.wavemodel import (
    Nonh,
    Roller,
    Stationary,
    Surfbeat,
)
from rompy_xbeach.types import XBeachBaseModel


logger = logging.getLogger(__name__)


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
    bedfriction: Optional[
        Union[Cf, Chezy, Manning, WhiteColebrook, WhiteColebrookGrainsize]
    ] = Field(
        default=None,
        description=("Bed friction formulation (XBeach default: manning)"),
        discriminator="model_type",
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
    roller: Optional[Union[bool, Roller]] = Field(
        default=None,
        description="Switch to enable roller model (XBeach default: 1)",
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
    vegetation: Optional[Union[bool, Vegetation]] = Field(
        default=None,
        description=(
            "Turn on interaction of waves and flow with vegetation (XBeach default: 0)"
        ),
    )
    viscosity: Optional[Union[bool, Viscosity]] = Field(
        default=None,
        description=(
            "Include viscosity in flow solver. Can be True/False to enable/disable, "
            "or a Viscosity object to enable with custom parameters (XBeach default: 1)"
        ),
    )
    wci: Optional[Union[bool, WaveCurrentInteraction]] = Field(
        default=None,
        description="Switch to turn on wave-current interaction (XBeach default: 0)",
    )
    wind: Optional[bool] = Field(
        default=None,
        description="Include wind in flow solver (XBeach default: 1)",
    )
    flow_numerics: Optional[FlowNumerics] = Field(
        default=None,
        description=(
            "Flow numerical parameters (eps, hmin, deltahmin, umin, secorder, etc.)"
        ),
    )
    wave_numerics: Optional[WaveNumerics] = Field(
        default=None,
        description="Wave numerical parameters (scheme, maxiter, maxerror, wavint)",
    )
    # Note: Non-hydrostatic parameters are configured via wavemodel=Nonh(...)
    constants: Optional[PhysicalConstants] = Field(
        default=None,
        description="Physical constants (g, rho, depthscale)",
    )
    coriolis: Optional[Coriolis] = Field(
        default=None,
        description="Coriolis force parameters (lat, wearth)",
    )

    @model_validator(mode="after")
    def swave_must_be_false_if_nonh(self) -> "Physics":
        """Warn if swave is not False when wavemodel is Nonh.

        Note: The legacy ``nonh`` parameter is deprecated in XBeach. Use
        ``wavemodel=Nonh(...)`` instead, which outputs ``wavemodel = nonh``.
        """
        if isinstance(self.wavemodel, Nonh):
            if self.swave is True:
                logger.warning(
                    "Parameter 'swave' should not be True when using Nonh wavemodel. "
                    "XBeach requires swave=0 for non-hydrostatic mode."
                )
            elif self.swave is None:
                logger.warning(
                    "Parameter 'swave' should be explicitly set to False when using Nonh "
                    "wavemodel. XBeach enables swave by default (swave=1), which conflicts "
                    "with non-hydrostatic mode. Consider setting swave=False explicitly."
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
