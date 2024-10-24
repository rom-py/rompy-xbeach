"""XBeach wave boundary conditions."""
from abc import ABC, abstractmethod
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field


class WaveBoundaryBase():
    """Base class for wave boundary conditions."""
    model_type: Literal["base"] = Field(
        default="base",
        description="Model type discriminator"
    )
    # wbctype: Literal[
    #     "params",
    #     "jonstable",
    #     "swan",
    #     "vardens",
    #     "ts_1",
    #     "ts_2",
    #     "ts_nonh",
    #     "reuse",
    #     "off"
    # ] = Field(
    #     description="Wave boundary condition type"
    # )
    # wbctype_params: Optional[Union[float, str]] = Field(
    #     description="Wave boundary condition parameters"
    # )


# Spectral
class WaveBoundarySpectral(WaveBoundaryBase, ABC):
    """Base class for spectral wave boundary conditions.

    Note
    ----
    XBeach will reuse the generated time series until the simulation is completed. The
    resolution of the time series should be enough to accurately represent the bound
    long wave, but need not be as small as the time step used in XBeach.

    """
    model_type: Literal["spectral_base"] = Field(
        default="spectral_base",
        description="Model type discriminator"
    )
    bcfile: Optional[str] = Field(
        default="spectrum.txt",
        description="Name of spectrum file",
        examples=["spectrum.txt"],
    )
    rt: Optional[float] = Field(
        default=None,
        description=(
            "Duration (s) of wave spectrum at offshore boundary, in morphological "
            "time (XBeach default: min(3600.d0, par\%tstop))"
        ),
        ge=1200.0,
        le=7200.0,
        examples=[3600.0],
    )
    dbtc: Optional[float] = Field(
        default=None,
        description=(
            "Timestep (s) used to describe time series of wave energy and long wave "
            "flux at offshore boundary (not affected by morfac) (XBeach default: 1.0)"
        ),
        ge=0.1,
        le=2.0,
        examples=[1.0],
    )
    tm01switch: Optional[bool] = Field(
        default=None,
        description="Switch to enable tm01 rather than tm-10 (XBeach default: 0)",
    )
    correcthm0: Optional[bool] = Field(
        default=None,
        description="Switch to enable hm0 correction (XBeach default: 1)",
    )
    dthetas_xb: Optional[float] = Field(
        description="The (counter-clockwise) angle in the degrees needed to rotate from the x-axis in swan to the x-axis pointing east",
    )


class WaveBoundarySpectralParametric(WaveBoundarySpectral):
    """"""


class WaveBoundarySpectralSWAN(WaveBoundarySpectral):
    pass


class WaveBoundarySpectralGeneral(WaveBoundarySpectral):
    pass


# Non-spectral
class WaveBoundaryStationary(WaveBoundaryBase):
    pass


class WaveBoundaryStationaryUniform(WaveBoundaryStationary):
    pass


class WaveBoundaryStationaryTimeseries(WaveBoundaryStationary):
    pass


# Special cases
class WaveBoundaryBichrom(WaveBoundaryBase):
    pass


class WaveBoundaryOff(WaveBoundaryBase):
    pass


class WaveBoundaryReuse(WaveBoundaryBase):
    pass


