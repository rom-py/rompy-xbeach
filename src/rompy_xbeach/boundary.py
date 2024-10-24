"""XBeach wave boundary conditions."""
from abc import ABC, abstractmethod
from typing import Literal, Union, Optional
from pydantic import BaseModel, Field


# TODO: Add support for time/space varying boundary with FILELIST and LOCLIST


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
        default=None,
        description=(
            "The (counter-clockwise) angle in the degrees needed to rotate from the "
            "x-axis in swan to the x-axis pointing east (XBeach default: 0.0)",
        ),
        ge=-360.0,
        le=360.0,
    )
    fcutoff: Optional[float] = Field(
        default=None,
        description=(
            "Low-freq cutoff frequency in Hz for jons, swan or vardens boundary "
            "conditions (XBeach default: 0.0)"
        ),
        ge=0.0,
        le=40.0,
    )
    nonhspectrum: Optional[Literal[0, 1]] = Field(
        default=None,
        description=(
            "Spectrum format for wave action balance of nonhydrostatic waves "
            "(XBeach default: 0)"
        ),
    )
    nspectrumloc: Optional[int] = Field(
        default=None,
        description=(
            "Number of input spectrum locations (XBeach default: 1)"
        ),
        ge=1,
    )
    nspr: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable long wave direction forced into centres of short wave "
            "bins (XBeach default: 0)",
        ),
    )
    random: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable random seed for jons, swan or vardens boundary "
            "conditions (XBeach default: 1)",
        ),
    )
    sprdthr: Optional[float] = Field(
        default=None,
        description=(
            "Threshold ratio to maximum value of s above which spectrum densities "
            "are read in (XBeach default: 0.08)"
        ),
        ge=0.0,
        le=1.0,
    )
    trepfac: Optional[float] = Field(
        default=None,
        description=(
            "Compute mean wave period over energy band: par%trepfac*maxval(sf) for "
            "jons, swan or vardens; converges to tm01 for trepfac = 0.0 "
            "(XBeach default: 0.01)",
        ),
        ge=0.0,
        le=1.0
    )
    wbcversion: Optional[Literal[1, 2, 3]] = Field(
        default=None,
        description="Version of wave boundary conditions (XBeach default: 3)",

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


