"""XBeach wave boundary conditions."""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Annotated
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from pydantic import Field, model_validator
from pydantic_numpy.typing import Np1DArray, Np2DArray

from rompy.core.types import RompyBaseModel


logger = logging.getLogger(__name__)


JONS_MAPPING = dict(
    hm0="Hm0",
    tp="Tp",
    mainang="mainang",
    gammajsp="gammajsp",
    s="s",
    fnyq="fnyq",
    dfj="dfj",
)


class WaveBoundaryBase(RompyBaseModel, ABC):
    """Base class for wave boundary conditions."""

    model_type: Literal["base"] = Field(
        default="base", description="Model type discriminator"
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

    @abstractmethod
    def write(self, destdir: str | Path) -> str:
        """Write the boundary data to the bcfile file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.

        Returns
        -------
        bcfile : Path
            Path to the bcfile.

        """
        pass


# Spectral: values jons, swan, vardens or jons_table
class WaveBoundarySpectral(WaveBoundaryBase, ABC):
    """Base class for spectral wave boundary conditions.

    Note
    ----
    XBeach will reuse the generated time series until the simulation is completed. The
    resolution of the time series should be enough to accurately represent the bound
    long wave, but need not be as small as the time step used in XBeach.

    """

    model_type: Literal["spectral_base"] = Field(
        default="spectral_base", description="Model type discriminator"
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
            "time (XBeach default: min(3600.d0, tstop))"
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
        description=("Number of input spectrum locations (XBeach default: 1)"),
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
        le=1.0,
    )
    wbcversion: Optional[Literal[1, 2, 3]] = Field(
        default=None,
        description="Version of wave boundary conditions (XBeach default: 3)",
    )


class WaveBoundaryJons(WaveBoundarySpectral):
    """Wave boundary conditions specified as a single Jonswap spectrum."""

    model_type: Literal["jons"] = Field(
        default="jons",
        description="Model type discriminator",
    )
    hm0: Optional[float] = Field(
        default=None,
        description=(
            "Hm0 of the wave spectrum, significant wave height [m] "
            "(XBeach default: 0.0)"
        ),
        ge=0.0,
        le=5.0,
    )
    tp: Optional[float] = Field(
        default=None,
        description="Peak period of the wave spectrum [s] (XBeach default: 12.5)",
        ge=0.0,
        le=25.0,
    )
    mainang: Optional[float] = Field(
        default=None,
        description=(
            "Main wave angle (nautical convention) [degrees] (XBeach default: 270.0)"
        ),
        ge=-180.0,
        le=360.0,
    )
    gammajsp: Optional[float] = Field(
        default=None,
        description=(
            "Peak enhancement factor in the JONSWAP expression (XBeach default: 3.3)"
        ),
        ge=1.0,
        le=5.0,
    )
    s: Optional[float] = Field(
        default=None,
        description=(
            "Directional spreading coefficient, {cos}^{2s} law [-] "
            "(XBeach default: 10.0)"
        ),
        ge=1.0,
        le=1000.0,
    )
    fnyq: Optional[float] = Field(
        default=None,
        description=(
            "Highest frequency used to create JONSWAP spectrum [Hz] "
            "(XBeach default: 0.3)"
        ),
        ge=0.2,
        le=1.0,
    )
    dfj: Optional[float] = Field(
        default=None,
        description=(
            "Step size frequency used to create JONSWAP spectrum [Hz] within the "
            "range fnyq/1000 - fnyq/20 (XBeach default: fnyq/200)"
        ),
    )

    @model_validator(mode="after")
    def validate_dfj(self) -> "WaveBoundaryJons":
        if self.dfj is not None:
            logger.warning(
                "It is advised not to specify the keyword dfj and allow XBeach "
                "to calculate the default value"
            )
            if not (self.dfj / 1000 <= self.dfj <= self.fnyq / 20):
                raise ValueError("dfj must be in the range fnyq/1000 to fnyq/20")
        return self

    def write(self, destdir: Path) -> str:
        """Write the boundary data to the bcfile file.

        Parameters
        ----------
        destdir : Path
            Destination directory for the netcdf file.

        Returns
        -------
        bcfile : Path
            Path to the bcfile.

        """
        bcfile = Path(destdir) / self.bcfile
        params = {"hm0", "tp", "mainang", "gammajsp", "s", "fnyq", "dfj"}
        with bcfile.open("w") as f:
            for param in params:
                if param not in self.model_fields_set or getattr(self, param) is None:
                    continue
                f.write(f"{JONS_MAPPING[param]} = {getattr(self, param):g}\n")
        return bcfile


class WaveBoundaryJonstable(WaveBoundarySpectral):
    """Wave boundary conditions specified as a time-varying Jonswap spectrum.

    .. code-block:: text

        <Hm0> <Tp> <mainang> <gammajsp> <s> <duration> <dtbc>

    Each line in the spectrum definition file contains a parametric definition of a
    spectrum, like in a regular JONSWAP definition file, plus the duration for which
    that spectrum is used during the simulation and the timestep.

    """

    model_type: Literal["jonstable"] = Field(
        default="jonstable",
        description="Model type discriminator",
    )
    hm0: list[Annotated[float, Field(ge=0.0, le=5.0)]] = Field(
        description="Hm0 of the wave spectrum, significant wave height [m]",
    )
    tp: list[Annotated[float, Field(ge=0.0, le=25.0)]] = Field(
        description="Peak period of the wave spectrum [s]",
    )
    mainang: list[Annotated[float, Field(ge=-180.0, le=360.0)]] = Field(
        description="Main wave angle (nautical convention) [degrees]",
    )
    gammajsp: list[Annotated[float, Field(ge=1.0, le=5.0)]] = Field(
        description="Peak enhancement factor in the JONSWAP expression",
    )
    s: list[Annotated[float, Field(ge=1.0, le=1000.0)]] = Field(
        description="Directional spreading coefficient, {cos}^{2s} law [-]",
    )
    duration: list[Annotated[float, Field(ge=0.0)]] = Field(
        description=(
            "Duration for which that spectrum is used during the simulation, XBeach "
            "does not reuse time-varying spectrum files, therefore the total duration "
            "of all spectra should at least match the duration of the simulation"
        ),
    )
    dtbc: list[Annotated[float, Field(ge=0.0)]] = Field(
        description="Boundary condition time step",
    )

    @model_validator(mode="after")
    def lists_are_the_same_sizes(self) -> "WaveBoundaryJonstable":
        for param in ["tp", "mainang", "gammajsp", "s", "duration", "dtbc"]:
            param_size = len(getattr(self, param))
            if param_size != len(self):
                raise ValueError(
                    f"All jonswap parameters must be the same size but size(hm0)="
                    f"{len(self)} size({param})={param_size}"
                )
        return self

    def __iter__(self):
        return zip(
            self.hm0,
            self.tp,
            self.mainang,
            self.gammajsp,
            self.s,
            self.duration,
            self.dtbc,
        )

    def __len__(self):
        return len(self.hm0)

    def write(self, destdir: Path) -> str:
        """Write the boundary data to the bcfile file.

        Parameters
        ----------
        destdir : Path
            Destination directory for the netcdf file.

        Returns
        -------
        bcfile : Path
            Path to the bcfile.

        """
        bcfile = Path(destdir) / self.bcfile
        with bcfile.open("w") as f:
            for params in self:
                f.write(f"{' '.join(f'{x:g}' for x in params)}\n")
        return bcfile


class WaveBoundarySWAN(WaveBoundarySpectral):
    """Wave boundary conditions specified as a SWAN spectrum."""

    model_type: Literal["swan"] = Field(
        default="swan",
        description="Model type discriminator",
    )
    freq: Np1DArray = Field(
        description="Wave frequency [Hz]",
    )
    dir: Np1DArray = Field(
        description="Wave direction [degrees]",
    )
    efth: Np2DArray = Field(
        description="Energy density [m^2/Hz]",
    )
    lon: Optional[float] = Field(
        default=0.0,
        description="Longitude of the spectral data",
    )
    lat: Optional[float] = Field(
        default=0.0,
        description="Latitude of the spectral data",
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

    @property
    def ds(self) -> xr.DataArray:
        """Return the SWAN spectrum as an xarray DataArray."""
        dset = xr.DataArray(
            np.expand_dims(self.efth, 0),
            coords={"site": [0.0], "freq": self.freq, "dir": self.dir},
            dims=["site", "freq", "dir"],
        ).to_dataset(name="efth")
        dset["lon"] = [self.lon]
        dset["lat"] = [self.lat]
        return dset

    def write(self, destdir: Path) -> str:
        """Write the boundary data to the bcfile file.

        Parameters
        ----------
        destdir : Path
            Destination directory for the netcdf file.

        Returns
        -------
        bcfile : Path
            Path to the bcfile.

        """
        bcfile = Path(destdir) / self.bcfile
        self.ds.spec.to_swan(bcfile)
        return bcfile


class WaveBoundaryGeneral(WaveBoundarySpectral):
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
