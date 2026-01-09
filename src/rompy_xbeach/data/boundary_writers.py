"""XBeach wave boundary file writers.

These classes are responsible for writing the various boundary condition files
required by XBeach (JONSWAP, JONSTABLE, SWAN, etc.). They do NOT contain XBeach
parameter configuration - those are in components.boundary.parameters.

These file writers are used by the boundary data classes in this package (data.boundary)
to generate the actual boundary files from processed data.

Note: These are utility classes for file I/O, not XBeach parameter components.
"""

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


class BoundaryWriterBase(RompyBaseModel, ABC):
    """Base class for wave boundary file writers.

    This class defines the interface for writing XBeach boundary condition files.
    Subclasses implement specific file formats (JONSWAP, SWAN, etc.).

    Note: XBeach parameter configuration (rt, dtbc, random, etc.) is handled by
    WaveBoundaryConditions in components.boundary.parameters, not here.
    """

    model_type: Literal["base"] = Field(
        default="base", description="Model type discriminator"
    )

    # Class attribute: subclasses can override to specify which fields to serialize
    # These are typically data-derived fields needed in params.txt (e.g., dtbc for jonstable)
    _serializable_fields: tuple[str, ...] = ()

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

    def get(self, destdir: Path) -> dict:
        """Write the boundary file and return required model parameters.

        This method writes the boundary file and returns a dictionary containing
        the bcfile path plus any additional fields specified in _serializable_fields.

        Parameters
        ----------
        destdir : Path
            Destination directory for the boundary file.

        Returns
        -------
        params : dict
            Dictionary of required model parameters, always includes 'bcfile'.

        """
        result = {"bcfile": self.write(destdir)}

        # Add any fields specified in _serializable_fields
        for field_name in self._serializable_fields:
            value = getattr(self, field_name, None)
            if value is not None:
                result[field_name] = value

        return result


# Spectral boundary file writers
class SpectralWriter(BoundaryWriterBase, ABC):
    """Base class for spectral wave boundary file writers.

    Handles common file naming for spectral boundary types (JONSWAP, SWAN, etc.).
    """

    model_type: Literal["spectral_base"] = Field(
        default="spectral_base", description="Model type discriminator"
    )
    bcfile: Optional[str] = Field(
        default="spectrum.txt",
        description="Name of spectrum file",
        examples=["spectrum.txt"],
    )


class JonsWriter(SpectralWriter):
    """File writer for single JONSWAP spectrum boundary conditions.

    Writes a JONSWAP parameter file with format:
        Hm0 = <value>
        Tp = <value>
        mainang = <value>
        gammajsp = <value>
        s = <value>
        fnyq = <value>
        dfj = <value>
    """

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
    _serializable_fields: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_dfj(self) -> "JonsWriter":
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


class JonstableWriter(SpectralWriter):
    """File writer for time-varying JONSWAP spectrum boundary conditions.

    Writes a JONSTABLE file with format:
        <Hm0> <Tp> <mainang> <gammajsp> <s> <duration> <dtbc>

    Each line contains a parametric definition of a spectrum, plus the duration
    for which that spectrum is used and the timestep.

    Note: dtbc values are written to the file AND returned in get() for params.txt.
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
    _serializable_fields: tuple[str, ...] = ("dtbc",)

    @model_validator(mode="after")
    def lists_are_the_same_sizes(self) -> "JonstableWriter":
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


class SwanWriter(SpectralWriter):
    """File writer for SWAN spectrum boundary conditions.

    Writes a SWAN spectral file using wavespectra library.

    Note: lat and dthetas_xb are returned in get() for params.txt.
    """

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
        description="Latitude at model location for computing coriolis",
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
    _serializable_fields: tuple[str, ...] = ("lat", "dthetas_xb")

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


class VardensWriter(SpectralWriter):
    """File writer for VARDENS spectrum boundary conditions."""

    pass


# Non-spectral boundary file writers
class StationaryWriter(BoundaryWriterBase):
    """File writer for stationary boundary conditions."""

    pass


class TimeSeriesWriter(BoundaryWriterBase):
    """File writer for time series boundary conditions."""

    pass


# Special case boundary file writers
class BichromWriter(BoundaryWriterBase):
    """File writer for bichromatic boundary conditions."""

    pass
