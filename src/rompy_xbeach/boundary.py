"""XBeach wave boundary conditions."""

from abc import ABC, abstractmethod
from typing import Literal, Union, Optional
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from pydantic import Field, model_validator, field_validator
from rompy.core.source import SourceTimeseriesCSV, SourceTimeseriesDataFrame
from rompy.core.types import DatasetCoords
from rompy.core.time import TimeRange
from rompy_xbeach.data import BaseDataStation, BaseDataPoint, BaseDataGrid

from rompy_xbeach.source import (
    SourceCRSFile,
    SourceCRSIntake,
    SourceCRSDataset,
    SourceCRSWavespectra,
)
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.components.boundary import (
    WaveBoundaryJons,
    WaveBoundaryJonstable,
    WaveBoundarySWAN,
)


logger = logging.getLogger(__name__)


SOURCE_TIMESERIES_TYPES = Union[
    SourceTimeseriesCSV,
    SourceTimeseriesDataFrame,
]

SOURCE_PARAM_TYPES = Union[
    SourceCRSFile,
    SourceCRSIntake,
    SourceCRSDataset,
]

SOURCE_SPECTRA_TYPES = Union[
    SourceCRSWavespectra,
    SourceCRSFile,
    SourceCRSIntake,
    SourceCRSDataset,
]


def dspr_to_s(dspr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the Jonswap spreading coefficient from the directional spread.

    Parameters
    ----------
    dspr: float | np.ndarray
        The directional spread in degrees.

    Returns
    -------
    s : float | np.ndarray
        The Jonswap spreading coefficient.

    """
    return (2 / np.radians(dspr) ** 2) - 1


def s_to_dspr(s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the directional spread from the Jonswap spreading coefficient.

    Parameters
    ----------
    s: float | np.ndarray
        The Jonswap spreading coefficient.

    Returns
    -------
    dspr : float | np.ndarray
        The directional spread in degrees.

    """
    return np.degrees(np.sqrt(2 / (s + 1)))


# =====================================================================================
# Base and Mixin classes
# =====================================================================================
class BoundaryBase:
    """Base class for wave boundary interfaces."""

    dbtc: Optional[float] = Field(
        default=1.0,
        description=(
            "Timestep (s) used to describe time series of wave energy and long wave "
            "flux at offshore boundary"
        ),
        ge=0.1,
        le=2.0,
        examples=[1.0],
    )
    location: Literal["offshore"] = Field(
        default="offshore",
        description="Location to extract the data from the source dataset",
    )


class BoundaryBaseGrid(BoundaryBase, BaseDataGrid):
    """Base class to construct XBeach wave boundary from gridded data."""


class BoundaryBaseStation(BoundaryBase, BaseDataStation):
    """Base class to construct XBeach wave boundary from stations type data."""


class BoundaryBasePoint(BoundaryBase, BaseDataPoint):
    """Base class to construct XBeach wave boundary from point timeseries type data."""


class SpectraMixin:
    """Mixin class to calculate wave statistics from spectral data."""

    source: SOURCE_SPECTRA_TYPES = Field(
        description=(
            "Dataset source reader, must support CRS and have wavespectra accessor "
        ),
        discriminator="model_type",
    )
    coords: DatasetCoords = Field(
        default=DatasetCoords(x="lon", y="lat", t="time", s="site"),
        description="Names of the coordinates in the dataset",
    )

    @field_validator("source")
    def _validate_source_wavespectra(cls, source, values):
        if not hasattr(source.open(), "spec"):
            raise ValueError("source must have wavespectra accessor")
        return source

    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the wave statistics from the spectral data.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary spectral data.

        """
        stats = ds.chunk(freq=-1, dir=-1).spec.stats(["hs", "tp", "dpm", "gamma", "dspr"])
        stats["s"] = dspr_to_s(stats.dspr)
        return stats.rename(hs="hm0", dpm="mainang", gamma="gammajsp")


class ParamMixin:
    """Mixin class to get Jonswap statistics from parameter data."""

    source: SOURCE_PARAM_TYPES = Field(
        description="Dataset source reader, must support CRS",
        discriminator="model_type",
    )
    hm0: Union[str, float] = Field(
        # default="hs",
        description=(
            "Variable name of the significant wave height Hm0 in the source data, "
            "or alternatively a constant value to use for all times"
        ),
    )
    tp: Union[str, float] = Field(
        # default="tp",
        description=(
            "Variable name of the peak period Tp in the source data, "
            "or alternatively a constant value to use for all times"
        ),
    )
    mainang: Union[str, float] = Field(
        # default="dpm",
        description=(
            "Variable name of the main wave direction in the source data, "
            "or alternatively a constant  value to use for all times"
        ),
    )
    gammajsp: Optional[Union[str, float]] = Field(
        default=None,
        description=(
            "Variable name of the gamma parameter in the source data, "
            "or alternatively a constant value to use for all times"
        ),
    )
    dspr: Optional[Union[str, float]] = Field(
        default=None,
        description=(
            "Variable name of the directional spreading in the source data, used to "
            "calculate the Jonswap spreading coefficient, "
            "or alternatively a constant value to use for all times"
        ),
    )

    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the wave statistics from the spectral data.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary spectral data.

        """
        stats = xr.Dataset()
        for param in ["hm0", "tp", "mainang", "gammajsp"]:
            if isinstance(getattr(self, param), str):
                stats[param] = ds[getattr(self, param)]
            elif isinstance(getattr(self, param), float):
                stats[param] = xr.DataArray(getattr(self, param), coords=stats.coords)
        if self.dspr is not None:
            if isinstance(self.dspr, str):
                stats["s"] = dspr_to_s(ds[self.dspr])
            elif isinstance(self.dspr, float):
                stats["s"] = dspr_to_s([self.dspr] * ds.time.size)
        return stats


class FilelistMixin:
    """Mixin class to write a filelist for multiple boundary files."""

    filelist: Optional[bool] = Field(
        default=True,
        description=(
            "If True, create one bcfile for each timestep in the filtered dataset and "
            "return a FILELIST.txt file with the list of bcfiles, otherwise return a "
            "single bcfile with the wave parameters interpolated at time.start"
        ),
    )

    def _write_filelist(
        self, destdir: Path, bcfiles: list[str], durations: list[float]
    ) -> Path:
        """Write a filelist with the bcfiles.

        Parameters
        ----------
        destdir : Path
            Destination directory for the filelist.
        bcfiles : list[Path]
            List of bcfiles to include in the filelist.
        durations : list[float]
            List of durations for each bcfile.

        Returns
        -------
        filename : Path
            Path to the filelist file.

        """
        filename = Path(destdir) / f"{self.id}-filelist.txt"
        with open(filename, "w") as f:
            f.write("FILELIST\n")
            for bcfile, duration in zip(bcfiles, durations):
                f.write(f"{duration:g} {self.dbtc:g} {bcfile.name}\n")
        return filename


class BoundaryJons(FilelistMixin, ABC):
    """Base class for JONS wave boundary from station type dataset such as SMC."""

    id: Literal["jons", "parametric"] = Field(
        default="jons",
        description="Boundary type identifier, used to define the wbctype",
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
    dbtc: Optional[float] = Field(
        default=1.0,
        description=(
            "Timestep (s) used to describe time series of wave energy and long wave "
            "flux at offshore boundary"
        ),
        ge=0.1,
        le=2.0,
        examples=[1.0],
    )

    @abstractmethod
    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the Jonswap parameters from the data.

        This method should be implemented in the subclass as it will be different for
        spectra and params data types.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary spectral data.

        """
        pass

    def _instantiate_boundary(self, data: xr.Dataset) -> "BoundaryJons":
        """Instantiate the boundary object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing single time for the boundary spectral data.

        """
        assert data.time.size == 1
        t = data.time.to_index().to_pydatetime()[0]
        logger.debug(f"Creating boundary for time {t}")
        kwargs = {}
        for param in ["hm0", "tp", "mainang", "gammajsp", "s"]:
            if hasattr(self, param) and getattr(self, param) is None:
                continue
            elif param in data and not np.isnan(data[param]):
                kwargs[param] = float(data[param].squeeze())
            elif param in data and np.isnan(data[param]):
                raise ValueError(f"Parameter {param} is NaN for {data.time}")
        bcfile = f"{self.id}-{t:%Y%m%dT%H%M%S}.txt"
        return WaveBoundaryJons(bcfile=bcfile, fnyq=self.fnyq, dfj=self.dfj, **kwargs)

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Write the selected boundary data to file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the boundary bcfile data.

        """
        ds = super().get(destdir, grid, time)
        if not self.filelist:
            # Write a single bcfile at the timerange start
            ds = ds.interp({self.coords.t: [time.start]})
            data = self._calculate_stats(ds)
            wb = self._instantiate_boundary(data)
            bcfile = wb.write(destdir)
        else:
            # Write a bcfile for each timestep in the timerange
            ds = self._adjust_time(ds, time)
            stats = self._calculate_stats(ds)
            times = stats.time.to_index().to_pydatetime()
            bcfiles = []
            durations = []
            for t0, t1 in zip(times[:-1], times[1:]):
                # Boundary data
                data = stats.sel(time=[t0])
                wb = self._instantiate_boundary(data)
                bcfiles.append(wb.write(destdir))
                # Boundary duration
                durations.append((t1 - t0).total_seconds())
            bcfile = self._write_filelist(destdir, bcfiles, durations)
        return {"wbctype": self.id, "bcfile": bcfile.name}


class BoundaryJonstable(ABC):
    """Base class for JONSTABLE wave boundary from station type dataset such as SMC."""

    id: Literal["jonstable"] = Field(
        default="jonstable", description="Boundary type identifier"
    )

    @model_validator(mode="after")
    def default_params(self) -> "BoundaryStationParamJonstable":
        if hasattr(self, "gammajsp") and self.gammajsp is None:
            logger.debug("Setting default value for gammajsp of 3.3")
            self.gammajsp = 3.3
        if hasattr(self, "dspr") and self.dspr is None:
            logger.debug("Setting default value for dspr of 24.431 (s=10.0)")
            self.dspr = 24.43100247268452
        return self

    def _instantiate_boundary(self, data: xr.Dataset) -> "BoundaryJonstable":
        """Instantiate the boundary object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing single time for the boundary spectral data.

        """
        times = data.time.to_index().to_pydatetime()
        logger.debug(f"Creating jonstable boundary for times {times}")
        dts = [dt.total_seconds() for dt in np.diff(times)]
        bcfile = f"{self.id}-{times[0]:%Y%m%dT%H%M%S}-{times[-1]:%Y%m%dT%H%M%S}.txt"
        kwargs = dict(
            hm0=data.hm0.squeeze().values,
            tp=data.tp.squeeze().values,
            mainang=data.mainang.squeeze().values,
            gammajsp=data.gammajsp.squeeze().values,
            s=data.s.squeeze().values,
            duration=dts + [dts[-1]],
            dtbc=[self.dbtc] * len(times),
        )
        for key, val in kwargs.items():
            if any(np.isnan(val)):
                raise ValueError(
                    f"Parameter {key} has NaN for one or more times ({list(zip(times, val))})"
                )
        return WaveBoundaryJonstable(bcfile=bcfile, **kwargs)

    @abstractmethod
    def _calculate_stats(self, ds: xr.Dataset) -> xr.Dataset:
        """Calculate the Jonswap parameters from the data.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary spectral data.

        """
        pass

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Write the selected boundary data to file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the boundary bcfile data.

        """
        ds = super().get(destdir, grid, time)
        ds = self._adjust_time(ds, time)
        data = self._calculate_stats(ds)
        wb = self._instantiate_boundary(data)
        bcfile = wb.write(destdir)
        return {"wbctype": self.id, "bcfile": bcfile.name}


# =====================================================================================
# JONS bctype
# =====================================================================================
class BoundaryStationSpectraJons(SpectraMixin, BoundaryJons, BoundaryBaseStation):
    """Wave boundary conditions from station type spectra dataset such as SMC."""

    model_type: Literal["station_spectra_jons"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )


class BoundaryStationParamJons(ParamMixin, BoundaryJons, BoundaryBaseStation):
    """Wave boundary conditions from station type parameters dataset such as SMC."""

    model_type: Literal["station_param_jons"] = Field(
        default="station_param_jons",
        description="Model type discriminator",
    )


class BoundaryPointParamJons(ParamMixin, BoundaryJons, BoundaryBasePoint):
    """Wave boundary conditions from point timeseries type parameters dataset."""

    model_type: Literal["point_param_jons"] = Field(
        default="point_param_jons",
        description="Model type discriminator",
    )
    source: SOURCE_TIMESERIES_TYPES = Field(
        description="Dataset source reader for point timeseries type data",
        discriminator="model_type",
    )


class BoundaryGridParamJons(ParamMixin, BoundaryJons, BoundaryBaseGrid):
    """Wave boundary conditions from grid type parameters dataset."""

    model_type: Literal["grid_param_jons"] = Field(
        default="grid_param_jons",
        description="Model type discriminator",
    )


# =====================================================================================
# JONSTABLE bctype
# =====================================================================================
class BoundaryStationSpectraJonstable(
    SpectraMixin, BoundaryJonstable, BoundaryBaseStation
):
    """Wave boundary conditions from station type parameters dataset such as SMC."""

    model_type: Literal["station_spectra_jonstable"] = Field(
        default="station_spectra_jonstable",
        description="Model type discriminator",
    )


class BoundaryStationParamJonstable(ParamMixin, BoundaryJonstable, BoundaryBaseStation):
    """Wave boundary conditions from station type parameters dataset such as SMC."""

    model_type: Literal["station_param_jonstable"] = Field(
        default="station_param_jonstable",
        description="Model type discriminator",
    )


class BoundaryPointParamJonstable(ParamMixin, BoundaryJonstable, BoundaryBasePoint):
    """Wave boundary conditions from point timeseries type parameters dataset."""

    model_type: Literal["point_param_jonstable"] = Field(
        default="point_param_jonstable",
        description="Model type discriminator",
    )
    source: SOURCE_TIMESERIES_TYPES = Field(
        description="Dataset source reader for point timeseries type data",
        discriminator="model_type",
    )


class BoundaryGridParamJonstable(ParamMixin, BoundaryJonstable, BoundaryBaseGrid):
    """Generate XBeach JONSTABLE wave boundary conditions from gridded parameter data.

    This class reads wave parameters (Hm0, Tp, Dir, Spread, Gamma) from a gridded data
    source, selects/interpolates the data at the offshore boundary location of the
    XBeach grid, and writes the time-varying parameters to a JONSTABLE format file (`bcfile`).

    Inherits spatial selection from BoundaryBaseGrid/BaseDataGrid, parameter handling
    from ParamMixin, and JONSTABLE file writing logic from BoundaryJonstable.

    """

    model_type: Literal["grid_param_jonstable"] = Field(
        default="grid_param_jonstable",
        description="Model type discriminator",
    )


# =====================================================================================
# SWAN bctype
# =====================================================================================
class BoundaryStationSpectraSwan(FilelistMixin, SpectraMixin, BoundaryBaseStation):
    """Base class for SWAN wave boundary from station type dataset such as SMC."""

    id: Literal["swan"] = Field(default="swan", description="Boundary type identifier")
    model_type: Literal["station_spectra_swan"] = Field(
        default="station_spectra_swan",
        description="Model type discriminator",
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

    def _instantiate_boundary(self, data: xr.Dataset) -> "BoundaryJons":
        """Instantiate the boundary object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing single time for the boundary spectral data.

        """
        assert data.time.size == 1
        t = data.time.to_index().to_pydatetime()[0]
        logger.debug(f"Creating boundary for time {t}")
        bcfile = f"{self.id}-{t:%Y%m%dT%H%M%S}.txt"
        return WaveBoundarySWAN(
            bcfile=bcfile,
            freq=data.freq.squeeze().values,
            dir=data.dir.squeeze().values,
            efth=data.efth.squeeze().values,
            lon=float(data.lon.values),
            lat=float(data.lat.values),
        )

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Write the selected boundary data to file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the boundary bcfile data.

        """
        ds = super().get(destdir, grid, time)
        if not self.filelist:
            # Write a single bcfile at the timerange start
            ds = ds.interp({self.coords.t: [time.start]})
            wb = self._instantiate_boundary(ds)
            bcfile = wb.write(destdir)
        else:
            # Write a bcfile for each timestep in the timerange
            ds = self._adjust_time(ds, time)
            times = ds.time.to_index().to_pydatetime()
            bcfiles = []
            durations = []
            for t0, t1 in zip(times[:-1], times[1:]):
                # Boundary data
                data = ds.sel(time=[t0])
                wb = self._instantiate_boundary(data)
                bcfiles.append(wb.write(destdir))
                # Boundary duration
                durations.append((t1 - t0).total_seconds())
            bcfile = self._write_filelist(destdir, bcfiles, durations)
        namelist = {"wbctype": self.id, "bcfile": bcfile.name}
        if self.dthetas_xb is not None:
            namelist["dthetas_xb"] = self.dthetas_xb
        return namelist
