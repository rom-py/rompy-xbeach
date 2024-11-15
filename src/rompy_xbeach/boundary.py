"""XBeach wave boundary conditions."""

from abc import ABC, abstractmethod
from typing import Literal, Union, Optional, Annotated
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from pydantic import BaseModel, Field, model_validator, field_validator

from wavespectra.core import select

from rompy.core.types import DatasetCoords, RompyBaseModel
from rompy.core.time import TimeRange
from rompy.core.boundary import BoundaryWaveStation
from rompy.core.data import DataGrid

from rompy_xbeach.source import SourceCRSFile, SourceCRSIntake, SourceCRSDataset, SourceCRSWavespectra
from rompy_xbeach.grid import RegularGrid, Ori
from rompy_xbeach.components.boundary import WaveBoundaryJons


logger = logging.getLogger(__name__)


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
        return (2 / np.radians(dspr)**2) - 1


class BCFile(RompyBaseModel):
    """Base class for writing XBeach boundary condition files."""

    bcfile: Optional[Path] = Field(
        default=None,
        description="Path to the boundary condition file",
    )
    filelist: Optional[Path] = Field(
        default=None,
        description="Path to the filelist file",
    )

    @model_validator(mode="after")
    def bcfile_or_filelist(self) -> "BCFile":
        if not any([self.bcfile, self.filelist]):
            raise ValueError("Either bcfile or filelist must be set")
        return self

    @property
    def namelist(self):
        """Return the namelist representation of the bcfile."""
        if self.filelist is not None:
            return dict(filelist=self.filelist.name)
        else:
            return dict(bcfile=self.bcfile.name)

    def write(self, destdir: str | Path) -> Path:
        """Write the boundary condition file to the destination directory.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the bcfile.

        Returns
        -------
        outfile : Path
            Path to the bcfile.

        """
        raise NotImplementedError


class BoundaryStation(DataGrid, ABC):
    """Base class to construct XBeach wave boundary from stations type data.

    This object provides similar functionality to the `BoundaryWaveStation` object in
    that it uses wavespectra to select points from a stations (non-gridded) type source
    data, but it also supports non-spectral data.

    Notes
    -----
    The `time_buffer` field is redefined from the base class to define new default
    values that ensure the time range is always buffered by one timestep.

    """

    model_type: Literal["station"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    source: SOURCE_SPECTRA_TYPES = Field(
        description="Dataset source reader, must support CRS",
        discriminator="model_type",
    )
    sel_method: Literal["idw", "nearest"] = Field(
        default="idw",
        description=(
            "Defines which function from wavespectra.core.select to use for data "
            "selection: 'idw' uses sel_idw() for inverse distance weighting, "
            "'nearest' uses sel_nearest() for nearest neighbor selection"
        ),
    )
    sel_method_kwargs: dict = Field(
        default={},
        description="Keyword arguments for sel_method"
    )
    time_buffer: list[int] = Field(
        default=[1, 1],
        description=(
            "Number of source data timesteps to buffer the time range "
            "if `filter_time` is True"
        ),
    )

    # Validate coords are in dataset

    def _boundary_points(self, grid: RegularGrid) -> Ori:
        """Return the x, y point of the offshore boundary in the source crs."""
        xoff, yoff = grid.offshore
        bnd = Ori(x=xoff, y=yoff, crs=grid.crs).reproject(self.source.crs)
        return bnd.x, bnd.y

    def _sel_boundary(self, grid) -> xr.Dataset:
        """Select the offshore boundary point from the source dataset."""
        xbnd, ybnd = self._boundary_points(grid=grid)
        ds = getattr(select, f"sel_{self.sel_method}")(
            self.ds,
            lons=xbnd,
            lats=ybnd,
            sitename=self.coords.s,
            lonname=self.coords.x,
            latname=self.coords.y,
            **self.sel_method_kwargs,
        )
        return ds

    def _validate_time(self, time):
        if self.coords.t not in self.source.coordinates:
            raise ValueError(f"Time coordinate {self.coords.t} not in source")
        t0, t1 = self.ds.time.to_index().to_pydatetime()[[0, -1]]
        if time.start < t0 or time.end > t1:
            raise ValueError(
                f"time range {time} outside of source time range {t0} - {t1}"
            )

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> xr.Dataset:
        """Return a dataset with the boundary data.

        Parameters
        ----------
        destdir : str | Path
            Placeholder for the destination directory for saving the boundary data.
        grid: RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        ds: xr.Dataset
            The boundary dataset selected from the source. This method is abstract and
            must be implemented by the subclass to generate the expected bcfile output.

        Notes
        -----
        The `destdir` parameter is a placeholder for the output directory, but is not
        used in this method. The method is designed to return the dataset for further
        processing.

        """
        # Slice the times
        if self.crop_data and time is not None:
            self._validate_time(time)
            self._filter_time(time)
        # Select the boundary point
        return self._sel_boundary(grid)


class BoundaryStationSpectraJons(BoundaryStation):
    """Wave boundary conditions from station type spectra dataset such as SMC."""

    model_type: Literal["station_spectra_jons"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    source: SOURCE_SPECTRA_TYPES = Field(
        description=(
            "Dataset source reader, must support CRS and have wavespectra accessor "
        ),
        discriminator="model_type",
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
    coords: DatasetCoords = Field(
        default=DatasetCoords(x="lon", y="lat", t="time", s="site"),
        description="Names of the coordinates in the dataset",
    )
    filelist: Optional[bool] = Field(
        default=True,
        description=(
            "If True, create one bcfile for each timestep in the filtered dataset and "
            "return a FILELIST.txt file with the list of bcfiles, otherwise return a "
            "single bcfile with the wave parameters interpolated at time.start"
        )
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
        stats = ds.spec.stats(["hs", "tp", "dpm", "gamma", "dspr"])
        stats["s"] = dspr_to_s(stats.dspr)
        return stats

    def _adjust_time(self, ds: xr.Dataset, time: TimeRange) -> xr.Dataset:
        """Modify the dataset so the start and end times are included.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the boundary data to adjust.
        time : TimeRange
            The time range to adjust the dataset to.

        Returns
        -------
        dsout : xr.Dataset
            Dataset with the adjusted time range.

        """
        dsout = ds.sel(time=slice(time.start, time.end))
        kwargs = {"fill_value": "extrapolate"}
        times = ds.time.to_index().to_pydatetime()
        if time.start not in times:
            ds_start = ds.interp({self.coords.t: [time.start]}, kwargs=kwargs)
            dsout = xr.concat([ds_start, dsout], dim=self.coords.t)
        if time.end not in times:
            ds_end = ds.interp({self.coords.t: [time.end]}, kwargs=kwargs)
            dsout = xr.concat([dsout, ds_end], dim=self.coords.t)
        return dsout

    def _write_filelist(self, destdir: Path, bcfiles: list[str], durations: list[float]) -> Path:
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
        filename = destdir / "filelist.txt"
        with open(filename, "w") as f:
            f.write("FILELIST\n")
            for bcfile, duration in zip(bcfiles, durations):
                f.write(f"{duration:g} {self.dbtc:g} {bcfile.name}\n")
        return filename

    def _instantiate_boundary(self, data: xr.Dataset) -> WaveBoundaryJons:
        """Instantiate the boundary object.

        Parameters
        ----------
        data : xr.Dataset
            Dataset containing single time for the boundary spectral data.

        """
        assert data.time.size == 1
        t = data.time.to_index().to_pydatetime()[0]
        logger.debug(f"Creating boundary for time {t}")
        return WaveBoundaryJons(
            bcfile=f"jons-{t:%Y%m%dT%H%M%S}.txt",
            hm0=float(data.hs),
            tp=float(data.tp),
            mainang=float(data.dpm),
            gammajsp=float(data.gamma),
            s=float(data.s),
            fnyq=self.fnyq,
            dfj=self.dfj,
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
            data = self._calculate_stats(ds)
            wb = self._instantiate_boundary(data)
            bcfile = BCFile(bcfile=wb.write(destdir))
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
            bcfile = BCFile(filelist=self._write_filelist(destdir, bcfiles, durations))
        return bcfile.namelist







class XBeachSpectraStationMulti(BoundaryWaveStation):
    """Wave boundary conditions from station type spectra dataset."""

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    source: Union[SourceCRSFile, SourceCRSIntake, SourceCRSDataset] = Field(
        description=(
            "Dataset source reader, must support CRS and return a wavespectra-enabled "
            "xarray dataset in the open method"
        ),
        discriminator="model_type",
    )
    kind: Literal["jons", "jonstable", "swan"] = Field(
        description="XBeach wave boundary type",
    )

    def _validate_time(self, time):
        if self.coords.t not in self.source.coordinates:
            raise ValueError(f"Time coordinate {self.coords.t} not in source")
        t0, t1 = self.ds.time.to_index().to_pydatetime()[[0, -1]]
        if time.start < t0 or time.end > t1:
            raise ValueError(f"Times {time} outside of source time range {t0} - {t1}")

    def _dspr_to_s(self, dspr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the Jonswap spreading coefficient from the directional spread."""
        return (2 / np.radians(dspr)**2) - 1

    def _write_jons(self, ds: xr.Dataset, destdir: Path):
        """Write the boundary data as JONS type bcfile.

        Parameters
        ----------
        ds: xr.Dataset
            Dataset containing the boundary spectral data.
        destdir : Path
            Destination directory for the netcdf file.

        Returns
        -------
        bcfile : Path
            Path to the bcfile.

        """
        # Timestep
        if ds[self.coords.t].size > 1:
            times = ds[self.coords.t].to_index().to_pydatetime()
            dt = times[1] - times[0]

        stats = ds.spec.stats(["hs", "tp", "dpm", "gamma", "dspr"])
        stats["s"] = self._dspr_to_s(stats.dspr)
        filelist = []
        for time in stats.time.to_index().to_pydatetime():
            if stats.time.size > 1:
                bcfilename = f"jonswap-{time:%Y%m%dT%H%M%S}.txt"
            else:
                bcfilename = "jonswap.txt"
                filelist = None
            data = stats.sel(time=time)
            wb = WaveBoundarySpectralJons(
                bcfile=bcfilename,
                hm0=float(data.hs),
                tp=float(data.tp),
                mainang=float(data.dpm),
                gammajsp=float(data.gamma),
                s=float(data.s),
                fnyq=0.3,
            )
            bcfile = wb.write(destdir)
            filelist.append(bcfile)

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> str:
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
            Path to the netcdf file.

        """
        # Crop times
        if time is not None:
            self._filter_time(time)
        # # Interpolate at times
        # self._validate_time(time)
        # logger.debug(f"Interpolating boundary data to times {time.date_range}")        
        # ds = self.ds.interp({self.coords.t: time.date_range})

        # Interpolate the the centre of the offshore boundary using wavespectra
        bnd = Ori(x=grid.offshore[0], y=grid.offshore[1], crs=grid.crs).reproject(4326)
        ds = self.ds.spec.sel(
            lons=[bnd.x],
            lats=[bnd.y],
            method=self.sel_method,
            **self.sel_method_kwargs,
        )
        # Write the boundary data
        if self.kind in ["jons"]:
            stats = ds.spec.stats(["hs", "tp", "dpm", "gamma", "dspr"])
            stats["s"] = self._dspr_to_s(stats.dspr)

            filelist = []
            for time in stats.time.to_index().to_pydatetime():
                if stats.time.size > 1:
                    bcfilename = f"jonswap-{time:%Y%m%dT%H%M%S}.txt"
                else:
                    bcfilename = "jonswap.txt"
                data = stats.sel(time=time)
                wb = WaveBoundarySpectralJons(
                    bcfile=bcfilename,
                    hm0=float(data.hs),
                    tp=float(data.tp),
                    mainang=float(data.dpm),
                    gammajsp=float(data.gamma),
                    s=float(data.s),
                    fnyq=0.3,
                )
                bcfile = wb.write(destdir)
                filelist.append(bcfile)
        return filelist


class XBeachParamSingle(BoundaryWaveStation):
    """Wave boundary conditions from station type spectra dataset."""

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    source: Union[SourceCRSFile, SourceCRSIntake, SourceCRSDataset] = Field(
        description=(
            "Dataset source reader, must support CRS and return a wavespectra-enabled "
            "xarray dataset in the open method"
        ),
        discriminator="model_type",
    )
    kind: Literal["jons", "jonstable", "swan"] = Field(
        description="XBeach wave boundary type",
    )

    def _validate_time(self, time):
        if self.coords.t not in self.source.coordinates:
            raise ValueError(f"Time coordinate {self.coords.t} not in source")
        t0, t1 = self.ds.time.to_index().to_pydatetime()[[0, -1]]
        if time.start < t0 or time.end > t1:
            raise ValueError(f"Times {time} outside of source time range {t0} - {t1}")

    def _dspr_to_s(self, dspr: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Calculate the Jonswap spreading coefficient from the directional spread."""
        return (2 / np.radians(dspr)**2) - 1

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> str:
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
            Path to the netcdf file.

        """
        # Interpolate at time
        self._validate_time(time)
        logger.debug(f"Interpolating boundary data to the start time {time.start}")        
        ds = self.ds.interp({self.coords.t: time.start})

        # Interpolate the the centre of the offshore boundary using wavespectra
        bnd = Ori(x=grid.offshore[0], y=grid.offshore[1], crs=grid.crs).reproject(4326)
        ds = ds.spec.sel(
            lons=[bnd.x],
            lats=[bnd.y],
            method=self.sel_method,
            **self.sel_method_kwargs,
        )
        # Write the boundary data
        filename = f"{self.kind}.txt"
        if self.kind == "jons":
            data = ds.spec.stats(["hs", "tp", "dpm", "gamma", "dspr"])
            data["s"] = self._dspr_to_s(data.dspr)
            wb = WaveBoundarySpectralJons(
                bcfile=filename,
                hm0=float(data.hs),
                tp=float(data.tp),
                mainang=float(data.dpm),
                gammajsp=float(data.gamma),
                s=float(data.s),
                fnyq=0.3,
            )
        elif self.kind == "swan":
            pass
        bcfile = wb.write(destdir)
        return bcfile
