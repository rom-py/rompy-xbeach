"""XBeach wave boundary conditions."""

from abc import ABC, abstractmethod
from typing import Literal, Union, Optional, Annotated
from pathlib import Path
import logging
import numpy as np
import xarray as xr
from pydantic import BaseModel, Field, model_validator

from rompy.core.time import TimeRange
from rompy.core.boundary import BoundaryWaveStation

from rompy_xbeach.source import SourceCRSFile, SourceCRSIntake, SourceCRSDataset
from rompy_xbeach.grid import RegularGrid, Ori
from rompy_xbeach.components.boundary import WaveBoundarySpectralJons


logger = logging.getLogger(__name__)


# Data interfaces
class XBeachSpectraStationSingle(BoundaryWaveStation):
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
