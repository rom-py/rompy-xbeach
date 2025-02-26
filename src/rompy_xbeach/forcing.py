"""XBeach forcing."""

from abc import ABC, abstractmethod
from typing import Literal, Union, Optional
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import xarray as xr
from wavespectra.core.utils import uv_to_spddir
from pydantic import Field, model_validator, field_validator


from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy.utils import load_entry_points

from rompy_xbeach.source import SourceCRSOceantide
from rompy_xbeach.data import BaseData, BaseDataGrid, BaseDataStation, BaseDataPoint
from rompy_xbeach.grid import RegularGrid, Ori
from rompy_xbeach.components.forcing import WindFile, TideFile

logger = logging.getLogger(__name__)


# =====================================================================================
# Wind
# =====================================================================================
class WindVector(RompyBaseModel):
    """Wind vector variables."""

    model_type: Literal["wind_vector"] = Field(
        default="wind_vector",
        description="Model type discriminator",
    )
    u: str = Field(
        description="Name of the u-component wind variable",
    )
    v: str = Field(
        description="Name of the v-component wind variable",
    )


class WindScalar(RompyBaseModel):
    """Wind scalar variables."""

    model_type: Literal["wind_scalar"] = Field(
        default="wind_scalar",
        description="Model type discriminator",
    )
    spd: str = Field(
        description="Name of the wind speed wind variable",
    )
    dir: str = Field(
        description="Name of the wind coming from direction variable",
    )


class BaseWind(BaseData, ABC):
    """Wind forcing from gridded data."""

    id: Literal["wind"] = Field(
        default="wind",
        description="Identifier for the wind forcing",
    )
    wind_vars: Union[WindVector, WindScalar] = Field(
        description="Wind forcing variables",
        discriminator="model_type",
    )

    @model_validator(mode="after")
    def set_variables(self):
        """Set the variables attribute based on the wind_vars attribute."""
        logger.debug("Setting wind variables")
        if self.variables:
            logger.warning(f"Overwriting wind variables from {self.wind_vars}")
        # Add the coordinates if stations
        variables = []
        if self.coords.x in self.ds.data_vars and self.coords.y in self.ds.data_vars:
            variables.extend([self.coords.x, self.coords.y])
        if isinstance(self.wind_vars, WindVector):
            variables.extend([self.wind_vars.u, self.wind_vars.v])
        else:
            variables.extend([self.wind_vars.spd, self.wind_vars.dir])
        self.variables = variables
        return self

    def spddir(self, ds) -> tuple[xr.DataArray, xr.DataArray]:
        """Wind speed and direction.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset containing the wind data.

        Returns
        -------
        spd : xr.DataArray
            Wind speed.
        dir : xr.DataArray
            Wind coming_from direction.

        """
        if isinstance(self.wind_vars, WindVector):
            return uv_to_spddir(
                ds[self.wind_vars.u], ds[self.wind_vars.v], coming_from=True
            )
        else:
            return ds[self.wind_vars.spd], ds[self.wind_vars.dir]

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Generate the wind file.

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
        # Select the data at the grid location and times
        ds = super().get(destdir, grid, time)

        # Write the data
        times = ds.time.to_index().to_pydatetime()
        wspd, wdir = self.spddir(ds)
        filename = f"{self.id}-{time.start:%Y%m%dT%H%M%S}-{time.end:%Y%m%dT%H%M%S}.txt"
        logger.debug(f"Creating wind file {filename} with times {times}")
        wf = WindFile(
            filename=filename,
            tsec=[(t - times[0]).total_seconds() for t in times],
            windv=wspd.squeeze().values,
            windth=wdir.squeeze().values,
        )
        wf.write(destdir)

        return {"windfile": filename}


class WindGrid(BaseDataGrid, BaseWind):
    """Wind forcing from gridded data.

    Namelist
    --------
    - windfile : str
        Name of file with non-stationary wind data.

    """

    model_type: Literal["wind_grid"] = Field(
        default="wind_grid",
        description="Model type discriminator",
    )


class WindStation(BaseDataStation, BaseWind):
    """Wind forcing from station data.

    Namelist
    --------
    - windfile : str
        Name of file with non-stationary wind data.

    """

    model_type: Literal["wind_station"] = Field(
        default="wind_station",
        description="Model type discriminator",
    )


class WindPoint(BaseDataPoint, BaseWind):
    """Wind forcing from point timeseries data.

    Namelist
    --------
    - windfile : str
        Name of file with non-stationary wind data.

    """

    model_type: Literal["wind_point"] = Field(
        default="wind_point",
        description="Model type discriminator",
    )


# =====================================================================================
# Water level
# =====================================================================================
class BaseTide(BaseData):
    """Base class for Water level forcing from tide based on oceantide.

    Namelist
    --------
    - zs0file : str
        Name of tide boundary condition series.
    - tideloc : int
        Number of corner points on which a tide time series is specified.
    - tidelen : int
        Number of time steps in the tide time series.

    """

    id: Literal["tide"] = Field(
        default="tide",
        description="Identifier for the tide forcing",
    )
    source: SourceCRSOceantide = Field(
        description="Source of the tide data",
    )
    tideloc: Literal[0, 1, 2, 4] = Field(
        default=1,
        description="Number of corner points on which a tide time series is specified",
    )
    freq: str = Field(
        default="1h",
        description="Frequency for generating the tide timeseries from constituents",
    )
    variables: list[str] = Field(
        default=["h"],
        description="Variables to extract from the dataset",
    )

    @field_validator("tideloc")
    @classmethod
    def raise_non_implemented(cls, v):
        """Only tideloc=1 is currently implemented."""
        if v != 1:
            raise NotImplementedError("Only tideloc=1 is currently implemented")
        return v

    @model_validator(mode="after")
    def set_variables(self):
        """Variable names in an Oceantide dataset should be fixed."""
        logger.debug("Setting oceantide variables")
        if self.variables:
            logger.debug(f"Overwriting tide variables to the oceantide convention")
        self.variables = ["h"]
        return self

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Generate the tide file.

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
        # Select the cons data at the grid location
        ds = super().get(destdir, grid, time=None)

        # Calculate the surface elevation
        times = pd.date_range(time.start, time.end, freq=self.freq)
        ds = ds.tide.predict(times=times, components=["h"], time_chunk=None)

        # Write the data
        filename = f"{self.id}-{time.start:%Y%m%dT%H%M%S}-{time.end:%Y%m%dT%H%M%S}.txt"
        logger.debug(f"Creating wind file {filename} with times {times}")
        tf = TideFile(
            filename=filename,
            tsec=[(t - times[0]).total_seconds() for t in times],
            zs=ds.h.squeeze().values,
        )
        tf.write(destdir)

        return {"zs0file": filename, "tideloc": self.tideloc, "tidelen": ds.time.size}


class TideGrid(BaseTide, BaseDataGrid):
    """Water level forcing from gridded tide cons.

    Namelist
    --------
    - zs0file : str
        Name of tide boundary condition series.
    - tideloc : int
        Number of corner points on which a tide time series is specified.
    - tidelen : int
        Number of time steps in the tide time series.

    """

    model_type: Literal["tide_grid"] = Field(
        default="tide_grid",
        description="Model type discriminator",
    )


# SOURCES_TIDE_STATION = load_entry_points("rompy_xbeach.sources", "tide_station")
from rompy_xbeach.source import SourceTidePointCSV

class TidePoint(BaseTide, BaseDataPoint):
    """Water level forcing from single tide cons station.

    Namelist
    --------
    - zs0file : str
        Name of tide boundary condition series.
    - tideloc : int
        Number of corner points on which a tide time series is specified.
    - tidelen : int
        Number of time steps in the tide time series.

    """

    model_type: Literal["tide_point"] = Field(
        default="tide_point",
        description="Model type discriminator",
    )
    source: SourceTidePointCSV = Field(
        description="Source of the tide data",
    )
