"""XBEACH Rompy data."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union, Optional
from functools import cached_property
import xarray as xr

from pydantic import Field, model_validator

from wavespectra.core import select

from rompy.utils import load_entry_points
from rompy.core.data import DataGrid
from rompy.core.time import TimeRange
from rompy_xbeach.grid import RegularGrid, GeoPoint


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

# Load the source types from the entry points
SOURCES = Union[load_entry_points("xbeach.source")]
SOURCES_TS = Union[load_entry_points("rompy.source", etype="timeseries")]


class BaseData(DataGrid, ABC):
    """Xbeach data class."""

    source: SOURCES = Field(
        description=(
            "Source reader, must return a dataset with "
            "the rioxarray accessor in the open method"
        ),
        discriminator="model_type",
    )
    location: Literal["centre", "offshore", "grid"] = Field(
        default="centre",
        description=(
            "Location to extract the data from the source dataset: 'centre' extracts "
            "the data at the centre of the grid, 'offshore' extracts the data at the "
            "middle of the offshore grid boundary, 'grid' at all grid points"
        ),
    )
    time_buffer: list[int] = Field(
        default=[1, 1],
        description=(
            "Number of source data timesteps to buffer the time range "
            "if `filter_time` is True"
        ),
    )

    @cached_property
    def crs(self):
        """Return the coordinate reference system of the data source."""
        return self.ds.rio.crs

    def _validate_time(self, time):
        if self.coords.t not in self.source.coordinates:
            raise ValueError(f"Time coordinate {self.coords.t} not in source")
        t0, t1 = self.ds.time.to_index().to_pydatetime()[[0, -1]]
        if time.start < t0 or time.end > t1:
            raise ValueError(
                f"time range {time} outside of source time range {t0} - {t1}"
            )

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
            dsout = xr.concat([ds_start, dsout], dim=self.coords.t, data_vars="all")
        if time.end not in times:
            ds_end = ds.interp({self.coords.t: [time.end]}, kwargs=kwargs)
            dsout = xr.concat([dsout, ds_end], dim=self.coords.t, data_vars="all")
        return dsout

    def _locations(self, grid: RegularGrid) -> tuple[list[float], list[float]]:
        """Return the x, y locations to generate the data in the source crs."""
        if self.location == "grid":
            # return self.grid.x, self.grid.y
            raise NotImplementedError("Location 'grid' not implemented")
        else:
            x, y = getattr(grid, self.location)
            bnd = GeoPoint(x=x, y=y, crs=grid.crs).reproject(self.crs)
            return [bnd.x], [bnd.y]

    @abstractmethod
    def _sel_locations(self, grid) -> xr.Dataset:
        """Select the data from the source dataset."""
        pass

    @abstractmethod
    def get(
        self,
        destdir: str | Path,
        grid: RegularGrid,
        time: Optional[TimeRange] = None,
    ) -> xr.Dataset:
        """Write the data source to a new location.

        Parameters
        ----------
        destdir : str | Path
            The destination directory to write data file to.
        grid: rompy_xbeach.grid.RegularGrid
            The grid to interpolate the data to.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.filter_time` is True.

        Returns
        -------
        data: xr.Dataset
            The dataset selected from the grid and times. This method is abstract and
            must be implemented by the subclass to generate the expected xbeach output.

        """
        # Slice the times
        if self.crop_data and time is not None:
            self._validate_time(time)
            self._filter_time(time)
        # Select the boundary point
        ds = self._sel_locations(grid)
        # Ensure time exist at the time boundaries
        if time is not None:
            ds = self._adjust_time(ds, time)
        return ds


class BaseDataPoint(BaseData):
    """Base class to construct XBeach input from point timeseries type data."""

    source: SOURCES_TS = Field(
        description=(
            "Source reader, must return an xarray timeseries dataset in the open method"
        ),
    )

    def _sel_locations(self, grid) -> xr.Dataset:
        """Just a placeholder given no spatial selection needs to be performed."""
        return self.ds


class BaseDataStation(BaseData):
    """Base class to construct XBeach input from stations type data."""

    sel_method: Literal["idw", "nearest"] = Field(
        default="idw",
        description=(
            "Defines which function from wavespectra.core.select to use for data "
            "selection: 'idw' uses sel_idw() for inverse distance weighting, "
            "'nearest' uses sel_nearest() for nearest neighbor selection"
        ),
    )
    sel_method_kwargs: dict = Field(
        default={}, description="Keyword arguments for sel_method"
    )

    @model_validator(mode="after")
    def validate_coords(self) -> "BaseDataStation":
        ds = self.ds.copy().reset_coords()
        for coord in [self.coords.t, self.coords.s]:
            if coord not in ds.dims:
                raise ValueError(
                    f"Coordinate '{coord}' not in source dataset, available "
                    f"coordinates are {dict(ds.sizes)} - is this a gridded source?"
                )
        for coord in [self.coords.x, self.coords.y]:
            if coord in ds.dims:
                raise ValueError(
                    f"'{coord}' must not be a dimension in the stations source "
                    f"dataset, but it is: {dict(ds.sizes)} - is this a gridded source?"
                )
            if coord not in ds.data_vars:
                raise ValueError(
                    f"'{coord}' must be a variable in the stations source dataset "
                    f"but available variables are {list(ds.data_vars)}"
                )
        return self

    def _sel_locations(self, grid) -> xr.Dataset:
        """Select the offshore boundary point from the stations source dataset."""
        xbnd, ybnd = self._locations(grid=grid)
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


class BaseDataGrid(BaseData):
    """Base class to construct XBeach input from gridded type data."""

    sel_method: Literal["interp", "sel"] = Field(
        default="sel",
        description=(
            "Defines which function from xarray to use for data selection: 'interp' "
            "uses interp() for interpolation, 'sel' uses sel() for selection"
        ),
    )
    sel_method_kwargs: dict = Field(
        default={"method": "nearest"}, description="Keyword arguments for sel_method"
    )

    @cached_property
    def x_dim(self):
        """Return the x dimension name."""
        return self.ds.rio.x_dim

    @cached_property
    def y_dim(self):
        """Return the y dimension name."""
        return self.ds.rio.y_dim

    def _sel_locations(self, grid) -> xr.Dataset:
        """Select the offshore boundary point from the stations source dataset."""
        xi, yi = self._locations(grid=grid)
        ds = self.ds.copy()
        ds = getattr(ds, self.sel_method)(
            {self.x_dim: xi, self.y_dim: yi}, **self.sel_method_kwargs
        )
        return ds
