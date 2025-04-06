"""XBEACH Rompy data."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Union, Optional
from functools import cached_property
import numpy as np
import xarray as xr
from importlib.metadata import entry_points

import matplotlib.pyplot as plt
from matplotlib import gridspec

from pydantic import Field, model_validator
from pydantic_numpy.typing import Np2DArray

from wavespectra.core import select

from rompy.utils import load_entry_points
from rompy.core.types import RompyBaseModel
from rompy.core.data import DataGrid
from rompy.core.time import TimeRange
from rompy_xbeach.grid import RegularGrid, GeoPoint


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

# Load the source types from the entry points
SOURCES = Union[load_entry_points("xbeach.source")]
SOURCES_TS = Union[load_entry_points("rompy.source", etype="timeseries")]

# Load the interpolator types from the entry points
INTERPOLATORS = Union[load_entry_points("xbeach.interpolator")]
default_interpolator = entry_points(group="xbeach.interpolator")["regular_grid"].load()


class SeawardExtensionBase(RompyBaseModel, ABC):
    """Base class for extending the data grid seaward."""

    model_type: Literal["base"] = Field(
        default="base",
        description="Model type discriminator",
    )

    def get(
        self,
        data: Np2DArray,
        grid: RegularGrid,
        posdwn: bool,
    ) -> tuple[Np2DArray, RegularGrid]:
        """Override to implement the method to extend the data in the seaward direction.

        Parameters
        ----------
        data: np.ndarray
            The data grid to extend.
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the data.
        posdwn: bool
            Bathymetry is positive down if True, positive up if False.

        Returns
        -------
        data_ext: np.ndarray
            The extended data grid.
        grid_ext: rompy_xbeach.grid.RegularGrid
            The grid associated with the extended data.

        """
        return data, grid


class SeawardExtensionLinear(SeawardExtensionBase):
    """Linear extension of the data grid.

    This class extends the data grid in the seaward direction by linearly
    interpolating between the existing offshore boundary and the extended
    boundary. The length of the extension is calculated based on the shallowest
    offshore depth, the depth at the extended boundary, and the maximum slope.

    """

    model_type: Literal["linear"] = Field(
        default="linear",
        description="Model type discriminator",
    )
    depth: float = Field(
        default=25,
        description="Depth at the offshore boundary of the extended grid",
        gt=0,
    )
    slope: float = Field(
        default=0.3,
        description="Slope of the linear extension",
        gt=0,
    )

    def xlen(self, h_profile: float) -> float:
        """Calculate the 1D extension length.

        Parameters
        ----------
        h_profile: float
            Depth at the offshore boundary of the existing cross shore profile.

        Returns
        -------
        xlen : float
            The extension length in the cross shore direction (m).

        """
        deltaz = self.depth - h_profile
        if deltaz < 0:
            logger.warning(
                f"The offshore depth ({h_profile}) is greater than the extended depth "
                f"({self.depth}), is posdwn set correctly for this bathy data?"
            )

        return np.maximum(deltaz / self.slope, 0)

    def get(
        self,
        data: Np2DArray,
        grid: RegularGrid,
        posdwn: bool,
    ) -> tuple[Np2DArray, RegularGrid]:
        """Extend the data in the seaward direction.

        Parameters
        ----------
        data: np.ndarray
            The data grid to extend.
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the data.
        posdwn: bool
            Bathymetry is positive down if True, positive up if False.

        Returns
        -------
        data_ext: np.ndarray
            The extended data grid.
        grid_ext: rompy_xbeach.grid.RegularGrid
            The grid associated with the extended data.

        """
        # Ensure positive down bathymetry
        data_posdwn = -data.copy() if not posdwn else data.copy()

        # Calculate extension length based on the shallowest offshore depth
        h_profile = data_posdwn[:, 0].min()
        extension_length = self.xlen(h_profile)

        # Define the extended grid
        nx = int(np.ceil(extension_length / grid.dx))
        grid_ext = grid.expand(front=nx)

        # Initialise an extension array that includes the offshore grid column
        ext = np.full((int(grid.ny), nx + 1), np.nan)
        ext[:, 0] = 25
        ext[:, -1] = data_posdwn[:, 0]

        # Linearly interpolate the data between the existing and extended boundary
        ext = np.linspace(ext[:, 0], ext[:, -1], nx + 1).T

        # Concatenate the extension to the data
        data_ext = np.concatenate((ext[:, :-1], data_posdwn), axis=1)

        # Convert depth to the original sign convention
        data_ext = -data_ext if not posdwn else data_ext

        return data_ext, grid_ext


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
            dsout = xr.concat([ds_start, dsout], dim=self.coords.t)
        if time.end not in times:
            ds_end = ds.interp({self.coords.t: [time.end]}, kwargs=kwargs)
            dsout = xr.concat([dsout, ds_end], dim=self.coords.t)
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


class XBeachDataGrid(DataGrid):
    """Xbeach data class."""

    model_type: Literal["xbeach_data_grid"] = Field(
        default="xbeach_data_grid",
        description="Model type discriminator",
    )
    source: SOURCES = Field(
        description=(
            "Source reader, must return a dataset with "
            "the rioxarray accessor in the open method"
        ),
        discriminator="model_type",
    )
    interpolator: INTERPOLATORS = Field(
        default_factory=default_interpolator,
        description="Interpolator for the data",
        discriminator="model_type",
    )

    @cached_property
    def crs(self):
        """Return the coordinate reference system of the data source."""
        return self.ds.rio.crs

    @cached_property
    def x_dim(self):
        """Return the x dimension name."""
        return self.ds.rio.x_dim

    @cached_property
    def y_dim(self):
        """Return the y dimension name."""
        return self.ds.rio.y_dim

    def get(
        self,
        destdir: str | Path,
        grid: RegularGrid,
        time: Optional[TimeRange] = None,
    ) -> tuple[Path, Path, list[Path], RegularGrid]:
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
        xfile: Path
            The path to the generated x-coordinate data file.
        yfile: Path
            The path to the generated y-coordinate data file.
        datafiles: list[Path]
            The paths to the generated data file for each variable.
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the bathymetry data.

        """
        if self.crop_data:
            if time is not None:
                self._filter_time(time)

        # Reproject to the model grid
        if grid.crs is not None:
            logger.info(f"Reprojecting {self.source.filename} to {grid.crs}")
            dset = self.ds.rio.reproject(grid.crs)
        else:
            dset = self.ds.copy()

        # Interpolate to the model grid
        datai = {}
        for variable in self.variables:
            datai[variable] = self.interpolator.get(
                x=dset[self.x_dim].values,
                y=dset[self.y_dim].values,
                data=dset[variable].values,
                xi=grid.x,
                yi=grid.y,
            )

        # Save to disk
        xfile = Path(destdir) / "xdata.txt"
        yfile = Path(destdir) / "ydata.txt"
        np.savetxt(xfile, grid.x)
        np.savetxt(yfile, grid.y)
        datafiles = []
        for variable, data in datai.items():
            datafile = Path(destdir) / f"data-{variable}.txt"
            np.savetxt(datafile, data)
            datafiles.append(datafile)

        return xfile, yfile, datafiles, grid


class XBeachBathy(XBeachDataGrid):
    """XBeach bathymetry data class."""

    model_type: Literal["xbeach_bathy"] = Field(
        default="xbeach_bathy",
        description="Model type discriminator",
    )
    variables: Union[str, list] = Field(
        default="data",
        description="The variable name in the source dataset",
    )
    posdwn: bool = Field(
        default=True,
        description="Bathymerty is positive down if True, positive up if False",
    )
    left: int = Field(
        default=0,
        description="Number of points to extend the left lateral boundary",
        ge=0,
    )
    right: int = Field(
        default=0,
        description="Number of points to extend the right lateral boundary",
        ge=0,
    )
    extension: Union[SeawardExtensionBase, SeawardExtensionLinear] = Field(
        default_factory=SeawardExtensionBase,
        description="Method to extend the data seaward",
        discriminator="model_type",
    )
    interpolate_na: bool = Field(
        default=True,
        description="Interpolate NaN values in the source data",
    )
    interpolate_na_kwargs: dict = Field(
        default_factory=dict,
        description="Keyword arguments for the interpolate_na method",
    )

    @model_validator(mode="after")
    def single_variable(self) -> "XBeachBathy":
        """Ensure a single variable is provided."""
        if isinstance(self.variables, str):
            self.variables = [self.variables]
        if len(self.variables) > 1:
            raise ValueError(
                "XBeachBathy only supports one single variable but multiple "
                f"variables {list(self.variables)} have been prescribed"
            )
        return self

    @property
    def namelist(self):
        """Return the namelist representation of the bathy data."""
        return dict(
            posdwn=1 if self.posdwn else -1,
        )

    def expand_lateral(
        self,
        data: Np2DArray,
        grid: RegularGrid,
    ) -> tuple[Np2DArray, RegularGrid]:
        """Extend the data laterally.

        Parameters
        ----------
        data: Np2DArray
            The data grid to extend.
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the data.

        Returns
        -------
        data_ext: Np2DArray
            The extended data grid.
        grid_ext: rompy_xbeach.grid.RegularGrid
            The grid associated with the extended data.

        """
        grid_ext = grid.expand(left=self.left, right=self.right)
        right_ext = np.tile(data[0, :], (self.right, 1))
        left_ext = np.tile(data[-1, :], (self.right, 1))
        data_ext = np.concatenate((right_ext, data, left_ext), axis=0)
        return data_ext, grid_ext

    def get(
        self,
        destdir: str | Path,
        grid: RegularGrid,
        time: Optional[TimeRange] = None,
    ) -> tuple[Path, Path, Path, RegularGrid]:
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
        xfile: Path
            The path to the generated x-coordinate data file.
        yfile: Path
            The path to the generated y-coordinate data file.
        depfile: Path
            The path to the generated bathymetry data file.
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the bathymetry data.

        """
        if self.crop_data:
            if time is not None:
                self._filter_time(time)

        # Reproject to the model grid
        if grid.crs is not None and grid.crs.to_epsg() != self.crs.to_epsg():
            logger.debug(f"Reprojecting data to {grid.crs}")
            dset = self.ds.rio.reproject(grid.crs).rename(x=self.x_dim, y=self.y_dim)
        else:
            dset = self.ds.copy()

        # Interpolate nan values
        if self.interpolate_na:
            dset = (
                dset.sortby([self.x_dim, self.y_dim])
                .interpolate_na(dim=self.x_dim, **self.interpolate_na_kwargs)
                .interpolate_na(dim=self.y_dim, **self.interpolate_na_kwargs)
            )

        # Interpolate to the model grid
        variable = self.variables[0]
        data = self.interpolator.get(
            x=dset[self.x_dim].values,
            y=dset[self.y_dim].values,
            data=dset[variable].values,
            xi=grid.x,
            yi=grid.y,
        )

        # Extend offshore boundary
        data, grid = self.extension.get(data, grid, self.posdwn)

        # Extend the lateral boundaries
        data, grid = self.expand_lateral(data, grid)

        # Save to disk
        xfile = Path(destdir) / "xdata.txt"
        yfile = Path(destdir) / "ydata.txt"
        depfile = Path(destdir) / "bathy.txt"
        np.savetxt(xfile, grid.x)
        np.savetxt(yfile, grid.y)
        np.savetxt(depfile, data)

        return xfile, yfile, depfile, grid


@xr.register_dataset_accessor("xbeach")
class XBeach_accessor(object):
    """XBeach accessor for xarray datasets with the model data."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot_model_bathy(
        self,
        grid: RegularGrid,
        variable: str = "dep",
        posdwn: bool = True,
        figsize=(15, 12),
        vmin: float = -20,
        vmax: float = 20,
        cmap: str = "terrain",
    ):
        """Plot the model bathy.

        This method plots the bathymetry in real and model coordinates, and the
        cross-shore profiles and slopes.

        Parameters
        ----------
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the data.
        variable: str
            The variable to plot.
        posdwn: bool
            Bathymetry is positive down if True, positive up if False.
        figsize: tuple
            The figure size.
        vmin: float
            The minimum value for the colormap.
        vmax: float
            The maximum value for the colormap.
        cmap: str
            The colormap.

        """

        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1], width_ratios=[1, 1])
        dep = self._obj[variable]

        # =========================
        # Plot in real coordinates
        # =========================
        ax = plt.subplot(gs[0], projection=grid.projection)
        ax.pcolormesh(
            grid.x,
            grid.y,
            dep,
            transform=grid.transform,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax = grid.plot(
            ax=ax,
            buffer=50,
            grid_kwargs=dict(edgecolor="k", facecolor="none"),
            show_mesh=True,
            mesh_step=5,
            mesh_kwargs=dict(color="k", linewidth=0.3, alpha=0.5),
        )
        ax.set_title("Real coordinates")

        # ==========================
        # Plot in model coordinates
        # ==========================
        ax = plt.subplot(gs[1])
        x = np.arange(grid.nx) * grid.dx
        y = np.arange(grid.ny) * grid.dy
        p = ax.pcolormesh(x, y, dep, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(p)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        xlim = 0, x[-1]
        ax.set_xlim(xlim)
        ax.set_aspect("equal", "box")
        ax.set_title("Model coordinates")

        # =============================
        # Plot the cross-shore profile
        # =============================
        ax = plt.subplot(gs[1, :])
        slopes = np.zeros((grid.ny, grid.nx))
        for iy in range(grid.ny):
            ax.plot(x, dep[iy, :], color="0.5", alpha=0.5)
            slopes[iy, :] = np.gradient(dep[iy, :], grid.dx)
        z = np.nanmean(dep, axis=0)
        ax.plot(x, z, color="k", linewidth=3)
        ax.plot(x, x * 0, "k--")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        if posdwn:
            ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_title("Cross-shore profile")
        # Plot the cross-shore slopes on the right axis
        ax2 = ax.twinx()
        slope = np.gradient(z, grid.dx)
        ax2.fill_between(
            x, np.nanmin(slopes, 0), np.nanmax(slopes, 0), color="r", alpha=0.2
        )
        ax2.plot(x, slope, color="r")
        ax2.yaxis.label.set_color("r")
        ax2.tick_params(axis="y", colors="r")
        ax2.set_ylabel(r"$\tan(\alpha)$")
        ax2.set_xlim(xlim)

    @classmethod
    def from_xbeach(cls, datafile, grid):
        """Construct an xarray bathy dataset from a XBeach data file.

        Parameters
        ----------
        datafile : str | Path
            The path to the XBeach bathy data file.
        grid : rompy_xbeach.grid.RegularGrid
            The grid associated with the data.

        Returns
        -------
        dset : xr.Dataset
            The xarray dataset with the bathy data.

        """
        dset = xr.Dataset()
        dset["xc"] = xr.DataArray(grid.x, dims=("y", "x"))
        dset["yc"] = xr.DataArray(grid.y, dims=("y", "x"))
        dset["dep"] = xr.DataArray(np.loadtxt(datafile), dims=("y", "x"))
        return dset.rio.write_crs(grid.crs).set_coords(["xc", "yc"])
