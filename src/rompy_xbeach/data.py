"""XBEACH Rompy data."""

import logging
from abc import ABC
from pathlib import Path
from typing import Literal, Union, Optional
from functools import cached_property
import numpy as np
import xarray as xr
from importlib.metadata import entry_points

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import gridspec

from pydantic import Field, field_validator, ConfigDict
from pydantic_numpy.typing import Np2DArray

from rompy.core.types import RompyBaseModel
from rompy.core.data import DataGrid
from rompy.core.time import TimeRange
from rompy_xbeach.grid import RegularGrid


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

# Load the source types from the entry points
eps = entry_points(group="xbeach.source")
Sources = Union[tuple([e.load() for e in eps])]

# Load the interpolator types from the entry points
eps = entry_points(group="xbeach.interpolator")
Interpolators = Union[tuple([e.load() for e in eps])]


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
        ext = np.full((int(grid.ny), nx+1), np.nan)
        ext[:, 0] = 25
        ext[:, -1] = data_posdwn[:, 0]

        # Linearly interpolate the data between the existing and extended boundary
        ext = np.linspace(ext[:, 0], ext[:, -1], nx+1).T

        # Concatenate the extension to the data
        data_ext = np.concatenate((ext[:, :-1], data_posdwn), axis=1)

        # Convert depth to the original sign convention
        data_ext = -data_ext if not posdwn else data_ext

        return data_ext, grid_ext


class XBeachDataGrid(DataGrid):
    """Xbeach data class."""
    model_config = ConfigDict(extra="forbid")
    model_type: Literal["xbeach_data_grid"] = Field(
        default="xbeach_data_grid",
        description="Model type discriminator",
    )
    source: Sources = Field(
        description=(
            "Source reader, must return a dataset with "
            "the rioxarray accessor in the open method"
        ),
        discriminator="model_type",
    )
    interpolator: Interpolators = Field(
        default_factory=eps["regular_grid"].load(),
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
        datafile: Path
            The path to the generated bathymetry data file.
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
        datai = self.interpolator.get(
            x=dset[self.x_dim].values,
            y=dset[self.y_dim].values,
            data=dset.data.values,
            xi=grid.x,
            yi=grid.y,
        )

        # Save to disk
        xfile = Path(destdir) / "xdata.txt"
        yfile = Path(destdir) / "ydata.txt"
        datafile = Path(destdir) / "data.txt"
        np.savetxt(xfile, grid.x)
        np.savetxt(yfile, grid.y)
        np.savetxt(datafile, datai)

        return xfile, yfile, datafile, grid


class XBeachBathy(XBeachDataGrid):
    """XBeach bathymetry data class."""
    model_type: Literal["xbeach_bathy"] = Field(
        default="xbeach_bathy",
        description="Model type discriminator",
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
        if grid.crs is not None:
            logger.debug(f"Reprojecting {self.source.filename} to {grid.crs}")
            dset = self.ds.rio.reproject(grid.crs)
        else:
            dset = self.ds.copy()

        # Interpolate to the model grid
        data = self.interpolator.get(
            x=dset[self.x_dim].values,
            y=dset[self.y_dim].values,
            data=dset.data.values,
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

        fig = plt.figure(figsize=figsize)
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
        ax.plot(x, x*0, "k--")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        if posdwn:
            ax.invert_yaxis()
        ax.set_xlim(xlim)
        ax.set_title("Cross-shore profile")
        # Plot the cross-shore slopes on the right axis
        ax2 = ax.twinx()
        slope = np.gradient(z, grid.dx)
        ax2.fill_between(x, np.nanmin(slopes, 0), np.nanmax(slopes, 0), color="r", alpha=0.2)
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
