"""XBEACH Rompy data."""

import logging
from abc import ABC
from pathlib import Path
from typing import Literal, Union, Optional
from functools import cached_property
import numpy as np
import xarray as xr
from importlib.metadata import entry_points

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
    ) -> Path:
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
        cmd: str
            The command line string with the INPGRID/READINP commands ready to be
            written to the SWAN input file.

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
        datai = self.interpolator.interpolate(
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

        return xfile, yfile, datafile


class XBeachBathy(XBeachDataGrid):

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

    def expand_lateral(
        self,
        data: Np2DArray,
        grid: RegularGrid,
    ) -> tuple[Np2DArray, RegularGrid]:
        """Extend the data laterally.

        Parameters
        ----------
        data: np.ndarray
            The data grid to extend.
        grid: rompy_xbeach.grid.RegularGrid
            The grid associated with the data.

        Returns
        -------
        data_ext: np.ndarray
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
    ) -> Path:
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
        cmd: str
            The command line string with the INPGRID/READINP commands ready to be
            written to the SWAN input file.

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
        data = self.interpolator.interpolate(
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
        datafile = Path(destdir) / "data.txt"
        np.savetxt(xfile, grid.x)
        np.savetxt(yfile, grid.y)
        np.savetxt(datafile, data)

        return xfile, yfile, datafile, grid


@xr.register_dataset_accessor("xbeach")
class XBeach_accessor(object):
    """XBeach accessor for xarray datasets."""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def x(self):
        """Return the x coord."""
        return self._obj[self._obj.rio.x_dim]

    @property
    def y(self):
        """Return the y coord."""
        return self._obj[self._obj.rio.y_dim]
