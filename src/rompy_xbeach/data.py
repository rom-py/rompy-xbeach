"""XBEACH Rompy data."""

import logging
from pathlib import Path
from typing import Literal, Union, Optional
from functools import cached_property
from pydantic import Field, field_validator
import numpy as np
import xarray as xr
from importlib.metadata import entry_points

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

class XBeachDataGrid(DataGrid):
    """Xbeach data class."""

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
