"""XBEACH Rompy grid."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union
from pydantic import Field, field_validator
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
import geopandas

from rompy.core.types import RompyBaseModel
from rompy.core.grid import BaseGrid


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


def validate_crs(crs: Optional[Union[str, int]]) -> int:
    """Validate the coordinate reference system input."""
    if crs is None:
        return crs
    if isinstance(crs, str):
        crs = crs.split(":")[-1]
    return ccrs.CRS(str(crs))


class Ori(RompyBaseModel):
    """Origin of the grid in geographic space."""

    x: float = Field(
        description="X coordinate of the origin",
    )
    y: float = Field(
        description="Y coordinate of the origin",
    )
    crs: Optional[Union[int, str]] = Field(
        default=None,
        description="EPSG code for the coordinate reference system",
    )
    _validate_crs = field_validator("crs")(validate_crs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, crs={self.crs})"

    def reproject(self, epsg: int) -> "Ori":
        """Transform the origin to a new coordinate reference system."""
        if self.crs is None:
            raise ValueError("No CRS defined for the origin")
        transformer = Transformer.from_crs(self.crs, epsg, always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        return Ori(x=x, y=y, crs=str(epsg))


# TODO: Method to extend the boundaries of the grid
class RegularGrid(BaseGrid):
    """Xbeach regular grid class."""

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    ori: Ori = Field(
        description="Origin of the grid in geographic space",
    )
    alfa: float = Field(
        description="Angle of x-axis from east in degrees",
    )
    dx: float = Field(
        description="Grid spacing in the x-direction in meters",
    )
    dy: float = Field(
        description="Grid spacing in the y-direction in meters",
    )
    nx: int = Field(
        description="Number of grid points in the x-direction",
    )
    ny: int = Field(
        description="Number of grid points in the y-direction",
    )
    crs: Optional[Union[str, int]] = Field(
        default=None,
        description="EPSG code for the grid projection",
    )
    _validate_crs = field_validator("crs")(validate_crs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(x0={self.x0}, y0={self.y0}, alfa={self.alfa}, "
            f"nx={self.nx}, ny={self.ny}, dx={self.dx}, dy={self.dy}, crs={self.crs})"
        )

    @property
    def x0(self) -> float:
        """X coordinate of the grid origin in the grid crs."""
        if self.crs is not None:
            return self.ori.reproject(self.crs).x
        else:
            return self.ori.x

    @property
    def y0(self) -> float:
        """Y coordinate of the grid origin in the grid crs."""
        if self.crs is not None:
            return self.ori.reproject(self.crs).y
        else:
            return self.ori.y

    @property
    def x(self) -> np.ndarray:
        """X coordinates of the grid."""
        return self._generate()[0]

    @property
    def y(self) -> np.ndarray:
        """Y coordinates of the grid."""
        return self._generate()[1]

    @property
    def left(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the left (lateral) boundary of the grid."""
        return self.x[-1, :], self.y[-1, :]

    @property
    def right(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the right (lateral) boundary of the grid."""
        return self.x[0, :], self.y[0, :]

    @property
    def back(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the back (offshore) boundary of the grid."""
        return self.x[:, -1], self.y[:, -1]

    @property
    def front(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the front (land) boundary of the grid."""
        return self.x[:, 0], self.y[:, 0]

    @property
    def transform(self):
        """Cartopy transformation for the grid."""
        return ccrs.epsg(self.crs.to_epsg())

    @property
    def namelist(self):
        """Return the namelist representation of the grid."""
        return dict(
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy,
            alfa=self.alfa
        )

    def expand(self, left=0, right=0, back=0, front=0) -> "RegularGrid":
        """Expand the grid boundaries."""
        x, y = self._generate(left, right, back, front)
        crs = self.crs.to_epsg()
        ori = Ori(x=x[0, 0], y=y[0, 0], crs=crs).reproject(self.ori.crs.to_epsg())
        return RegularGrid(
            ori=ori,
            alfa=self.alfa,
            dx=self.dx,
            dy=self.dy,
            nx=self.nx + back + front,
            ny=self.ny + left + right,
            crs=crs,
        )

    def _generate(self, left=0, right=0, back=0, front=0) -> tuple[np.ndarray, np.ndarray]:
        """Generate the grid coordinates.

        Parameters
        ----------
        left : int, optional
            Number of points to extend the left lateral boundary, by default 0.
        right : int, optional
            Number of points to extend the right lateral boundary, by default 0.
        back : int, optional
            Number of points to extend the back offshore boundary, by default 0.
        front : int, optional
            Number of points to extend the front inshore boundary, by default 0.

        """
        # Grid at origin
        i = np.arange(0.0, self.dx * self.nx, self.dx)
        j = np.arange(0.0, self.dy * self.ny, self.dy)

        # Expanding the grid
        if front > 0:
            i = np.concatenate([np.arange(-front * self.dx, 0, self.dx), i])
        if back > 0:
            i = np.concatenate([i, np.arange(self.dx * self.nx, self.dx * self.nx + back * self.dx, self.dx)])
        if right > 0:
            j = np.concatenate([np.arange(-right * self.dy, 0, self.dy), j])
        if left > 0:
            j = np.concatenate([j, np.arange(self.dy * self.ny, self.dy * self.ny + left * self.dy, self.dy)])

        # 2D grid indices
        ii, jj = np.meshgrid(i, j)

        # Rotation
        alpha = -self.alfa * np.pi / 180.0
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        gg = np.dot(np.vstack([ii.ravel(), jj.ravel()]).T, R)

        # Translation
        x = gg[:, 0] + self.x0
        y = gg[:, 1] + self.y0

        x = np.reshape(x, ii.shape)
        y = np.reshape(y, ii.shape)
        return x, y

    def plot(
        self,
        ax=None,
        scale=None,
        projection=None,
        buffer=500,
        set_extent=True,
        set_gridlines=True,
        grid_kwargs=dict(alpha=0.5, zorder=2),
        figsize=None,
        show_offshore=True,
    ) -> GeoAxes:
        """Plot the grid optionally overlaid with GSHHS coastlines.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on, by default create a new figure and axes.
        scale : str, optional
            Scale for the GSHHS coastline feature, one of 'c', 'l', 'i', 'h', 'f',
            by default None which implies no coastlines are plotted.
        projection : cartopy.crs.Projection, optional
            Map projection, by default use a stereographic projection.
        buffer : float, optional
            Buffer around the grid in meters, by default 500.
        set_extent : bool, optional
            Set the extent of the axes to the grid bbox and buffer, by default True.
        set_gridlines : bool, optional
            Add gridlines to the plot, by default True.
        grid_kwargs : dict, optional
            Keyword arguments for the grid plot, by default dict(alpha=0.5, zorder=2).
        figsize : tuple, optional
            Figure size in inches, by default None.
        show_offshore : bool, optional
            Show the offshore boundary, by default True.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.

        """
        # Define the projection if not provided
        if projection is None:
            ori = self.ori.reproject(4326)
            projection = ccrs.Stereographic(
                central_longitude=ori.x, central_latitude=ori.y,
            )

        # Define axis if not provided
        if ax is None:
            __, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=projection))

        # Add coastlines
        if scale is not None:
            coast = cfeature.GSHHSFeature(scale=scale)
            ax.add_feature(coast, facecolor="0.7", edgecolor="0.3", linewidth=0.5)

        # Add the model grid
        bnd = geopandas.GeoSeries(self.boundary(), crs=self.transform)
        bnd.plot(ax=ax, transform=self.transform, **grid_kwargs)

        # Add the offshore boundary
        if show_offshore:
            x, y = self.front
            ax.plot(x, y, color="red", transform=self.transform, label="Front")

        # Set extent
        if set_extent:
            x0, y0, x1, y1 = self.bbox()
            ax.set_extent(
                [x0-buffer, x1+buffer, y0-buffer, y1+buffer],
                crs=self.transform,
            )

        # set grid lines
        if set_gridlines is not None:
            ax.gridlines(crs=self.transform, linewidth=0.5, color="gray", alpha=0.5)

        return ax