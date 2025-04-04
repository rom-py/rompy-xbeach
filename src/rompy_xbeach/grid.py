"""XBEACH Rompy grid."""

import logging
import ast
from pathlib import Path
from typing import Literal, Optional, Union
from pydantic import Field, field_validator, ConfigDict
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from pyproj import CRS
import rasterio
import cartopy
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
import geopandas
from functools import cached_property

from rompy.core.types import RompyBaseModel
from rompy.core.grid import BaseGrid


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

CRS_TYPES = Union[str, int, CRS, cartopy.crs.CRS, rasterio.crs.CRS]


def validate_crs(crs: Optional[CRS_TYPES]) -> CRS:
    """Validate the coordinate reference system input."""
    if crs is None:
        return crs
    if isinstance(crs, str):
        crs = crs.split(":")[-1]
    return CRS.from_user_input(crs)


class GeoPoint(RompyBaseModel):
    """Origin of the grid in geographic space."""

    x: float = Field(
        description="X coordinate of the origin",
    )
    y: float = Field(
        description="Y coordinate of the origin",
    )
    crs: Optional[CRS_TYPES] = Field(
        default=None,
        description="Coordinate reference system of the origin",
    )
    _validate_crs = field_validator("crs")(validate_crs)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_dump(self, *args, **kwargs):
        """Dump the model representation of the object taking care of the CRS."""
        d = super().model_dump(*args, **kwargs)
        if d["crs"] is not None:
            d["crs"] = str(self.crs)
        return d

    def __repr__(self) -> str:
        crs = str(self.crs) if self.crs is not None else None
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, crs='{crs}')"

    def __str__(self) -> str:
        return self.__repr__()

    def reproject(self, crs: CRS_TYPES) -> "GeoPoint":
        """Transform the origin to a new coordinate reference system."""
        if self.crs is None:
            raise ValueError(f"No CRS defined, cannot reproject onto {crs}")
        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
        x, y = transformer.transform(self.x, self.y)
        return GeoPoint(x=x, y=y, crs=str(crs))


class RegularGrid(BaseGrid):
    """Xbeach regular grid class."""

    model_type: Literal["regular"] = Field(
        default="regular",
        description="Model type discriminator",
    )
    ori: GeoPoint = Field(
        description="Origin of the grid in geographic space",
    )
    alfa: float = Field(
        description="Angle of x-axis from east in degrees",
    )
    dx: float = Field(
        description=(
            "Grid spacing in the x-direction, for projected CRS, this value is "
            "defined in meters, for geographic CRS, it is defined in degrees"
        ),
    )
    dy: float = Field(
        description=(
            "Grid spacing in the y-direction, for projected CRS, this value is "
            "defined in meters, for geographic CRS, it is defined in degrees"
        ),
    )
    nx: int = Field(
        description=(
            "Number of grid points in the x-direction, the 'nx' paramter "
            "in the XBeach namelist will be one less than this value (nx - 1)"
        ),
    )
    ny: int = Field(
        description=(
            "Number of grid points in the y-direction, the 'ny' paramter "
            "in the XBeach namelist will be one less than this value (ny - 1)"
        )
    )
    crs: Optional[CRS_TYPES] = Field(
        default=None,
        description="Coordinate reference system of the grid",
    )
    _validate_crs = field_validator("crs")(validate_crs)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        epsg = str(self.crs) if self.crs is not None else None
        return (
            f"{self.__class__.__name__}(ori={self.ori}, alfa={self.alfa}, "
            f"dx={self.dx}, dy={self.dy}, nx={self.nx}, ny={self.ny}, crs='{epsg}')"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def model_dump(self, *args, **kwargs):
        """Dump the model representation of the object taking care of the CRS."""
        d = super().model_dump(*args, **kwargs)
        if d["crs"] is not None:
            d["crs"] = str(self.crs)
        if d["ori"]["crs"] is not None:
            d["ori"]["crs"] = str(self.ori.crs)
        return d

    @cached_property
    def x0(self) -> float:
        """X coordinate of the grid origin in the grid crs."""
        if self.crs is not None:
            return self.ori.reproject(self.crs).x
        else:
            return self.ori.x

    @cached_property
    def y0(self) -> float:
        """Y coordinate of the grid origin in the grid crs."""
        if self.crs is not None:
            return self.ori.reproject(self.crs).y
        else:
            return self.ori.y

    @cached_property
    def x(self) -> np.ndarray:
        """X coordinates of the grid."""
        return self._generate()[0]

    @cached_property
    def y(self) -> np.ndarray:
        """Y coordinates of the grid."""
        return self._generate()[1]

    @cached_property
    def shape(self) -> tuple[int, int]:
        """Shape of the grid."""
        return self.x.shape

    @cached_property
    def gdf(self):
        """Geodataframe representation with multi-polygons for each grid cell."""
        # Define the multi-polygon geodataframe
        i, j = np.meshgrid(
            range(self.shape[0] - 1), range(self.shape[1] - 1), indexing="ij"
        )
        corners = np.stack(
            [
                np.stack((self.x[i, j], self.y[i, j]), axis=-1),
                np.stack((self.x[i, j + 1], self.y[i, j + 1]), axis=-1),
                np.stack((self.x[i + 1, j + 1], self.y[i + 1, j + 1]), axis=-1),
                np.stack((self.x[i + 1, j], self.y[i + 1, j]), axis=-1),
            ],
            axis=-2,
        )
        corners_2d = corners.reshape(-1, 4, 2)
        multi_polygon = MultiPolygon([Polygon(cell) for cell in corners_2d])
        gdf = gpd.GeoDataFrame(geometry=[multi_polygon], crs=self.crs)
        # Add the grid kwargs in the Name column
        gdf["Name"] = (self.model_dump(),)
        return gdf

    @cached_property
    def left(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the left (lateral) boundary of the grid."""
        return self.x[-1, :], self.y[-1, :]

    @cached_property
    def right(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the right (lateral) boundary of the grid."""
        return self.x[0, :], self.y[0, :]

    @cached_property
    def back(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the back (land) boundary of the grid."""
        return self.x[:, -1], self.y[:, -1]

    @cached_property
    def front(self) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates of the front (offshore) boundary of the grid."""
        return self.x[:, 0], self.y[:, 0]

    @cached_property
    def offshore(self) -> tuple[float, float]:
        """Coordinates at the centre of the offshore boundary."""
        x, y = self.front
        return float(x.mean()), float(y.mean())

    @cached_property
    def centre(self) -> tuple[float, float]:
        """Coordinates at the centre of the grid."""
        return float(self.x.mean()), float(self.y.mean())

    @cached_property
    def transform(self):
        """Cartopy transformation for the grid."""
        _epsg = self.crs.to_epsg()

        # If EPSG exists, use it
        if _epsg is not None and self.crs.is_projected:
            return ccrs.epsg(_epsg)

        # If no EPSG, use Cartopy's Stereographic projection
        elif "stere" in self.crs.to_proj4():
            return ccrs.Stereographic(
                central_longitude=self.ori.x, central_latitude=self.ori.y
            )

        # If CRS is geographic (lat/lon), use PlateCarree
        elif self.crs.is_geographic:
            return ccrs.PlateCarree()

        else:
            raise ValueError(f"Unsupported CRS: {self.crs}")

    @cached_property
    def projection(self):
        """Cartopy stereographic projection for this grid."""
        if self.ori.crs is None:
            raise ValueError("No CRS defined for the grid origin")
        ori = self.ori.reproject(4326)
        return ccrs.Stereographic(central_longitude=ori.x, central_latitude=ori.y)

    @cached_property
    def namelist(self):
        """Return the namelist representation of the grid."""
        return dict(
            nx=self.nx - 1,
            ny=self.ny - 1,
            dx=self.dx,
            dy=self.dy,
            xori=self.x0,
            yori=self.y0,
            alfa=self.alfa,
            projection=self.crs.to_proj4(),
        )

    def expand(self, left=0, right=0, back=0, front=0) -> "RegularGrid":
        """Expand the grid boundaries."""
        x, y = self._generate(left, right, back, front)
        ori = GeoPoint(x=x[0, 0], y=y[0, 0], crs=self.crs).reproject(self.ori.crs.to_epsg())
        return RegularGrid(
            ori=ori,
            alfa=self.alfa,
            dx=self.dx,
            dy=self.dy,
            nx=self.nx + back + front,
            ny=self.ny + left + right,
            crs=self.crs,
        )

    def _generate(
        self, left=0, right=0, back=0, front=0
    ) -> tuple[np.ndarray, np.ndarray]:
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
            i = np.concatenate(
                [
                    i,
                    np.arange(
                        self.dx * self.nx, self.dx * self.nx + back * self.dx, self.dx
                    ),
                ]
            )
        if right > 0:
            j = np.concatenate([np.arange(-right * self.dy, 0, self.dy), j])
        if left > 0:
            j = np.concatenate(
                [
                    j,
                    np.arange(
                        self.dy * self.ny, self.dy * self.ny + left * self.dy, self.dy
                    ),
                ]
            )

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
        ax: cartopy.mpl.geoaxes.GeoAxes = None,
        scale: str = None,
        projection: cartopy.crs.PlateCarree = None,
        buffer: float = 500,
        set_extent: bool = True,
        set_gridlines: bool = True,
        grid_kwargs: dict = dict(alpha=0.5, zorder=2),
        figsize: tuple = None,
        show_mesh: bool = False,
        mesh_step: int = 1,
        mesh_kwargs: dict = dict(color="k", linewidth=0.5),
        show_offshore: bool = True,
        show_origin: bool = True,
    ) -> GeoAxes:
        """Plot the grid optionally overlaid with GSHHS coastlines.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            Axes object to plot on, by default create a new figure and axes.
        scale: str, optional
            Scale for the GSHHS coastline feature, one of 'c', 'l', 'i', 'h', 'f',
            by default None which implies no coastlines are plotted.
        projection: cartopy.crs.Projection, optional
            Map projection, by default use a stereographic projection.
        buffer: float, optional
            Buffer around the grid in meters, by default 500.
        set_extent: bool, optional
            Set the extent of the axes to the grid bbox and buffer, by default True.
        set_gridlines: bool, optional
            Add gridlines to the plot, by default True.
        grid_kwargs: dict, optional
            Keyword arguments for the grid plot, by default dict(alpha=0.5, zorder=2).
        figsize: tuple, optional
            Figure size in inches, by default None.
        show_mesh: bool, optional
            Show the model grid mesh, by default False.
        mesh_step: int, optional
            Step for the mesh plot, by default 1.
        mesh_kwargs: dict, optional
            Keyword arguments for the mesh, by default dict(color="k", linewidth=0.5).
        show_offshore: bool, optional
            Show the offshore boundary, by default True.
        show_origin: bool, optional
            Show the origin of the grid, by default True.

        Returns
        -------
        matplotlib.axes.Axes
            Axes object with the plot.

        """
        # Define the projection if not provided
        if projection is None:
            projection = self.projection

        # Define axis if not provided
        if ax is None:
            __, ax = plt.subplots(
                figsize=figsize, subplot_kw=dict(projection=projection)
            )

        # Add coastlines
        if scale is not None:
            coast = cfeature.GSHHSFeature(scale=scale)
            ax.add_feature(coast, facecolor="0.7", edgecolor="0.3", linewidth=0.5)

        # Add the model grid
        bnd = geopandas.GeoSeries(self.boundary(), crs=self.transform)
        bnd.plot(ax=ax, transform=self.transform, **grid_kwargs)

        # Draw the model grid mesh
        if show_mesh:
            ix = np.unique(np.append(np.arange(0, self.nx, mesh_step), self.nx - 1))
            iy = np.unique(np.append(np.arange(0, self.ny, mesh_step), self.ny - 1))
            x = self.x[np.ix_(iy, ix)]
            y = self.y[np.ix_(iy, ix)]
            ax.plot(x, y, transform=self.transform, **mesh_kwargs)
            ax.plot(x.T, y.T, transform=self.transform, **mesh_kwargs)

        # Add the offshore boundary
        if show_offshore:
            x, y = self.front
            ax.plot(x, y, color="red", transform=self.transform, label="Front")

        # Add the grid origin
        if show_origin:
            ax.plot(self.x0, self.y0, "ro", transform=self.transform, label="Origin")

        # Set extent
        if set_extent:
            x0, y0, x1, y1 = self.bbox()
            ax.set_extent(
                [x0 - buffer, x1 + buffer, y0 - buffer, y1 + buffer],
                crs=self.transform,
            )

        # set grid lines
        if set_gridlines is not None:
            ax.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                linewidth=0.5,
                color="gray",
                alpha=0.5,
            )

        return ax

    def to_file(self, filename, **kwargs):
        self.gdf.to_file(filename, **kwargs)

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> "RegularGrid":
        """Read a grid from a file.

        Parameters
        ----------
        filename : str
            Path to the file.
        kwargs : dict
            Additional keyword arguments for the geopandas read_file method.

        Returns
        -------
        RegularGrid
            RegularGrid object.

        """
        gdf = gpd.read_file(filename, **kwargs)
        kwargs = ast.literal_eval(gdf["Name"].values[0])
        return RegularGrid(**kwargs)
