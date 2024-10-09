"""XBEACH Rompy sources."""

import logging
from pathlib import Path
from typing import Literal, Union, Optional
from pydantic import Field, field_validator
import cartopy.crs as ccrs
import rioxarray
import xarray as xr

from rompy.core.source import SourceBase, SourceDataset, SourceFile, SourceIntake


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


class SourceGeotiff(SourceBase):
    """Geotiff source class."""

    model_type: Literal["geotiff"] = Field(
        default="geotiff",
        description="Model type discriminator",
    )
    filename: str | Path = Field(description="Path to the geotiff dataset")
    band: int = Field(
        default=1,
        description="Band to read from the dataset after opening it with rasterio",
        ge=1,
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to rioxarray.open_rasterio",
    )

    @field_validator("kwargs")
    @classmethod
    def validate_rasterio_kwargs(cls, v) -> dict:
        """Validate the rasterio keyword arguments."""
        # Ensure Dataset is not returned
        v["band_as_variable"] = False
        # Ensure band name is not set
        if "default_name" in v:
            logger.info("Ignoring `default_name` from rasterio kwargs")
            v.pop("default_name")
        return v

    def _open(self) -> xr.Dataset:
        """This method needs to return an xarray Dataset object."""
        xds = rioxarray.open_rasterio(self.filename, **self.kwargs)
        xds = xds.sel(band=self.band).to_dataset(name="data")
        return xds


class SourceMixin:
    """Mixin class for crs aware source objects."""
    crs: Union[int, str] = Field(
        description="EPSG code for the coordinate reference system",
    )
    x_dim: str = Field(
        default="x",
        description="Name of the x dimension",
    )
    y_dim: str = Field(
        default="y",
        description="Name of the y dimension",
    )

    @field_validator("crs")
    @classmethod
    def validate_crs(cls, v):
        """Validate the coordinate reference system input."""
        if isinstance(v, str):
            v = v.split(":")[-1]
        return ccrs.CRS(str(v))

    def _open(self):
        """Return a CRS aware dataset."""
        ds = super()._open()
        return ds.rio.set_spatial_dims(self.x_dim, self.y_dim).rio.write_crs(self.crs)


class SourceCRSDataset(SourceMixin, SourceDataset):
    """Source dataset with CRS support from an existing xarray Dataset object."""


class SourceCRSFile(SourceMixin, SourceFile):
    """Source dataset with CRS support from file to open with xarray.open_dataset."""


class SourceCRSIntake(SourceMixin, SourceIntake):
    """Source dataset with CRS support from intake catalog."""
