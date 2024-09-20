"""XBEACH Rompy sources."""

import logging
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
import rioxarray
import xarray as xr

from rompy.core.source import SourceBase


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


class SourceRasterio(SourceBase):
    """Rioxarray source class."""

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    filename: str | Path = Field(description="Path to the rasterio dataset")
    band: int = Field(
        default=1,
        description="Band to read from the rasterio dataset",
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
