"""XBEACH Rompy sources."""

import logging
from pathlib import Path
from typing import Literal, Union
from pydantic import Field, field_validator, model_validator, ConfigDict
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from scipy.interpolate import griddata
import oceantide

from rompy.core.filters import Filter
from rompy.core.source import (
    SourceBase,
    SourceDataset,
    SourceFile,
    SourceIntake,
    SourceWavespectra,
)
from rompy_xbeach.grid import CRS_TYPES, validate_crs


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


class SourceXYZ(SourceBase):
    """XYZ source class."""

    model_type: Literal["xyz"] = Field(
        default="xyz",
        description="Model type discriminator",
    )
    filename: str | Path = Field(description="Path to the xyz dataset")
    crs: Union[int, str] = Field(
        description="Coordinate reference system of the source data",
    )
    res: float = Field(
        description="Resolution of the regular grid to interpolate onto",
    )
    xcol: str = Field(
        default="x",
        description="Name of the column containing the x data",
    )
    ycol: str = Field(
        default="y",
        description="Name of the column containing the y data",
    )
    zcol: str = Field(
        default="z",
        description="Name of the column containing the z data",
    )
    read_csv_kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to pandas.read_csv",
    )
    griddata_kwargs: dict = Field(
        default={"method": "linear"},
        description="Keyword arguments to pass to scipy.interpolate.griddata",
    )

    def _open_dataframe(self) -> pd.DataFrame:
        """Read the xyz data from the file."""
        df = pd.read_csv(self.filename, **self.read_csv_kwargs)
        try:
            df = df[[self.xcol, self.ycol, self.zcol]]
        except KeyError as e:
            raise ValueError(
                f"Columns ({self.xcol}, {self.ycol}, {self.zcol}) must be present in "
                f"the dataframe, got ({list(df.columns)}), make sure xcol, ycol and "
                "zcol fields are correctly specified and the headers can be correctly "
                f"parsed from read_csv_kwargs ({self.read_csv_kwargs})"
            ) from e
        df.columns = ["x", "y", "z"]
        return df

    def _open(self) -> xr.Dataset:
        """Interpolate the xyz data onto a regular grid."""
        df = self._open_dataframe()

        # Define the grid
        xgrid = np.unique(np.arange(df.x.min(), df.x.max() + self.res, self.res))
        ygrid = np.unique(np.arange(df.y.min(), df.y.max() + self.res, self.res))
        if xgrid.size < 3:
            raise ValueError(
                "The resolution is too high for the provided data, the grid must have "
                f"at least 3 points in each dimension, got xgrid={xgrid}, ygrid={ygrid}"
            )
        logger.info(f"Interpolating onto a grid of shape ({len(ygrid)}, {len(xgrid)})")

        # Interpolate the data
        zgrid = griddata(
            points=(df.x, df.y),
            values=df.z,
            xi=tuple(np.meshgrid(xgrid, ygrid)),
            **self.griddata_kwargs,
        )

        # Create the dataset
        ds = xr.Dataset(
            data_vars={"data": (["y", "x"], zgrid)}, coords={"y": ygrid, "x": xgrid}
        )
        return ds.rio.write_crs(self.crs)


class SourceMixin:
    """Mixin class for crs aware source objects."""

    crs: CRS_TYPES = Field(
        description="Coordinate reference system of the source data",
    )
    x_dim: str = Field(
        default="x",
        description="Name of the x dimension",
    )
    y_dim: str = Field(
        default="y",
        description="Name of the y dimension",
    )
    _validate_crs = field_validator("crs")(validate_crs)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def open(self, variables: list = [], filters: Filter = {}, **kwargs) -> xr.Dataset:
        """Return the filtered dataset object.

        Parameters
        ----------
        variables : list, optional
            List of variables to select from the dataset.
        filters : Filter, optional
            Filters to apply to the dataset.

        Notes
        -----
        The kwargs are only a placeholder in case a subclass needs to pass additional
        arguments to the open method.

        """
        ds = super().open(variables=variables, filters=filters, **kwargs)
        # Set spatial dims, wavespectra won't have the dims so we allow it to pass
        if self.x_dim in ds.dims and self.y_dim in ds.dims:
            ds = ds.rio.set_spatial_dims(self.x_dim, self.y_dim)
        else:
            logger.debug(f"Spatial dims ({self.x_dim}, {self.y_dim}) not available")
        return ds.rio.write_crs(self.crs)


class SourceCRSDataset(SourceMixin, SourceDataset):
    """Source dataset with CRS support from an existing xarray Dataset object."""


class SourceCRSFile(SourceMixin, SourceFile):
    """Source dataset with CRS support from file to open with xarray.open_dataset."""


class SourceCRSIntake(SourceMixin, SourceIntake):
    """Source dataset with CRS support from intake catalog."""


class SourceCRSWavespectra(SourceMixin, SourceWavespectra):
    """Source dataset with CRS support from wavespectra reader.

    Note
    ----
    Default values are provided for crs, x_dim and y_dim fields as they are common
    to most wavespectra datasets.

    """

    crs: CRS_TYPES = Field(
        default=4326,
        description="Coordinate reference system of the source data",
    )
    x_dim: str = Field(
        default="lon",
        description="Name of the x dimension",
    )
    y_dim: str = Field(
        default="lat",
        description="Name of the y dimension",
    )


class SourceOceantide(SourceBase):
    """Geotiff source class."""

    model_type: Literal["oceantide"] = Field(
        default="oceantide",
        description="Model type discriminator",
    )
    reader: str = Field(
        description="Name of the oceantide reader to use, e.g., read_swan",
    )
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to the oceantide reader",
    )

    def _open(self) -> xr.Dataset:
        """This method needs to return an xarray Dataset object."""
        ds = getattr(oceantide, self.reader)(**self.kwargs)
        return ds


class SourceCRSOceantide(SourceMixin, SourceOceantide):
    """Source dataset with CRS support from intake catalog."""

    crs: CRS_TYPES = Field(
        default=4326,
        description="Coordinate reference system of the source data",
    )
    x_dim: str = Field(
        default="lon",
        description="Name of the x dimension",
    )
    y_dim: str = Field(
        default="lat",
        description="Name of the y dimension",
    )


class SourceTideConsPointCSV(SourceBase):
    """Tide point source class."""

    model_type: Literal["tide_cons_point_csv"] = Field(
        default="tide_cons_point_csv",
        description="Model type discriminator",
    )
    filename: str | Path = Field(description="Path to the csv stations file")
    acol: str = Field(
        default="amplitude",
        description="Name of the column containing the amplitude data",
    )
    pcol: str = Field(
        default="phase",
        description="Name of the column containing the phase data",
    )
    ccol: str = Field(
        default="constituent",
        description="Name of the column containing the constituent data",
    )
    read_csv_kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to pandas.read_csv",
    )

    @model_validator(mode="after")
    def validate_kwargs(self) -> "SourceTideConsPointCSV":
        """Validate the keyword arguments."""
        if "index_col" not in self.read_csv_kwargs:
            self.read_csv_kwargs["index_col"] = self.ccol
        return self

    def _to_oceantide(self, df):
        """."""
        df["h"] = df.amplitude * np.exp(1j * np.deg2rad(df.phase))
        df.index = df.index.str.upper()
        df.index.name = "con"
        ds = xr.Dataset.from_dataframe(df[["h"]])
        return ds.assign_coords(con=ds.con.astype("U4"))

    def _open_dataframe(self) -> pd.DataFrame:
        """Read the data from the csv file."""
        return pd.read_csv(self.filename, **self.read_csv_kwargs)

    def _open(self) -> xr.Dataset:
        """Interpolate the xyz data onto a regular grid."""
        df = self._open_dataframe()
        ds = self._to_oceantide(df)
        return ds
