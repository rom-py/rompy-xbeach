"""XBEACH Rompy sources."""

import logging
from pathlib import Path
from typing import Literal
from pydantic import Field
import xarray as xr

from rompy.core.source import SourceBase


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


class XbeachSource(SourceBase):
    """Xbeach source class.

    This is a dummy class to demonstrate how to create a source type to use with the
    Data objects. The functionality in this class is already provided by the SourceFile
    class in rompy.core.source so this class is not necessary. It is included here as
    an example.

    """

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    uri: str | Path = Field(description="Path to the dataset")
    kwargs: dict = Field(
        default={},
        description="Keyword arguments to pass to xarray.open_dataset",
    )

    def __str__(self) -> str:
        """String representation for this source class."""
        return f"SourceFile(uri={self.uri})"

    def _open(self) -> xr.Dataset:
        """This method needs to return an xarray Dataset object."""
        return xr.open_dataset(self.uri, **self.kwargs)
