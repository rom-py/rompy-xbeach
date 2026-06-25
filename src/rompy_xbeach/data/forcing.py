"""XBeach forcing file writers.

This module contains helper classes for writing XBeach forcing files:
- BaseFile: Abstract base class for time-series file writing
- Wind: Simple constant wind parameters
- WindFile: Time-varying wind file writer
- TideFile: Time-varying tide/water level file writer

These classes are used internally by the data interface classes in
rompy_xbeach.data.wind and rompy_xbeach.data.waterlevel.
"""

import logging
from abc import ABC
from typing import Optional, Union
from pathlib import Path
from pydantic import Field, model_validator
import numpy as np

from rompy.core.types import RompyBaseModel


logger = logging.getLogger(__name__)


class BaseFile(RompyBaseModel, ABC):
    """Base file definition."""

    filename: str = Field(
        description="File name",
    )
    tsec: list[float] = Field(
        description="Time (s)",
    )
    fmt: str = Field(
        default="%10.2f",
        description="Format string for writing the wind file",
    )
    _params = []

    @model_validator(mode="after")
    def same_sizes(self) -> "BaseFile":
        for param in self._params:
            param_size = len(getattr(self, param))
            if param_size != len(self):
                raise ValueError("All input parameters must be the same size")
        return self

    def __len__(self):
        return len(self.tsec)

    @property
    def data(self):
        cols = [self.tsec]
        for param in self._params:
            cols.append(getattr(self, param))
        return np.column_stack(cols)

    @property
    def params(self) -> dict:
        """XBeach parameters to write to the params.txt file."""
        pass

    def write(self, destdir: str | Path):
        """Write the wind file."""
        filename = Path(destdir) / self.filename
        np.savetxt(filename, self.data, fmt=self.fmt)
        return filename


class Wind(RompyBaseModel):
    """XBeach basic wind definition."""

    windv: float = Field(
        description="Wind velocity",
        ge=0.0,
    )
    windth: float = Field(
        description="Wind direction",
        ge=-180.0,
        le=360.0,
    )

    @property
    def params(self) -> dict:
        """Return the XBeach wind parameters."""
        return {
            "windv": self.windv,
            "windth": self.windth,
        }

    def write(self, destdir: Optional[Union[str | Path]] = None):
        """Write the wind file."""
        return self.params


class WindFile(BaseFile):
    """XBeach wind file definition."""

    windv: list[float] = Field(
        description="Wind velocity (m/s)",
    )
    windth: list[float] = Field(
        description="Wind direction (degrees)",
    )
    fmt: str = Field(
        default="%10.2f",
        description="Format string for writing the wind file",
    )
    _params = ["windv", "windth"]

    @property
    def params(self) -> dict:
        """Return the XBeach wind file parameters."""
        return {"windfile": self.filename}


class TideFile(BaseFile):
    """XBeach tide file definition."""

    tsec: list[float] = Field(
        description="Time (s)",
    )
    zs: list[float] = Field(
        description="Tide elevation (m)",
    )
    _params = ["zs"]

    @property
    def params(self) -> dict:
        """Return the XBeach tide file parameters."""
        return {
            "filename": self.filename,
        }
