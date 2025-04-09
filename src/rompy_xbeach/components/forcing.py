"""XBeach wave boundary conditions."""

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
    def namelist(self) -> dict:
        """Namelist to write to the params.txt file."""
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
    def namelist(self) -> dict:
        """Return the wind namelist."""
        return {
            "windv": self.windv,
            "windth": self.windth,
        }

    def write(self, destdir: Optional[Union[str | Path]] = None):
        """Write the wind file."""
        return self.namelist


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
    def namelist(self) -> dict:
        """Return the wind file namelist."""
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
    def namelist(self) -> dict:
        """Return the wind file namelist."""
        return {
            "filename": self.filename,
        }
