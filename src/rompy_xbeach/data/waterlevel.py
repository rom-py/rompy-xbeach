"""XBeach water level and tide forcing."""

from typing import Literal, Optional
from pathlib import Path
import logging
import pandas as pd
from pydantic import Field, model_validator, field_validator

from rompy.core.time import TimeRange

from rompy_xbeach.source import SourceCRSOceantide, SourceTideConsPointCSV
from rompy_xbeach.data.base import BaseDataGrid, BaseDataStation, BaseDataPoint
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.forcing import TideFile


logger = logging.getLogger(__name__)


class ZS0Mixin:
    """Mixin class for zs0file parameters.

    Namelist
    --------
    - zs0file : str
        Name of tide boundary condition series.
    - tideloc : int
        Number of corner points on which a tide time series is specified.
    - tidelen : int
        Number of time steps in the tide time series.

    """

    id: Literal["tide"] = Field(
        default="tide",
        description="Identifier for the tide forcing",
    )
    tideloc: Literal[0, 1, 2, 4] = Field(
        default=1,
        description="Number of corner points on which a tide time series is specified",
    )
    freq: str = Field(
        default="1h",
        description="Frequency for generating the tide timeseries from constituents",
    )

    @field_validator("tideloc")
    @classmethod
    def raise_non_implemented(cls, v):
        """Only tideloc=1 is currently implemented."""
        if v != 1:
            raise NotImplementedError("Only tideloc=1 is currently implemented")
        return v


class WaterLevelBase(ZS0Mixin):
    """Mixin class for Water level forcing from timeseries data."""

    variables: list[str] = Field(
        default=["h"],
        description="Variables to extract from the dataset",
        min_length=1,
        max_length=1,
    )

    @property
    def h(self):
        return self.variables[0]

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Generate the tide file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the boundary bcfile data.

        """
        # Select the cons data at the grid location
        ds = super().get(destdir, grid, time=None)

        # Write the data
        filename = f"{self.id}-{time.start:%Y%m%dT%H%M%S}-{time.end:%Y%m%dT%H%M%S}.txt"
        times = ds.time.to_index().to_pydatetime()
        logger.debug(f"Creating wind file {filename} with times {times}")
        tf = TideFile(
            filename=filename,
            tsec=[(t - times[0]).total_seconds() for t in times],
            zs=ds[self.h].squeeze().values,
        )
        tf.write(destdir)

        return {"zs0file": filename, "tideloc": self.tideloc, "tidelen": ds.time.size}


class WaterLevelGrid(WaterLevelBase, BaseDataGrid):
    """Water level forcing from gridded timeseries."""

    model_type: Literal["water_level_grid"] = Field(
        default="water_level_grid",
        description="Model type discriminator",
    )


class WaterLevelStation(WaterLevelBase, BaseDataStation):
    """Water level forcing from station data."""

    model_type: Literal["water_level_station"] = Field(
        default="water_level_station",
        description="Model type discriminator",
    )


class WaterLevelPoint(WaterLevelBase, BaseDataPoint):
    """Water level forcing from point timeseries data."""

    model_type: Literal["water_level_point"] = Field(
        default="water_level_point",
        description="Model type discriminator",
    )


class TideConsBase(ZS0Mixin):
    """Mixin class to generate timeseries from cons using oceantide."""

    variables: list[str] = Field(
        default=["h"],
        description="Variables to extract from the dataset",
    )

    @model_validator(mode="after")
    def set_variables(self) -> "TideConsBase":
        """Variable names in an Oceantide dataset should be fixed."""
        logger.debug("Setting oceantide variables")
        if self.variables:
            logger.debug("Overwriting tide variables to the oceantide convention")
        self.variables = ["h"]
        return self

    def get(
        self, destdir: str | Path, grid: RegularGrid, time: Optional[TimeRange] = None
    ) -> dict:
        """Generate the tide file.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : RegularGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the boundary bcfile data.

        """
        # Select the cons data at the grid location
        ds = super().get(destdir, grid, time=None)

        # Calculate the surface elevation
        times = pd.date_range(time.start, time.end, freq=self.freq)
        ds = ds.tide.predict(times=times, components=["h"], time_chunk=None)

        # Write the data
        filename = f"{self.id}-{time.start:%Y%m%dT%H%M%S}-{time.end:%Y%m%dT%H%M%S}.txt"
        logger.debug(f"Creating wind file {filename} with times {times}")
        tf = TideFile(
            filename=filename,
            tsec=[(t - times[0]).total_seconds() for t in times],
            zs=ds.h.squeeze().values,
        )
        tf.write(destdir)

        return {"zs0file": filename, "tideloc": self.tideloc, "tidelen": ds.time.size}


class TideConsGrid(TideConsBase, BaseDataGrid):
    """Water level forcing from gridded tide cons processed with oceantide."""

    model_type: Literal["tide_cons_grid"] = Field(
        default="tide_cons_grid",
        description="Model type discriminator",
    )
    source: SourceCRSOceantide = Field(
        description="Source of the tide data",
    )


class TideConsPoint(TideConsBase, BaseDataPoint):
    """Water level forcing from single tide cons point processed with oceantide."""

    model_type: Literal["tide_cons_point"] = Field(
        default="tide_cons_point",
        description="Model type discriminator",
    )
    source: SourceTideConsPointCSV = Field(
        description="Source of the tide data",
    )
