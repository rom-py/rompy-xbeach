"""XBEACH Rompy config."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union, Annotated
from pydantic import Field, model_validator

from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy.utils import load_entry_points

from rompy_xbeach.types import XBeachBaseConfig
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.bathy import XBeachBathy

from rompy_xbeach.components.mpi import Mpi
from rompy_xbeach.components.output import Output
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.hotstart import Hotstart
from rompy_xbeach.components.boundary.parameters import (
    FlowBoundaryConditions,
    TideBoundaryConditions,
)


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


# TODO: Remove the 'rugdepth' parameter? (confirm it with CSIRO)
# TODO: Make 'random' part of the wave boundary conditions objects

# TODO: Split numerics into a separate component?


WindType = Annotated[
    Union[load_entry_points("xbeach.data", etype="wind")],
    Field(description="Wind input data", discriminator="model_type"),
]
WaveType = Annotated[
    Union[load_entry_points("xbeach.data", etype="wave")],
    Field(description="Wave input data", discriminator="model_type"),
]
TideType = Annotated[
    Union[load_entry_points("xbeach.data", etype="tide")],
    Field(description="Tide input data", discriminator="model_type"),
]


# TODO: Add the bathy here, need to change the return type of the get method
class DataInterface(RompyBaseModel):
    """XBeach forcing data interface.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy_xbeach.interface import DataInterface

    """

    model_type: Literal["data"] = Field(
        default="data", description="Model type discriminator"
    )
    # bathy: XBeachBathy = Field(
    #     description="Bathymetry data",
    # )
    wave: Optional[WaveType] = Field(default=None)
    wind: Optional[WindType] = Field(default=None)
    tide: Optional[TideType] = Field(default=None)

    def get(self, staging_dir: Path, grid: RegularGrid, period: TimeRange):
        """Generate each input data and return the XBeach params."""
        params = {}
        if self.wave is not None:
            logger.info("Generating wave boundary data")
            params.update(self.wave.get(staging_dir, grid, period))
        if self.wind is not None:
            logger.info("Generating wind forcing data")
            params.update(self.wind.get(staging_dir, grid, period))
        if self.tide is not None:
            logger.info("Generating tide forcing data")
            params.update(self.tide.get(staging_dir, grid, period))
        return params


class Config(XBeachBaseConfig):
    """Xbeach config class."""

    model_type: Literal["xbeach"] = Field(
        default="xbeach",
        description="Model type discriminator",
    )
    template: str = Field(
        default=str(HERE / "templates" / "base"),
        description="The model config template",
    )
    grid: RegularGrid = Field(
        description="The XBeach grid object",
    )
    bathy: XBeachBathy = Field(
        description="The XBeach bathymetry object",
    )
    input: Optional[DataInterface] = Field(
        default=None,
        description="Input data including wave, wind, and tide boundary conditions",
    )
    physics: Optional[Physics] = Field(
        default=None,
        description="Physical processes configuration",
    )
    sediment: Optional[Sediment] = Field(
        default_factory=Sediment,
        description="Sediment transport configuration",
    )
    mpi: Optional[Mpi] = Field(
        default_factory=Mpi,
        description="MPI parallelisation configuration",
    )
    output: Output = Field(
        default_factory=Output,
        description="Output configuration",
    )
    flow_boundary: Optional[FlowBoundaryConditions] = Field(
        default=None,
        description="Flow boundary conditions for shallow water equations",
    )
    tide_boundary: Optional[TideBoundaryConditions] = Field(
        default=None,
        description="Tide and surge boundary conditions",
    )
    hotstart: Optional[Union[bool, Hotstart]] = Field(
        default=None,
        description=(
            "Hotstart configuration. Set to True to enable hotstart with files "
            "already in run directory, or provide a Hotstart object to specify "
            "source directory and file number."
        ),
    )
    tunits: Optional[str] = Field(
        default=None,
        description=(
            "Time units in udunits format, if not provided it is constructed based on "
            "the simulation start time (XBeach default: s)"
        ),
        examples=["seconds since 1970-01-01 00:00:00.00 +1:00"],
    )

    @model_validator(mode="after")
    def set_dtheta_if_surfbeat(self) -> "Config":
        """Placeholder validator for future dtheta logic."""
        return self

    @model_validator(mode="after")
    def warn_wave_direction_params_without_swave(self) -> "Config":
        """Warn if wave directional parameters are set but swave is disabled.

        The wave directional grid parameters (thetamin, thetamax, dtheta, thetanaut)
        are only used when short waves are enabled (swave=1). Setting these when
        swave=0 has no effect.
        """
        # Collect directional params from input.wave
        dir_params = {}
        if self.input and self.input.wave:
            wave = self.input.wave
            for k in ["thetamin", "thetamax", "dtheta", "thetanaut"]:
                v = getattr(wave, k, None)
                if v is not None:
                    dir_params[k] = v

        if self.physics and self.physics.swave is False and dir_params:
            logger.warning(
                f"Wave directional parameters ({', '.join(dir_params.keys())}) are set "
                "but swave=0. These parameters only apply when short waves are enabled "
                "(swave=1) and will be ignored."
            )

        return self

    @property
    def params(self) -> dict:
        """Return the XBeach configuration parameters."""
        return self._params

    def __call__(self, runtime) -> dict:
        """Serialise the config to generate the params file."""

        # Model times and staging dir from the ModelRun object
        period = runtime.period
        staging_dir = runtime.staging_dir

        # Initial params dict
        self._params = self.model_dump(
            exclude=[
                "model_type",
                "template",
                "checkout",
                "grid",
                "bathy",
                "input",
                "flow_boundary",
                "tide_boundary",
                "hotstart",
                "output",
                "physics",
                "sediment",
                "mpi",
            ],
            exclude_none=True,
            by_alias=True,
        )

        # Simulation time
        self._params["tstop"] = (period.end - period.start).total_seconds()

        # tunits
        if self.tunits is None:
            self._params["tunits"] = f"seconds since {period.start:%Y-%m-%d %H:%M:%S}"

        # Handle input data (wave, wind, tide)
        if self.input:
            if self.input.wave:
                logger.info("Generating wave boundary data")
                self._params.update(self.input.wave.get(staging_dir, self.grid, period))
            if self.input.wind:
                logger.info("Generating wind forcing data")
                self._params.update(self.input.wind.get(staging_dir, self.grid, period))
            if self.input.tide:
                logger.info("Generating tide forcing data")
                self._params.update(self.input.tide.get(staging_dir, self.grid, period))
        # Update flow and tide boundary parameters
        if self.flow_boundary:
            self._params.update(self.flow_boundary.get(staging_dir))
        if self.tide_boundary:
            self._params.update(self.tide_boundary.get(staging_dir))

        # Hotstart configuration
        if self.hotstart is True:
            self._params["hotstart"] = 1
        elif isinstance(self.hotstart, Hotstart):
            self._params.update(self.hotstart.get(staging_dir))

        # Bathy data interface
        # TODO: Make this consistent with the other input data
        self._params.update(self.bathy.params)
        __, __, depfile, grid = self.bathy.get(destdir=staging_dir, grid=self.grid)
        self._params.update(grid.params)
        self._params.update({"depfile": depfile.name})

        # Physics configuration
        self._params.update(self.physics.get(staging_dir))

        # Sediment configuration
        self._params.update(self.sediment.get(staging_dir))

        # Output configuration
        self._params.update(self.output.get(staging_dir))

        # MPI configuration
        self._params.update(self.mpi.get(staging_dir))

        # XBeach expects booleans as 0/1. Normalise here, at the single point where
        # all component params are aggregated, so data interfaces (wave, wind, tide)
        # that don't go through the XBeachBaseModel serializer are also covered.
        self._params = {
            k: int(v) if isinstance(v, bool) else v for k, v in self._params.items()
        }

        return self._params
