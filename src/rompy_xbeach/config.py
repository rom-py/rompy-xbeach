"""XBEACH Rompy config."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union, Annotated
from pydantic import Field, field_serializer, model_validator

from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy.utils import load_entry_points

from rompy_xbeach.types import XBeachBaseConfig
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data.base import XBeachBathy

from rompy_xbeach.components.mpi import Mpi
from rompy_xbeach.components.output import Output
from rompy_xbeach.components.physics import Physics
from rompy_xbeach.components.sediment import Sediment
from rompy_xbeach.components.boundary.specification import (
    SpectralWaveBoundary,
    NonSpectralWaveBoundary,
    OffWaveBoundary,
    ReuseWaveBoundary,
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

WaveBoundaryType = Annotated[
    Union[
        SpectralWaveBoundary,
        NonSpectralWaveBoundary,
        OffWaveBoundary,
        ReuseWaveBoundary,
    ],
    Field(description="Wave boundary specification", discriminator="model_type"),
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


FrontType = Literal["abs_1d", "abs_2d", "wall", "wlevel", "nonh_1d", "waveflume"]
BackType = Literal["wall", "abs_1d", "abs_2d", "wlevel"]
LeftRightType = Literal["neumann", "wall", "no_advec", "neumann_v", "abs_1d"]
LateralWaveType = Literal["neumann", "wavecrest", "cyclic"]


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
        description="Input data (generate boundary conditions from data sources)",
    )
    wave_boundary: Optional[WaveBoundaryType] = Field(
        default=None,
        description="Wave boundary specification (manual specification or pre-existing files)",
    )
    physics: Physics = Field(
        default_factory=Physics,
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
    zs0: Optional[float] = Field(
        default=None,
        description="Initial water level (m) (XB default: 0.0)",
        ge=-5.0,
        le=5.0,
    )
    front: Optional[FrontType] = Field(
        default=None,
        description="Switch for seaward flow boundary (XBeach default: abs_2d)",
    )
    back: Optional[BackType] = Field(
        default=None,
        description="Switch for boundary at bay side (XBeach default: abs_2d)",
    )
    left: Optional[LeftRightType] = Field(
        default=None,
        description="Switch for lateral boundary at ny+1 (XBeach default: neumann)",
    )
    right: Optional[LeftRightType] = Field(
        default=None,
        description="Switch for lateral boundary at 0 (XBeach default: neumann)",
    )
    lateralwave: Optional[LateralWaveType] = Field(
        default=None,
        description="Switch for lateral boundary at left (XBeach default: neumann)",
    )
    rugdepth: Optional[float] = Field(
        default=None,
        description="To be defined",
        ge=0,
        le=1,
    )
    tunits: Optional[str] = Field(
        default=None,
        description=(
            "Time units in udunits format, if not provided it is constructed based on "
            "the simulation start time (XBeach default: s)"
        ),
        examples=["seconds since 1970-01-01 00:00:00.00 +1:00"],
    )
    order: Optional[Literal[1, 2]] = Field(
        default=None,
        description=(
            "Switch for order of wave steering, first order wave steering (short wave "
            "energy only), second oder wave steering (bound long wave corresponding "
            "to short wave forcing is added) (XBeach default: 2)",
        ),
    )
    thetamin: Optional[float] = Field(
        default=None,
        description=(
            "Lower directional limit (angle w.r.t computational x-axis) (deg) "
            "(XBeach default: -90.0)"
        ),
        ge=-360.0,
        le=360.0,
    )
    thetamax: Optional[float] = Field(
        default=None,
        description=(
            "Higher directional limit (angle w.r.t computational x-axis) (deg) "
            "(XBeach default: 90.0)"
        ),
        ge=-360.0,
        le=360.0,
    )
    dtheta: Optional[float] = Field(
        default=None,
        description="Directional resolution (deg) (XBeach default: 10.0)",
        ge=0.1,
        le=180.0,
    )
    cfl: Optional[float] = Field(
        default=None,
        description="Maximum courant-friedrichs-lewy number (XBeach default: 0.7)",
        ge=0.1,
        le=0.9,
    )
    # TODO: Make this part of the Tide object
    paulrevere: Optional[Literal["land", "sea"]] = Field(
        default=None,
        description=(
            "Specifies the sea or land boundary for tide boundary conditions "
            "(XBeach default: land)"
        ),
    )

    @model_validator(mode='after')
    def validate_wave_boundary(self):
        """Ensure only one wave boundary source is specified."""
        has_input_wave = self.input and self.input.wave
        has_wave_boundary = self.wave_boundary is not None
        
        if has_input_wave and has_wave_boundary:
            raise ValueError(
                "Cannot specify both input.wave and wave_boundary. "
                "Use input.wave to generate from data, or wave_boundary for manual specification."
            )
        
        return self

    @model_validator(mode="after")
    def set_dtheta_if_surfbeat(self) -> "Config":
        """Placeholder validator for future dtheta logic."""
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
                "wave_boundary",
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

        # Handle wave boundary conditions
        if self.input and self.input.wave:
            # Generate from data - returns WaveBoundary object
            logger.info("Generating wave boundary data from input.wave")
            wave_boundary = self.input.wave.get(staging_dir, self.grid, period)
            self._params.update(wave_boundary.get(staging_dir))
        elif self.wave_boundary:
            # Use manual specification
            logger.info("Using manual wave_boundary specification")
            self._params.update(self.wave_boundary.get(staging_dir))
        
        # Generate other input data (wind, tide)
        if self.input:
            if self.input.wind:
                logger.info("Generating wind forcing data")
                self._params.update(self.input.wind.get(staging_dir, self.grid, period))
            if self.input.tide:
                logger.info("Generating tide forcing data")
                self._params.update(self.input.tide.get(staging_dir, self.grid, period))

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

        return self._params
