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
from rompy_xbeach.data import XBeachBathy

from rompy_xbeach.components.output import Output
from rompy_xbeach.components.physics import Physics


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


# TODO: Remove the 'rugdepth' parameter? (confirm it with CSIRO)
# TODO: Remove the 'cf' parameter (confirm it with CSIRO)
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
    input: DataInterface = Field(
        description="Input data",
    )
    physics: Physics = Field(
        default_factory=Physics,
        description="Physical processes configuration",
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
    random: Optional[bool] = Field(
        default=None,
        description=(
            "Switch to enable random seed for instat = jons, swan or vardens "
            "boundary conditions (XBeach default: 1)",
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
    gammax: Optional[float] = Field(
        default=None,
        description="Maximum ratio wave height to water depth (XBeach default: 2.0)",
        ge=0.4,
        le=5.0,
    )
    morfac: Optional[float] = Field(
        default=None,
        description="Morphological acceleration factor (XBeach default: 1.0)",
        ge=0.0,
        le=1000.0,
    )
    cf: Optional[float] = Field(
        default=None,
        description="Friction coefficient?",
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
            "Specifies tide on sea and land or two sea points if tideloc = 2"
            "(XBeach default: land)"
        ),
    )
    _params = {}

    @model_validator(mode="after")
    def set_dtheta_if_surfbeat(self) -> "Config":
        """Placeholder validator for future dtheta logic."""
        return self

    @field_serializer("random")
    def serialize_random(self, value: Optional[bool]):
        """Serialise bool to int."""
        if value is None:
            return None
        return int(value)

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
                "output",
                "physics",
            ],
            exclude_none=True,
            by_alias=True,
        )

        # Simulation time
        self._params["tstop"] = (period.end - period.start).total_seconds()

        # tunits
        if self.tunits is None:
            self._params["tunits"] = f"seconds since {period.start:%Y-%m-%d %H:%M:%S}"

        # Generate the input data
        self._params.update(self.input.get(staging_dir, self.grid, period))

        # Bathy data interface
        # TODO: Make this consistent with the other input data
        self._params.update(self.bathy.params)
        __, __, depfile, grid = self.bathy.get(destdir=staging_dir, grid=self.grid)
        self._params.update(grid.params)
        self._params.update({"depfile": depfile.name})

        # Physics configuration
        self._params.update(self.physics.get(staging_dir))

        # Output configuration
        self._params.update(self.output.get(staging_dir))

        return self._params
