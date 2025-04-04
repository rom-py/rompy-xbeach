"""XBEACH Rompy config."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union, Annotated
from pydantic import Field, field_serializer

from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy.utils import load_entry_points

from rompy_xbeach.types import XBeachBaseConfig
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data import XBeachBathy


logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


# TODO: Remove the 'rugdepth' parameter? (confirm it with CSIRO)
# TODO: Remove the 'cf' parameter (confirm it with CSIRO)
# TODO: Make 'random' part of the wave boundary conditions objects


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
    """SWAN forcing data interface.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.interface import DataInterface

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
        """Generate each input data and return the namelist params."""
        namelist = {}
        if self.wave is not None:
            logger.info("Generating wave boundary data")
            namelist.update(self.wave.get(staging_dir, grid, period))
        if self.wind is not None:
            logger.info("Generating wind forcing data")
            namelist.update(self.wind.get(staging_dir, grid, period))
        if self.tide is not None:
            logger.info("Generating tide forcing data")
            namelist.update(self.tide.get(staging_dir, grid, period))
        return namelist


BreakType = Literal["roelvink1", "baldock", "roelvink2", "roelvink_daly", "janssen"]
FrontType = Literal["abs_1d", "abs_2d", "wall", "wlevel", "nonh_1d", "waveflume"]
BackType = Literal["wall", "abs_1d", "abs_2d", "wlevel"]
LeftRightType = Literal["neumann", "wall", "no_advec", "neumann_v", "abs_1d"]
LateralWaveType = Literal["neumann", "wavecrest", "cyclic"]
SchemeType = Literal["upwind_1", "lax_wendroff", "upwind_2", "warmbeam"]
OutputFormatType = Literal["fortran", "netcdf", "debug"]


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
    breaktype: Optional[BreakType] = Field(
        default=None,
        description="Type of breaker formulation (XBeach default: roelvink_daly)",
        alias="break",
    )
    scheme: Optional[SchemeType] = Field(
        default=None,
        description="Numerical scheme for wave propagation (XBeach default: warmbeam)",
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
    hmin: Optional[float] = Field(
        default=None,
        description=(
            "Threshold water depth above which stokes drift is included (m) "
            "(XBeach default: 0.0)",
        ),
        ge=0.001,
        le=1.0,
    )
    wci: Optional[bool] = Field(
        default=None,
        description="Turns on wave-current interaction (XBeach default: 0)",
    )
    alpha: Optional[float] = Field(
        default=None,
        description=(
            "Wave dissipation coefficient in roelvink formulation"
            "(XBeach default: 1.38)"
        ),
        ge=0.5,
        le=2.0,
    )
    delta: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of wave height to add to water depth (XBeach default: 0.0)"
        ),
        ge=0.0,
        le=1.0,
    )
    n: Optional[float] = Field(
        default=None,
        description="Power in roelvink dissipation model (Xbeach default: 10.0)",
        ge=5.0,
        le=20.0,
    )
    rho: Optional[float] = Field(
        default=None,
        description="Density of water (kgm-3) (XBeach default: 1025.0)",
        ge=1000.0,
        le=1040.0,
    )
    g: Optional[float] = Field(
        default=None,
        description="Gravitational acceleration (ms^-2) (XBeach default: 9.81)",
        ge=9.7,
        le=9.9,
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
    beta: Optional[float] = Field(
        default=None,
        description="Breaker slope coefficient in roller model (XBeach default: 0.08)",
        ge=0.05,
        le=0.3,
    )
    roller: Optional[bool] = Field(
        default=None,
        description="Switch to enable roller model (XBeach default: 1)",
    )
    gamma: Optional[float] = Field(
        default=None,
        description=(
            "Breaker parameter in baldock or roelvink formulation "
            "(XBeach default: 0.46)"
        ),
        ge=0.4,
        le=0.9,
    )
    gammax: Optional[float] = Field(
        default=None,
        description="Maximum ratio wave height to water depth (XBeach default: 2.0)",
        ge=0.4,
        le=5.0,
    )
    sedtrans: Optional[bool] = Field(
        default=None,
        description="Turn on sediment transport (XBeach default: 1)",
    )
    morfac: Optional[float] = Field(
        default=None,
        description="Morphological acceleration factor (XBeach default: 1.0)",
        ge=0.0,
        le=1000.0,
    )
    morphology: Optional[bool] = Field(
        default=None,
        description="Turn on morphology (XBeach default: 1)",
    )
    cf: Optional[float] = Field(
        default=None,
        description="Friction coefficient?",
    )
    eps: Optional[float] = Field(
        default=None,
        description=(
            "Threshold water depth above which cells are considered wet (m) "
            "(XBeach default: 0.005)"
        ),
        ge=0.001,
        le=0.1,
    )
    epsi: Optional[float] = Field(
        default=None,
        description=(
            "Ratio of mean current to time varying current through offshore boundary "
            "(XBeach default: -1.0)"
        ),
        ge=-1.0,
        le=0.2,
    )
    cfl: Optional[float] = Field(
        default=None,
        description="Maximum courant-friedrichs-lewy number (XBeach default: 0.7)",
        ge=0.1,
        le=0.9,
    )
    umin: Optional[float] = Field(
        default=None,
        description=(
            "Threshold velocity for upwind velocity detection and for vmag2 in "
            "equilibrium sediment concentration (m/s) (XB default: 0.0)"
        ),
        ge=0.0,
        le=0.2,
    )
    oldhu: Optional[bool] = Field(
        default=None,
        description="Switch to enable old hu calculation (XBeach default: 0)",
    )
    outputformat: Optional[OutputFormatType] = Field(
        default=None,
        description="Output file format (XBeach default: fortran)",
    )
    ncfilename: Optional[str] = Field(
        default=None,
        description="Xbeach netcdf output file name (XBeach default: xboutput.nc)",
    )
    # TODO: Make the fields below part of the Output object
    # nmeanvar: int = Field(
    #     description="Number of mean, min, max, var output variables",
    #     default=0,
    #     ge=0,
    #     le=15,
    # )
    tstart: Optional[float] = Field(
        default=None,
        description=(
            "Start time of output, in morphological time (s) " "(XBeach default: 0.0)"
        ),
        ge=0.0,
    )
    tintc: Optional[float] = Field(
        default=None,
        description="Interval time of cross section output (s)",
        gt=0.0,
    )
    tintg: Optional[float] = Field(
        default=None,
        description="Interval time of global output (s) (XBeach default: 1.0)",
        gt=0.0,
    )
    tintm: Optional[float] = Field(
        default=None,
        description=(
            "Interval time of mean, var, max, min output (s) "
            "(XBeach default: tstop - tstart)"
        ),
        gt=0.0,
    )
    tintp: Optional[float] = Field(
        default=None,
        description=(
            "Interval time of point and runup gauge output (s) " "(XBeach default: 1.0)"
        ),
        gt=0.0,
    )
    # TODO: Make this part of the Tide object
    paulrevere: Optional[Literal["land", "sea"]] = Field(
        default=None,
        description=(
            "Specifies tide on sea and land or two sea points if tideloc = 2"
            "(XBeach default: land)"
        ),
    )
    _namelist = {}

    @field_serializer("random")
    def serialize_random(self, value: Optional[bool]):
        if value is None:
            return None
        return int(value)

    @field_serializer("wci")
    def serialize_wci(self, value: Optional[bool]):
        if value is None:
            return None
        return int(value)

    @field_serializer("roller")
    def serialize_roller(self, value: Optional[bool]):
        if value is None:
            return None
        return int(value)

    @field_serializer("sedtrans")
    def serialize_sedtrans(self, value: Optional[bool]):
        if value is None:
            return None
        return int(value)

    @field_serializer("morphology")
    def serialize_morphology(self, value: Optional[bool]):
        if value is None:
            return None
        return int(value)

    @field_serializer("oldhu")
    def serialize_oldhu(self, value: Optional[bool]):
        if value is None:
            return None
        return int(value)

    @property
    def namelist(self) -> dict:
        """Return the config namelist."""
        return self._namelist

    def __call__(self, runtime) -> dict:
        """Serialise the config to generate the params file."""
        # Model times and staging dir from the ModelRun object
        period = runtime.period
        staging_dir = runtime.staging_dir

        # Initial namelist
        self._namelist = self.model_dump(
            exclude=["model_type", "template", "checkout", "grid", "bathy", "input"],
            exclude_none=True,
            by_alias=True,
        )

        # Simulation time
        self._namelist["tstop"] = (period.end - period.start).total_seconds()

        # tunits
        if self.tunits is None:
            self._namelist["tunits"] = f"seconds since {period.start:%Y-%m-%d %H:%M:%S}"

        # Generate the input data
        self._namelist.update(self.input.get(staging_dir, self.grid, period))

        # Bathy data interface
        # TODO: Make this consistent with the other input data
        self._namelist.update(self.bathy.namelist)
        __, __, depfile, grid = self.bathy.get(destdir=staging_dir, grid=self.grid)
        self._namelist.update(grid.namelist)
        self._namelist.update({"depfile": depfile.name})

        return self._namelist
