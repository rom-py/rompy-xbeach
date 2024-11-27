"""XBEACH Rompy config."""

import logging
from pathlib import Path
from typing import Literal, Optional, Union, Annotated
from pydantic import Field

from rompy.core.types import RompyBaseModel
from rompy.core.time import TimeRange
from rompy_xbeach.types import XBeachBaseConfig, WbcEnum
from rompy_xbeach.grid import RegularGrid
from rompy_xbeach.data import XBeachBathy
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent


# TODO: What xbeach version? Does it matter? Docker available?
# TODO: rugdetpth vs nrugdepth
# TODO: instat: example says 41, manual says stat, bichrom, ts_1, ts_2, jons, swan, vardens, reuse, ts_nonh, off, stat_table, jons_table
# TODO: break: example says 1, manual says roelvink1, baldock, roelvink2, roelvink_daly, janssen
# TODO: scheme: example says 1, manual says upwind_1, lax_wendroff, upwind_2, warmbeam
# TODO: leftwave: not in manual (lateralwave)
# TODO: rightwave: not in manual
# TODO: tidelen: not in manual, what is it?
# TODO: roh should be rho
# TODO: cf: not in manual
# TODO: paulrevere: example says 0, manual says land, sea


from rompy_xbeach.forcing import WindGrid, WindStation, TideGrid
from rompy_xbeach.boundary import (
    BoundaryStationSpectraJons,
    BoundaryStationParamJons,
    BoundaryStationSpectraJonstable,
    BoundaryStationParamJonstable,
    BoundaryStationSpectraSwan,
)

WindType = Annotated[
    Union[
        WindGrid,
        WindStation,
        TideGrid,
    ],
    Field(description="Wind input data", discriminator="model_type"),
]
WaveType = Annotated[
    Union[
        BoundaryStationSpectraJons,
        BoundaryStationParamJons,
        BoundaryStationSpectraJonstable,
        BoundaryStationParamJonstable,
        BoundaryStationSpectraSwan,
    ],
    Field(description="Wave input data", discriminator="model_type"),
]
TideType = Annotated[
    TideGrid,
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
            namelist.update(self.wave.get(staging_dir, grid, period))
        if self.wind is not None:
            namelist.update(self.wind.get(staging_dir, grid, period))
        if self.tide is not None:
            namelist.update(self.tide.get(staging_dir, grid, period))
        return namelist


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
        description="Initial water level (m)",
        ge=-5.0,
        le=5.0,
    )
    front: Literal["abs_1d", "abs_2d", "wall", "wlevel", "nonh_1d", "waveflume"] = Field(
        description="Switch for seaward flow boundary",
        default="abs_2d",
    )
    back: Literal["wall", "abs_1d", "abs_2d", "wlevel"] = Field(
        description="Switch for boundary at bay side",
        default="abs_2d",
    )
    left: Literal["neumann", "wall", "no_advec", "neumann_v", "abs_1d"] = Field(
        description="Switch for lateral boundary at ny+1",
        default="neumann",
    )
    right: Literal["neumann", "wall", "no_advec", "neumann_v", "abs_1d"] = Field(
        description="Switch for lateral boundary at 0",
        default="neumann",
    )
    rugdepth: float = Field(
        description="To be defined",
        ge=0,
        le=1,
    )
    tunits: str = Field(
        description=(
            "Time units in udunits format (seconds since 1970-01-01 00:00:00.00 +1:00)"
        ),
        default="s",
    )
    breaker: int = Field(
        description="Type of breaker formulation",
        alias="breaker",
    )
    scheme: int = Field(
        description="Numerical scheme for wave propagation",
    )
    order: Literal[1, 2] = Field(
        description=(
            "Switch for order of wave steering, first order wave steering (short wave "
            "energy only), second oder wave steering (bound long wave corresponding "
            "to short wave forcing is added)",
        ),
        default=2,
    )
    leftwave: Literal["neumann", "wavecrest", "cyclic"] = Field(
        description="Switch for lateral boundary at left",
        default="neumann",
    )
    rightwave: Literal["neumann", "wavecrest", "cyclic"] = Field(
        description="Switch for lateral boundary at left",
        default="neumann",
    )
    random: Literal[0, 1] = Field(
        description=(
            "Switch to enable random seed for instat = jons, swan or vardens "
            "boundary conditions"
        ),
        default=1,
    )
    hmin: float = Field(
        description="Threshold water depth above which stokes drift is included (m)",
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    wci: Literal[0, 1] = Field(
        description="Turns on wave-current interaction",
        default=0,
    )
    alpha: float = Field(
        description="Wave dissipation coefficient in roelvink formulation",
        default=1.38,
        ge=0.5,
        le=2.0,
    )
    delta: float = Field(
        description="Fraction of wave height to add to water depth",
        default=0.0,
        ge=0.0,
        le=1.0,
    )
    n: float = Field(
        description="Power in roelvink dissipation model",
        default=10.0,
        ge=5.0,
        le=20.0,
    )
    rho: float = Field(
        description="Density of water (kgm-3)",
        default=1025.0,
        ge=1000.0,
        le=1040.0,
    )
    g: float = Field(
        description="Gravitational acceleration (ms^-2)",
        default=9.81,
        ge=9.7,
        le=9.9,
    )
    thetamin: float = Field(
        description="Lower directional limit (angle w.r.t computational x-axis) (deg)",
        default=-90.0,
        ge=-360.0,
        le=360.0,
    )
    thetamax: float = Field(
        description="Higher directional limit (angle w.r.t computational x-axis) (deg)",
        default=90.0,
        ge=-360.0,
        le=360.0,
    )
    dtheta: float = Field(
        description="Directional resolution (deg)",
        default=10.0,
        ge=0.1,
        le=180.0,
    )
    beta: float = Field(
        description="Breaker slope coefficient in roller model",
        default=0.08,
        ge=0.05,
        le=0.3,
    )
    roller: Literal[0, 1] = Field(
        description="Switch to enable roller model",
        default=1,
    )
    gamma: float = Field(
        description="Breaker parameter in baldock or roelvink formulation",
        default=0.46,
        ge=0.4,
        le=0.9,
    )
    gammax: float = Field(
        description="Maximum ratio wave height to water depth",
        default=2.0,
        ge=0.4,
        le=5.0,
    )
    sedtrans: Literal[0, 1] = Field(
        description="Turn on sediment transport",
        default=1,
    )
    morfac: float = Field(
        description="Morphological acceleration factor",
        default=1.0,
        ge=0.0,
        le=1000.0,
    )
    morphology: Literal[0, 1] = Field(
        description="Turn on morphology",
        default=1,
    )
    cf: float = Field(
        description="Friction coefficient?",
        default=0.01,
    )
    paulrevere: Literal[0, 1] = Field(
        description="Specifies tide on sea and land or two sea points if tideloc = 2",
        default=0,
    )
    eps: float = Field(
        description="Threshold water depth above which cells are considered wet (m)",
        default=0.005,
        ge=0.001,
        le=0.1,
    )
    epsi: float = Field(
        description=(
            "Ratio of mean current to time varying current through offshore boundary"
        ),
        default=-1.0,
        ge=-1.0,
        le=0.2,
    )
    cfl: float = Field(
        description="Maximum courant-friedrichs-lewy number",
        default=0.7,
        ge=0.1,
        le=0.9,
    )
    umin: float = Field(
        description=(
            "Threshold velocity for upwind velocity detection and for vmag2 in "
            "equilibrium sediment concentration (m/s)"
        ),
        default=0.0,
        ge=0.0,
        le=0.2,
    )
    oldhu: Literal[0, 1] = Field(
        description="Switch to enable old hu calculation",
        default=0,
    )
    outputformat: Literal["fortran", "netcdf", "debug"] = Field(
        description="Output file format",
        default="fortran",
    )
    ncfilename: str = Field(
        description="Xbeach netcdf output file name",
    )
    tintm: float = Field(
        description="Interval time of mean, var, max, min output (s)",
    )
    # nmeanvar: int = Field(
    #     description="Number of mean, min, max, var output variables",
    #     default=0,
    #     ge=0,
    #     le=15,
    # )
    tstart: Optional[float] = Field(
        default=None,
        description=(
            "Start time of output, in morphological time (s) "
            "(XBeach default: 0.0)"
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
            "Interval time of point and runup gauge output (s) "
            "(XBeach default: 1.0)"
        ),
        gt=0.0,
    )

    def __call__(self, runtime) -> dict:
        """Callable where data and config are interfaced and CMD is rendered."""
        # Model times and staging dir from the ModelRun object
        period = runtime.period
        staging_dir = runtime.staging_dir

        # Initial namelist
        namelist = self.model_dump(
            exclude=["model_type", "template", "checkout", "grid", "bathy", "input"],
            exclude_none=True,
        )

        # Simulation time
        namelist["tstop"] = (period.end - period.start).total_seconds()

        # Generate the input data
        namelist.update(self.input.get(staging_dir, self.grid, period))

        # Bathy data interface
        # TODO: Make this consistent with the other input data
        namelist.update(self.bathy.namelist)
        __, __, depfile, grid = self.bathy.get(destdir=staging_dir, grid=self.grid)
        namelist.update(grid.namelist)
        namelist.update({"depfile": depfile.name})

        return namelist