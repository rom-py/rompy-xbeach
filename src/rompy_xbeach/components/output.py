"""XBeach output."""

import logging
from pathlib import Path
from typing import Literal, Optional, Any
from pydantic import (
    Field,
    model_validator,
    field_validator,
    model_serializer,
    field_serializer,
)

from rompy.core.types import RompyBaseModel
from rompy.core.data import DataBlob
from rompy_xbeach.types import OutputVarsEnum

logger = logging.getLogger(__name__)

# TODO: Interface to allow fetching output times from files


class Output(RompyBaseModel):
    """XBeach output configuration.

    XBeach supports four different types of output: 1) instantaneous spatial output 2)
    time-averaged spatial output 3) fixed point output or 4) run-up gauge output. In
    principle any variable in XBeach can be outputted as long as it is part of the
    spaceparams structure defined in variables.f90 in the XBeach source code.

    This class simplifies output configuration by using list fields (`meanvars`,
    `globalvars`, `pointvars`, `points`, `rugauges`) to define variables and locations.
    The corresponding count keywords (`nmeanvar`, `nglobalvar`, `npointvar`, `npoints`,
    `nrugauge`) are automatically set based on list lengths when generating params.txt.
    Empty lists disable that output type.

    Instantaneous spatial output
    ----------------------------
    Instantaneous spatial output (`globalvars`) describes the instantaneous state of
    variables across the entire model domain at various points in time.

    Time-averaged spatial output
    ----------------------------
    Time-averaged spatial output (`meanvars`)describes the time-averaged state of
    variables across the entire model domain at various points in time. The user can
    define the averaging period using the `tintm` field.

    Fixed point output
    ------------------
    Fixed point output (`pointvars`) allows the user to select one or more locations for
    which a time series of data is stored. This output describes a time-series of one or
    more variables at one point in the model domain. To make use of this option, the
    user must specify the output locations using the `points` field describing the
    location coordinates given as the x-coordinate and y-coordinate and in world
    coordinates. XBeach will link the output location to the nearest computational point.

    Run-up gauge output
    -------------------
    Run-up gauge output describes a time-series of a number of variables at the (moving)
    waterline. In this case XBeach scans in an x-directional transect defined by the
    user for the location of the waterline. Output information is recorded for this
    moving point. This is particularly useful to keep track of run-up levels in
    cross-shore transects.

    The definition of run-up gauges is similar to the definition of fixed point output.
    The user needs to specify the run-up gauge locations using the `rugauges` field,
    describing the coordinates of the initial location of the run-up gauge. XBeach will
    subsequently link the initial run-up gauge location to the nearest computational
    cross-shore transect rather than just the nearest computational point.

    Run-up gauges share their selection of output variables with regular point output.
    However, in the case of run-up gauges, XBeach will automatically also include the
    variables xw, yw and zs to the point output variables, if these variables were not
    specified using the npointvar keyword in params.txt. Note that the user should refer
    to the pointvars.idx output file to check order of output variables for points and
    run-up gauges.

    See https://xbeach.readthedocs.io/en/latest/xbeach_manual.html#output-selection
    for more information.

    """

    model_type: Literal["output"] = Field(
        default="output",
        description="Model type discriminator",
    )
    outputformat: Optional[Literal["fortran", "netcdf", "debug"]] = Field(
        default="netcdf",
        description="Output file format (XBeach default: fortran)",
    )
    outputprecision: Optional[Literal["single", "double"]] = Field(
        default=None,
        description="Netcdf output precision (XBeach default: double)",
    )
    ncfilename: Optional[str] = Field(
        default=None,
        description="Xbeach netcdf output file name (XBeach default: xboutput.nc)",
    )
    meanvars: list[OutputVarsEnum] = Field(
        default=[],
        description=(
            "Mean output variables (sets `nmeanvar` and variable list in params.txt)"
        ),
    )
    globalvars: list[OutputVarsEnum] = Field(
        default=[],
        description=(
            "Global output variables (sets `nglobalvar` and variable list in params.txt)"
        ),
    )
    points: list[tuple[float, float]] = Field(
        default=[],
        description=(
            "Point locations as (x, y) coordinate pairs "
            "(sets `npoints` and point coordinates in params.txt)"
        ),
    )
    pointvars: list[OutputVarsEnum] = Field(
        default=[],
        description=(
            "Point output variables (sets `npointvar` and variable list in params.txt)"
        ),
    )
    rugauges: list[tuple[float, float]] = Field(
        default=[],
        description=(
            "Runup gauge locations as (x, y) coordinate pairs "
            "(sets `nrugauge` and gauge coordinates in params.txt)"
        ),
    )
    nrugdepth: Optional[int] = Field(
        default=None,
        description="Number of depths to compute runup in runup gauge",
        ge=1,
        le=10,
    )
    timings: Optional[bool] = Field(
        default=None,
        description="Switch enable progress output to screen (XBeach default: True)",
    )
    tstart: Optional[float] = Field(
        default=None,
        description="Start time (s) of output, in morphological time (XBeach default: 0)",
        ge=0.0,
    )
    tintc: Optional[float] = Field(
        default=None,
        description="Interval time (s) of cross section output (XBeach default: -123)",
        gt=0.0,
    )
    tintg: Optional[float] = Field(
        default=None,
        description=(
            "Interval time (s) of global output (XBeach default: 1), the first output "
            "is given at tstart"
        ),
        gt=0.0,
    )
    tintm: Optional[float] = Field(
        default=None,
        description=(
            "Interval time (s) of mean, var, max, min output (XBeach default: tstop - "
            "tstart), the first output is given at tstart+tintm and represents the "
            "average condition over the interval between tstart and tstart+tintm"
        ),
        gt=0.0,
    )
    tintp: Optional[float] = Field(
        default=None,
        description=(
            "Interval time (s) of point and runup gauge output (XBeach default: the "
            "value defined for tintg), the first output is given at tstart"
        ),
        gt=0.0,
    )
    tsglobal: Optional[DataBlob] = Field(
        default=None,
        description="File source containing timings of global output",
    )
    tsmean: Optional[DataBlob] = Field(
        default=None,
        description="File source containing timings of mean, max, min and var output",
    )
    tspoint: Optional[DataBlob] = Field(
        default=None,
        description="File source containing timings of point output",
    )

    @field_validator("meanvars", "globalvars", "pointvars")
    @classmethod
    def check_no_duplicate_variables(cls, v, info):
        """Validate that variable lists don't contain duplicates."""
        if len(v) != len(set(v)):
            raise ValueError(
                f"Duplicate variables found in {info.field_name}. "
                f"Each variable should only be specified once."
            )
        return v

    @field_validator("meanvars", "globalvars", "pointvars", "points", "rugauges")
    @classmethod
    def check_variable_limits(cls, v, info):
        """Validate that variable lists don't exceed XBeach limits."""
        limits = {
            "meanvars": 15,
            "globalvars": 20,
            "pointvars": 50,
            "points": 50,
            "rugauges": 50,
        }

        field_name = info.field_name
        max_vars = limits.get(field_name)

        if max_vars and len(v) > max_vars:
            logger.warning(
                f"More than {max_vars} {field_name} requested. XBeach only "
                f"supports up to {max_vars}. Beware of possible unexpected "
                f"results in the model."
            )
        return v

    @model_validator(mode="after")
    def validate_point_output_consistency(self) -> "Output":
        """Validate consistency between pointvars and point/rugauge locations."""

        # Check if pointvars are set but no locations defined
        if self.pointvars and not (self.points or self.rugauges):
            logger.warning(
                "Point output variables (pointvars) are defined, but no point "
                "locations (points) or runup gauge locations (rugauges) have been "
                "prescribed. Output will not be generated."
            )

        # Check if locations are set but no variables defined
        if (self.points or self.rugauges) and not self.pointvars:
            logger.warning(
                "Point locations (points) or runup gauge locations (rugauges) are "
                "defined, but no point output variables (pointvars) have been "
                "prescribed. No point/runup output will be generated."
            )

        return self

    @model_validator(mode="after")
    def fixed_or_file_times(self) -> "Output":
        """Validate that either fixed times or file times are specified."""
        # Check global output times
        if self.tintg is not None and self.tsglobal is not None:
            logger.warning(
                "Global times defined by both fixed (tintg) and file (tsglobal) times. "
                "The file-based times (tsglobal) will supersede the fixed interval."
            )

        # Check mean output times
        if self.tintm is not None and self.tsmean is not None:
            logger.warning(
                "Mean times defined by both fixed (tintm) and file (tsmean) times. "
                "The file-based times (tsmean) will supersede the fixed interval."
            )

        # Check point output times
        if self.tintp is not None and self.tspoint is not None:
            logger.warning(
                "Point times defined by both fixed (tintp) and file (tspoint) times. "
                "The file-based times (tspoint) will supersede the fixed interval."
            )

        return self

    @field_serializer("timings")
    def serialize_timings(self, value: Optional[bool]):
        """Serialise bool to int."""
        if value is None:
            return None
        return int(value)

    @model_serializer(mode="wrap")
    def _serialize_for_params(self, serializer: Any) -> dict:
        """Transforms variable lists into XBeach params format with count keys."""
        data = serializer(self)

        # Coordinate pairs (points, rugauges, etc.)
        coord_fields = {
            "points": "npoints",
            "rugauges": "nrugauge",
        }
        for field_name, count_key in coord_fields.items():
            if field_name in data and data[field_name]:
                coord_list = data.pop(field_name)
                data[count_key] = len(coord_list)
                data[field_name] = [f"{x} {y}" for x, y in coord_list]
            elif field_name in data:
                data.pop(field_name)

        # Variables definitions
        var_fields = {
            "meanvars": "nmeanvar",
            "globalvars": "nglobalvar",
            "pointvars": "npointvar",
        }
        for field_name, count_key in var_fields.items():
            if field_name in data and data[field_name]:
                var_list = data.pop(field_name)
                data[count_key] = len(var_list)
                data[field_name] = [var.value for var in var_list]
            elif field_name in data:
                data.pop(field_name)

        return data

    @property
    def params(self) -> dict:
        """Return the XBeach parameters for the output component."""
        return self.model_dump(exclude_none=True, exclude=["model_type"])

    def get(self, destdir: str | Path) -> dict:
        """Fetch external timing files if specified, and return the params dict."""
        params = self.params.copy()
        if self.tsglobal:
            params["tsglobal"] = self.tsglobal.get(destdir)
        if self.tsmean:
            params["tsmean"] = self.tsmean.get(destdir)
        if self.tspoint:
            params["tspoint"] = self.tspoint.get(destdir)
        return params
