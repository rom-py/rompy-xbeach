"""XBeach output."""

import logging
from typing import Literal, Optional, Any
from pydantic import Field, field_validator, model_serializer, field_serializer
from rompy.core.types import RompyBaseModel
from rompy_xbeach.types import OutputVarsEnum

logger = logging.getLogger(__name__)



class Output(RompyBaseModel):
    """XBeach output configuration.

    XBeach supports four different types of output: 1) instantaneous spatial output 2)
    time-averaged spatial output 3) fixed point output or 4) run-up gauge output. In
    principle any variable in XBeach can be outputted as long as it is part of the
    spaceparams structure defined in variables.f90 in the XBeach source code.

    The amount of output variables used for each type is determined by the keywords
    nglobalvar, nmeanvar, npoints and nrugauge. Each of these keywords takes a number
    indicating the number of parameters or locations that should be written to file. If
    any of the keywords is set to zero, the output type is effectively disabled. If
    nglovalvar is set to -1 then a standard set of output variables is used, being H,
    zs, zs0, zb, hh, u, v, ue, ve, urms, Fc, Fy, ccg, ceqsg, ceqbg, Susg, Svsg, E, R, D
    and DR. If nglobalvar is not set it defaults to -1. The lines in the params.txt file
    immediately following these keywords determine what parameters or locations are used.

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
    user must specify the number of output locations using the `npoints` field,
    immediately followed by one line per output location describing the location
    coordinates given as the x-coordinate and y-coordinate and in world coordinates.
    XBeach will link the output location to the nearest computational point.

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
        description="Mean output variables",
        default=[],
    )
    globalvars: list[OutputVarsEnum] = Field(
        description="Global output variables",
        default=[],
    )
    pointvars: list[OutputVarsEnum] = Field(
        description="Point output variables",
        default=[],
    )
    npoints: Optional[int] = Field(
        default=None,
        description="Number of output point locations",
    )
    nrugauge: Optional[int] = Field(
        default=None,
        description="Number of output runup gauge locations",
    )
    nrugdepth: Optional[int] = Field(
        default=None,
        description="Number of depths to compute runup in runup gauge",
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
        description="Interval time (s) of global output (XBeach default: 1)",
        gt=0.0,
    )
    tintm: Optional[float] = Field(
        default=None,
        description=(
            "Interval time (s) of mean, var, max, min output "
            "(XBeach default: tstop-tstart)"
        ),
        gt=0.0,
    )
    tintp: Optional[float] = Field(
        default=None,
        description=(
            "Interval time (s) of point and runup gauge output (XBeach default: 1)"
        ),
        gt=0.0,
    )
    tsglobal: Optional[str] = Field(
        default=None,
        description="Name of file containing timings of global output",
    )
    tsmean: Optional[str] = Field(
        default=None,
        description="Name of file containing timings of mean, max, min and var output",
    )
    tspoint: Optional[str] = Field(
        default=None,
        description="Name of file containing timings of point output",
    )

    @field_validator(
        "meanvars", "globalvars", "pointvars", "npoints", "nrugauge", "nrugdepth"
    )
    @classmethod
    def check_variable_limits(cls, v, info):
        """Validate that variable lists don't exceed XBeach limits."""
        limits = {
            "meanvars": 15,
            "globalvars": 20,
            "pointvars": 50,
            "npoints": 50,
            "nrugauge": 50,
            "nrugdepth": 10,
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

    @field_serializer("timings")
    def serialize_timings(self, value: Optional[bool]):
        """Serialise bool to int."""
        if value is None:
            return None
        return int(value)

    @model_serializer(mode="wrap")
    def _serialize_for_namelist(self, serializer: Any) -> dict:
        """Transforms variable lists into XBeach format with count keys."""
        data = serializer(self)
        var_fields = ["meanvars", "globalvars", "pointvars"]
        for field_name in var_fields:
            if field_name in data and data[field_name]:
                var_list = data.pop(field_name)
                # Add count key-value pair
                count_key = f"n{field_name[:-1]}"
                data[count_key] = len(var_list)
                # Add list with enum values
                data[field_name] = [var.value for var in var_list]
            elif field_name in data:
                # Remove empty lists
                data.pop(field_name)

        return data

    @property
    def namelist(self):
        """Return the namelist representation of the output component."""
        return self.model_dump(exclude_none=True, exclude=["model_type"])
