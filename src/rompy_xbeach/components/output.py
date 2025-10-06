"""XBeach output."""

from typing import Literal, Optional
from pydantic import Field, field_validator
from rompy.core.types import RompyBaseModel
from rompy_xbeach.types import OutputVarsEnum


DEFAULT_MEANVARS = [
    OutputVarsEnum.H,
    OutputVarsEnum.THETAMEAN,
    OutputVarsEnum.HH,
    OutputVarsEnum.U,
    OutputVarsEnum.V,
    OutputVarsEnum.D,
    OutputVarsEnum.R,
    OutputVarsEnum.K,
    OutputVarsEnum.UE,
    OutputVarsEnum.VE,
    OutputVarsEnum.URMS,
    OutputVarsEnum.QB,
    OutputVarsEnum.ZB,
    OutputVarsEnum.ZS,
]


class Output(RompyBaseModel):
    """XBeach output configuration."""

    model_type: Literal["output"] = Field(
        default="output",
        description="Model type discriminator",
    )
    outputformat: Optional[Literal["fortran", "netcdf", "debug"]] = Field(
        default="netcdf",
        description="Output file format (XBeach default: fortran)",
    )
    ncfilename: Optional[str] = Field(
        default=None,
        description="Xbeach netcdf output file name (XBeach default: xboutput.nc)",
    )
    meanvars: list[OutputVarsEnum] = Field(
        description="Mean output variables",
        default=DEFAULT_MEANVARS,
    )
    globalvars: list[OutputVarsEnum] = Field(
        description="Global output variables",
        default=[],
    )
    pointvars: list[OutputVarsEnum] = Field(
        description="Point output variables",
        default=[],
    )

    @field_validator("meanvars")
    @classmethod
    def warning_if_more_than_15_meanvars(cls, v):
        if len(v) > 15:
            logger.warning(
                "More than 15 mean variables requested, XBeach only supports up to 15, "
                "beware of possible unexpected results in the model."
            )
        return v

    @field_validator("globalvars")
    @classmethod
    def warning_if_more_than_20_globalvars(cls, v):
        if len(v) > 20:
            logger.warning(
                "More than 20 global variables requested, XBeach only supports up to "
                "20, beware of possible unexpected results in the model."
            )
        return v

    @field_validator("pointvars")
    @classmethod
    def warning_if_more_than_50_pointvars(cls, v):
        if len(v) > 50:
            logger.warning(
                "More than 50 point variables requested, XBeach only supports up to "
                "50, beware of possible unexpected results in the model."
            )
        return v

    @property
    def nmeanvar(self):
        """Return the of mean output variables."""
        if len(self.meanvars) == 0:
            return {}
        return {
            "nmeanvar": len(self.meanvars),
            "meanvars": [var.value for var in self.meanvars]
        }

    @property
    def nglobalvar(self):
        """Return the of global output variables."""
        if len(self.globalvars) == 0:
            return {}
        return {
            "nglobalvar": len(self.globalvars),
            "globalvars": [var.value for var in self.globalvars]
        }

    @property
    def namelist(self):
        """Return the namelist representation of the output component."""
        _namelist = {}
        if self.outputformat is not None:
            _namelist.update({"outputformat": self.outputformat})
        if self.ncfilename is not None:
            _namelist.update({"ncfilename": self.ncfilename})
        return {**_namelist, **self.nmeanvar, **self.nglobalvar}
