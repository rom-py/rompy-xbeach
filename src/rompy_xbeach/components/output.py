"""XBeach output."""

from typing import Literal, Optional
from pydantic import Field
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
        if self.ncfilename is not None:
            _namelist["ncfilename"] = self.ncfilename
        return {**_namelist, **self.nmeanvar, **self.nglobalvar}
