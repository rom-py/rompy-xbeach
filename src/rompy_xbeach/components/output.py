"""XBeach output."""

import logging
from typing import Literal, Optional
from pydantic import Field, field_validator
from rompy.core.types import RompyBaseModel
from rompy_xbeach.types import OutputVarsEnum

logger = logging.getLogger(__name__)


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

    @field_validator("meanvars", "globalvars", "pointvars")
    @classmethod
    def check_variable_limits(cls, v, info):
        """Validate that variable lists don't exceed XBeach limits."""
        limits = {
            "meanvars": 15,
            "globalvars": 20,
            "pointvars": 50,
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
