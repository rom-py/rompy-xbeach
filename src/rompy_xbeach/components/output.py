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
    npoints: Optional[int] = Field(
        default=None,
        description="Number of output point locations",
    )

    @field_validator("meanvars", "globalvars", "pointvars", "npoints")
    @classmethod
    def check_variable_limits(cls, v, info):
        """Validate that variable lists don't exceed XBeach limits."""
        limits = {
            "meanvars": 15,
            "globalvars": 20,
            "pointvars": 50,
            "npoints": 50,
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

    def _build_var_dict(self, field_name: str) -> dict:
        """Build output variable dictionary with count and list.

        Parameters
        ----------
        field_name : str
            Name of the field (e.g., 'meanvars', 'globalvars', 'pointvars')

        Returns
        -------
        dict
            Dictionary with count key (e.g., 'nmeanvar') and variable list.

        """
        var_list = getattr(self, field_name, [])
        if not var_list:
            return {}
        count_key = f"n{field_name[:-1]}"
        return {count_key: len(var_list), field_name: [var.value for var in var_list]}

    @property
    def namelist(self):
        """Return the namelist representation of the output component."""
        _namelist = {}

        # Direct key-value pairs
        if self.outputformat is not None:
            _namelist["outputformat"] = self.outputformat
        if self.ncfilename is not None:
            _namelist["ncfilename"] = self.ncfilename
        if self.npoints is not None:
            _namelist["npoints"] = self.npoints

        # Variable lists
        for var_type in ["meanvars", "globalvars", "pointvars"]:
            _namelist.update(self._build_var_dict(var_type))

        return _namelist
